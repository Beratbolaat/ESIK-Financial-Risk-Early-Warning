"""EŞİK: eş kat ve eş XGBoost ayarlarıyla keşifsel PCA karşılaştırması.

python esik_pca_karsilastirmasi.py --project-root PATH --output-dir NEW_PATH

Tarihsel test skorlanmaz. Ana model ve önceki deney kayıtları değiştirilmez.
Her dış katın mevcut, yalnız iç katlardan seçilmiş XGBoost ayarları kullanılır.
PCA için ayrı hiperparametre optimizasyonu yapılmaz. Bu, ön işleme ablation'ıdır.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ARMS = [
    ('raw_64', None, False),
    ('scaled_64', None, True),
    ('pca_90', .90, True),
    ('pca_95', .95, True),
    ('pca_99', .99, True),
    ('pca_full_64', 64, True),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root,out=args.project_root.resolve(),args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Yeni ve boş bir çıktı klasörü kullanın.')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    from esik_core import load_data, create_pipeline, score_metrics
    X,y,development,holdout,data=load_data()
    Xdev,ydev=X.loc[development].copy(),y.loc[development].copy()
    del X,y
    model_path=root/'models/esik_final_model_pipeline.pkl'
    original_model_sha=sha(model_path)
    stages=[]
    for i in range(5):
        split_path=root/f'experiments/optuna-v2/outer_{i}/splits.json'
        params_path=root/f'experiments/optuna-v2/outer_{i}/xgboost/best.json'
        s=json.loads(split_path.read_text(encoding='utf-8'))
        best=json.loads(params_path.read_text(encoding='utf-8'))
        a,b=set(s['train_ids']),set(s['valid_ids'])
        assert not a&b and a|b==set(development)
        assert not (a|b)&set(holdout)
        for inner in s['inner']:
            ia,ib=set(inner['train']),set(inner['valid'])
            assert not ia&ib and ia|ib==a
        stages.append({'fold':i,'train_ids':s['train_ids'],'valid_ids':s['valid_ids'],
                       'parameters':best['parameters'],'split_sha256':sha(split_path),
                       'parameters_sha256':sha(params_path)})
    protocol={
        'created_before_fit_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'PCA ön işleme adımının mevcut XGBoost yaklaşımındaki etkisini ölçen keşifsel karşılaştırma.',
        'arms':[{'name':n,'pca_n_components':c,'standardize':s} for n,c,s in ARMS],
        'imputation':'Training-fold median only',
        'scaling':'Training-fold StandardScaler only, no clipping or nonlinear transform',
        'pca':'Training-fold PCA, full SVD, whiten=False',
        'model_parameters':'Each fold uses its pre-existing inner-CV-selected XGBoost parameters; all arms match within fold.',
        'primary_metric':'Recall@Top10','secondary_metric':'Average Precision',
        'component_fraction_meaning':'Retained TRAINING feature variance; not retained predictive accuracy.',
        'holdout_scored':False,'external_api_called':False,
        'limitation':'PCA-specific model tuning was not performed. Existing development folds and prior data understanding are reused. Not new independent validation.',
        'source_sha256':sha(Path(__file__)), 'dataset_sha256':data['dataset_sha256'],
        'original_model_sha256':original_model_sha,
        'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','xgboost']},
        'stages':stages,
    }
    save(out/'pca_protocol.json',protocol)
    rows,predictions,variance_rows=[],[],[]
    # Prevent native BLAS libraries from oversubscribing the machine.
    with threadpool_limits(limits=2):
        for stage in stages:
            i,train,valid=stage['fold'],stage['train_ids'],stage['valid_ids']
            xtrain,xvalid=Xdev.loc[train],Xdev.loc[valid]
            ytrain,yvalid=ydev.loc[train],ydev.loc[valid]
            for arm,components,scale in ARMS:
                start=time.perf_counter()
                template=create_pipeline('xgboost',stage['parameters'],ytrain,threads=2)
                steps=[('imputer',SimpleImputer(strategy='median'))]
                if scale:
                    steps.append(('scaler',StandardScaler()))
                if components is not None:
                    steps.append(('pca',PCA(n_components=components,svd_solver='full',whiten=False)))
                steps.append(('model',template.named_steps['model']))
                pipe=Pipeline(steps)
                pipe.fit(xtrain,ytrain)
                scores=pipe.predict_proba(xvalid)[:,1]
                metrics=score_metrics(yvalid,scores)
                imputed=pipe.named_steps['imputer'].transform(xtrain)
                np.testing.assert_allclose(pipe.named_steps['imputer'].statistics_,np.nanmedian(xtrain.to_numpy(),axis=0))
                if scale:
                    scaler=pipe.named_steps['scaler']
                    np.testing.assert_allclose(scaler.mean_,imputed.mean(axis=0),rtol=1e-10,atol=1e-10)
                    assert int(scaler.n_samples_seen_)==len(train)
                count,variance=64,1.0
                if components is not None:
                    pca=pipe.named_steps['pca']
                    scaled=pipe.named_steps['scaler'].transform(imputed)
                    np.testing.assert_allclose(pca.mean_,scaled.mean(axis=0),atol=1e-10)
                    assert pca.n_samples_==len(train)
                    count=int(pca.n_components_)
                    variance=float(pca.explained_variance_ratio_.sum())
                    if isinstance(components,float):
                        assert variance>=components-1e-12
                    if arm=='pca_full_64':
                        for j,v in enumerate(pca.explained_variance_ratio_):
                            variance_rows.append({'fold':i,'component':j+1,'explained_variance_ratio':float(v)})
                row={'fold':i,'arm':arm,'input_dimensions':count,'retained_training_variance':variance,
                     'seconds':time.perf_counter()-start,'training_rows':len(train),
                     'preprocessing_fit_on_training_only':True,**metrics}
                rows.append(row)
                predictions.extend({'fold':i,'arm':arm,'row_id':int(r),'actual':int(t),'score':float(s)}
                                   for r,t,s in zip(valid,yvalid,scores))
                pd.DataFrame(rows).to_csv(out/'pca_fold_results.csv',index=False)
                print(json.dumps({k:row[k] for k in ['fold','arm','input_dimensions','recall_at_10','average_precision','seconds']}),flush=True)
    assert sha(model_path)==original_model_sha
    frame=pd.DataFrame(rows)
    assert len(frame)==30
    pred=pd.DataFrame(predictions)
    for name,g in pred.groupby('arm'):
        assert len(g)==len(development) and g.row_id.is_unique
        assert set(g.row_id)==set(development)
    pred.to_csv(out/'pca_oof_predictions.csv',index=False)
    pd.DataFrame(variance_rows).to_csv(out/'pca_explained_variance.csv',index=False)
    result={}
    baseline=frame[frame.arm=='raw_64'].set_index('fold')
    for name,_,_ in ARMS:
        g=frame[frame.arm==name].set_index('fold')
        delta=g.recall_at_10-baseline.recall_at_10
        result[name]={'mean_recall_at_10':float(g.recall_at_10.mean()),
                     'std_recall_at_10_population':float(g.recall_at_10.std(ddof=0)),
                     'mean_average_precision':float(g.average_precision.mean()),
                     'mean_roc_auc':float(g.roc_auc.mean()),
                     'mean_precision_at_10':float(g.precision_at_10.mean()),
                     'total_captured_across_folds':int(g.captured.sum()),
                     'total_reviewed_across_folds':int(g.reviewed.sum()),
                     'dimensions_by_fold':g.input_dimensions.tolist(),
                     'mean_retained_training_variance':float(g.retained_training_variance.mean()),
                     'delta_recall_percentage_points_vs_raw':float(delta.mean()*100),
                     'folds_better_than_raw':int((delta>1e-12).sum()),
                     'folds_equal_to_raw':int((delta.abs()<1e-12).sum()),
                     'folds_worse_than_raw':int((delta< -1e-12).sum()),
                     'sum_fit_predict_check_seconds':float(g.seconds.sum())}
    historical=pd.read_csv(root/'experiments/optuna-v2/nested_fold_metrics.csv')
    old=historical[historical.model=='xgboost'].set_index('outer_fold')
    np.testing.assert_allclose(baseline.recall_at_10,old.recall_at_10,atol=1e-12)
    np.testing.assert_allclose(baseline.average_precision,old.average_precision,atol=1e-10)
    report={'protocol_sha256':sha(out/'pca_protocol.json'),'complete_model_fits':30,
            'holdout_scored':False,'production_model_unchanged':True,
            'recorded_xgboost_baseline_reproduced':True,
            'preprocessing_training_only_checks_passed':True,
            'comparison_scope':protocol['limitation'],'results':result}
    save(out/'pca_summary.json',report)
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':
    main()
