"""EŞİK için sınırlı, keşifsel özellik mühendisliği karşılaştırması.

Kullanım:
  python esik_feature_engineering.py --project-root /path/to/esik-profesyonel --output-dir /path/to/fe-results

Ana model ve tarihsel test skorları değiştirilmez. 64 ham oran ile 68 girdi,
aynı beş geliştirme katında ve sabit XGBoost ayarlarıyla karşılaştırılır.
Sonuçlar yeni bağımsız test veya nested CV sonucu değildir.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

RAW = [f'Attr{i}' for i in range(1,65)]
ADDED = ['FE_missing_fraction','FE_short_term_debt_share',
         'FE_loss_and_liquidity_shortfall','FE_loss_and_sales_contraction']

class FinancialFeatures(TransformerMixin, BaseEstimator):
    """Hedefi ve diğer satırları kullanmayan finansal dönüşümler."""
    def fit(self, X, y=None):
        if list(X.columns) != RAW:
            raise ValueError('Girdiler Attr1..Attr64 sırasıyla sunulmalı.')
        self.feature_names_in_ = np.array(RAW,dtype=object)
        self.n_features_in_ = len(RAW)
        return self

    def transform(self, X):
        z = X.loc[:,RAW].copy()
        z[ADDED[0]] = z.isna().mean(axis=1)
        # Attr51: kısa vadeli yükümlülük / varlık, Attr2: toplam yükümlülük / varlık.
        # Sıfır, negatif veya çok küçük payda tanımsız kabul edilir.
        denominator = z['Attr2'].where(z['Attr2'] > 1e-9)
        z[ADDED[1]] = (z['Attr51']/denominator).replace([np.inf,-np.inf],np.nan)
        joint = ((z['Attr1']<0)&(z['Attr4']<1)).astype(float)
        z[ADDED[2]] = joint.where(z[['Attr1','Attr4']].notna().all(axis=1))
        contraction = ((z['Attr1']<0)&(z['Attr21']<1)).astype(float)
        z[ADDED[3]] = contraction.where(z[['Attr1','Attr21']].notna().all(axis=1))
        return z

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW+ADDED,dtype=object)

def boundary_checks():
    a = pd.DataFrame(2.0,index=range(3),columns=RAW)
    a.loc[0,['Attr1','Attr4','Attr21','Attr2','Attr51']] = [-.1,.8,.9,.5,.2]
    a.loc[1,['Attr1','Attr2']] = [np.nan,0]
    t = FinancialFeatures().fit(a)
    b = t.transform(a)
    assert b.loc[0,ADDED[1]] == .4
    assert b.loc[0,ADDED[2]] == b.loc[0,ADDED[3]] == 1
    assert np.isnan(b.loc[1,ADDED[1]]) and np.isnan(b.loc[1,ADDED[2]])
    assert b.loc[1,ADDED[0]] == 1/64
    assert b.shape[1] == 68
    pd.testing.assert_frame_equal(t.transform(a.iloc[[0]]),b.iloc[[0]])

def main():
    args=argparse.ArgumentParser()
    args.add_argument('--project-root',type=Path,required=True)
    args.add_argument('--output-dir',type=Path,required=True)
    cfg=args.parse_args()
    root,out=cfg.project_root.resolve(),cfg.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'fe_results.json').exists() or (out/'fe_protocol.json').exists():
        raise FileExistsError('Sonucu/protokolü korumak için yeni bir çıktı klasörü kullanın.')
    sys.path.insert(0,str(root))
    from esik_core import load_data, BASELINE_PARAMS, create_pipeline
    boundary_checks()
    X,y,development,holdout,_=load_data()
    # Bütün fit ve metrik işlemleri bu iki geliştirme nesnesi ile yapılır.
    Xdev,ydev=X.loc[development].copy(),y.loc[development].copy()
    del X,y
    splits=[]
    for i in range(5):
        s=json.loads((root/f'experiments/optuna-v2/outer_{i}/splits.json').read_text(encoding='utf-8'))
        a,b=set(s['train_ids']),set(s['valid_ids'])
        assert not a&b and a|b==set(development)
        assert not (a|b)&set(holdout)
        splits.append((s['train_ids'],s['valid_ids']))
    protocol={
      'purpose':'Keşifsel geliştirme verisi ablation deneyi. Üretim modeli seçimi yapılmaz.',
      'predefined_arms':['raw_64','engineered_68'], 'added_features':ADDED,
      'fixed_model_family':'xgboost', 'fixed_parameters':BASELINE_PARAMS['xgboost'],
      'fold_count':5,'fold_source':'optuna-v2/outer_0..4/splits.json',
      'primary_metric':'Recall at top 10 percent','secondary_metric':'Average Precision',
      'holdout_scored':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'versions':{p:importlib.metadata.version(p) for p in ['scikit-learn','numpy','pandas','xgboost']},
      'interpretation_limit':'Özellikler önceki EDA bulgularından esinlenir. Yeni bağımsız doğrulama veya nested CV değildir.',
    }
    (out/'fe_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for i,(train,valid) in enumerate(splits):
        for arm in protocol['predefined_arms']:
            start=time.perf_counter()
            baseline=create_pipeline('xgboost',BASELINE_PARAMS['xgboost'],ydev.loc[train],threads=2)
            if arm=='engineered_68':
                pipeline=Pipeline([('features',FinancialFeatures()),
                                   ('imputer',SimpleImputer(strategy='median')),
                                   ('model',baseline.named_steps['model'])])
            else:
                pipeline=baseline
            pipeline.fit(Xdev.loc[train],ydev.loc[train])
            scores=pipeline.predict_proba(Xdev.loc[valid])[:,1]
            labels=ydev.loc[valid].to_numpy()
            k=math.ceil(len(valid)*.1)
            caught=int(labels[np.argsort(-scores,kind='stable')[:k]].sum())
            row={'fold':i,'arm':arm,'features':64 if arm=='raw_64' else 68,
                 'reviewed':k,'captured':caught,'positives':int(labels.sum()),
                 'recall_at_10':caught/int(labels.sum()),'precision_at_10':caught/k,
                 'average_precision':average_precision_score(labels,scores),
                 'seconds':time.perf_counter()-start}
            rows.append(row)
            print(json.dumps(row),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'fe_fold_results.csv',index=False)
    summary=frame.groupby('arm')[['recall_at_10','precision_at_10','average_precision']].mean()
    result={'experiment_scope':protocol['interpretation_limit'],
            'production_model_changed':False,'holdout_scored':False,'boundary_checks_passed':True,
            'mean_metrics':summary.to_dict(orient='index'),
            'recall_delta_percentage_points':float((summary.loc['engineered_68','recall_at_10']-summary.loc['raw_64','recall_at_10'])*100),
            'ap_delta':float(summary.loc['engineered_68','average_precision']-summary.loc['raw_64','average_precision'])}
    (out/'fe_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
