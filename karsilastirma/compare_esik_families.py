"""Controlled ESIK family refits, complete releases and separate calibrations.

No new hyperparameter search or historical-test based parameter selection.
The original project is read-only. All outputs go to a new comparison directory.
"""
from pathlib import Path
import argparse, copy, json, sys, time, warnings
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
SOURCE, OUT = args.source.resolve(), args.output.resolve()
assert SOURCE != OUT and not OUT.is_relative_to(SOURCE)
sys.path.insert(0, str(SOURCE))
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss
from esik_core import (load_data, create_pipeline, score_metrics, summarize_folds,
    sha256, write_json, FEATURES, capacity_metrics)
from esik_calibration import raw_margin, fit_sigmoid, probability_from_margin
from esik_artifacts import generate_release, validate_artifacts

warnings.filterwarnings('ignore', message='X does not have valid feature names')
OUT.mkdir(parents=True, exist_ok=True)
EXP = SOURCE / 'experiments/optuna-v2'
old_summary = json.loads((EXP/'summary.json').read_text(encoding='utf-8'))
old_protocol = json.loads((EXP/'protocol.json').read_text(encoding='utf-8'))
old_metrics = pd.read_csv(EXP/'nested_fold_metrics.csv')
X, y, dev, test, data_report = load_data()
assert dev.tolist() == old_protocol['development_ids']
assert test.tolist() == old_protocol['holdout_ids']
assert data_report['dataset_sha256'] == old_protocol['data']['dataset_sha256']
source_names = ['models/esik_final_model_pipeline.pkl', 'outputs/artifact_manifest.json',
                'outputs/esik_model_metadata.json', 'outputs/calibration_v1/manifest.json',
                'app.py', 'esik_core.py', 'esik_optuna.py']
source_before = {name: sha256(SOURCE/name) for name in source_names}
selection_rule = ('Primary: recorded five-outer-fold mean Recall@Top10, higher is better; '
    'tie-break mean Average Precision. Family choice uses previously inspected development '
    'folds, so these folds are selection evidence, not a new independent test. '
    'Historical test metrics will be reported descriptively and will not tune parameters.')
protocol = dict(version='paired-family-comparison-v1', created_utc=datetime.now(timezone.utc).isoformat(),
    selection_rule=selection_rule, source_project=str(SOURCE), source_hashes=source_before,
    source_protocol_sha256=sha256(EXP/'protocol.json'), dataset=data_report,
    development_ids=dev.tolist(), historical_test_ids=test.tolist(),
    historical_test_status='Previously inspected, not fresh independent validation',
    families=['xgboost','lightgbm'], original_trials_per_family_per_search=40,
    new_hyperparameter_trials=0, outer_folds=5, calibration_inner_folds=3,
    final_calibration_folds=5, final_calibration_seed=20260914,
    calibration='Positive-slope sigmoid on raw margin; fit only on training-side OOF predictions',
    calibration_gate='Both pooled outer Brier and log loss must improve over that family raw output',
    prediction_uncertainty='Paired stratified bootstrap on reused historical records is descriptive only',
    builder_sha256=sha256(Path(__file__)))
protocol_file = OUT/'comparison_protocol.json'
if not protocol_file.exists():
    write_json(protocol_file, protocol)
else:
    protocol = json.loads(protocol_file.read_text(encoding='utf-8'))
    assert protocol['source_hashes'] == source_before
    assert protocol['builder_sha256'] == sha256(Path(__file__))

def probability_metrics(labels, p):
    return dict(brier=float(brier_score_loss(labels,p)),
                log_loss=float(log_loss(labels,p,labels=[0,1])),
                mean_prediction=float(np.mean(p)), observed_rate=float(np.mean(labels)))

def reliability(labels,p):
    labels, p = np.asarray(labels), np.asarray(p)
    results=[]
    for lo,hi in zip([0,.05,.1,.2,.4,.6,.8],[.05,.1,.2,.4,.6,.8,1]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        n=int(mask.sum())
        if not n: continue
        events=int(labels[mask].sum()); rate=events/n; z=1.959963984540054
        denominator=1+z*z/n; center=(rate+z*z/(2*n))/denominator
        half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denominator
        results.append(dict(lower=lo,upper=hi,n=n,events=events,
            mean_prediction=float(p[mask].mean()),observed_rate=rate,
            wilson95_lower=max(0.,center-half),wilson95_upper=min(1.,center+half)))
    return results

def oof_margins(ids, family, parameters, splits, cache):
    if cache.exists():
        frame=pd.read_csv(cache,index_col='row_id')
        assert set(frame.index)==set(ids)
        return frame.raw_margin.loc[ids]
    result=pd.Series(np.nan,index=ids,dtype=float)
    seen=set()
    for split in splits:
        tr,va=np.asarray(split['train']),np.asarray(split['valid'])
        assert not set(tr)&set(va) and not (set(tr)|set(va))&set(test)
        assert set(tr)|set(va)==set(ids) and not set(va)&seen
        model=create_pipeline(family,parameters,y.loc[tr],threads=2)
        model.fit(X.loc[tr],y.loc[tr])
        result.loc[va]=raw_margin(model,X.loc[va]); seen.update(va)
    assert result.notna().all() and seen==set(ids)
    result.rename('raw_margin').to_csv(cache,index_label='row_id')
    return result

records={}
for family in protocol['families']:
    started=time.perf_counter()
    destination=OUT/family
    checkpoint=destination/'comparison_evidence'
    checkpoint.mkdir(parents=True,exist_ok=True)
    per_fold=[]; predictions=[]
    print(f'{family}: starting five matched outer refits and separate calibration',flush=True)
    for k in range(5):
        original_fold=EXP/f'outer_{k}'
        split=json.loads((original_fold/'splits.json').read_text(encoding='utf-8'))
        best=json.loads((original_fold/family/'best.json').read_text(encoding='utf-8'))
        parameters=best['parameters']
        tr,va=np.asarray(split['train_ids']),np.asarray(split['valid_ids'])
        assert not set(tr)&set(va) and set(tr)|set(va)==set(dev)
        fold_file=checkpoint/f'outer_{k}.json'
        prediction_file=checkpoint/f'outer_{k}_predictions.csv'
        if fold_file.exists() and prediction_file.exists():
            fold_record=json.loads(fold_file.read_text(encoding='utf-8'))
            prediction=pd.read_csv(prediction_file)
        else:
            train_margins=oof_margins(tr,family,parameters,split['inner'],checkpoint/f'outer_{k}_training_oof.csv')
            calibration=fit_sigmoid(train_margins.to_numpy(),y.loc[tr].to_numpy())
            model=create_pipeline(family,parameters,y.loc[tr],threads=2)
            tick=time.perf_counter(); model.fit(X.loc[tr],y.loc[tr]); fit_seconds=time.perf_counter()-tick
            raw=raw_margin(model,X.loc[va]); uncal=expit(raw)
            calibrated=probability_from_margin(raw,calibration)
            metric=score_metrics(y.loc[va],uncal)
            expected=old_metrics[(old_metrics.outer_fold==k)&(old_metrics.model==family)].iloc[0]
            for name in ['recall_at_10','average_precision','roc_auc']:
                assert abs(metric[name]-float(expected[name]))<2e-8,(family,k,name,metric[name],expected[name])
            fold_record=dict(outer_fold=k,family=family,parameters=parameters,
                training_rows=len(tr),evaluation_rows=len(va),calibrator=calibration,
                metrics=metric,raw=probability_metrics(y.loc[va],uncal),
                calibrated=probability_metrics(y.loc[va],calibrated),
                fit_seconds=fit_seconds,fit_evaluation_disjoint=True,
                matches_original_outer_metrics=True)
            prediction=pd.DataFrame(dict(row_id=va,outer_fold=k,actual=y.loc[va].to_numpy(),
                raw_probability=uncal,raw_margin=raw,calibrated_probability=calibrated))
            write_json(fold_file,fold_record);prediction.to_csv(prediction_file,index=False)
        per_fold.append(fold_record); predictions.append(prediction)
        print(f'{family}: outer {k+1}/5 reproduced; Recall@10={fold_record["metrics"]["recall_at_10"]:.6f}',flush=True)
    outer=pd.concat(predictions,ignore_index=True)
    assert outer.row_id.is_unique and set(outer.row_id)==set(dev)
    outer_summary=summarize_folds([dict(r['metrics'],seconds=r['fit_seconds']) for r in per_fold])
    raw_summary=probability_metrics(outer.actual,outer.raw_probability)
    calibrated_summary=probability_metrics(outer.actual,outer.calibrated_probability)
    gate=calibrated_summary['brier']<raw_summary['brier'] and calibrated_summary['log_loss']<raw_summary['log_loss']
    write_json(checkpoint/'outer_summary_before_final_test.json',dict(family=family,
        outer=outer_summary,raw=raw_summary,calibrated=calibrated_summary,calibration_gate=bool(gate),
        note='Gate computed without historical test labels; outer folds previously inspected.'))

    # Parameters already fixed by the original equal-budget full-development search.
    family_result=next(r for r in old_summary['final_comparison'] if r['family']==family)
    pkl=destination/'models/esik_final_model_pipeline.pkl'
    release_protocol=copy.deepcopy(old_protocol)
    release_protocol.update(selection_rule=selection_rule,artifact_family=family,
        parameter_origin=f'original full_development/{family}/best.json',
        comparison_protocol_sha256=sha256(protocol_file),
        original_joint_search_winner=old_summary['winner']['family'])
    release_protocol_file=checkpoint/'release_protocol.json'
    if not release_protocol_file.exists(): write_json(release_protocol_file,release_protocol)
    if not (destination/'outputs/artifact_manifest.json').exists():
        model=create_pipeline(family,family_result['parameters'],y.loc[dev],threads=2)
        tick=time.perf_counter(); model.fit(X.loc[dev],y.loc[dev]); final_fit_seconds=time.perf_counter()-tick
        search=copy.deepcopy(old_summary)
        search['winner']=copy.deepcopy(family_result)
        search['protocol_sha256']=sha256(release_protocol_file)
        search['selection_rule']=selection_rule
        search['selection_scope']='Family fixed for paired comparison; per-family hyperparameters from recorded inner search'
        candidate=dict(pipeline=model,search_summary=search,development_ids=dev.tolist(),
                       holdout_ids=test.tolist(),feature_columns=FEATURES)
        print(f'{family}: building final scores, SHAP and model package',flush=True)
        metadata=generate_release(candidate,destination,f'family-comparison-{family}-v1',release_protocol_file)
        # Keep original mixed-procedure results visible as historical evidence, but the
        # runtime report must display the deployed family's outer validation key.
        metadata.update(validation_key=family,model_selection_rule=selection_rule,
            model_selection_status='Family-specific comparison build; previous joint inner winner was LightGBM',
            comparison_protocol_sha256=sha256(protocol_file),final_fit_seconds=final_fit_seconds,
            score_definition='Ham model olasılık tahmini; öncelik sıralaması için kullanılır. Kalibre tahmin ayrı bir katmandır.')
        package=joblib.load(pkl);package['metadata']=metadata;joblib.dump(package,pkl)
        write_json(destination/'outputs/esik_model_metadata.json',metadata)
        manifest=json.loads((destination/'outputs/artifact_manifest.json').read_text(encoding='utf-8'))
        manifest['model_sha256']=sha256(pkl)
        manifest['files']={name:sha256(destination/name) for name in manifest['files']}
        write_json(destination/'outputs/artifact_manifest.json',manifest)
    else:
        metadata=json.loads((destination/'outputs/esik_model_metadata.json').read_text(encoding='utf-8'))
    validate_artifacts(destination)
    model=joblib.load(pkl)['pipeline']
    if family=='lightgbm':
        original_model=joblib.load(SOURCE/'models/esik_final_model_pipeline.pkl')['pipeline']
        delta=float(np.max(np.abs(model.predict_proba(X.loc[test])[:,1]-original_model.predict_proba(X.loc[test])[:,1])))
        assert delta<1e-10,delta
        print(f'LightGBM final refit matches existing production scores: max delta {delta}',flush=True)

    cal_folder=destination/'outputs/calibration_v1'
    cal_folder.mkdir(parents=True,exist_ok=True)
    # Exactly the same final calibration splits as the existing LightGBM layer.
    final_splits=json.loads((SOURCE/'outputs/calibration_v1/final_calibration_splits.json').read_text(encoding='utf-8'))
    final_margins=oof_margins(dev,family,family_result['parameters'],final_splits,checkpoint/'final_calibration_oof.csv')
    final_cal=fit_sigmoid(final_margins.to_numpy(),y.loc[dev].to_numpy())
    scored=pd.read_csv(destination/'outputs/esik_scored_test_companies.csv')
    ids=scored.COMPANY_ID.str.removeprefix('ESIK-').astype(int).to_numpy()
    margin=raw_margin(model,X.loc[ids]); prob=probability_from_margin(margin,final_cal)
    assert np.max(np.abs(expit(margin)-scored.RISK_SCORE.to_numpy()))<2e-7
    assert np.all(np.diff(prob)<=1e-10), 'Calibration changed ranking'
    company=pd.DataFrame(dict(COMPANY_ID=scored.COMPANY_ID,RAW_MARGIN=margin,CALIBRATED_PROBABILITY=prob))
    labels=y.loc[ids].to_numpy()
    historical=dict(status='Previously inspected descriptive benchmark, not fresh external validation',
        raw=probability_metrics(labels,scored.RISK_SCORE),calibrated=probability_metrics(labels,prob))
    cal_report=dict(protocol_version='paired-family-calibration-v1',family=family,model_sha256=sha256(pkl),
        scope='Family-specific post-selection calibration diagnostic on previously inspected development folds',
        deployment_gate=dict(passed=bool(gate),criteria=protocol['calibration_gate']),
        outer_raw=raw_summary,outer_calibrated=calibrated_summary,
        outer_reliability=dict(raw=reliability(outer.actual,outer.raw_probability),
                              calibrated=reliability(outer.actual,outer.calibrated_probability)),
        outer_folds=per_fold,final_calibrator=final_cal,historical_test=historical,
        original_ranking_preserved=True,top10_original=capacity_metrics(labels,scored.RISK_SCORE),
        top10_probability_mean=float(prob[:117].mean()),top10_expected_count=float(prob[:117].sum()),
        final_model_sha256_unchanged=True,
        limitations=['Outer folds previously inspected; not a fresh independent validation.',
            'Per-fold hyperparameters were selected on their training labels before OOF calibration.',
            'Final mapping uses OOF predictions with full-development-selected parameters.',
            'No contemporary local external validation; individual probabilities are exploratory.'])
    for name,value in [('report.json',cal_report),('report-before-historical-test.json',dict(
        family=family,outer_raw=raw_summary,outer_calibrated=calibrated_summary,
        deployment_gate=cal_report['deployment_gate'],scope=cal_report['scope'])),
        ('calibrator.json',final_cal),('final_calibration_splits.json',final_splits),
        ('fold_results.json',per_fold),('protocol.json',dict(protocol,family=family,model_sha256=sha256(pkl)))]:
        write_json(cal_folder/name,value)
    outer.to_csv(cal_folder/'outer_diagnostic_predictions.csv',index=False)
    final_margins.rename('raw_margin').to_csv(cal_folder/'final_calibration_oof.csv',index_label='row_id')
    company.to_csv(cal_folder/'company_probabilities.csv',index=False)
    reference={f:dict(median=float(X.loc[dev,f].median()),q05=float(X.loc[dev,f].quantile(.05)),
        q95=float(X.loc[dev,f].quantile(.95)),observed_count=int(X.loc[dev,f].notna().sum())) for f in FEATURES}
    write_json(cal_folder/'development_reference.json',reference)
    write_json(cal_folder/'manifest.json',dict(model_sha256=sha256(pkl),files={
        file.name:sha256(file) for file in cal_folder.iterdir() if file.is_file() and file.name!='manifest.json'}))
    historical_ranking=score_metrics(labels,scored.RISK_SCORE)
    records[family]=dict(outer=outer_summary,final_inner=family_result['summary'],
        historical_ranking=historical_ranking,calibration=cal_report,
        model_bytes=pkl.stat().st_size,total_seconds=time.perf_counter()-started)
    print(f'{family}: complete. Historical Recall@10={historical_ranking["recall_at_10"]:.6f}; '
          f'calibration gate={gate}; calibrated Brier={historical["calibrated"]["brier"]:.6f}',flush=True)

# Pair observations by ID; never compare rankings from different samples.
frames={f:pd.read_csv(OUT/f/'outputs/esik_scored_test_companies.csv').set_index('COMPANY_ID') for f in records}
paired=pd.DataFrame(index=sorted(frames['xgboost'].index))
paired.index.name='COMPANY_ID'
for f in records:
    frame=frames[f].loc[paired.index]
    paired[f'{f}_score']=frame.RISK_SCORE
    paired[f'{f}_rank']=frame.RISK_RANK
    p=pd.read_csv(OUT/f/'outputs/calibration_v1/company_probabilities.csv').set_index('COMPANY_ID')
    paired[f'{f}_probability']=p.loc[paired.index,'CALIBRATED_PROBABILITY']
    if 'actual' not in paired: paired['actual']=frame.ACTUAL_BANKRUPT
    else: assert np.array_equal(paired.actual,frame.ACTUAL_BANKRUPT)
    paired[f'{f}_in_top10']=frame.RISK_RANK<=117
paired['review_difference']=np.select([
    paired.xgboost_in_top10&paired.lightgbm_in_top10,
    paired.xgboost_in_top10&~paired.lightgbm_in_top10,
    ~paired.xgboost_in_top10&paired.lightgbm_in_top10],
    ['both','xgboost_only','lightgbm_only'],default='neither')
paired.to_csv(OUT/'paired_historical_company_results.csv')
disagreement=paired.groupby(['actual','review_difference']).size().rename('companies').reset_index()
disagreement.to_csv(OUT/'paired_list_disagreements.csv',index=False)

# Descriptive sampling sensitivity of the *fixed* two fitted models; no tuning.
rng=np.random.default_rng(20260915)
labels=paired.actual.to_numpy(); pos=np.flatnonzero(labels==1); neg=np.flatnonzero(labels==0)
sx=paired.xgboost_score.to_numpy(); sl=paired.lightgbm_score.to_numpy()
differences=[]
for _ in range(3000):
    ix=np.concatenate([rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)])
    differences.append(capacity_metrics(labels[ix],sx[ix])['recall']-capacity_metrics(labels[ix],sl[ix])['recall'])
bootstrap=dict(method='Paired stratified percentile bootstrap, 3000 resamples; fixed models; reused historical sample',
    recall_difference_xgb_minus_lgb=records['xgboost']['historical_ranking']['recall_at_10']-records['lightgbm']['historical_ranking']['recall_at_10'],
    interval_95=[float(x) for x in np.quantile(differences,[.025,.975])],
    limitation='Descriptive sampling sensitivity only; does not remove prior model-selection bias or establish performance on new populations.')
fold_comparison=[]
for k in range(5):
    a=json.loads((OUT/'xgboost/comparison_evidence'/f'outer_{k}.json').read_text(encoding='utf-8'))
    b=json.loads((OUT/'lightgbm/comparison_evidence'/f'outer_{k}.json').read_text(encoding='utf-8'))
    fold_comparison.append(dict(fold=k+1,xgboost=a['metrics']['recall_at_10'],lightgbm=b['metrics']['recall_at_10'],
        difference=a['metrics']['recall_at_10']-b['metrics']['recall_at_10']))
pd.DataFrame(fold_comparison).to_csv(OUT/'paired_outer_folds.csv',index=False)
preferred=max(records,key=lambda f:(records[f]['outer']['mean_recall_at_10'],records[f]['outer']['mean_average_precision']))
report=dict(protocol=protocol,models=records,paired_outer_folds=fold_comparison,
    historical_bootstrap=bootstrap,disagreement=disagreement.to_dict('records'),
    preference_by_predeclared_primary_rule=preferred,
    conclusion_scope='Preference on reused outer mean Recall@Top10, not proof of universal or statistically established superiority.',
    source_project_unchanged=all(sha256(SOURCE/name)==digest for name,digest in source_before.items()))
assert report['source_project_unchanged']
write_json(OUT/'comparison_report.json',report)
print(json.dumps(dict(preference=preferred,models={f:dict(outer_recall=r['outer']['mean_recall_at_10'],
    outer_ap=r['outer']['mean_average_precision'],test_recall=r['historical_ranking']['recall_at_10'],
    captured=r['historical_ranking']['captured'],calibrated_test=r['calibration']['historical_test']['calibrated'])
    for f,r in records.items()},bootstrap=bootstrap,source_unchanged=True),ensure_ascii=False,indent=2),flush=True)
