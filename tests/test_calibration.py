import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from esik_calibration import fit_sigmoid, probability_from_margin, load_probability_layer
from esik_assessment import probability_answer, evidence_table
from esik_artifacts import validate_artifacts
from esik_company_query import CompanyQuery
ROOT=Path(__file__).resolve().parents[1]


def test_sigmoid_recovers_known_probability_scale():
    rng=np.random.default_rng(41);x=rng.normal(0,2,30000)
    truth=1/(1+np.exp(-(.8*x-1.4)));y=rng.binomial(1,truth)
    calibration=fit_sigmoid(x,y);pred=probability_from_margin(x,calibration)
    assert np.mean(np.abs(pred-truth))<.015
    assert np.all(np.diff(probability_from_margin(np.linspace(-12,12,300),calibration))>0)


@pytest.mark.parametrize('scores,labels',[([1,1],[0,1]),([1,2],[1,1]),([1,float('nan')],[0,1])])
def test_calibration_rejects_invalid_training(scores,labels):
    with pytest.raises(ValueError):fit_sigmoid(scores,labels)


def test_probability_release_matches_original_model_and_outer_evaluation():
    digest=validate_artifacts(ROOT)['model_sha256'];layer=load_probability_layer(ROOT,digest)
    assert layer is not None
    r=layer['report'];assert r['outer_calibrated']['brier']<r['outer_raw']['brier']
    assert r['outer_calibrated']['log_loss']<r['outer_raw']['log_loss']
    assert r['final_model_sha256_unchanged'] and r['original_ranking_preserved']
    assert r['top10_original']['captured']==67
    with pytest.raises(ValueError):load_probability_layer(ROOT,'wrong-model')


def test_protocol_excludes_historical_test_and_covers_each_calibration_row_once():
    folder=ROOT/'outputs/calibration_v1';protocol=json.loads((folder/'protocol.json').read_text())
    train=set(protocol['development_ids']);test=set(protocol['historical_test_ids'])
    assert not train&test
    splits=json.loads((folder/'final_calibration_splits.json').read_text())
    seen=[]
    for fold in splits:
        a,b=set(fold['train']),set(fold['valid']);assert not a&b and a|b==train and not (a|b)&test
        seen+=fold['valid']
    assert len(seen)==len(set(seen))==4680 and set(seen)==train
    oof=pd.read_csv(folder/'outer_diagnostic_predictions.csv')
    assert oof.row_id.is_unique and set(oof.row_id)==train


def test_company_probability_response_never_uses_test_label_or_calls_ai(monkeypatch):
    q=CompanyQuery(ROOT,chat_url='http://unused')
    def prohibited(**kw):raise AssertionError('Probability response must use numeric evidence')
    monkeypatch.setattr('esik_company_query.send_chat_message',prohibited)
    result=q.query('ESIK-05511 için iflas olasılığı kaç?')
    assert result['answer_mode']=='calibrated_probability_report'
    assert abs(result['context']['company']['bankruptcy_probability']['estimate']-.147162620840441)<1e-12
    serialized=json.dumps(result,ensure_ascii=False)
    assert 'ACTUAL_BANKRUPT' not in serialized and 'ACTUAL_STATUS' not in serialized
    assert 'güven aralığı değildir' in serialized
    before=result['answer'];q.rows.loc['ESIK-05511','ACTUAL_BANKRUPT']=1-q.rows.loc['ESIK-05511','ACTUAL_BANKRUPT']
    assert q.query('ESIK-05511 için iflas olasılığı kaç?')['answer']==before


def test_evidence_marks_imputed_inputs_as_unobserved():
    q=CompanyQuery(ROOT);values=q.shap.loc[q.shap.COMPANY_ID=='ESIK-05817']
    evidence=evidence_table(values,{})
    row=evidence.loc[evidence['Değişken']=='Attr21'].iloc[0]
    assert pd.isna(row['Gözlenen değer']) and 'Eksik' in row['Girdi kaynağı']
    assert np.isfinite(row['Modelin kullandığı değer'])


def test_nonprobability_questions_still_use_existing_route():
    assert probability_answer('En önemli 3 risk faktörü ne?',{}) is None
    assert probability_answer('İflas olasılığı ne?',{}) is None
