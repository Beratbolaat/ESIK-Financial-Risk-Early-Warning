"""Regressions for observed wrong-company and wrong-intent answers."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from esik_assessment import probability_answer
from esik_company_query import CompanyQuery, CompanyQueryError

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def lookup():
    return CompanyQuery(ROOT)

@pytest.mark.parametrize('question', [
    'Sigmoid kalibrasyonunu neden yaptık, nasıl uyguladık?',
    'Olasılığı tekrar yazma. En önemli üç SHAP katkısını sayılarıyla açıkla.',
    'Bu ihtimalin güvenilirliği nasıl ölçüldü?',
    'Şirketin olasılığı kaç ve neden LightGBM yerine XGBoost seçtiniz?',
    'iflas olasılığı ne? Brier hatası nasıl hesaplandı?',
])
def test_free_questions_are_not_swallowed_by_numeric_shortcut(lookup, question):
    assert probability_answer(question, lookup.company_context('ESIK-05817')) is None

def test_shortcut_never_substitutes_selected_company(lookup):
    question = 'ESIK-05511 için bir yıllık iflas ihtimali kaç?'
    assert probability_answer(question, lookup.company_context('ESIK-05817')) is None
    target = lookup.resolve_company_id(question, 'ESIK-05817')
    answer = probability_answer(question, lookup.company_context(target))
    assert target == 'ESIK-05511' and '%14,7' in answer and 'ESIK-05817' not in answer

@pytest.mark.parametrize('question', [
    'ESIK-05511 ve ESIK-05817 riskleri?', 'ESIK-99999 için risk?',
    'ESIK-055111 için risk?', 'ESIK-05511x için risk?',
    'ESIK-05511 ve ESIK-999999 riskleri?',
])
def test_invalid_and_ambiguous_codes_cannot_use_selected_company(lookup, question):
    with pytest.raises(CompanyQueryError):
        lookup.resolve_company_id(question, 'ESIK-05817')

def test_followup_uses_current_company_and_scenario_does_not_cross_records(lookup):
    assert lookup.resolve_company_id('Bu şirket neden riskli?', 'ESIK-05511') == 'ESIK-05511'
    scenario = dict(company_id='ESIK-05817', current_score=.9, scenario_score=.8, score_change_points=-10)
    assert lookup.company_context('ESIK-05817', scenario)['latest_what_if'] == scenario
    assert lookup.company_context('ESIK-05511', scenario)['latest_what_if'] is None

def test_company_has_full_missingness_and_two_explicit_probability_fields(lookup):
    company = lookup.company_context('ESIK-05817')
    assert company['missing_feature_count'] == len(company['missing_features']) == 2
    assert company['feature_count'] == 64 and 'Attr21' in company['missing_features']
    assert company['raw_probability'] == company['risk_score']
    assert company['bankruptcy_probability']['estimate'] != company['raw_probability']
    assert '99,84' in company['risk_score_display']
    assert 'ACTUAL_BANKRUPT' not in json.dumps(company)

def test_validation_distinguishes_comparison_calibration_and_history(lookup):
    validation = lookup.model_validation()
    comparison = validation['model_comparison']
    assert comparison['xgboost']['historical_test']['captured'] == 67
    assert comparison['lightgbm']['historical_test']['captured'] == 70
    assert comparison['xgboost']['outer_cv']['mean_recall_at_10'] > comparison['lightgbm']['outer_cv']['mean_recall_at_10']
    assert 'sonradan' in validation['selection']['status']
    assert validation['calibration']['raw']['brier'] > validation['calibration']['calibrated']['brier']
    assert validation['calibration']['ranking_preserved']
    assert 'Şirket Detayı' in validation['application_guide']
    assert 'ACTUAL_BANKRUPT' not in json.dumps(validation)

def test_direct_query_sends_same_full_validation_and_routes_method_question_to_ai(lookup):
    connected = copy.copy(lookup)
    connected.chat_url = 'http://localhost/not-called'
    with patch('esik_company_query.send_chat_message', return_value='Sigmoid kalibrasyonunun açıklaması.') as send:
        result = connected.query('ESIK-05511 için kalibrasyon neden kullanıldı?')
    assert result['answer_mode'] == 'model_grounded_ai'
    payload = send.call_args.kwargs['payload']
    assert payload['model_validation'] == connected.model_validation()
    assert payload['context']['company']['company_id'] == 'ESIK-05511'

def test_workflow_prompts_preserve_probability_semantics_and_new_data_sections():
    for name in ('esik-ai-workflow.json', 'esik-ai-v3-workflow.json'):
        workflow = json.loads((ROOT/'n8n'/name).read_text(encoding='utf-8'))
        node = next(n for n in workflow['nodes'] if n['type']=='@n8n/n8n-nodes-langchain.chainLlm')
        prompt = node['parameters']['messages']['messageValues'][0]['message']
        assert 'KALİBRE EDİLMEMİŞ iflas olasılığı tahminidir' in prompt
        assert 'iflas yüzdesi deme' not in prompt
        assert '$json.model_validation_json' in node['parameters']['text']
