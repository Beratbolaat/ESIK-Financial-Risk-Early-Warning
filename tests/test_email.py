import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import esik_email


PAYLOAD = {'recipient': 'demo@example.com', 'request_id': 'test-id'}


def test_smtp_acceptance_is_required(monkeypatch):
    response = dict(status='sent', recipient='demo@example.com', request_id='test-id', message_id='<smtp-id>')
    opener = Mock(return_value=io.BytesIO(json.dumps(response).encode()))
    monkeypatch.setattr(esik_email, 'urlopen', opener)
    result = esik_email.send_report_email('http://127.0.0.1:5678/webhook/demo', 'test-token', PAYLOAD)
    assert result == response
    request = opener.call_args.args[0]
    assert request.method == 'POST'
    assert json.loads(request.data) == PAYLOAD
    assert dict(request.header_items())['X-esik-webhook-token'] == 'test-token'
    assert opener.call_count == 1


@pytest.mark.parametrize('response', [
    {'answer': 'Mail gönderildi'}, {'status': 'sent'},
    {'status':'sent','recipient':'other@example.com','request_id':'test-id','message_id':'id'},
    {'status':'sent','recipient':'demo@example.com','request_id':'other-id','message_id':'id'},
])
def test_no_false_success(monkeypatch, response):
    monkeypatch.setattr(esik_email, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(esik_email.EmailSendError) as caught:
        esik_email.send_report_email('http://127.0.0.1:5678/webhook/demo', 'token', PAYLOAD)
    assert caught.value.delivery_unknown


def test_timeout_never_retries(monkeypatch):
    opener = Mock(side_effect=TimeoutError)
    monkeypatch.setattr(esik_email, 'urlopen', opener)
    with pytest.raises(esik_email.EmailSendError) as caught:
        esik_email.send_report_email('http://127.0.0.1:5678/webhook/demo', 'token', PAYLOAD)
    assert caught.value.delivery_unknown and opener.call_count == 1


def test_report_uses_verified_outputs_not_individual_outcome():
    from esik_company_query import CompanyQuery
    query = CompanyQuery(Path(__file__).resolve().parents[1])
    row = query.rows.loc['ESIK-05511'].copy()
    row['COMPANY_ID'] = 'ESIK-05511'
    shap = query.shap[query.shap.COMPANY_ID == row['COMPANY_ID']]
    report = esik_email.build_email_report(row, shap, query.probability_layer, query.metadata['model_name'])
    assert '%17,4' in report['report_text'] and '%14,7' in report['report_text']
    assert '125/1170' in report['report_text']
    assert 'SHAP' in report['report_text'] and 'Polonya' in report['report_text']
    row['ACTUAL_BANKRUPT'] = 1 - int(row['ACTUAL_BANKRUPT'])
    assert esik_email.build_email_report(row, shap, query.probability_layer, query.metadata['model_name']) == report


def test_portable_workflow_has_auth_and_no_credentials():
    workflow = json.loads((Path(__file__).resolve().parents[1]/'n8n/esik-email-workflow.json').read_text(encoding='utf-8'))
    by_name = {n['name']:n for n in workflow['nodes']}
    assert by_name['Rapor Webhook']['parameters']['authentication'] == 'headerAuth'
    assert all(not n.get('credentials') for n in workflow['nodes'])
    assert workflow['active'] is False
    assert 'sender@example.com' in by_name['Rapor ve Alıcı Kontrolü']['parameters']['jsCode']
    assert by_name['Gmail ile Rapor Gönder']['onError'] == 'continueErrorOutput'
