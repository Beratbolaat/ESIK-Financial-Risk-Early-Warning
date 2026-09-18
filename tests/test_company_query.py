import copy
import http.client
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from esik_company_query import CompanyQuery, CompanyQueryError, extract_company_id, private_session_id
from esik_query_server import build_server
from n8n_client import N8nClientError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def query():
    return CompanyQuery(ROOT)


@pytest.mark.parametrize('question', ['ESIK-05511 risk?', 'eşik 5511 için risk?', 'ESIK:5511', 'ＥＳＩＫ-０５５１１', 'ESIK-05511 ve ESIK-5511'])
def test_code_normalization(question):
    assert extract_company_id(question) == 'ESIK-05511'


@pytest.mark.parametrize('question', ['', 'Şirketim nasıl?', 'ESIK-00001 ve ESIK-00002', 'ESIK-055111', 'xESIK-05511', 'ESIK-05511x', 'x'*2001])
def test_ambiguous_or_invalid_codes_are_rejected(question):
    with pytest.raises(CompanyQueryError):
        extract_company_id(question)


def test_record_facts_and_provenance(query):
    result = query.query('ESIK-05511 için en önemli 3 risk nedir?')
    row = query.rows.loc['ESIK-05511']
    company = result['context']['company']
    assert result['status'] == 'ok' and result['answer_mode'] == 'record_summary'
    assert company['risk_rank'] == 125 and company['risk_score'] == row.RISK_SCORE
    assert company['portfolio_size'] == 1170
    assert company['missing_feature_count'] == row[query.metadata['feature_columns']].isna().sum()
    assert result['source']['model_sha256'] == '281067c0fbc4a060b9ec3d1b4ce91a4b7f7cdcf4d8d36d5933aafe2a14d5ed41'
    assert 'ACTUAL_BANKRUPT' not in json.dumps(result)
    assert 'ACTUAL_STATUS' not in json.dumps(result)
    assert 'Tarihsel demo' in result['answer'] and len(result['answer']) <= 3900
    for factor in company['risk_increasing_factors']:
        assert factor['shap'] > 0
    for factor in company['risk_reducing_factors']:
        assert factor['shap'] < 0


def test_missing_record_does_not_call_ai_or_substitute(query):
    connected = copy.copy(query)
    connected.chat_url = 'http://localhost/not-called'
    with patch('esik_company_query.send_chat_message') as chat:
        result = connected.query('ESIK-99999 risk?')
    assert result['status'] == 'needs_input' and result['company_id'] is None
    assert 'bulunamadı' in result['answer'] and not chat.called


def test_ai_failure_is_explicit_and_keeps_record_facts(query):
    connected = copy.copy(query)
    connected.chat_url = 'http://localhost/not-called'
    with patch('esik_company_query.send_chat_message', side_effect=N8nClientError('offline')):
        result = connected.query('ESIK-05511 risk?')
    assert result['answer_mode'] == 'record_summary_fallback'
    assert 'ulaşılamadı' in result['answer'] and '125/1170' in result['answer']


def test_each_question_builds_fresh_company_context(query):
    connected = copy.copy(query)
    connected.chat_url = 'http://localhost/not-called'
    second = str(query.rows.index[0])
    with patch('esik_company_query.send_chat_message', return_value='Kayıt üzerinden kısa yorum.') as chat:
        connected.query('ESIK-05511 risk?', session_id='shared-session')
        result = connected.query(f'{second} risk?', session_id='shared-session')
    payload = chat.call_args.kwargs['payload']
    assert payload['context']['company']['company_id'] == second
    assert payload['conversation_history'] == []
    assert result['answer_mode'] == 'model_grounded_ai'
    assert 'ACTUAL_BANKRUPT' not in json.dumps(payload)
    assert '905551234567' not in private_session_id('whatsapp','905551234567')


@pytest.fixture(scope='module')
def http_service(query):
    settings = dict(ESIK_QUERY_TOKEN='local-test-token', ESIK_TELEGRAM_ALLOWED_IDS='123', ESIK_WHATSAPP_ALLOWED_IDS='+905551234567')
    with patch('esik_query_server.CompanyQuery', return_value=query):
        server = build_server(ROOT, port=0, settings=settings)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_port
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def call(port, body, *, token='local-test-token', content_type='application/json', path='/v1/company-query'):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        connection.request('POST', path, body=body if isinstance(body, bytes) else json.dumps(body).encode(),
                           headers={'X-ESIK-QUERY-TOKEN':token, 'Content-Type':content_type})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_real_local_response(http_service):
    status, result = call(http_service, {'question':'ESIK-05511 risk?', 'channel':'local'})
    assert status == 200 and result['should_reply'] is True and result['company_id'] == 'ESIK-05511'


@pytest.mark.parametrize('body,options,expected', [
    ({'question':'ESIK-05511'}, {'token':'incorrect'}, 401),
    ({'question':'ESIK-05511'}, {'path':'/unknown'}, 404),
    ({'question':'ESIK-05511'}, {'content_type':'text/plain'}, 415),
    (b'x'*12001, {}, 413),
    (b'{invalid', {}, 400),
    ({'question':4}, {}, 400),
    ({'question':'ESIK-05511', 'channel':[]}, {}, 400),
    ({'question':'ESIK-05511', 'channel':'email'}, {}, 400),
], ids=['auth','path','content-type','size','json','question-type','channel-type','unknown-channel'])
def test_http_rejects_bad_requests(http_service, body, options, expected):
    status, result = call(http_service, body, **options)
    assert status == expected and 'context' not in result


@pytest.mark.parametrize('channel,sender,allowed', [('telegram','123',True),('telegram','999',False),
    ('whatsapp','905551234567',True),('whatsapp','999',False),('telegram','',False)])
def test_channel_allowlists(http_service, channel, sender, allowed):
    status, result = call(http_service, {'question':'ESIK-05511', 'channel':channel, 'sender_id':sender})
    assert status == 200 and result['should_reply'] is allowed
    if not allowed:
        assert 'answer' not in result and 'context' not in result


def test_server_requires_token():
    with pytest.raises(ValueError, match='ESIK_QUERY_TOKEN'):
        build_server(ROOT, port=0, settings={})
