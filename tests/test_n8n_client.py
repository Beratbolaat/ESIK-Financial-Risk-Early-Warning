import pytest

from n8n_client import N8nClientError, extract_answer


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"answer": "Risk açıklaması"}, "Risk açıklaması"),
        ({"output": "Analist özeti"}, "Analist özeti"),
        ([{"text": "Toplu analiz"}], "Toplu analiz"),
        ("Düz metin", "Düz metin"),
    ],
)
def test_extract_answer_supports_common_n8n_shapes(payload, expected):
    assert extract_answer(payload) == expected


def test_extract_answer_rejects_empty_payload():
    with pytest.raises(N8nClientError):
        extract_answer({"answer": ""})


def test_client_sends_json_post_and_optional_header(monkeypatch):
    import json
    import io
    import n8n_client
    captured = {}
    def fake_open(request, timeout):
        captured.update(method=request.method, data=json.loads(request.data),
                        headers=dict(request.header_items()), timeout=timeout)
        return io.BytesIO('{"answer":"Kontrol edildi"}'.encode("utf-8"))
    monkeypatch.setattr(n8n_client, "urlopen", fake_open)
    payload = {"question": "Şirketi açıkla", "context": {"risk_score": .2}}
    assert n8n_client.send_chat_message(webhook_url="http://localhost:5678/webhook/test",
               payload=payload, webhook_token="test-only", timeout_seconds=5) == "Kontrol edildi"
    assert captured["method"] == "POST" and captured["data"] == payload
    assert captured["headers"]["X-esik-webhook-token"] == "test-only"
    assert captured["timeout"] == 5


def test_client_reports_timeout_without_retrying(monkeypatch):
    import n8n_client
    calls = []
    def fake_open(*args, **kwargs):
        calls.append(1)
        raise TimeoutError()
    monkeypatch.setattr(n8n_client, "urlopen", fake_open)
    with pytest.raises(N8nClientError, match="yanıt süresini aştı"):
        n8n_client.send_chat_message(webhook_url="http://localhost:5678/webhook/test", payload={})
    assert len(calls) == 1
