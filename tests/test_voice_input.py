import io
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
import wave

import pytest
import esik_voice as voice


def wav(seconds=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(b"\x00\x00" * (16000 * seconds))
    return output.getvalue()


def test_sends_only_audio_with_auth_and_returns_transcript(monkeypatch):
    calls = []
    audio = wav()
    def fake_open(request, timeout):
        calls.append(request)
        assert timeout == 45
        assert request.data == audio
        assert request.get_header("Content-type") == "audio/wav"
        assert request.get_header("X-esik-webhook-token") == "test-only"
        assert "question.wav" in request.get_header("Content-disposition")
        return io.BytesIO(json.dumps({"text": " Bu şirket neden riskli? "}).encode())
    monkeypatch.setattr(voice, "urlopen", fake_open)
    assert voice.transcribe_recording(audio=audio, webhook_url="http://localhost:5678/webhook/esik-voice", webhook_token="test-only") == "Bu şirket neden riskli?"
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [{"answer": "Not a transcript"}, {"text": " "}, {"error": "unauthorized"}, ["wrong shape"]])
def test_error_payload_is_never_used_as_a_question(monkeypatch, payload):
    monkeypatch.setattr(voice, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(voice.VoiceInputError, match="soru elde edilemedi"):
        voice.transcribe_recording(audio=wav(), webhook_url="http://localhost:5678/webhook/esik-voice", webhook_token="test-only")


@pytest.mark.parametrize("data", [b"", b"not audio", wav(0), wav(61), wav()[:-2]],
                         ids=["empty", "invalid", "zero_frames", "too_long", "truncated"])
def test_invalid_recordings_never_reach_network(monkeypatch, data):
    def forbidden(*a, **k):
        raise AssertionError("must validate before network")
    monkeypatch.setattr(voice, "urlopen", forbidden)
    with pytest.raises(voice.VoiceInputError):
        voice.transcribe_recording(audio=data, webhook_url="http://localhost:5678/webhook/esik-voice", webhook_token="test-only")


@pytest.mark.parametrize("failure", [TimeoutError(), HTTPError("http://localhost", 401, "Unauthorized", {}, None)])
def test_network_failures_do_not_retry(monkeypatch, failure):
    calls = []
    def fail(*a, **k):
        calls.append(1)
        raise failure
    monkeypatch.setattr(voice, "urlopen", fail)
    with pytest.raises(voice.VoiceInputError):
        voice.transcribe_recording(audio=wav(), webhook_url="http://localhost:5678/webhook/esik-voice", webhook_token="test-only")
    assert calls == [1]


class Rerun(BaseException):
    pass


def fake_streamlit(submission, state):
    def rerun():
        raise Rerun()
    return SimpleNamespace(
        session_state=state, caption=lambda *a: None,
        info=lambda *a: None, error=lambda *a: None,
        spinner=lambda *a: nullcontext(),
        chat_input=lambda *a, **k: submission, rerun=rerun,
    )


def render(**overrides):
    return voice.voice_chat_input(**dict(
        context_key="company:ESIK-05511",
        transcription_url="http://localhost:5678/webhook/esik-voice",
        webhook_token="test-only", **overrides))


def test_audio_becomes_editable_draft_and_requires_text_submission(monkeypatch):
    state = {}
    submission = SimpleNamespace(text="Kısaca:", audio=io.BytesIO(wav()))
    monkeypatch.setattr(voice, "st", fake_streamlit(submission, state))
    calls = []
    def fake_transcribe(**kwargs):
        calls.append(1)
        return "Bu şirket neden riskli?"
    monkeypatch.setattr(voice, "transcribe_recording", fake_transcribe)
    with pytest.raises(Rerun):
        render()
    assert state["esik_question_company:ESIK-05511_voice_draft"] == "Kısaca:\nBu şirket neden riskli?"
    monkeypatch.setattr(voice, "st", fake_streamlit(None, state))
    assert render() is None
    assert state["esik_question_company:ESIK-05511"] == "Kısaca:\nBu şirket neden riskli?"
    assert calls == [1]
    monkeypatch.setattr(voice, "st", fake_streamlit(SimpleNamespace(text="Düzeltilmiş soru", audio=None), state))
    assert render() == "Düzeltilmiş soru"
    assert calls == [1]


def test_transcription_failure_preserves_typed_text_without_submitting(monkeypatch):
    state = {}
    monkeypatch.setattr(voice, "st", fake_streamlit(SimpleNamespace(text="Yazılı bölüm", audio=io.BytesIO(wav())), state))
    def fail(**kwargs):
        raise voice.VoiceInputError("service unavailable")
    monkeypatch.setattr(voice, "transcribe_recording", fail)
    with pytest.raises(Rerun):
        render()
    assert state["esik_question_company:ESIK-05511_voice_draft"] == "Yazılı bölüm"


def test_voice_draft_stays_with_its_company(monkeypatch):
    state = {"esik_question_company:ESIK-00001_voice_draft": "Önceki şirket sorusu"}
    monkeypatch.setattr(voice, "st", fake_streamlit(None, state))
    assert render() is None
    assert "esik_question_company:ESIK-05511" not in state


def test_text_works_when_voice_is_not_configured(monkeypatch):
    monkeypatch.setattr(voice, "st", fake_streamlit(" Yazılı soru ", {}))
    assert voice.voice_chat_input(context_key="company:1", transcription_url="", webhook_token="") == "Yazılı soru"


def test_voice_workflow_uses_audio_route_without_credentials_or_chat_side_effects():
    path = Path(__file__).resolve().parents[1] / "n8n/esik-voice-workflow.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["active"] is False
    nodes = {node["name"]: node for node in workflow["nodes"]}
    assert nodes["Ses Kaydı"]["parameters"]["authentication"] == "headerAuth"
    assert nodes["Ses Kaydı"]["parameters"]["options"]["binaryPropertyName"] == "audio"
    assert nodes["Türkçe Metne Çevir"]["parameters"]["operation"] == "transcribe"
    assert nodes["Türkçe Metne Çevir"]["parameters"]["binaryPropertyName"] == "audio"
    assert all("credentials" not in node for node in nodes.values())
    for source, connections in workflow["connections"].items():
        assert source in nodes
        for branch in connections["main"]:
            for target in branch:
                assert target["node"] in nodes
