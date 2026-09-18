"""Microphone questions: transcribe once, then let the user edit and submit."""
from __future__ import annotations

import io
import json
import socket
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import streamlit as st

MAX_AUDIO_BYTES = 4 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
MAX_QUESTION_CHARS = 2000


class VoiceInputError(RuntimeError):
    """A usable voice question could not be produced."""


def validate_recording(audio: bytes) -> None:
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise VoiceInputError("Ses kaydı boş veya 4 MB sınırını aşıyor.")
    try:
        with wave.open(io.BytesIO(audio), "rb") as recording:
            frames = recording.getnframes()
            rate = recording.getframerate()
            if frames <= 0 or rate <= 0:
                raise VoiceInputError("Ses kaydı boş. Sorunuzu yeniden kaydedin.")
            if frames / rate > MAX_AUDIO_SECONDS:
                raise VoiceInputError("Sorunuzu en fazla 60 saniyelik bir kayıtla iletin.")
            expected_size = frames * recording.getnchannels() * recording.getsampwidth()
            if len(recording.readframes(frames)) != expected_size:
                raise VoiceInputError("Ses kaydı tamamlanmamış. Lütfen yeniden kaydedin.")
    except (wave.Error, EOFError) as error:
        raise VoiceInputError("Ses kaydı okunamadı. Lütfen yeniden kaydedin.") from error


def transcribe_recording(*, audio: bytes, webhook_url: str,
                         webhook_token: str, timeout_seconds: int = 45) -> str:
    """Send a WAV to the separately configured, authenticated n8n voice endpoint."""
    target = webhook_url.strip()
    parsed = urlparse(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VoiceInputError("Sesle soru bağlantısı yapılandırılmamış.")
    if not webhook_token.strip():
        raise VoiceInputError("Sesle soru bağlantısının erişim anahtarı eksik.")
    validate_recording(audio)
    request = Request(target, data=audio, method="POST", headers={
        "Content-Type": "audio/wav",
        "Content-Disposition": 'attachment; filename="question.wav"',
        "X-ESIK-WEBHOOK-TOKEN": webhook_token.strip(),
    })
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(65537)
    except (socket.timeout, TimeoutError) as error:
        raise VoiceInputError("Sesin yazıya çevrilmesi zaman aşımına uğradı. Tekrar deneyebilir veya sorunuzu yazabilirsiniz.") from error
    except HTTPError as error:
        raise VoiceInputError(f"Sesle soru hizmeti yanıt veremedi (HTTP {error.code}). Sorunuzu yazarak devam edebilirsiniz.") from error
    except (URLError, OSError) as error:
        raise VoiceInputError("Sesle soru hizmetine bağlanılamadı. Sorunuzu yazarak devam edebilirsiniz.") from error
    try:
        if len(raw) > 65536:
            raise ValueError("oversized response")
        payload = json.loads(raw.decode("utf-8"))
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("missing transcript")
    except (ValueError, UnicodeDecodeError) as error:
        raise VoiceInputError("Kayıttan bir soru elde edilemedi. Yeniden kaydedebilir veya yazabilirsiniz.") from error
    text = text.strip()
    if len(text) > MAX_QUESTION_CHARS:
        raise VoiceInputError("Ses kaydı çok uzun bir metin üretti. Daha kısa bir soru kaydedin.")
    return text


def voice_chat_input(*, context_key: str, transcription_url: str,
                     webhook_token: str, disabled: bool = False) -> str | None:
    """Audio fills an editable draft; only a later text submission returns a question."""
    widget_key = f"esik_question_{context_key}"
    draft_key = f"{widget_key}_voice_draft"
    notice_key = f"{widget_key}_voice_notice"
    if draft_key in st.session_state:
        st.session_state[widget_key] = st.session_state.pop(draft_key)
    if notice_key in st.session_state:
        st.info(st.session_state.pop(notice_key))

    audio_enabled = bool(transcription_url.strip() and webhook_token.strip())
    if audio_enabled:
        st.caption("Mikrofon kaydı n8n üzerinden OpenAI ile yazıya çevrilir. Metni kontrol edip gönderin. En fazla 60 saniye.")

    submission = st.chat_input(
        "Eşik AI'ya model çıktıları hakkında soru sorun...",
        key=widget_key, disabled=disabled, accept_audio=audio_enabled,
        audio_sample_rate=16000, max_upload_size=4,
        max_chars=MAX_QUESTION_CHARS, submit_mode="disable",
    )
    if submission is None:
        return None
    if isinstance(submission, str):
        return submission.strip() or None

    text = submission.text.strip()
    audio = submission.audio
    if audio is None:
        return text or None
    try:
        with st.spinner("Sesiniz yazıya çevriliyor..."):
            transcript = transcribe_recording(
                audio=audio.getvalue(), webhook_url=transcription_url,
                webhook_token=webhook_token,
            )
        draft = "\n".join(part for part in (text, transcript) if part)
        if len(draft) > MAX_QUESTION_CHARS:
            raise VoiceInputError("Yazılı ve sesli sorunun toplamı çok uzun. Sorunuzu kısaltın.")
    except VoiceInputError as error:
        # Preserve accompanying text but never send it without the failed audio part.
        if text:
            st.session_state[draft_key] = text
            st.session_state[notice_key] = str(error)
            st.rerun()
        st.error(str(error))
        return None

    st.session_state[draft_key] = draft
    st.session_state[notice_key] = "Sesiniz yazıya çevrildi. Şirket kodunu ve soruyu kontrol edip gönderin."
    st.rerun()
    return None
