"""Small, explicit client for the Eşik n8n chatbot workflow."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class N8nClientError(RuntimeError):
    """Raised when the n8n workflow cannot return a usable answer."""


def _validate_webhook_url(webhook_url: str) -> str:
    normalized_url = webhook_url.strip()
    parsed_url = urlparse(normalized_url)

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise N8nClientError(
            "N8N_WEBHOOK_URL geçerli bir http/https adresi olmalıdır."
        )

    return normalized_url


def extract_answer(response_payload: Any) -> str:
    """Accept common n8n response shapes and return the assistant text."""

    if isinstance(response_payload, list) and response_payload:
        response_payload = response_payload[0]

    if isinstance(response_payload, str):
        answer = response_payload.strip()
    elif isinstance(response_payload, dict):
        answer = ""
        for key in ("answer", "output", "text", "response"):
            candidate = response_payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                answer = candidate.strip()
                break
    else:
        answer = ""

    if not answer:
        raise N8nClientError("n8n geçerli bir sohbet yanıtı döndürmedi.")

    return answer


def send_chat_message(
    *,
    webhook_url: str,
    payload: dict[str, Any],
    webhook_token: str = "",
    timeout_seconds: int = 60,
) -> str:
    """Send model-grounded context to n8n and return the chatbot answer."""

    target_url = _validate_webhook_url(webhook_url)
    headers = {"Content-Type": "application/json"}

    if webhook_token.strip():
        headers["X-ESIK-WEBHOOK-TOKEN"] = webhook_token.strip()

    request = Request(
        target_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except (socket.timeout, TimeoutError) as error:
        raise N8nClientError(
            "n8n yanıt süresini aştı. Workflow durumunu kontrol edin."
        ) from error
    except HTTPError as error:
        raise N8nClientError(
            f"n8n workflow'u HTTP {error.code} hatası döndürdü."
        ) from error
    except (URLError, OSError) as error:
        raise N8nClientError(
            "n8n workflow'una bağlanılamadı. Webhook adresini ve workflow'un "
            "aktif olduğunu kontrol edin."
        ) from error

    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError:
        response_payload = response_text

    return extract_answer(response_payload)
