"""Preview-first report email; n8n owns SMTP credentials and transport."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from esik_report import build_report, render_text, render_html, REPORT_VERSION


class EmailSendError(RuntimeError):
    def __init__(self, message, *, delivery_unknown=False):
        super().__init__(message)
        self.delivery_unknown = delivery_unknown


def build_email_report(row, shap_rows, layer, model_name):
    data = build_report(row, shap_rows, layer, model_name)
    return {'company_id': data['company_id'],
            'subject': f"EŞİK | {data['company_id']} | Şirket risk raporu",
            'report_text': render_text(data), 'report_html': render_html(data),
            'report_version': REPORT_VERSION}


def send_report_email(webhook_url, webhook_token, payload, timeout_seconds=30):
    """One HTTP attempt. A timeout is ambiguous and must never be auto-retried."""
    parsed = urlparse(webhook_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or not webhook_token:
        raise EmailSendError('E-posta webhook adresi veya erişim anahtarı eksik.')
    recipient = payload.get('recipient', '')
    if not re.fullmatch(r'[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+', recipient):
        raise EmailSendError('Alıcı adresini kontrol edin.')
    request = Request(webhook_url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type': 'application/json', 'X-ESIK-WEBHOOK-TOKEN': webhook_token}, method='POST')
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        known_not_sent = error.code in (400, 401, 403, 404, 422)
        raise EmailSendError(f'Gönderim tamamlanamadı (HTTP {error.code}). n8n akışını kontrol edin.',
                             delivery_unknown=not known_not_sent) from error
    except (URLError, OSError, TimeoutError, ValueError) as error:
        raise EmailSendError('Gönderim sonucu alınamadı. Tekrar göndermeden önce Gmail Gönderilmiş klasörünü ve n8n çalışmasını kontrol edin.',
                             delivery_unknown=True) from error
    if not (isinstance(result, dict) and result.get('status') == 'sent'
            and result.get('recipient') == recipient
            and result.get('request_id') == payload.get('request_id')
            and result.get('message_id')):
        raise EmailSendError('SMTP kabul bilgisi doğrulanamadı. Gmail Gönderilmiş klasörünü ve n8n çalışmasını kontrol edin.',
                             delivery_unknown=True)
    return result


def render_report_email(row, shap_rows, layer, model_name):
    import streamlit as st

    def setting(name):
        value = os.getenv(name, '').strip()
        try:
            return value or str(st.secrets.get(name, '')).strip()
        except Exception:
            return value

    with st.expander('Analiz raporunu e-postayla gönder'):
        recipient = setting('ESIK_REPORT_EMAIL')
        url, token = setting('N8N_EMAIL_WEBHOOK_URL'), setting('N8N_WEBHOOK_TOKEN')
        if not recipient or not url or not token:
            st.info('E-posta bağlantısı henüz ayarlanmamış. Kurulum: docs/EPOSTA_KURULUMU.md')
            return
        if layer is None:
            st.warning('Doğrulanmış rapor verisi yüklenemedi; gönderim kapalı.')
            return
        report = build_email_report(row, shap_rows, layer, model_name)
        fingerprint = hashlib.sha256((recipient + json.dumps(report, ensure_ascii=False, sort_keys=True)).encode()).hexdigest()
        state_key = 'esik_mail_' + fingerprint
        state = st.session_state.get(state_key, {})
        st.write(f'**Alıcı:** {recipient}')
        st.write(f"**Konu:** {report['subject']}")
        preview_tab, text_tab = st.tabs(['Rapor önizlemesi', 'Metin sürümü'])
        with preview_tab:
            import streamlit.components.v1 as components
            components.html(report['report_html'], height=660, scrolling=True)
        with text_tab:
            st.text_area('Düz metin alternatifi', report['report_text'], height=260, disabled=True,
                         key='mail_preview_' + fingerprint)
        st.download_button('Tasarımlı raporu indir', report['report_html'],
                           file_name=f"{report['company_id']}_inceleme_raporu.html", mime='text/html',
                           key='mail_download_' + fingerprint)
        st.caption('Gönderim yalnız bu düğmeyle başlar. Rapor n8n üzerinden SMTP bağlantısına iletilir.')
        if state.get('status') == 'sent':
            st.success(f"Rapor {recipient} adresine gönderilmek üzere SMTP sunucusu tarafından kabul edildi. Gelen kutusu ve spam klasörünü kontrol edebilirsin.")
            st.caption('İleti kimliği: ' + state['message_id'])
        elif state.get('status') in ('unknown', 'sending'):
            st.warning('Önceki gönderimin sonucu belirsiz. Çift e-posta oluşmaması için bu oturumda yeniden gönderim durduruldu. Gmail Gönderilmiş klasörünü ve n8n çalışmasını kontrol edin.')
        if st.button('E-postayla gönder', key='mail_send_' + fingerprint,
                     disabled=state.get('status') in ('sent', 'unknown', 'sending'), type='primary') and not state:
            payload = dict(report, recipient=recipient, request_id=str(uuid.uuid4()))
            st.session_state[state_key] = {'status': 'sending'}
            try:
                with st.spinner('Rapor n8n üzerinden gönderiliyor…'):
                    result = send_report_email(url, token, payload)
            except EmailSendError as error:
                if error.delivery_unknown:
                    st.session_state[state_key] = {'status': 'unknown'}
                else:
                    st.session_state.pop(state_key, None)
                st.error(str(error))
            else:
                st.session_state[state_key] = result
                st.rerun()
