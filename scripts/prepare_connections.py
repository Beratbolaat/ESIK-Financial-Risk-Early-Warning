"""Create local tokens and draft workflow imports. Does not contact external services."""
import argparse
import json
from pathlib import Path
import secrets
import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    'esik-ai-v3-workflow.json':'ESIKv3Chat000001',
    'esik-voice-workflow.json':'ESIKv3Voice00001',
    'esik-telegram-workflow.json':'ESIKv3Telegram1',
    'esik-whatsapp-workflow.json':'ESIKv3WhatsApp1',
}
QUERY_AUTH = {'id':'ESIKv3QueryAuth1', 'name':'EŞİK v3 - Yerel Şirket Sorgusu'}
WEBHOOK_AUTH = {'id':'ESIKv3WebAuth001', 'name':'EŞİK v3 - Uygulama Webhook'}


def prepare(root=ROOT, *, enable_ai=False, telegram_id=None, whatsapp_id=None):
    root = Path(root)
    secret_path = root/'.streamlit/secrets.toml'
    secret_path.parent.mkdir(exist_ok=True)
    original = secret_path.read_text(encoding='utf-8-sig') if secret_path.exists() else ''
    settings = tomllib.loads(original)
    updates = {}
    for key in ('ESIK_QUERY_TOKEN','N8N_WEBHOOK_TOKEN'):
        if not settings.get(key):
            updates[key] = secrets.token_urlsafe(32)
    for key in ('N8N_WEBHOOK_URL','N8N_TRANSCRIBE_WEBHOOK_URL','ESIK_TELEGRAM_ALLOWED_IDS','ESIK_WHATSAPP_ALLOWED_IDS'):
        if key not in settings:
            updates[key] = ''
    if enable_ai:
        updates['N8N_WEBHOOK_URL'] = 'http://127.0.0.1:5678/webhook/esik-ai-v3'
        updates['N8N_TRANSCRIBE_WEBHOOK_URL'] = 'http://127.0.0.1:5678/webhook/esik-voice'
    if telegram_id is not None:
        if not all(value.strip().isdigit() for value in telegram_id.split(',')):
            raise ValueError('Telegram kullanıcı kimliği yalnızca rakamlardan oluşmalı.')
        updates['ESIK_TELEGRAM_ALLOWED_IDS'] = telegram_id
    if whatsapp_id is not None:
        values = [value.strip().lstrip('+') for value in whatsapp_id.split(',')]
        if not all(value.isdigit() and 7 <= len(value) <= 15 for value in values):
            raise ValueError('WhatsApp numarasını ülke koduyla ve boşluksuz yazın.')
        updates['ESIK_WHATSAPP_ALLOWED_IDS'] = ','.join(values)
    # Preserve every unrelated setting and comment. New keys precede TOML tables.
    import re
    remaining = dict(updates)
    lines = original.splitlines()
    in_table = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith('['):
            in_table = True
        if not in_table:
            for key in list(remaining):
                if re.match(r'^\s*' + re.escape(key) + r'\s*=', line):
                    lines[index] = f'{key} = {json.dumps(remaining.pop(key), ensure_ascii=False)}'
                    break
    text = '\n'.join(f'{key} = {json.dumps(value, ensure_ascii=False)}' for key, value in remaining.items())
    if text:
        text += '\n'
    text += '\n'.join(lines) + '\n'
    verified = tomllib.loads(text)
    secret_path.write_text(text, encoding='utf-8')
    runtime = root/'.runtime'
    workflow_dir = runtime/'workflows'
    workflow_dir.mkdir(parents=True, exist_ok=True)
    credentials = [dict(**QUERY_AUTH, type='httpHeaderAuth', data={'name':'X-ESIK-QUERY-TOKEN', 'value':verified['ESIK_QUERY_TOKEN']}),
                   dict(**WEBHOOK_AUTH, type='httpHeaderAuth', data={'name':'X-ESIK-WEBHOOK-TOKEN', 'value':verified['N8N_WEBHOOK_TOKEN']})]
    (runtime/'credentials.json').write_text(json.dumps(credentials, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename, workflow_id in WORKFLOWS.items():
        workflow = json.loads((root/'n8n'/filename).read_text(encoding='utf-8'))
        workflow.update(id=workflow_id, active=False)
        for node in workflow['nodes']:
            if node['type'] == 'n8n-nodes-base.webhook':
                node['credentials'] = {'httpHeaderAuth':WEBHOOK_AUTH}
            if node['type'] == 'n8n-nodes-base.httpRequest':
                node['credentials'] = {'httpHeaderAuth':QUERY_AUTH}
        (workflow_dir/filename).write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'local_settings_created':True, 'draft_workflow_count':len(WORKFLOWS),
            'ai_urls_enabled':bool(verified.get('N8N_WEBHOOK_URL')),
            'telegram_allowlist_set':bool(verified.get('ESIK_TELEGRAM_ALLOWED_IDS')),
            'whatsapp_allowlist_set':bool(verified.get('ESIK_WHATSAPP_ALLOWED_IDS')),
            'external_account_created':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-ai', action='store_true', help='Enable local URLs only after OpenAI is connected and workflows published.')
    parser.add_argument('--telegram-id')
    parser.add_argument('--whatsapp-id')
    args = parser.parse_args()
    print(json.dumps(prepare(enable_ai=args.enable_ai, telegram_id=args.telegram_id, whatsapp_id=args.whatsapp_id), ensure_ascii=False))
