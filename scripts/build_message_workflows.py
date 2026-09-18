"""Rebuild shareable, inactive n8n templates; no credentials are embedded."""
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(executionOrder='v1', saveDataSuccessExecution='none',
                saveDataErrorExecution='none', saveManualExecutions=False)


def node(name, kind, version, parameters, x):
    return dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL, 'esik-v3/' + name + '/' + kind)),
                name=name, type=kind, typeVersion=version, position=[x, 0], parameters=parameters)


def make_workflow(channel):
    title = 'Telegram' if channel == 'telegram' else 'WhatsApp'
    trigger = node(title + ' Mesajı', 'n8n-nodes-base.' + ('telegramTrigger' if channel == 'telegram' else 'whatsAppTrigger'),
                   1.5 if channel == 'telegram' else 1, dict(updates=['message' if channel == 'telegram' else 'messages'],
                   **({'additionalFields':{}} if channel == 'telegram' else {'options':{}})), 0)
    trigger['webhookId'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'esik-v3/' + channel))
    normalize = node('Mesajı Hazırla', 'n8n-nodes-base.code', 2,
                     dict(mode='runOnceForAllItems', jsCode=(ROOT/f'n8n/normalize_{channel}.js').read_text(encoding='utf-8')), 260)
    query = node('Şirketi Sorgula', 'n8n-nodes-base.httpRequest', 4.2, dict(
        method='POST', url='http://127.0.0.1:8765/v1/company-query', authentication='genericCredentialType',
        genericAuthType='httpHeaderAuth', sendBody=True, specifyBody='json',
        jsonBody='={{ { question: $json.question, channel: $json.channel, sender_id: $json.sender_id } }}',
        options={'timeout':45000}), 520)
    check = node('Yanıt İzni', 'n8n-nodes-base.code', 2, dict(mode='runOnceForAllItems',
        jsCode="return $input.all().flatMap((item, index) => item.json.should_reply === true && typeof item.json.answer === 'string' && item.json.answer.trim() ? [{json: item.json, pairedItem: {item: index}}] : []);"), 780)
    if channel == 'telegram':
        send = node('Yanıtı Gönder', 'n8n-nodes-base.telegram', 1.2, dict(resource='message', operation='sendMessage',
            chatId="={{ $('Mesajı Hazırla').item.json.reply_to }}",
            text="={{ $json.answer.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') }}",
            additionalFields={'appendAttribution':False, 'parse_mode':'HTML'}), 1040)
    else:
        send = node('Yanıtı Gönder', 'n8n-nodes-base.whatsApp', 1.1, dict(resource='message', operation='send',
            messageType='text', phoneNumberId="={{ $('Mesajı Hazırla').item.json.phone_number_id }}",
            recipientPhoneNumber="={{ $('Mesajı Hazırla').item.json.reply_to }}", textBody='={{ $json.answer }}'), 1040)
    nodes = [trigger, normalize, query, check, send]
    connections = {a['name']:{'main':[[dict(node=b['name'], type='main', index=0)]]} for a,b in zip(nodes, nodes[1:])}
    return dict(name=f'EŞİK v3 - {title} Şirket Sorgusu', active=False, nodes=nodes,
                connections=connections, settings=SETTINGS, pinData={})


if __name__ == '__main__':
    for channel in ('telegram','whatsapp'):
        (ROOT/f'n8n/esik-{channel}-workflow.json').write_text(json.dumps(make_workflow(channel), ensure_ascii=False, indent=2), encoding='utf-8')
    source = json.loads((ROOT/'n8n/esik-ai-workflow.json').read_text(encoding='utf-8'))
    for key in ('id','versionId','activeVersionId','meta','tags','createdAt','updatedAt','versionCounter','shared'):
        source.pop(key, None)
    source.update(name='EŞİK v3 - Finansal Risk Asistanı', active=False, settings=SETTINGS, pinData={})
    for item in source['nodes']:
        item.pop('credentials', None)
        if item['type'] == 'n8n-nodes-base.webhook':
            item['parameters'].update(path='esik-ai-v3', authentication='headerAuth')
            item['webhookId'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'esik-v3/chat'))
    (ROOT/'n8n/esik-ai-v3-workflow.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Telegram, WhatsApp ve AI v3 pasif şablonları oluşturuldu.')
