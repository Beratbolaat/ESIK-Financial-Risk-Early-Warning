// Run with node --test tests/test_message_workflows.js; no network or accounts required.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const root = path.resolve(__dirname, '..');
const load = channel => JSON.parse(fs.readFileSync(path.join(root, 'n8n', `esik-${channel}-workflow.json`), 'utf8'));
const execute = (workflow, name, items) => new Function('$input', workflow.nodes.find(n => n.name === name).parameters.jsCode)({all: () => items});
const tg = (overrides={}) => ({json:{message:{from:{id:123,is_bot:false},chat:{id:123,type:'private'},text:'ESIK-05511 risk?',message_id:7,...overrides}}});
const wa = (messages) => ({json:{metadata:{phone_number_id:'phone-1'},messages}});
const waText = (from, body) => ({id:'message-1',from,type:'text',text:{body}});

test('Telegram accepts private human text and preserves sender/reply routing', () => {
  const values = execute(load('telegram'), 'Mesajı Hazırla', [tg(), tg({chat:{id:-99,type:'group'}}), tg({from:{id:1,is_bot:true}}), tg({text:undefined}), {json:{}}]);
  assert.equal(values.length, 1);
  assert.deepEqual(values[0], {json:{question:'ESIK-05511 risk?',channel:'telegram',sender_id:'123',reply_to:'123',message_id:'7'},pairedItem:{item:0}});
});

test('WhatsApp handles native trigger structure, multiple messages and status events', () => {
  const values = execute(load('whatsapp'), 'Mesajı Hazırla', [wa([waText('905550001111','ESIK-05511 risk?'), waText('905550002222','ESIK-99999 risk?'), {type:'image',from:'5'}]), {json:{statuses:[{status:'delivered'}]}}]);
  assert.equal(values.length, 2);
  assert.equal(values[0].json.phone_number_id, 'phone-1');
  assert.equal(values[1].json.reply_to, '905550002222');
  assert.equal(values[1].json.question, 'ESIK-99999 risk?');
  assert.deepEqual(values[1].pairedItem, {item:0});
});

test('Unapproved senders and empty responses cannot reach send node', () => {
  for (const channel of ['telegram','whatsapp']) {
    const values = execute(load(channel), 'Yanıt İzni', [{json:{should_reply:false}}, {json:{should_reply:true,answer:' '}}, {json:{should_reply:true,answer:'OK'}}]);
    assert.deepEqual(values, [{json:{should_reply:true,answer:'OK'},pairedItem:{item:2}}]);
  }
});

test('Telegram escapes model text for native HTML parse mode', () => {
  const params = load('telegram').nodes.find(n => n.name === 'Yanıtı Gönder').parameters;
  assert.equal(params.additionalFields.parse_mode, 'HTML');
  const expression = params.text.slice(3,-2).trim();
  const text = new Function('$json', `return (${expression});`)({answer:'A&B <risk> [x] _y_'});
  assert.equal(text, 'A&amp;B &lt;risk&gt; [x] _y_');
});

test('Templates contain resolvable connections, no secrets, and authenticated local query', () => {
  for (const channel of ['telegram','whatsapp','ai-v3','voice']) {
    const workflow = load(channel);
    assert.equal(workflow.active, false);
    assert.equal(workflow.settings.saveDataSuccessExecution, 'none');
    assert.equal(workflow.settings.saveDataErrorExecution, 'none');
    const names = new Set(workflow.nodes.map(n => n.name));
    assert.equal(names.size, workflow.nodes.length);
    for (const node of workflow.nodes) assert.equal(node.credentials, undefined);
    for (const [from, kinds] of Object.entries(workflow.connections)) {
      assert(names.has(from));
      for (const branches of Object.values(kinds)) for (const branch of branches) for (const edge of branch) assert(names.has(edge.node));
    }
    for (const node of workflow.nodes.filter(n => n.type === 'n8n-nodes-base.httpRequest')) {
      assert.equal(node.parameters.genericAuthType, 'httpHeaderAuth');
      assert.equal(node.parameters.url, 'http://127.0.0.1:8765/v1/company-query');
    }
  }
});
