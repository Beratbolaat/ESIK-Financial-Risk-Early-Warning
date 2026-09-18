// WhatsApp Trigger emits change.value, not the outer Meta entry envelope.
return $input.all().flatMap((item, index) => {
  const event = item.json;
  const phoneNumberId = event.metadata?.phone_number_id;
  if (!phoneNumberId || !Array.isArray(event.messages)) return [];
  return event.messages.flatMap(message => {
    if (message.type !== 'text' || typeof message.text?.body !== 'string' || !message.from) return [];
    const question = message.text.body.trim();
    if (!question) return [];
    return [{json: {question, channel: 'whatsapp', sender_id: String(message.from),
      reply_to: String(message.from), phone_number_id: String(phoneNumberId),
      message_id: String(message.id || '')}, pairedItem: {item: index}}];
  });
});
