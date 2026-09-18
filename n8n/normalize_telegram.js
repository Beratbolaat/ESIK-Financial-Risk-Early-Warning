// Telegram Trigger output. Only private, human-authored text is accepted.
return $input.all().flatMap((item, index) => {
  const message = item.json.message;
  if (!message || message.chat?.type !== 'private' || message.from?.is_bot !== false
      || !message.from?.id || !message.chat?.id || typeof message.text !== 'string') return [];
  const question = message.text.trim();
  if (!question) return [];
  return [{json: {question, channel: 'telegram', sender_id: String(message.from.id),
    reply_to: String(message.chat.id), message_id: String(message.message_id || '')},
    pairedItem: {item: index}}];
});
