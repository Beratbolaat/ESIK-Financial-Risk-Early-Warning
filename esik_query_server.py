"""Local authenticated adapter for n8n Telegram/WhatsApp company queries."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tomllib

from esik_company_query import CompanyQuery, private_session_id


def load_settings(root):
    path = Path(root)/'.streamlit/secrets.toml'
    values = tomllib.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    for key in ['ESIK_QUERY_TOKEN', 'N8N_WEBHOOK_URL', 'N8N_WEBHOOK_TOKEN',
                'ESIK_TELEGRAM_ALLOWED_IDS', 'ESIK_WHATSAPP_ALLOWED_IDS']:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def build_server(root, *, port=8765, settings=None):
    settings = load_settings(root) if settings is None else settings
    token = str(settings.get('ESIK_QUERY_TOKEN', '')).strip()
    if not token:
        raise ValueError('ESIK_QUERY_TOKEN tanımlanmalı. scripts/prepare_connections.py ile yerel ayarları hazırlayın.')
    query = CompanyQuery(root, chat_url=str(settings.get('N8N_WEBHOOK_URL', '')),
                         chat_token=str(settings.get('N8N_WEBHOOK_TOKEN', '')))
    allowlists = {
        'telegram':{s.strip() for s in str(settings.get('ESIK_TELEGRAM_ALLOWED_IDS', '')).split(',') if s.strip()},
        'whatsapp':{s.strip().lstrip('+') for s in str(settings.get('ESIK_WHATSAPP_ALLOWED_IDS', '')).split(',') if s.strip()},
    }

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Do not log questions, sender identities or headers.

        def respond(self, status, body):
            content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == '/health':
                self.respond(200, {'status':'ok', 'service':'esik-company-query'})
            else:
                self.respond(404, {'error':'not_found'})

        def do_POST(self):
            if self.path != '/v1/company-query':
                return self.respond(404, {'error':'not_found'})
            provided = self.headers.get('X-ESIK-QUERY-TOKEN', '')
            if not hmac.compare_digest(provided.encode(), token.encode()):
                return self.respond(401, {'error':'unauthorized'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > 12000:
                    return self.respond(413, {'error':'invalid_body_size'})
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self.respond(415, {'error':'expected_json'})
                body = json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(body, dict) or not isinstance(body.get('question'), str):
                    return self.respond(400, {'error':'question_required'})
                channel = body.get('channel', 'local')
                if not isinstance(channel, str):
                    return self.respond(400, {'error':'unknown_channel'})
                sender = str(body.get('sender_id', '')).lstrip('+')
                if channel not in {'local','telegram','whatsapp'}:
                    return self.respond(400, {'error':'unknown_channel'})
                if channel in allowlists and sender not in allowlists[channel]:
                    return self.respond(200, {'status':'ignored', 'should_reply':False})
                result = query.query(body['question'], session_id=private_session_id(channel, sender))
                result['should_reply'] = True
                self.respond(200, result)
            except (ValueError, UnicodeDecodeError):
                self.respond(400, {'error':'invalid_request'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = build_server(args.project_root, port=args.port)
    print(f'EŞİK şirket sorgusu: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
