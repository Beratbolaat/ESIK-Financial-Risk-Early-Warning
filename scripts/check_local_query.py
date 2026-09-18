"""Verify that the local adapter uses this project's token, without sending data to AI."""
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esik_query_server import load_settings

settings = load_settings(Path(__file__).resolve().parents[1])
request = Request('http://127.0.0.1:8765/v1/company-query',
    data=json.dumps({'question':'ESIK-99999', 'channel':'local'}).encode(),
    headers={'Content-Type':'application/json','X-ESIK-QUERY-TOKEN':settings.get('ESIK_QUERY_TOKEN','')})
try:
    with urlopen(request, timeout=3) as response:
        body = json.loads(response.read(10000))
    sys.exit(0 if body.get('answer_mode') == 'validation' else 1)
except HTTPError:
    sys.exit(1)
except (URLError, TimeoutError):
    sys.exit(2)
