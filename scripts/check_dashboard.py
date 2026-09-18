"""Sekiz sayfayı gerçek yerel model dosyalarıyla açar; dış API çağrısı yapmaz."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=90).run()
results = []
pages = app.sidebar.radio[0].options
for page in pages:
    app.sidebar.radio[0].set_value(page).run()
    errors = [x.message for x in app.exception]
    results.append(dict(page=page, errors=errors))
    if errors:
        raise RuntimeError(f"{page}: {errors}")
report = dict(pages=results, external_api_called=False)
(ROOT/"outputs/dashboard_check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
