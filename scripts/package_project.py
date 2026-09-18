"""Tamamlanan, test edilmiş çalışma kopyasını sırlar/venv olmadan paketler."""
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from esik_artifacts import validate_artifacts
from esik_core import sha256, write_json

manifest=validate_artifacts(ROOT)
version=manifest["model_version"]
summary=json.loads((ROOT/"experiments"/version/"summary.json").read_text(encoding="utf-8"))
assert summary["status"]=="complete"
test_log=(ROOT/"tests-latest.log").read_text(encoding="utf-8")
match=re.search(r"(\d+) passed",test_log)
assert match and not re.search(r"\d+ (failed|error)",test_log)
dashboard=json.loads((ROOT/"outputs/dashboard_check.json").read_text(encoding="utf-8"))
assert len(dashboard["pages"])==8 and all(not p["errors"] for p in dashboard["pages"])
assert (ROOT/"docs/optuna-sonuclari.md").is_file()

# Eski çalışma rehberleri V1 metrikleri içerir; sürüm farkını baştan belirt.
guide_root=ROOT.parents[1]/"ESIK-sunum"
note=("> **Güncel uygulama notu:** Optuna revizyonu tamamlandı. Bu belge V1 modelinin "
      "ve uygulama öncesi planın tarihsel açıklamasıdır. Yeni parametreler ve sonuçlar için "
      "[Optuna sonuç raporunu](../projects/esik-profesyonel/docs/optuna-sonuclari.md), "
      "[yöntem protokolünü](../projects/esik-profesyonel/docs/model-protokolu.md) ve "
      "[teknik savunma rehberini](../projects/esik-profesyonel/docs/teknik-savunma.md) kullanın.\n\n")
for name in ["ESIK-Anlama-ve-Sunum-Rehberi.md","ESIK-Teknik-Kararlar-ve-Hiperparametreler.md"]:
    path=guide_root/name
    if path.exists():
        content=path.read_text(encoding="utf-8")
        if not content.startswith("> **Güncel uygulama notu:"):
            content=note+content
        content=content.replace(
            "Henüz Optuna kodu uygulanmadı, yeni model eğitilmedi ve mevcut sonuçlar değiştirilmedi.",
            "Bu paragraf uygulama öncesindeki durumu kaydeder; Optuna revizyonu artık ayrı çalışma "
            "kopyasında tamamlanmıştır. Güncel deney için belgenin başındaki bağlantıları kullanın.")
        path.write_text(content,encoding="utf-8")

skip_dirs={".venv","venv","env","__pycache__",".pytest_cache",".git",".idea",".vscode","releases"}
skip_files={".env","secrets.toml","prefetch_final_search.py","prepare_portable_workflow.py",
            "update_entrypoints.py","write_project_docs.py","pilot.log","artifact-tests.log","changed-tests.log"}
files=[]
for current,dirs,names in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in skip_dirs and not d.startswith("pilot-")]
    for name in names:
        path=Path(current)/name
        relative=path.relative_to(ROOT)
        if name in skip_files or name.endswith((".pyc",".tmp",".sqlite3-wal",".sqlite3-shm")):
            continue
        if any(part=="screenshots" for part in relative.parts):
            continue
        if relative.as_posix()=="models/esik_gradient_boosting_pipeline.pkl":
            continue  # Eski model reference/v1/models altında zaten korunuyor.
        files.append(path)
archive=ROOT.parents[1]/"ESIK-Optuna-v2.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
    for path in sorted(files):
        output.write(path,Path(ROOT.name)/path.relative_to(ROOT))
with zipfile.ZipFile(archive) as packed:
    assert packed.testzip() is None
    assert not any("/.venv/" in name or name.endswith("/secrets.toml") or name.endswith("/.env")
                   for name in packed.namelist())
delivery=dict(archive=str(archive),sha256=sha256(archive),bytes=archive.stat().st_size,
              files=len(files),model_version=version,passed_tests=int(match.group(1)),
              dashboard_pages=8,external_api_tested=False)
write_json(ROOT.parents[1]/"ESIK-Optuna-v2-teslim.json",delivery)
print(json.dumps(delivery,ensure_ascii=False))
