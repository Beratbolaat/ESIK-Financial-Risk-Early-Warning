"""Kayıtlı modeldeki gerçek boosting turunu okur; yeniden eğitim yapmaz."""
from pathlib import Path
import json,sys,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from esik_core import sha256
model=joblib.load(ROOT/'models/esik_final_model_pipeline.pkl')['pipeline'].named_steps['model']
actual=model.get_booster().num_boosted_rounds() if hasattr(model,'get_booster') else int(model.n_estimators_)
report=dict(model=type(model).__name__,requested=model.n_estimators,actual=actual,
 model_sha256=sha256(ROOT/'models/esik_final_model_pipeline.pkl'),retrained=False)
(ROOT/'outputs/tree_count_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
