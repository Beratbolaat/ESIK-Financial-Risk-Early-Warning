"""Bu sürümün ayrı XGBoost kalibrasyonunu ve hash'lerini doğrular.
Yeniden eğitim için karsilastirma/compare_esik_families.py kullanılır.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from esik_artifacts import validate_artifacts
from esik_calibration import load_probability_layer
manifest=validate_artifacts(ROOT);layer=load_probability_layer(ROOT,manifest['model_sha256'])
assert layer is not None
r=layer['report']
print(json.dumps({k:r[k] for k in ['family','scope','outer_raw','outer_calibrated','deployment_gate','historical_test']},ensure_ascii=False,indent=2))
