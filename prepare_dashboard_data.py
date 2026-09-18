"""Kayıtlı XGBoost modelinden yeni bir klasöre tutarlı skor/SHAP çıktısı üretir.
Eğitim ve kalibrasyon yenilenmez. Kaynak model, çalışan uygulama ve çıktılar değişmez.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,joblib,shutil,copy
from esik_core import BASE_DIR,sha256,write_json
from esik_artifacts import generate_release,validate_artifacts
from esik_calibration import load_probability_layer

def rebuild_current():
 original=validate_artifacts(BASE_DIR)
 pkg=joblib.load(BASE_DIR/'models/esik_final_model_pipeline.pkl');metadata=pkg['metadata']
 family=metadata.get('validation_key')
 if family not in ('xgboost','lightgbm'):raise ValueError('Bu yenileme aile karşılaştırma sürümü içindir.')
 summary=json.loads((BASE_DIR/'experiments/optuna-v2/summary.json').read_text(encoding='utf-8'))
 summary['winner']=copy.deepcopy(next(v for v in summary['final_comparison'] if v['family']==family))
 protocol=BASE_DIR/'comparison_evidence/release_protocol.json'
 summary['protocol_sha256']=sha256(protocol)
 candidate=dict(pipeline=pkg['pipeline'],search_summary=summary,
  development_ids=metadata['development_ids'],holdout_ids=metadata['holdout_ids'])
 target=BASE_DIR/'releases'/('refresh-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
 generate_release(candidate,target,metadata['model_version'],protocol)
 # Preserve the exact fitted-model package bytes and its metadata. Only deterministic
 # model-derived tables are regenerated, so the existing calibrated mapping stays valid.
 for name in ['models/esik_final_model_pipeline.pkl','outputs/esik_model_metadata.json']:
  shutil.copy2(BASE_DIR/name,target/name)
 manifest=json.loads((target/'outputs/artifact_manifest.json').read_text(encoding='utf-8'))
 manifest['model_sha256']=original['model_sha256']
 manifest['files']={name:sha256(target/name) for name in manifest['files']}
 write_json(target/'outputs/artifact_manifest.json',manifest)
 shutil.copytree(BASE_DIR/'outputs/calibration_v1',target/'outputs/calibration_v1')
 validate_artifacts(target)
 assert load_probability_layer(target,original['model_sha256']) is not None
 print('Yeni skor ve SHAP çıktıları: '+str(target))
 return target

if __name__=='__main__':rebuild_current()
