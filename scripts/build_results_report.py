"""Güncel, eşlenmiş aile karşılaştırmasını kaydedilmiş sonuçlardan özetler."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'karsilastirma/comparison_report.json').read_text(encoding='utf-8'))
lines=['# EŞİK aile karşılaştırması','',
 '| Aile | Dış kat yakalama | Tarihsel yakalama | Yakalanan | Kalibre test Brier |',
 '|---|---:|---:|---:|---:|']
for family,v in r['models'].items():
 lines.append(f"| {family} | {v['outer']['mean_recall_at_10']:.2%} | {v['historical_ranking']['recall_at_10']:.2%} | {v['historical_ranking']['captured']}/82 | {v['calibration']['historical_test']['calibrated']['brier']:.5f} |")
lines+=['','Dış kat ortalamasında XGBoost; tarihsel testte LightGBM önde. İki veri bölümü de daha önce incelendi. Ayrıntılı karar: MODEL_KARARI_VE_KARSILASTIRMA.md.']
(ROOT/'docs/sonuc_ozeti.md').write_text('\n'.join(lines),encoding='utf-8')
print('docs/sonuc_ozeti.md güncellendi.')
