"""Optional probability layer with provenance; original ranking model stays immutable."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def raw_margin(pipeline, X):
    inputs = pipeline.named_steps['imputer'].transform(X)
    model = pipeline.named_steps['model']
    if model.__class__.__name__ == 'LGBMClassifier':
        values = model.predict(inputs, raw_score=True)
    elif model.__class__.__name__ == 'XGBClassifier':
        values = model.predict(inputs, output_margin=True)
    else:
        raise ValueError('Unsupported calibration model family')
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Raw margins must be finite and one-dimensional')
    return values


def fit_sigmoid(margins, labels):
    """Unweighted sigmoid on held-out raw margins, with positive slope constraint."""
    margins = np.asarray(margins, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if margins.ndim != 1 or margins.shape != labels.shape or not np.isfinite(margins).all():
        raise ValueError('Invalid calibration observations')
    if set(np.unique(labels)) != {0., 1.}:
        raise ValueError('Calibration requires both classes')
    center, scale = float(margins.mean()), float(margins.std())
    if scale < 1e-12:
        raise ValueError('Calibration needs nonconstant scores')
    x = (margins-center)/scale
    def objective(theta):
        z = theta[0]*x+theta[1]
        error = expit(z)-labels
        return float(np.mean(np.logaddexp(0, z)-labels*z)), np.array([np.mean(error*x), np.mean(error)])
    prior = labels.mean()
    result = minimize(objective, [1., np.log(prior/(1-prior))], jac=True,
                      method='L-BFGS-B', bounds=[(1e-6, None), (None, None)],
                      options={'maxiter':1000, 'ftol':1e-12, 'gtol':1e-9})
    if not result.success:
        raise ValueError('Sigmoid calibration did not converge')
    return dict(method='positive_slope_sigmoid_on_raw_margin', center=center, scale=scale,
                slope=float(result.x[0]), intercept=float(result.x[1]),
                calibration_rows=len(labels), calibration_positives=int(labels.sum()))


def probability_from_margin(margins, calibration):
    values = np.asarray(margins, dtype=float)
    if not np.isfinite(values).all() or calibration['scale'] <= 0 or calibration['slope'] <= 0:
        raise ValueError('Invalid margin or calibration parameters')
    return expit(calibration['slope']*(values-calibration['center'])/calibration['scale']+calibration['intercept'])


def load_probability_layer(root, model_hash):
    """Fail closed on mismatched files; no silent substitution of another model."""
    folder = Path(root)/'outputs/calibration_v1'
    if not (folder/'manifest.json').exists():
        return None
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['model_sha256'] != model_hash:
        raise ValueError('Olasılık katmanı farklı bir model sürümüne ait.')
    for name, digest in manifest['files'].items():
        file = (folder/name).resolve()
        if not file.is_relative_to(folder.resolve()) or not file.is_file():
            raise ValueError('Olasılık katmanı dosyası eksik.')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('Olasılık katmanı dosya bütünlüğü doğrulanamadı.')
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if not report['deployment_gate']['passed']:
        return None
    import pandas as pd
    rows = pd.read_csv(folder/'company_probabilities.csv').set_index('COMPANY_ID')
    if not rows.index.is_unique or not rows.CALIBRATED_PROBABILITY.between(0,1).all():
        raise ValueError('Olasılık kayıtları geçersiz.')
    return dict(report=report, rows=rows, folder=folder)


def company_probability(layer, company_id):
    if layer is None or company_id not in layer['rows'].index:
        return None
    row = layer['rows'].loc[company_id]
    probability = float(row.CALIBRATED_PROBABILITY)
    bins = layer['report']['outer_reliability']['calibrated']
    band = next((b for b in bins if b['lower'] <= probability and
                 (probability < b['upper'] or b['upper'] == 1)), None)
    return dict(estimate=probability, horizon='1 yıl',
                method='Geliştirme verisindeki kat dışı tahminlerle sigmoid kalibrasyon',
                status='Keşifsel kalibrasyon tahmini; güncel yerel dış doğrulama yapılmadı',
                validation_band=band,
                validation_band_note='Aralık, aynı tahmin bandındaki doğrulama grubunun iflas oranına aittir; bireysel şirket olasılığının güven aralığı değildir.')
