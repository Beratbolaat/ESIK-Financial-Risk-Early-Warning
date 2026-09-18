##################################################
# EŞİK - VERİ, PIPELINE VE DEĞERLENDİRME
##################################################
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, precision_recall_curve, auc)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = Path(__file__).resolve().parent
SEED = 17
CAPACITY = 0.10
FEATURES = [f"Attr{i}" for i in range(1, 65)]

# Eski modelin ayarları. Aynı protokolde yeniden eğitilen referans.
BASELINE_PARAMS = {
    "xgboost": dict(n_estimators=400, max_depth=5, learning_rate=.05,
                    min_child_weight=5., subsample=.8, colsample_bytree=.8,
                    reg_alpha=0., reg_lambda=1., gamma=0., weight_multiplier=1.),
    "lightgbm": dict(n_estimators=200, learning_rate=.05, num_leaves=31,
                     max_depth=-1, min_child_samples=20, subsample=1.,
                     colsample_bytree=1., reg_alpha=0., reg_lambda=0.,
                     min_split_gain=0., weight_multiplier=1.),
    "gradient_boosting": dict(n_estimators=400, learning_rate=.05,
                              max_depth=3, min_samples_leaf=5, subsample=.8),
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


##################################################
# VERİ VE SABİT EĞİTİM / TEST AYRIMI
##################################################
def load_data(path=None):
    path = Path(path or BASE_DIR / "datasets/5year.arff")
    frame = pd.DataFrame(arff.loadarff(path)[0])
    frame["class"] = frame["class"].map(
        lambda value: int(value.decode()) if isinstance(value, bytes) else int(value))
    raw_rows = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    X, y = frame[FEATURES].copy(), frame["class"].rename("BANKRUPT")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Sonsuz finansal değer var; veri kuralı açıkça belirlenmeli.")
    if set(y.unique()) != {0, 1} or X.isna().all().any():
        raise ValueError("Hedef veya tamamen eksik sütun kontrolü başarısız.")
    train_ids, test_ids = train_test_split(X.index.to_numpy(), test_size=.2,
                                         random_state=SEED, stratify=y)
    report = dict(raw_rows=raw_rows, clean_rows=len(frame),
                  duplicates_removed=raw_rows-len(frame), positive=int(y.sum()),
                  train_rows=len(train_ids), test_rows=len(test_ids),
                  missing_fraction=X.isna().mean().to_dict(), dataset_sha256=sha256(path))
    return X, y, np.asarray(train_ids), np.asarray(test_ids), report


def make_folds(X, y, n_splits=3, seed=SEED):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(X.index[a].to_numpy(), X.index[b].to_numpy()) for a,b in cv.split(X, y)]


##################################################
# KAPASİTE VE SINIFLANDIRMA METRİKLERİ
##################################################
def capacity_metrics(y_true, scores, capacity=CAPACITY):
    labels, scores = np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    if not len(labels) or len(labels) != len(scores) or not 0 < capacity <= 1:
        raise ValueError("Geçersiz değerlendirme boyutu veya kapasite.")
    if not np.isfinite(scores).all() or not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("Geçersiz skor veya hedef.")
    reviewed = math.ceil(len(labels) * capacity)
    # Eşit skorlarda giriş sırası korunur; metrikler aynı listeyi kullanır.
    top = np.argsort(-scores, kind="stable")[:reviewed]
    caught, total = int(labels[top].sum()), int(labels.sum())
    precision = caught / reviewed
    return dict(capacity=capacity, reviewed=reviewed, captured=caught, total_positive=total,
                recall=caught/total if total else 0., precision=precision,
                lift=precision/labels.mean() if total else 0.)


def score_metrics(y, scores):
    capacity = capacity_metrics(y, scores)
    predicted = np.asarray(scores) >= .5
    pr_precision, pr_recall, _ = precision_recall_curve(y, scores)
    return dict(recall_at_10=capacity["recall"], precision_at_10=capacity["precision"],
                lift_at_10=capacity["lift"], captured=capacity["captured"],
                reviewed=capacity["reviewed"], positives=capacity["total_positive"],
                average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)),
                pr_auc_trapezoid=float(auc(pr_recall, pr_precision)),
                accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                confusion_matrix=confusion_matrix(y, predicted, labels=[0,1]).tolist())


##################################################
# PIPELINE: MEDYAN YALNIZCA EĞİTİM KISMINDA ÖĞRENİLİR
##################################################
def create_pipeline(family, parameters, y_fit, threads=2):
    parameters = dict(parameters)
    multiplier = parameters.pop("weight_multiplier", 1.)
    negatives, positives = int((y_fit == 0).sum()), int((y_fit == 1).sum())
    if min(negatives, positives) == 0:
        raise ValueError("Eğitim katında her iki sınıf bulunmalı.")
    if family == "xgboost":
        model = XGBClassifier(**parameters, scale_pos_weight=negatives/positives*multiplier,
                              objective="binary:logistic", eval_metric="logloss",
                              random_state=SEED, n_jobs=threads, tree_method="hist")
    elif family == "lightgbm":
        # multiplier=1, sklearn class_weight='balanced' ile aynı ağırlıklar.
        class_weights = {0: len(y_fit)/(2*negatives),
                         1: len(y_fit)/(2*positives)*multiplier}
        model = LGBMClassifier(**parameters, class_weight=class_weights,
                               objective="binary", random_state=SEED,
                               subsample_freq=1 if parameters["subsample"] < 1 else 0,
                               n_jobs=threads, verbosity=-1, deterministic=True,
                               force_col_wise=True)
    elif family == "gradient_boosting":
        model = GradientBoostingClassifier(**parameters, random_state=SEED)
    else:
        raise ValueError(f"Bilinmeyen model ailesi: {family}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def fit_and_score(family, parameters, X, y, fit_ids, valid_ids, threads=2):
    fit_ids, valid_ids = np.asarray(fit_ids), np.asarray(valid_ids)
    if np.intersect1d(fit_ids, valid_ids).size:
        raise ValueError("Eğitim ve doğrulama kayıtları kesişiyor.")
    started = time.perf_counter()
    pipeline = create_pipeline(family, parameters, y.loc[fit_ids], threads)
    pipeline.fit(X.loc[fit_ids], y.loc[fit_ids])
    scores = pipeline.predict_proba(X.loc[valid_ids])[:,1]
    metrics = score_metrics(y.loc[valid_ids], scores)
    metrics.update(seconds=time.perf_counter()-started, train_rows=len(fit_ids),
                   valid_rows=len(valid_ids), train_positive=int(y.loc[fit_ids].sum()),
                   weight_ratio=float((y.loc[fit_ids] == 0).sum()/y.loc[fit_ids].sum()))
    return pipeline, scores, metrics


def summarize_folds(rows):
    names = ["recall_at_10", "precision_at_10", "lift_at_10", "average_precision", "roc_auc"]
    output = {f"mean_{key}": float(np.mean([r[key] for r in rows])) for key in names}
    output["std_recall_at_10"] = float(np.std([r["recall_at_10"] for r in rows]))
    output["seconds"] = float(sum(r["seconds"] for r in rows))
    return output
