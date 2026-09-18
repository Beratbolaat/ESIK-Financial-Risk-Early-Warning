import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from esik_core import (capacity_metrics, fit_and_score, BASELINE_PARAMS,
                       load_data, make_folds, create_pipeline)
from esik_optuna import prepare_protocol, run_study


def test_capacity_counts_and_stable_ties():
    y = np.array([1]*8+[0]*2+[1]*12+[0]*78)
    result = capacity_metrics(y, np.zeros(100))
    assert result["reviewed"] == 10
    assert result["captured"] == 8
    assert result["recall"] == .4
    assert result["precision"] == .8
    assert result["lift"] == 4
    with pytest.raises(ValueError):
        capacity_metrics(y, np.full(100, np.nan))


def test_validation_extremes_do_not_change_imputer_or_class_weight():
    X = pd.DataFrame({"Attr1": [1., 3., np.nan, 5., 100000., -100000.],
                      "Attr2": [2.,4.,6.,8.,20.,30.]})
    y = pd.Series([0,0,0,1,0,1])
    params = dict(BASELINE_PARAMS["xgboost"], n_estimators=2, max_depth=2)
    pipe, _, _ = fit_and_score("xgboost", params, X, y, [0,1,2,3], [4,5], 1)
    assert pipe.named_steps["imputer"].statistics_[0] == 3.
    assert pipe.named_steps["model"].get_params()["scale_pos_weight"] == 3.
    with pytest.raises(ValueError, match="kesişiyor"):
        fit_and_score("xgboost", params, X, y, [0,1,2,3], [3,4], 1)


def test_historical_holdout_never_enters_development_folds():
    X, y, development, holdout, _ = load_data()
    all_valid = []
    for train, valid in make_folds(X.loc[development], y.loc[development], 5):
        assert not set(train) & set(valid)
        assert not (set(train) | set(valid)) & set(holdout)
        all_valid.extend(valid)
    assert sorted(all_valid) == sorted(development)


def test_lightgbm_balancing_uses_only_fit_labels():
    pipe = create_pipeline("lightgbm", BASELINE_PARAMS["lightgbm"], pd.Series([0,0,0,1]), 1)
    assert pipe.named_steps["model"].class_weight == {0: 4/6, 1: 2.}


def test_protocol_rejects_changed_budget(tmp_path):
    X, y, development, holdout, report = load_data()
    args = SimpleNamespace(mode="pilot", trials=2, inner_folds=3, outer_folds=0, threads=1)
    prepare_protocol(tmp_path, X, y, development, holdout, report, args)
    args.trials = 3
    with pytest.raises(ValueError, match="protokolü değişmiş"):
        prepare_protocol(tmp_path, X, y, development, holdout, report, args)


def test_study_resume_does_not_repeat_completed_fits(tmp_path, monkeypatch):
    import esik_optuna
    calls = []
    def fake_fit(*args):
        calls.append(1)
        row = dict(recall_at_10=.7, precision_at_10=.5, lift_at_10=5.,
                   average_precision=.6, roc_auc=.8, seconds=.01)
        return None, None, row
    monkeypatch.setattr(esik_optuna, "fit_and_score", fake_fit)
    folds = [(np.array([0,1]), np.array([2,3]))]*3
    run_study("xgboost", None, None, folds, tmp_path, 2, 1, 17)
    assert len(calls) == 6
    resumed = run_study("xgboost", None, None, folds, tmp_path, 2, 1, 17)
    assert len(calls) == 6
    assert resumed["trials"] == 2
    assert len(pd.read_csv(tmp_path/"fold_metrics.csv")) == 6
