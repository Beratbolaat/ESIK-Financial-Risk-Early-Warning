import json
import joblib
import numpy as np
import pandas as pd
import pytest

from esik_core import BASELINE_PARAMS, load_data, create_pipeline, score_metrics, sha256
from esik_artifacts import (generate_release, validate_artifacts,
                            validate_training_provenance, priority_fields)


@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_release_scores_shap_and_manifest_use_same_model(tmp_path, family):
    # Küçük ağaç sayısı yalnız test süresi içindir; performans deneyi değildir.
    X, y, development, holdout, _ = load_data()
    params = dict(BASELINE_PARAMS[family], n_estimators=3)
    pipe = create_pipeline(family, params, y.loc[development], threads=1)
    pipe.fit(X.loc[development], y.loc[development])
    winner = dict(family=family, parameters=params,
                  summary={"mean_recall_at_10": 0.},
                  baseline_summary={"mean_recall_at_10": 0.})
    candidate = dict(pipeline=pipe, development_ids=development.tolist(),
                     holdout_ids=holdout.tolist(),
                     search_summary=dict(winner=winner, nested={},
                         final_comparison=[winner], protocol_sha256="test-fixture"))
    root = tmp_path/family
    metadata = generate_release(candidate, root, "test-fixture")
    manifest = validate_artifacts(root)
    assert manifest["shap"]["company_count"] == len(holdout)
    assert manifest["shap"]["local_rows"] == len(holdout)*64
    assert manifest["shap"]["maximum_probability_reconstruction_error"] < 2e-5
    scored = pd.read_csv(root/"outputs/esik_scored_test_companies.csv")
    loaded = joblib.load(root/"models/esik_final_model_pipeline.pkl")
    np.testing.assert_allclose(loaded["pipeline"].predict_proba(scored[X.columns])[:,1],
                               scored.RISK_SCORE, atol=1e-7, rtol=0)
    assert set(scored.COMPANY_ID) == {f"ESIK-{i:05d}" for i in holdout}
    direct = score_metrics(scored.ACTUAL_BANKRUPT, scored.RISK_SCORE)
    assert metadata["test_recall_at_top_10"] == direct["recall_at_10"]
    assert metadata["test_average_precision"] == direct["average_precision"]
    shap_rows = pd.read_csv(root/"outputs/esik_shap_local_values.csv")
    assert int(shap_rows.WAS_MISSING.sum()) == int(X.loc[holdout].isna().sum().sum())
    with pytest.raises(ValueError, match="zaten üretilmiş"):
        generate_release(candidate, root, "test-fixture")
    # Eski bir skor dosyasının yenisiyle karışmasını yakalamalı.
    scored.loc[0, "RISK_SCORE"] = -1
    scored.to_csv(root/"outputs/esik_scored_test_companies.csv", index=False)
    with pytest.raises(ValueError, match="sürümü uyuşmuyor"):
        validate_artifacts(root)


def test_export_rejects_changed_training_data(tmp_path):
    protocol_path = tmp_path/"protocol.json"
    protocol_path.write_text(json.dumps({"data":{"dataset_sha256":"different-data"}}),encoding="utf-8")
    candidate = {"search_summary":{"protocol_sha256":sha256(protocol_path)}}
    with pytest.raises(ValueError, match="kaynak veri değişmiş"):
        validate_training_provenance(candidate, protocol_path)
    protocol_path.write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError, match="protokolü değişmiş"):
        validate_training_provenance(candidate, protocol_path)


def test_priority_capacity_boundaries_and_ties():
    ranked = priority_fields(np.full(100, .5))
    assert ranked.RISK_RANK.tolist() == list(range(1,101))
    assert ranked.RISK_GROUP.value_counts().to_dict() == {
        "Kritik - İlk %5": 5, "Yüksek - %5 ile %10": 5,
        "Orta - %10 ile %20": 10, "Düşük - %20 sonrası": 80}
    assert ranked.ANALYST_REVIEW.sum() == 10
    assert priority_fields(np.zeros(1170)).ANALYST_REVIEW.sum() == 117
    assert priority_fields(np.zeros(936)).ANALYST_REVIEW.sum() == 94
