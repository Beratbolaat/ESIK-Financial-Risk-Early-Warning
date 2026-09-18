import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    required_paths = [
        ROOT / "app.py",
        ROOT / "n8n_client.py",
        ROOT / "requirements.txt",
        ROOT / "README.md",
        ROOT / ".gitignore",
        ROOT / "models" / "esik_final_model_pipeline.pkl",
        ROOT / "outputs" / "esik_model_metadata.json",
        ROOT / "outputs" / "esik_scored_test_companies.csv",
        ROOT / "n8n" / "esik-ai-workflow.json",
        ROOT / "docs" / "n8n_setup.md",
    ]
    assert all(path.exists() for path in required_paths)


def test_metadata_features_match_scored_data():
    with open(
        ROOT / "outputs" / "esik_model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    scored_df = pd.read_csv(
        ROOT / "outputs" / "esik_scored_test_companies.csv",
        nrows=2,
    )

    missing_features = [
        feature
        for feature in metadata["feature_columns"]
        if feature not in scored_df.columns
    ]

    assert not missing_features
    assert "RISK_SCORE" in scored_df.columns


def test_saved_model_scores_one_company():
    model_package = joblib.load(
        ROOT / "models" / "esik_final_model_pipeline.pkl"
    )
    pipeline = model_package["pipeline"]

    with open(
        ROOT / "outputs" / "esik_model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    scored_df = pd.read_csv(
        ROOT / "outputs" / "esik_scored_test_companies.csv",
        nrows=1,
    )
    sample = scored_df[metadata["feature_columns"]]

    score = float(pipeline.predict_proba(sample)[0, 1])
    assert 0.0 <= score <= 1.0
