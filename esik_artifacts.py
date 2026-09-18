##################################################
# EŞİK - MODEL, SKOR VE SHAP ÇIKTILARININ BİRLİKTE ÜRETİLMESİ
##################################################
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from esik_core import (BASE_DIR, FEATURES, load_data, capacity_metrics,
                       score_metrics, sha256, write_json)

MODEL_NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM"}
DESCRIPTIONS = json.loads((BASE_DIR/"feature_descriptions.json").read_text(encoding="utf-8"))


##################################################
# ŞİRKET KİMLİKLERİ VE ÖNCELİK LİSTESİ
##################################################
def priority_fields(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Öncelik listesi için sonlu skorlardan oluşan en az bir kayıt gerekir.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model skorları 0–1 aralığında olmalıdır.")
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores)+1)
    critical, high, medium = [int(np.ceil(len(scores)*q)) for q in [.05,.10,.20]]
    return pd.DataFrame(dict(RISK_RANK=ranks,
        RISK_PERCENTILE=(len(scores)-ranks+1)/len(scores)*100,
        RISK_GROUP=np.select([ranks<=critical, ranks<=high, ranks<=medium],
            ["Kritik - İlk %5", "Yüksek - %5 ile %10", "Orta - %10 ile %20"],
            default="Düşük - %20 sonrası"), ANALYST_REVIEW=(ranks<=high).astype(int)))


def scored_companies(pipeline, X_test, y_test):
    scores = pipeline.predict_proba(X_test)[:,1]
    frame = X_test.copy()
    frame.insert(0, "COMPANY_ID", [f"ESIK-{i:05d}" for i in X_test.index])
    frame.insert(1, "ACTUAL_BANKRUPT", y_test.to_numpy())
    frame.insert(2, "ACTUAL_STATUS", np.where(y_test.to_numpy() == 1, "İflas Etti", "İflas Etmedi"))
    frame.insert(3, "RISK_SCORE", scores)
    frame = frame.sort_values("RISK_SCORE", ascending=False, kind="stable").reset_index(drop=True)
    fields = priority_fields(frame.RISK_SCORE)
    for column in fields:
        frame[column] = fields[column].to_numpy()
    frame["RECOMMENDED_ACTION"] = frame.RISK_GROUP.map({
        "Kritik - İlk %5": "Acil analist incelemesi",
        "Yüksek - %5 ile %10": "Öncelikli analist incelemesi",
        "Orta - %10 ile %20": "Yakın izleme listesi", "Düşük - %20 sonrası": "Rutin izleme"})
    return frame


##################################################
# SHAP: AYNI MODEL, AYNI GİRDİLER, HAM LOG-ODDS ÖLÇEĞİ
##################################################
def generate_shap(pipeline, scored, output_dir):
    import shap
    original = scored[FEATURES]
    imputed = pd.DataFrame(pipeline.named_steps["imputer"].transform(original), columns=FEATURES)
    explanation = shap.TreeExplainer(pipeline.named_steps["model"], model_output="raw")(imputed)
    values, base = np.asarray(explanation.values), np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:,:,1]
        base = base[:,1]
    if values.shape != imputed.shape:
        raise ValueError("SHAP ve model girdilerinin boyutları uyuşmuyor.")
    reconstructed = expit(np.asarray(base).reshape(-1) + values.sum(axis=1))
    residual = float(np.max(np.abs(reconstructed-scored.RISK_SCORE.to_numpy())))
    if residual > 2e-5:
        raise ValueError(f"SHAP katkıları model skorunu yeniden oluşturmuyor: {residual}")
    global_frame = pd.DataFrame(dict(VARIABLE=FEATURES,
                         DESCRIPTION=[DESCRIPTIONS[x] for x in FEATURES],
                         MEAN_ABS_SHAP=np.abs(values).mean(axis=0)))
    global_frame["SHAP_IMPORTANCE_RATIO"] = global_frame.MEAN_ABS_SHAP/global_frame.MEAN_ABS_SHAP.sum()
    global_frame = global_frame.sort_values("MEAN_ABS_SHAP", ascending=False).reset_index(drop=True)
    global_frame["SHAP_RANK"] = np.arange(1, len(FEATURES)+1)
    local_frame = pd.DataFrame(dict(
        COMPANY_ID=np.repeat(scored.COMPANY_ID.to_numpy(), len(FEATURES)),
        VARIABLE=np.tile(FEATURES, len(scored)), SHAP_VALUE=values.reshape(-1),
        MODEL_INPUT_VALUE=imputed.to_numpy().reshape(-1),
        WAS_MISSING=original.isna().to_numpy().reshape(-1)))
    local_frame["ABS_SHAP_VALUE"] = local_frame.SHAP_VALUE.abs()
    local_frame["DESCRIPTION"] = local_frame.VARIABLE.map(DESCRIPTIONS)
    local_frame["SHAP_DIRECTION"] = np.select([local_frame.SHAP_VALUE > 0, local_frame.SHAP_VALUE < 0],
                                 ["Riski Artırıyor", "Riski Azaltıyor"], default="Nötr")
    local_frame = local_frame.sort_values(["COMPANY_ID", "ABS_SHAP_VALUE"], ascending=[True, False])
    global_frame.to_csv(output_dir/"esik_shap_global_importance.csv", index=False)
    local_frame.to_csv(output_dir/"esik_shap_local_values.csv", index=False)
    return dict(scale="raw_log_odds", maximum_probability_reconstruction_error=residual,
                company_count=len(scored), local_rows=len(local_frame))


##################################################
# TARİHSEL TEST RAPORU VE UYGULAMA DOSYALARI
##################################################
def validate_training_provenance(candidate, protocol_path):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(protocol_path) != candidate["search_summary"]["protocol_sha256"]:
        raise ValueError("Aday modelin deney protokolü değişmiş.")
    if sha256(BASE_DIR/"datasets/5year.arff") != protocol["data"]["dataset_sha256"]:
        raise ValueError("Eğitimden sonra kaynak veri değişmiş; bu modelle çıktı üretilemez.")
    for name, expected in protocol["training_code"].items():
        if name not in {"esik_core.py", "esik_optuna.py"} or sha256(BASE_DIR/name) != expected:
            raise ValueError("Kayıtlı eğitim kodu ile mevcut kod uyuşmuyor.")
    for key in ["development_ids", "holdout_ids"]:
        if candidate[key] != protocol[key]:
            raise ValueError("Adayın kayıt ayrımı protokolle eşleşmiyor.")
    return protocol


def generate_release(candidate, release_dir, model_version, protocol_path=None):
    if protocol_path is not None:
        validate_training_provenance(candidate, protocol_path)
    release_dir = Path(release_dir)
    output_dir, model_dir = release_dir/"outputs", release_dir/"models"
    if (output_dir/"artifact_manifest.json").exists():
        raise ValueError("Bu sürüm zaten üretilmiş. Üzerine yazmak için yeni sürüm adı seçin.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    X, y, development, holdout, data_report = load_data()
    if candidate["development_ids"] != development.tolist() or candidate["holdout_ids"] != holdout.tolist():
        raise ValueError("Modelin veri ayrımı mevcut veriyle eşleşmiyor.")
    pipeline, search = candidate["pipeline"], candidate["search_summary"]
    winner = search["winner"]
    scored = scored_companies(pipeline, X.loc[holdout], y.loc[holdout])
    metrics = score_metrics(scored.ACTUAL_BANKRUPT, scored.RISK_SCORE)
    metadata = dict(project_name="Eşik", model_version=model_version,
                    model_name=MODEL_NAMES[winner["family"]],
                    prediction_target="Bir yıl içerisinde iflas riski",
                    dataset="UCI Polish Companies Bankruptcy - 5year", feature_count=64,
                    feature_columns=FEATURES, analyst_capacity=.1,
                    best_parameters=winner["parameters"],
                    cv_recall_at_top_10=winner["summary"]["mean_recall_at_10"],
                    cv_score_role="Internal selection score; not independent generalization estimate",
                    validation_status="Tarihsel test daha önce incelendi; bağımsız dış veri doğrulaması yok.",
                    nested_validation=search["nested"],
                    protocol_sha256=search["protocol_sha256"],
                    dataset_sha256=data_report["dataset_sha256"],
                    feature_dictionary_sha256=sha256(BASE_DIR/"feature_descriptions.json"),
                    feature_dictionary_note="Attr48/49 kaynak tanımı belirsiz; docs/veri-sozlugu-notlari.md",
                    development_ids=development.tolist(), holdout_ids=holdout.tolist(),
                    score_definition="Kalibre edilmiş iflas olasılığı değil, analist incelemesi için sıralama skorudur.")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "pr_auc_trapezoid"]:
        metadata[f"test_{key}"] = metrics[key]
    metadata.update(test_recall_at_top_10=metrics["recall_at_10"],
                    test_precision_at_top_10=metrics["precision_at_10"],
                    test_lift_at_top_10=metrics["lift_at_10"],
                    test_pr_auc=metrics["average_precision"],
                    legacy_metric_note="test_pr_auc is retained only as a legacy alias of Average Precision.",
                    effective_model_parameters={
                        key: (str(value) if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
                        for key,value in pipeline.named_steps["model"].get_params().items()})
    metadata["shap_validation"] = generate_shap(pipeline, scored, output_dir)
    package = dict(pipeline=pipeline, metadata=metadata)
    model_path = model_dir/"esik_final_model_pipeline.pkl"
    joblib.dump(package, model_path)
    loaded = joblib.load(model_path)
    roundtrip_error = float(np.max(np.abs(loaded["pipeline"].predict_proba(scored[FEATURES])[:,1]-scored.RISK_SCORE)))
    if roundtrip_error > 1e-7:
        raise ValueError("Kaydet-yükle kontrolünde skorlar değişti.")
    write_json(output_dir/"esik_model_metadata.json", metadata)
    scored.to_csv(output_dir/"esik_scored_test_companies.csv", index=False)
    risk_columns = ["COMPANY_ID", "RISK_RANK", "RISK_SCORE", "RISK_PERCENTILE", "RISK_GROUP",
                    "RECOMMENDED_ACTION", "ANALYST_REVIEW", "ACTUAL_BANKRUPT", "ACTUAL_STATUS"]
    scored[risk_columns].to_csv(output_dir/"esik_dashboard_risk_list.csv", index=False)
    capacities = []
    for q in [.05,.1,.2]:
        row = capacity_metrics(scored.ACTUAL_BANKRUPT, scored.RISK_SCORE, q)
        capacities.append(dict(CAPACITY=q, REVIEWED_COMPANIES=row["reviewed"],
                  CAPTURED_BANKRUPT=row["captured"], TOTAL_BANKRUPT=row["total_positive"],
                  RECALL_AT_CAPACITY=row["recall"], PRECISION_AT_CAPACITY=row["precision"],
                  LIFT_AT_CAPACITY=row["lift"]))
    pd.DataFrame(capacities).to_csv(output_dir/"esik_capacity_results.csv", index=False)
    pd.DataFrame([dict(METRIC=k, TEST_SCORE=v) for k,v in metrics.items() if k != "confusion_matrix"]).to_csv(
        output_dir/"esik_final_test_metrics.csv", index=False)
    for key, filename in [("summary", "esik_tuned_model_comparison.csv"),
                           ("baseline_summary", "esik_baseline_model_comparison.csv")]:
        rows = [dict(MODEL=MODEL_NAMES[r["family"]], **r[key]) for r in search["final_comparison"]]
        pd.DataFrame(rows).to_csv(output_dir/filename, index=False)
    importance = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
    importance = importance/importance.sum() if importance.sum() else importance
    feature_frame = pd.DataFrame(dict(VARIABLE=FEATURES, DESCRIPTION=[DESCRIPTIONS[f] for f in FEATURES],
                                      IMPORTANCE=importance)).sort_values("IMPORTANCE", ascending=False)
    feature_frame.to_csv(output_dir/"esik_feature_importance.csv", index=False)
    direction = feature_frame.copy()
    for label, prefix in [(0,"NON_BANKRUPT"),(1,"BANKRUPT")]:
        subset = X.loc[development].loc[y.loc[development] == label]
        direction[f"{prefix}_MEDIAN"] = direction.VARIABLE.map(subset.median().to_dict())
        direction[f"{prefix}_MISSING_RATIO"] = direction.VARIABLE.map(subset.isna().mean().to_dict())
    direction["MEDIAN_DIFFERENCE"] = direction.BANKRUPT_MEDIAN-direction.NON_BANKRUPT_MEDIAN
    direction.to_csv(output_dir/"esik_financial_direction.csv", index=False)
    dictionary_path = release_dir/"feature_descriptions.json"
    shutil.copy2(BASE_DIR/"feature_descriptions.json", dictionary_path)
    files = [model_path, dictionary_path] + sorted(output_dir.glob("*.csv")) + [output_dir/"esik_model_metadata.json"]
    manifest = dict(model_version=model_version, model_sha256=sha256(model_path),
                    files={p.relative_to(release_dir).as_posix():sha256(p) for p in files},
                    roundtrip_max_error=roundtrip_error, shap=metadata["shap_validation"])
    write_json(output_dir/"artifact_manifest.json", manifest)
    validate_artifacts(release_dir)
    return metadata


def validate_artifacts(root):
    root = Path(root)
    manifest = json.loads((root/"outputs/artifact_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model/çıktı sürümü uyuşmuyor: {relative}")
    return manifest


def install_release(release_dir):
    release_dir = Path(release_dir)
    manifest = validate_artifacts(release_dir)
    for relative in manifest["files"]:
        destination = BASE_DIR/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release_dir/relative, destination)
    shutil.copy2(release_dir/"outputs/artifact_manifest.json", BASE_DIR/"outputs/artifact_manifest.json")
    validate_artifacts(BASE_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seçilmiş EŞİK modelinin uygulama çıktıları")
    parser.add_argument("experiment")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if not args.experiment.replace("-", "").replace("_", "").isalnum():
        parser.error("Geçersiz deney adı")
    candidate = joblib.load(BASE_DIR/"experiments"/args.experiment/"candidate.pkl")
    release = BASE_DIR/"releases"/args.experiment
    meta = generate_release(candidate, release, args.experiment,
                            BASE_DIR/"experiments"/args.experiment/"protocol.json")
    if args.install:
        install_release(release)
    print(json.dumps(dict(model=meta["model_name"], recall=meta["test_recall_at_top_10"],
                          average_precision=meta["test_average_precision"]), ensure_ascii=False))
