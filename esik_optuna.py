##################################################
# EŞİK - OPTUNA VE İÇ İÇE ÇAPRAZ DOĞRULAMA
# Her deneme aynı katlarda ölçülür. Test kümesi aramaya girmez.
##################################################
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import optuna

from esik_core import (BASE_DIR, SEED, BASELINE_PARAMS, FEATURES, load_data, make_folds,
                       fit_and_score, summarize_folds, create_pipeline, write_json, sha256)

FAMILIES = ("xgboost", "lightgbm")
SPACE = {
    "n_estimators": [100, 1600, 100], "learning_rate_log": [.01, .2],
    "subsample": [.6, 1.], "colsample_bytree": [.6, 1.],
    "weight_multiplier_log": [.5, 2.],
    "reg_alpha": [0., .01, .1, 1., 5., 10.],
    "xgboost": {"max_depth": [2,8], "min_child_weight_log": [1.,20.],
                "reg_lambda_log": [.1,20.], "gamma": [0.,.01,.1,.5,1.,5.]},
    "lightgbm": {"num_leaves_log": [8,128], "min_child_samples_log": [5,100],
                 "max_depth": -1, "reg_lambda": [0.,.01,.1,1.,5.,10.,20.],
                 "min_split_gain": [0.,.01,.1,.5,1.,5.]},
}


##################################################
# PARAMETRE ÖNERİSİ VE KARŞILAŞTIRMA KURALI
##################################################
def suggest_parameters(trial, family):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 1600, step=100),
             learning_rate=trial.suggest_float("learning_rate", .01, .2, log=True),
             subsample=trial.suggest_float("subsample", .6, 1.),
             colsample_bytree=trial.suggest_float("colsample_bytree", .6, 1.),
             reg_alpha=trial.suggest_categorical("reg_alpha", SPACE["reg_alpha"]),
             weight_multiplier=trial.suggest_float("weight_multiplier", .5, 2., log=True))
    if family == "xgboost":
        p.update(max_depth=trial.suggest_int("max_depth", 2, 8),
                 min_child_weight=trial.suggest_float("min_child_weight", 1., 20., log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", .1, 20., log=True),
                 gamma=trial.suggest_categorical("gamma", SPACE[family]["gamma"]))
    else:
        p.update(max_depth=-1,
                 num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
                 reg_lambda=trial.suggest_categorical("reg_lambda", SPACE[family]["reg_lambda"]),
                 min_split_gain=trial.suggest_categorical("min_split_gain", SPACE[family]["min_split_gain"]))
    return p


def selection_key(summary, parameters):
    # Eşit Recall: AP, ardından daha az ağaç. Kural sonuçlardan önce sabit.
    return (summary["mean_recall_at_10"], summary["mean_average_precision"],
            -parameters["n_estimators"])


def save_study_tables(study, directory):
    trials, folds = [], []
    for trial in study.trials:
        row = dict(trial=trial.number, state=trial.state.name, value=trial.value,
                   **trial.params, **trial.user_attrs.get("summary", {}))
        trials.append(row)
        for fold in trial.user_attrs.get("fold_metrics", []):
            folds.append(dict(trial=trial.number, state=trial.state.name, **fold))
    for filename, rows in [("trials.csv", trials), ("fold_metrics.csv", folds)]:
        temporary = directory / (filename + ".tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(directory / filename)


##################################################
# TEK MODEL AİLESİNİN ARANMASI
##################################################
def run_study(family, X, y, folds, directory, n_trials, threads, seed):
    directory.mkdir(parents=True, exist_ok=True)
    sqlite_path = (directory / "study.sqlite3").resolve().as_posix()
    sampler_path = directory / "sampler.pkl"
    study = optuna.create_study(direction="maximize", study_name=family,
                                storage=f"sqlite:///{sqlite_path}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    executed = [t for t in study.trials if t.state != optuna.trial.TrialState.WAITING]
    if executed:
        if not sampler_path.exists():
            raise ValueError("Sampler kaydı yok; sessizce farklı bir aramaya devam edilmedi.")
        saved = joblib.load(sampler_path)
        if saved["executed"] != len(executed):
            raise ValueError("Sampler ve çalışma kaydı tutarsız. Yeni deney adı kullanın.")
        study.sampler = saved["sampler"]
    if not study.trials:
        baseline = dict(BASELINE_PARAMS[family])
        study.enqueue_trial(baseline)
        # Kullanıcının sorduğu 600 ağaç, diğer referans ayarlar sabitken gerçekten denenir.
        study.enqueue_trial(dict(baseline, n_estimators=600))

    def objective(trial):
        parameters = suggest_parameters(trial, family)
        trial.set_user_attr("resolved_parameters", parameters)
        metrics = []
        for fold_number, (fit_ids, valid_ids) in enumerate(folds):
            _, _, row = fit_and_score(family, parameters, X, y, fit_ids, valid_ids, threads)
            metrics.append(dict(fold=fold_number, **row))
            trial.set_user_attr("fold_metrics", metrics)
            # Pruning kapalı: bütün tamamlanan adaylar aynı katlarda ölçülür.
        summary = summarize_folds(metrics)
        trial.set_user_attr("summary", summary)
        return summary["mean_recall_at_10"]

    def checkpoint(study, trial):
        count = sum(t.state != optuna.trial.TrialState.WAITING for t in study.trials)
        temporary = sampler_path.with_suffix(".tmp")
        joblib.dump(dict(executed=count, sampler=study.sampler), temporary)
        temporary.replace(sampler_path)
        save_study_tables(study, directory)
        write_json(directory / "progress.json", dict(family=family, completed=count,
                    requested=n_trials, latest_trial=trial.number, latest_value=trial.value,
                    latest_state=trial.state.name))
        print(f"{directory.parent.name}/{family}: {count}/{n_trials}, recall={trial.value}", flush=True)

    remaining = max(0, n_trials-len(executed))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, callbacks=[checkpoint])
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) != n_trials:
        raise ValueError("Deneme sayısı protokoldeki tamamlanmış deneme sayısıyla eşleşmiyor.")
    winner = max(complete, key=lambda t: selection_key(t.user_attrs["summary"],
                                                       t.user_attrs["resolved_parameters"]))
    result = dict(family=family, best_trial=winner.number,
                  parameters=winner.user_attrs["resolved_parameters"],
                  summary=winner.user_attrs["summary"],
                  baseline_summary=complete[0].user_attrs["summary"],
                  trials=len(complete))
    write_json(directory / "best.json", result)
    save_study_tables(study, directory)
    return result


def search_families(X, y, folds, directory, trials, threads, seed):
    # Ailelerin ayrı SQLite dosyaları ve ayrı RNG'leri vardır.
    # Her ailenin denemeleri sıralıdır; iki bağımsız aile aynı anda çalışabilir.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_study, family, X, y, folds,
                              directory/family, trials, threads, seed+i)
                   for i, family in enumerate(FAMILIES)]
        return [future.result() for future in futures]


def choose_family(results):
    return max(results, key=lambda r: selection_key(r["summary"], r["parameters"]))


##################################################
# PROTOKOL: VERİ, KOD, KATLAR VE BÜTÇE KİLİDİ
##################################################
def prepare_protocol(directory, X, y, train_ids, holdout_ids, data_report, args):
    versions = {name: importlib.metadata.version(name) for name in
                ["optuna", "xgboost", "lightgbm", "scikit-learn", "numpy", "pandas", "joblib"]}
    protocol = dict(schema=1, mode=args.mode, trials=args.trials,
                    inner_folds=args.inner_folds, outer_folds=args.outer_folds,
                    threads_per_model=args.threads, families=list(FAMILIES),
                    seed=SEED, capacity=.1, primary_metric="mean_recall_at_10",
                    selection_rule="Recall, then AP, then fewer trees; no holdout selection",
                    holdout_status="Previously inspected historical benchmark; not fresh external validation",
                    search_space=SPACE, versions=versions, data=data_report,
                    training_code={name:sha256(BASE_DIR/name) for name in ["esik_core.py", "esik_optuna.py"]},
                    development_ids=train_ids.tolist(), holdout_ids=holdout_ids.tolist())
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Deney protokolü değişmiş. Aynı çalışma kaydıyla devam edilemez.")
    write_json(path, protocol)
    return protocol


##################################################
# DIŞ DOĞRULAMA VE SON GELİŞTİRME ARAMASI
##################################################
def run_experiment(args):
    started = time.perf_counter()
    directory = BASE_DIR / "experiments" / args.name
    X, y, train_ids, holdout_ids, data_report = load_data()
    protocol = prepare_protocol(directory, X, y, train_ids, holdout_ids, data_report, args)
    X_dev, y_dev = X.loc[train_ids], y.loc[train_ids]
    outer_rows, predictions = [], []
    outer_folds = make_folds(X_dev, y_dev, args.outer_folds, SEED) if args.outer_folds else []
    for outer_number, (outer_train, outer_valid) in enumerate(outer_folds):
        outer_dir = directory / f"outer_{outer_number}"
        result_path = outer_dir / "result.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            outer_rows.extend(saved["rows"])
            predictions.extend(saved["predictions"])
            continue
        inner = make_folds(X.loc[outer_train], y.loc[outer_train], args.inner_folds, SEED+100+outer_number)
        write_json(outer_dir / "splits.json", dict(train_ids=outer_train.tolist(),
                    valid_ids=outer_valid.tolist(),
                    inner=[dict(train=a.tolist(), valid=b.tolist()) for a,b in inner]))
        results = search_families(X.loc[outer_train], y.loc[outer_train], inner,
                                  outer_dir, args.trials, args.threads, SEED+1000*outer_number)
        chosen = choose_family(results)  # Dış hedefleri görmeden seçim.
        local_rows, local_predictions = [], []
        for result in results:
            family = result["family"]
            _, scores, metrics = fit_and_score(family, result["parameters"], X_dev, y_dev,
                                               outer_train, outer_valid, args.threads)
            row = dict(outer_fold=outer_number, model=family, **metrics)
            local_rows.append(row)
            if family == chosen["family"]:
                local_rows.append(dict(row, model="selected_procedure"))
                local_predictions.extend(dict(row_id=int(i), outer_fold=outer_number,
                       actual=int(y.loc[i]), score=float(score), family=family)
                       for i,score in zip(outer_valid,scores))
        for family in ["xgboost", "gradient_boosting"]:
            _, _, metrics = fit_and_score(family, BASELINE_PARAMS[family], X_dev, y_dev,
                                          outer_train, outer_valid, args.threads)
            local_rows.append(dict(outer_fold=outer_number, model=f"fixed_{family}", **metrics))
        write_json(result_path, dict(selected_family=chosen["family"], rows=local_rows,
                                    predictions=local_predictions))
        outer_rows.extend(local_rows)
        predictions.extend(local_predictions)
        pd.DataFrame(outer_rows).to_csv(directory/"nested_fold_metrics.csv", index=False)
        print(f"Outer {outer_number+1}/{args.outer_folds} completed.", flush=True)

    final_folds = make_folds(X_dev, y_dev, args.inner_folds, SEED)
    write_json(directory/"full_development/splits.json",
               [dict(train=a.tolist(), valid=b.tolist()) for a,b in final_folds])
    final_results = search_families(X_dev, y_dev, final_folds, directory/"full_development",
                                    args.trials, args.threads, SEED+9000)
    winner = choose_family(final_results)
    nested_summary = {}
    if outer_rows:
        pd.DataFrame(outer_rows).to_csv(directory/"nested_fold_metrics.csv", index=False)
        pd.DataFrame(predictions).to_csv(directory/"nested_oof_predictions.csv", index=False)
        for name in sorted({r["model"] for r in outer_rows}):
            nested_summary[name] = summarize_folds([r for r in outer_rows if r["model"] == name])
    summary = dict(status="complete", mode=args.mode, winner=winner,
                    final_comparison=final_results, nested=nested_summary,
                    elapsed_seconds=time.perf_counter()-started,
                    protocol_sha256=sha256(directory/"protocol.json"),
                    holdout_used_for_selection=False)
    if args.mode == "run":
        pipeline = create_pipeline(winner["family"], winner["parameters"], y_dev, args.threads)
        pipeline.fit(X_dev, y_dev)
        joblib.dump(dict(pipeline=pipeline, search_summary=summary,
                         development_ids=train_ids.tolist(), holdout_ids=holdout_ids.tolist(),
                         feature_columns=FEATURES), directory/"candidate.pkl")
    write_json(directory/"summary.json", summary)
    print(json.dumps(dict(status="complete", family=winner["family"],
                          cv_recall=winner["summary"]["mean_recall_at_10"],
                          seconds=summary["elapsed_seconds"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description="EŞİK kayıtlı Optuna deneyi")
    parser.add_argument("mode", choices=["pilot", "run"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("Deney adı yalnızca harf, rakam, tire ve alt çizgi içermeli.")
    if args.trials < 2 or args.inner_folds < 2 or args.outer_folds not in [0, 3, 5]:
        parser.error("Geçersiz deneme veya kat sayısı.")
    if args.mode == "pilot":
        args.outer_folds = 0
    elif args.outer_folds == 0:
        parser.error("Tam çalışmada dış doğrulama gereklidir.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_experiment(args)


if __name__ == "__main__":
    main()
