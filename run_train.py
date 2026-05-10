from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.modeling import (
    build_error_analysis_table,
    compare_models,
    compute_permutation_importance,
    evaluate_ensembles,
    feature_count_ablation,
    final_test_evaluation,
    get_regressors,
    save_model,
    save_table,
    seed_sensitivity_check,
    tune_top_models,
)
from src.preprocessing import (
    load_data,
    load_yaml,
    prepare_train_val_test,
    save_preprocessing_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_output_dirs(paths: dict) -> None:
    for key in ["processed_data_dir", "model_dir", "figures_dir", "tables_dir"]:
        (PROJECT_ROOT / paths[key]).mkdir(parents=True, exist_ok=True)


def write_log(paths: dict, text: str) -> None:
    log_path = PROJECT_ROOT / paths["training_log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")


def write_model_card(paths: dict, final_name: str, test_metrics: dict, selected_features: list[str]) -> None:
    card = f"""# Model Card: Metro Traffic Volume Predictor

## Purpose
Predict hourly `traffic_volume` using time and weather features.

## Validation Strategy
The project uses a chronological 70/15/15 train-validation-test split. Model comparison uses the validation set, not random K-fold CV.

## Final Model
{final_name}

## Test Metrics
- MAE: {test_metrics['MAE']:.3f}
- RMSE: {test_metrics['RMSE']:.3f}
- R2: {test_metrics['R2']:.3f}

## Selected Features
{", ".join(selected_features)}

## Notes
Rainfall capping, rare category grouping, scaling, and feature selection are fitted on training data only.
"""
    output_path = PROJECT_ROOT / paths["model_card_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card, encoding="utf-8")


def main() -> None:
    paths = load_yaml(PROJECT_ROOT / "config" / "paths.yaml")
    params = load_yaml(PROJECT_ROOT / "config" / "params.yaml")
    ensure_output_dirs(paths)

    raw_path = PROJECT_ROOT / paths["raw_data"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Place the CSV there before training."
        )

    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = load_data(raw_path)
    prepared = prepare_train_val_test(df, params)

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    comparison, fitted_base = compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
        model_names=params["models"]["include"],
        random_state=params["random_seed"],
        max_rows_for_slow_models=params["models"].get("max_rows_for_slow_models"),
        slow_models=params["models"].get("slow_models"),
    )

    base_models = get_regressors(random_state=params["random_seed"])
    tuned_results, tuned_models = tune_top_models(
        comparison,
        base_models,
        params["tuning"]["grids"],
        X_train,
        y_train,
        X_val,
        y_val,
        top_n=params["tuning"]["top_n_models"],
        max_rows_for_slow_models=params["models"].get("max_rows_for_slow_models"),
        slow_models=params["models"].get("slow_models"),
    )

    candidates = {**tuned_models}
    if not candidates:
        best_base_name = comparison.iloc[0]["model"]
        candidates[best_base_name] = fitted_base[best_base_name]

    ensemble_results, ensemble_models = evaluate_ensembles(
        candidates,
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=params["random_seed"],
    )

    all_results = pd.concat(
        [
            comparison.assign(stage="baseline"),
            tuned_results.assign(stage="tuned"),
            ensemble_results.assign(stage="ensemble"),
        ],
        ignore_index=True,
    ).sort_values("RMSE")

    all_models = {**fitted_base, **tuned_models, **ensemble_models}
    final_name = all_results.iloc[0]["model"]
    final_model = all_models[final_name]
    test_metrics = final_test_evaluation(final_model, X_test, y_test)

    final_row = pd.DataFrame([{"model": final_name, "stage": "final_test", **test_metrics}])
    metrics_table = pd.concat([all_results, final_row], ignore_index=True)
    save_table(metrics_table, PROJECT_ROOT / paths["metrics_path"])

    permutation_table = compute_permutation_importance(
        final_model,
        X_val,
        y_val,
        random_state=params["random_seed"],
    )
    save_table(permutation_table, PROJECT_ROOT / paths["tables_dir"] / "permutation_importance.csv")

    error_table = build_error_analysis_table(
        final_model,
        X_test,
        y_test,
        prepared["test_df"],
        top_n=20,
    )
    save_table(error_table, PROJECT_ROOT / paths["tables_dir"] / "highest_test_errors.csv")

    seed_table = seed_sensitivity_check(
        "Random Forest",
        [7, 42, 99],
        X_train,
        y_train,
        X_val,
        y_val,
    )
    save_table(seed_table, PROJECT_ROOT / paths["tables_dir"] / "seed_sensitivity.csv")

    # Recreate full scaled matrices before feature subset for top-k ablation.
    # This uses only training-fitted artifacts and validation remains untouched.
    prepared_no_fs_params = {**params, "feature_selection": {**params["feature_selection"], "enabled": False}}
    prepared_no_fs = prepare_train_val_test(df, prepared_no_fs_params)
    ablation_table = feature_count_ablation(
        prepared["artifacts"]["feature_importance"],
        prepared_no_fs["X_train"],
        prepared_no_fs["y_train"],
        prepared_no_fs["X_val"],
        prepared_no_fs["y_val"],
        top_k_values=[10, 15, 20, 25],
        random_state=params["random_seed"],
    )
    save_table(ablation_table, PROJECT_ROOT / paths["tables_dir"] / "feature_count_ablation.csv")

    save_model(final_model, PROJECT_ROOT / paths["model_path"])
    save_preprocessing_artifacts(
        prepared["artifacts"],
        PROJECT_ROOT / paths["preprocessor_path"],
    )

    write_model_card(
        paths,
        final_name,
        test_metrics,
        prepared["artifacts"]["selected_features"],
    )
    write_log(
        paths,
        "\n".join(
            [
                f"Training started: {started}",
                f"Training finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Rows after cleaning: {len(prepared['train_df']) + len(prepared['val_df']) + len(prepared['test_df'])}",
                f"Selected model: {final_name}",
                f"Final test metrics: {test_metrics}",
            ]
        ),
    )
    print(metrics_table)


if __name__ == "__main__":
    main()
