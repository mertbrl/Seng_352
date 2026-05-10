from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def get_regressors(random_state: int = 42) -> dict[str, Any]:
    """Student-friendly model set for validation comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=5,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=5,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64,),
            random_state=random_state,
            max_iter=300,
            early_stopping=True,
        ),
    }


def train_and_evaluate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Any, dict[str, float]]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    predictions = fitted.predict(X_val)
    return fitted, regression_metrics(y_val, predictions)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_names: list[str] | None = None,
    random_state: int = 42,
    max_rows_for_slow_models: int | None = None,
    slow_models: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    models = get_regressors(random_state=random_state)
    if model_names:
        models = {name: model for name, model in models.items() if name in model_names}
    slow_models = slow_models or ["SVR", "KNN", "MLP"]

    fitted_models = {}
    rows = []
    for name, model in models.items():
        model_X_train = X_train
        model_y_train = y_train
        if max_rows_for_slow_models and name in slow_models and len(X_train) > max_rows_for_slow_models:
            # recent chronological subset for slow classroom-friendly models
            model_X_train = X_train.tail(max_rows_for_slow_models)
            model_y_train = y_train.tail(max_rows_for_slow_models)

        fitted, metrics = train_and_evaluate(model, model_X_train, model_y_train, X_val, y_val)
        fitted_models[name] = fitted
        rows.append({"model": name, "train_rows_used": len(model_X_train), **metrics})

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, fitted_models


def tune_model_with_validation(
    name: str,
    model: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_train_rows: int | None = None,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """Grid search using a fixed validation fold, not random K-fold CV."""
    if max_train_rows and len(X_train) > max_train_rows:
        # keep the most recent training rows and preserve time order
        X_train = X_train.tail(max_train_rows)
        y_train = y_train.tail(max_train_rows)

    if not param_grid:
        fitted, metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)
        return fitted, {}, metrics

    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    test_fold = np.r_[
        np.full(len(X_train), -1),
        np.zeros(len(X_val), dtype=int),
    ]
    validation_split = PredefinedSplit(test_fold)

    search = GridSearchCV(
        estimator=clone(model),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=validation_split,
        refit=False,
        n_jobs=-1,
    )
    search.fit(X_combined, y_combined)

    tuned = clone(model).set_params(**search.best_params_)
    tuned.fit(X_train, y_train)
    metrics = regression_metrics(y_val, tuned.predict(X_val))
    return tuned, search.best_params_, metrics


def tune_top_models(
    comparison_results: pd.DataFrame,
    base_models: dict[str, Any],
    grids: dict[str, dict[str, list[Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 3,
    max_rows_for_slow_models: int | None = None,
    slow_models: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tuned_models = {}
    rows = []
    slow_models = slow_models or ["SVR", "KNN", "MLP"]
    for name in comparison_results["model"].head(top_n):
        max_rows = max_rows_for_slow_models if name in slow_models else None
        tuned, best_params, metrics = tune_model_with_validation(
            name,
            base_models[name],
            grids.get(name, {}),
            X_train,
            y_train,
            X_val,
            y_val,
            max_train_rows=max_rows,
        )
        tuned_models[f"{name} Tuned"] = tuned
        rows.append(
            {
                "model": f"{name} Tuned",
                "train_rows_used": len(X_train.tail(max_rows)) if max_rows else len(X_train),
                "best_params": best_params,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, tuned_models


def evaluate_ensembles(
    candidate_models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
    max_models: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate Voting and Stacking regressors on the validation set."""
    selected_items = list(candidate_models.items())[:max_models]
    estimators = [(name.replace(" ", "_"), clone(model)) for name, model in selected_items]
    if len(estimators) < 2:
        return pd.DataFrame(), {}

    voting = VotingRegressor(estimators=estimators, n_jobs=-1)

    stack_cutoff = max(1, int(len(X_train) * 0.8))
    X_base, y_base = X_train.iloc[:stack_cutoff], y_train.iloc[:stack_cutoff]
    X_meta, y_meta = X_train.iloc[stack_cutoff:], y_train.iloc[stack_cutoff:]
    if len(X_meta) < 10:
        X_base, y_base = X_train, y_train
        X_meta, y_meta = X_train, y_train

    prefit_estimators = []
    for est_name, estimator in estimators:
        fitted_estimator = clone(estimator).fit(X_base, y_base)
        prefit_estimators.append((est_name, fitted_estimator))

    stacking = StackingRegressor(
        estimators=prefit_estimators,
        final_estimator=Ridge(random_state=random_state),
        cv="prefit",
        n_jobs=-1,
        passthrough=False,
    )

    ensembles = {
        "Voting Regressor": voting,
        "Stacking Regressor": stacking,
    }
    fitted = {}
    rows = []
    for name, model in ensembles.items():
        if name == "Stacking Regressor":
            trained = model.fit(X_meta, y_meta)
            metrics = regression_metrics(y_val, trained.predict(X_val))
        else:
            trained, metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)
        fitted[name] = trained
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True), fitted


def final_test_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    return regression_metrics(y_test, model.predict(X_test))


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_repeats: int = 5,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def build_error_analysis_table(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    source_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    predictions = model.predict(X)
    errors = pd.DataFrame(
        {
            "date_time": source_df["date_time"].reset_index(drop=True),
            "actual": y.reset_index(drop=True),
            "predicted": predictions,
        }
    )
    errors["error"] = errors["actual"] - errors["predicted"]
    errors["absolute_error"] = errors["error"].abs()
    errors["hour"] = errors["date_time"].dt.hour
    errors["month"] = errors["date_time"].dt.month
    return errors.sort_values("absolute_error", ascending=False).head(top_n)


def seed_sensitivity_check(
    model_name: str,
    seeds: list[int],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        model = get_regressors(random_state=seed)[model_name]
        _, metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)
        rows.append({"model": model_name, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def feature_count_ablation(
    feature_importance: pd.DataFrame,
    X_train_full: pd.DataFrame,
    y_train: pd.Series,
    X_val_full: pd.DataFrame,
    y_val: pd.Series,
    top_k_values: list[int],
    model: Any | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare validation performance for different top-k feature counts."""
    base_model = model or RandomForestRegressor(
        n_estimators=150,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    ranked_features = feature_importance["feature"].tolist()
    rows = []
    for top_k in top_k_values:
        selected = ranked_features[: min(top_k, len(ranked_features))]
        fitted, metrics = train_and_evaluate(
            base_model,
            X_train_full[selected],
            y_train,
            X_val_full[selected],
            y_val,
        )
        rows.append(
            {
                "top_k": len(selected),
                "model": fitted.__class__.__name__,
                "features": ", ".join(selected),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def save_table(df: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_model(model: Any, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
