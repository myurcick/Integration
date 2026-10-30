import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("trainer")

SEED = 42
DATA_PATH = Path("data/train.csv")
NUMERIC = ["gr_liv_area", "total_bsmt_sf", "garage_area", "year_built", "overall_qual"]
CATEGORICAL = ["ms_zoning"]
TARGET = "sale_price"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house-price")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price")


def load_or_generate_dataset() -> pd.DataFrame:
    """Завантажує датасет або нормалізує назви колонок Ames Housing."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        col_map = {
            "GrLivArea": "gr_liv_area",
            "TotalBsmtSF": "total_bsmt_sf",
            "GarageArea": "garage_area",
            "YearBuilt": "year_built",
            "OverallQual": "overall_qual",
            "MSZoning": "ms_zoning",
            "SalePrice": "sale_price",
        }
        df = df.rename(columns=col_map)
        return df[NUMERIC + CATEGORICAL + [TARGET]].dropna(subset=[TARGET])
    
    # Резервний синтетичний датасет, якщо CSV мав іншу структуру
    logger.warning("data/train.csv not found, generating deterministic synthetic housing dataset...")
    rng = np.random.default_rng(SEED)
    n = 1200
    df = pd.DataFrame({
        "gr_liv_area": rng.normal(1500, 450, n).clip(400, 4500),
        "total_bsmt_sf": rng.normal(1000, 350, n).clip(0, 3000),
        "garage_area": rng.normal(480, 180, n).clip(0, 1200),
        "year_built": rng.integers(1950, 2022, n),
        "overall_qual": rng.integers(1, 11, n),
        "ms_zoning": rng.choice(["RL", "RM", "FV", "C", "RH"], size=n, p=[0.7, 0.15, 0.08, 0.04, 0.03]),
    })
    base_price = (
        df["gr_liv_area"] * 75
        + df["total_bsmt_sf"] * 45
        + df["garage_area"] * 50
        + (df["year_built"] - 1950) * 600
        + (df["overall_qual"] ** 2) * 1800
    )
    zoning_mult = df["ms_zoning"].map({"RL": 1.1, "FV": 1.2, "RM": 0.9, "RH": 0.85, "C": 0.75})
    noise = rng.lognormal(0, 0.12, n)
    df[TARGET] = (base_price * zoning_mult * noise).round(2)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def compute_stats_and_figure(df: pd.DataFrame):
    target_raw = df[TARGET]
    target_log = np.log1p(target_raw)
    skew_before = float(target_raw.skew())
    skew_after = float(target_log.skew())

    stats = {
        "row_count": int(len(df)),
        "missing_fractions": {col: float(df[col].isna().mean()) for col in NUMERIC + CATEGORICAL},
        "numeric_summary": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "median": float(df[col].median()),
                "max": float(df[col].max()),
            }
            for col in NUMERIC
        },
        "categorical_counts": {col: df[col].value_counts().to_dict() for col in CATEGORICAL},
        "target_skewness_before_log1p": skew_before,
        "target_skewness_after_log1p": skew_after,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(target_raw, bins=35, edgecolor="black")
    axes[0].set_title(f"Target before log1p (skew={skew_before:.2f})")
    axes[1].hist(target_log, bins=35, edgecolor="black")
    axes[1].set_title(f"Target after log1p (skew={skew_after:.2f})")
    plt.tight_layout()
    return stats, fig


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(root_mean_squared_error(y_test, preds)),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
    }


def build_ridge_baseline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", Ridge(alpha=1.0, random_state=SEED)),
    ])


def build_model(params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", TransformedTargetRegressor(
            regressor=XGBRegressor(**params, random_state=SEED, n_jobs=2),
            func=np.log1p,
            inverse_func=np.expm1,
        )),
    ])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    df = load_or_generate_dataset()
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    stats, fig = compute_stats_and_figure(df)
    dataset_input = mlflow.data.from_pandas(df, source=str(DATA_PATH), targets=TARGET, name="house-price-train")

    ref_path = Path("reference_data.parquet")
    X_train.to_parquet(ref_path, index=False)

    # 1. BASELINE RUN (Ridge) - Версія 1 для порівняння та Rollback-експерименту
    logger.info("Running Ridge baseline...")
    with mlflow.start_run(run_name="ridge-baseline") as baseline_run:
        mlflow.log_input(dataset_input, context="train")
        ridge_model = build_ridge_baseline()
        ridge_model.fit(X_train, y_train)
        base_metrics = evaluate(ridge_model, X_test, y_test)
        mlflow.log_params({"model_type": "Ridge", "alpha": 1.0, "random_state": SEED})
        mlflow.log_metrics({
            **base_metrics,
            "row_count": stats["row_count"],
            "target_skew_before": stats["target_skewness_before_log1p"],
            "target_skew_after": stats["target_skewness_after_log1p"],
        })
        mlflow.log_dict(stats, "data_stats.json")
        mlflow.log_figure(fig, "target_distributions.png")
        mlflow.log_artifact(str(ref_path))

        sig_base = infer_signature(X_train, ridge_model.predict(X_train))
        mlflow.sklearn.log_model(ridge_model, name="model", signature=sig_base)
        v_base = mlflow.register_model(f"runs:/{baseline_run.info.run_id}/model", MODEL_NAME)
        logger.info(f"Baseline Ridge registered as version {v_base.version}: {base_metrics}")

    # 2. OPTUNA HYPERPARAMETER SEARCH (Parent + 20 Nested Child Runs)
    logger.info("Starting Optuna hyperparameter search (20 trials)...")
    with mlflow.start_run(run_name="xgboost-optuna-search"):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(params)
                model = build_model(params)
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=1)
                cv_r2 = float(np.mean(scores))
                mlflow.log_metric("cv_r2", cv_r2)
                return cv_r2

        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=20)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_r2", float(study.best_value))
        logger.info(f"Best Optuna params: {study.best_params} (cv_r2={study.best_value:.4f})")

    # 3. FINAL XGBOOST MODEL RUN & PROMOTION TO @prod
    logger.info("Training final XGBoost model...")
    with mlflow.start_run(run_name="xgboost-final") as run:
        mlflow.log_input(dataset_input, context="train")
        final_model = build_model(study.best_params)
        final_model.fit(X_train, y_train)
        final_metrics = evaluate(final_model, X_test, y_test)

        mlflow.log_params(study.best_params)
        mlflow.log_metrics({
            **final_metrics,
            "row_count": stats["row_count"],
            "target_skew_before": stats["target_skewness_before_log1p"],
            "target_skew_after": stats["target_skewness_after_log1p"],
        })
        mlflow.log_dict(stats, "data_stats.json")
        mlflow.log_figure(fig, "target_distributions.png")
        mlflow.log_artifact(str(ref_path))

        signature = infer_signature(X_train, final_model.predict(X_train))
        mlflow.sklearn.log_model(final_model, name="model", signature=signature)
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "prod", version.version)
        logger.info(f"Final XGBoost registered as version {version.version} and promoted to @prod! Metrics: {final_metrics}")

    plt.close(fig)


if __name__ == "__main__":
    main()