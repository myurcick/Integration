"""Entry point for the forecasting service (forecast-api)."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI

from .schemas import PredictRequest, PredictResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("forecast-api")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "house-price")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "prod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve @prod alias from MLflow Registry at startup (fail-fast)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        app.state.model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
        app.state.model_version = str(mv.version)

        stats_path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="data_stats.json"
        )
        app.state.data_stats = json.loads(Path(stats_path).read_text())
        logger.info(
            "Loaded model %s@%s (version=%s, run_id=%s)",
            MODEL_NAME,
            MODEL_ALIAS,
            mv.version,
            mv.run_id,
        )
    except Exception as exc:
        logger.critical(
            "FAIL-FAST: Cannot resolve or load model %s@%s: %s",
            MODEL_NAME,
            MODEL_ALIAS,
            exc,
        )
        raise RuntimeError(
            f"Startup failed: model {MODEL_NAME}@{MODEL_ALIAS} could not be loaded"
        ) from exc

    yield

    app.state.model = None


app = FastAPI(
    title="Forecast API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Health check returning the actual version resolved from the registry."""
    return {
        "status": "ok",
        "model_version": app.state.model_version,
    }


@app.get("/model/data-stats")
def get_data_stats():
    """Expose descriptive statistics of the currently served @prod model."""
    return app.state.data_stats


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict house sale price using the registry-loaded model."""
    t0 = time.perf_counter()

    # The MLflow model signature expects snake_case feature names matching PredictRequest
    df = pd.DataFrame([req.model_dump()])

    price = float(app.state.model.predict(df)[0])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        "predict (v%s): %.2f USD (%.2f ms)",
        app.state.model_version,
        price,
        latency_ms,
    )

    return PredictResponse(
        predicted_price=round(price, 2),
        model_version=app.state.model_version,
        latency_ms=latency_ms,
    )