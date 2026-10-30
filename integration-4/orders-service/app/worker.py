import os
import logging
import httpx
from arq.connections import RedisSettings
from .database import SessionLocal
from . import crud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

FORECAST_URL = os.getenv("FORECAST_URL", "http://forecast-api:8001")
FORECAST_TIMEOUT = float(os.getenv("FORECAST_TIMEOUT", "3.0"))

async def process_forecast(ctx, order_id: int, payload: dict):
    logger.info(f"Starting forecast for order {order_id}")
    price = None
    model_version = None
    status = "failed"
    
    async with httpx.AsyncClient(timeout=FORECAST_TIMEOUT) as client:
        try:
            response = await client.post(f"{FORECAST_URL}/predict", json=payload)
            response.raise_for_status()
            data = response.json()
            price = float(data["predicted_price"])
            model_version = str(data.get("model_version"))
            status = "priced"
            logger.info(f"Order {order_id} priced at {price} (model_version={model_version})")
        except Exception as e:
            logger.error(f"Forecast API failed for order {order_id}: {e}")

    db = SessionLocal()
    try:
        crud.set_order_status(db, order_id, status, price, model_version)
    finally:
        db.close()

class WorkerSettings:
    functions = [process_forecast]
    redis_settings = RedisSettings(host='redis', port=6379)