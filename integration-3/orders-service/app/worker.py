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
    status = "failed"
    
    async with httpx.AsyncClient(timeout=FORECAST_TIMEOUT) as client:
        try:
            response = await client.post(f"{FORECAST_URL}/predict", json=payload)
            response.raise_for_status()
            price = float(response.json()["predicted_price"])
            status = "priced"
            logger.info(f"Order {order_id} priced successfully at {price}")
        except Exception as e:
            logger.error(f"Forecast API failed for order {order_id}: {e}")

    db = SessionLocal()
    try:
        crud.set_order_status(db, order_id, status, price)
    finally:
        db.close()

class WorkerSettings:
    functions = [process_forecast]
    redis_settings = RedisSettings(host='redis', port=6379)