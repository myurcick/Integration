import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import crud, schemas
from .database import Base, engine, get_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("orders-api")

FORECAST_URL = os.getenv(
    "FORECAST_URL",
    "http://forecast-api:8001",
)

FORECAST_TIMEOUT = float(
    os.getenv("FORECAST_TIMEOUT", "3.0")
)

# Tables are created automatically for the initial version.
# In production, replace this with Alembic migrations.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Orders API",
    version="0.1.0",
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, max=2),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException)
    ),
    reraise=True,
)
def _request_prediction(payload: dict) -> float:
    """Request a price prediction from the forecasting service.

    Retry only transient network failures such as connection errors
    and timeouts. HTTP errors are not retried.
    """
    response = httpx.post(
        f"{FORECAST_URL}/predict",
        json=payload,
        timeout=FORECAST_TIMEOUT,
    )
    response.raise_for_status()

    return float(response.json()["predicted_price"])



@app.get("/health")
def health() -> dict[str, str]:
    """Return the health status of the Orders API."""
    return {"status": "ok"}


@app.post(
    "/orders",
    response_model=schemas.OrderRead,
    status_code=201,
)
def create_order(
    data: schemas.OrderCreate,
    db: Session = Depends(get_db),
):
    """Create an order and attempt to obtain a price prediction.

    If the forecasting service is unavailable, the order is still
    persisted with a null predicted price.
    """
    try:
        price = _request_prediction(data.model_dump())
    except Exception as exc:
        logger.error(
            "Forecast service is unavailable: %s",
            exc,
        )
        price = None

    return crud.create_order(
        db,
        data,
        price,
    )


@app.get(
    "/orders",
    response_model=list[schemas.OrderRead],
)
def list_orders(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Return a paginated list of orders."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset must be non-negative.",
        )

    return crud.list_orders(
        db,
        limit,
        offset,
    )


@app.get(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def get_order(
    oid: int,
    db: Session = Depends(get_db),
):
    """Return a single order by ID."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    return order


@app.patch(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def update_order(
    oid: int,
    data: schemas.OrderUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing order and re-calculate price if features change."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    # Отримуємо словник з новими даними, які прислав клієнт
    update_data = data.model_dump(exclude_unset=True)

    # Перевіряємо, чи є в оновленні параметри, що впливають на ціну
    forecast_features = {
        "gr_liv_area", "total_bsmt_sf", "garage_area", 
        "year_built", "overall_qual", "ms_zoning"
    }
    needs_reforecast = any(feature in update_data for feature in forecast_features)

    if needs_reforecast:
        # Збираємо актуальні параметри: беремо нові, якщо є, інакше старі з БД
        payload = {
            "gr_liv_area": update_data.get("gr_liv_area", order.gr_liv_area),
            "total_bsmt_sf": update_data.get("total_bsmt_sf", order.total_bsmt_sf),
            "garage_area": update_data.get("garage_area", order.garage_area),
            "year_built": update_data.get("year_built", order.year_built),
            "overall_qual": update_data.get("overall_qual", order.overall_qual),
            "ms_zoning": update_data.get("ms_zoning", order.ms_zoning),
        }
        
        # Намагаємося отримати нову ціну (з graceful degradation)
        try:
            new_price = _request_prediction(payload)
            order.predicted_price = new_price
        except Exception as exc:
            logger.error("Forecast service is unavailable during update: %s", exc)
            order.predicted_price = None

    # Зберігаємо зміни в БД
    return crud.update_order(
        db,
        order,
        data,
    )


@app.delete(
    "/orders/{oid}",
    status_code=204,
)
def delete_order(
    oid: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete an existing order."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    crud.delete_order(db, order)
