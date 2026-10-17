import logging
import os
import jwt
import httpx
import asyncio
from fastapi import Depends, FastAPI, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from arq import create_pool
from arq.connections import RedisSettings

from . import crud, schemas, auth, models
from .database import Base, engine, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orders-api")

FORECAST_URL = os.getenv("FORECAST_URL", "http://forecast-api:8001")
FORECAST_TIMEOUT = float(os.getenv("FORECAST_TIMEOUT", "3.0"))
BATCH_MAX_ITEMS = 50

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Orders API", version="0.1.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@app.on_event("startup")
async def startup():
    app.state.redis = await create_pool(RedisSettings(host='redis', port=6379))
    app.state.http_client = httpx.AsyncClient(timeout=FORECAST_TIMEOUT)

@app.on_event("shutdown")
async def shutdown():
    await app.state.http_client.aclose()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
    user = crud.get_user_by_username(db, username=username)
    if user is None or not user.is_active: raise credentials_exception
    return user

async def _fetch_price(payload: dict) -> float | Exception:
    try:
        res = await app.state.http_client.post(f"{FORECAST_URL}/predict", json=payload)
        res.raise_for_status()
        return float(res.json()["predicted_price"])
    except Exception as e:
        return e

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/auth/register", response_model=schemas.UserOut, status_code=201)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_username(db, username=user.username):
        raise HTTPException(status_code=409, detail="Username already registered")
    return crud.create_user(db=db, user=user)

@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return {"access_token": auth.create_access_token(data={"sub": user.username}), "token_type": "bearer"}

@app.get("/auth/me", response_model=schemas.UserOut)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

# ЗАВДАННЯ 1: ПАРАЛЕЛЬНА ПАКЕТНА ОБРОБКА
@app.post("/orders/batch", response_model=schemas.BatchResponse)
async def create_orders_batch(
    orders_data: list[schemas.OrderCreate],
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if len(orders_data) > BATCH_MAX_ITEMS:
        raise HTTPException(status_code=422, detail=f"Max {BATCH_MAX_ITEMS} items allowed.")
    
    db_orders = [crud.create_order(db, d, status="pending") for d in orders_data]
    tasks = [_fetch_price(d.model_dump()) for d in orders_data]
    
    # Виконуємо запити конкурентно
    prices = await asyncio.gather(*tasks)
    
    results = []
    for i, (order, price) in enumerate(zip(db_orders, prices)):
        if isinstance(price, Exception):
            crud.set_order_status(db, order.id, status="failed")
            results.append(schemas.BatchResultItem(index=i, order_id=order.id, status="failed", error="Forecast Unavailable"))
        else:
            crud.set_order_status(db, order.id, status="priced", price=price)
            results.append(schemas.BatchResultItem(index=i, order_id=order.id, status="priced", predicted_price=price))
    
    return {"total": len(results), "results": results}

# ЗАВДАННЯ 2: ВІДКЛАДЕНЕ ВИКОНАННЯ (Черга)
@app.post("/orders", response_model=schemas.OrderRead, status_code=202)
async def create_order(
    request: Request,
    data: schemas.OrderCreate,
    response: Response,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    order = crud.create_order(db, data, status="pending")
    response.headers["Location"] = f"/orders/{order.id}"
    
    # Передаємо завдання брокеру Redis
    await request.app.state.redis.enqueue_job('process_forecast', order.id, data.model_dump())
    return order

@app.get("/orders", response_model=list[schemas.OrderRead])
def list_orders(limit: int = 20, offset: int = 0, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return crud.list_orders(db, limit, offset)

@app.get("/orders/{oid}", response_model=schemas.OrderRead)
def get_order(oid: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = crud.get_order(db, oid)
    if not order: raise HTTPException(status_code=404, detail="Order not found.")
    return order

# ЗАВДАННЯ 3: ОНОВЛЕННЯ ЗАМОВЛЕННЯ З ЧЕРГОЮ
@app.patch("/orders/{oid}", response_model=schemas.OrderRead)
async def update_order(
    request: Request,
    oid: int,
    data: schemas.OrderUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    order = crud.get_order(db, oid)
    if not order: raise HTTPException(status_code=404)

    update_data = data.model_dump(exclude_unset=True)
    needs_reforecast = any(f in update_data for f in ["gr_liv_area", "total_bsmt_sf", "garage_area", "year_built", "overall_qual", "ms_zoning"])
    order = crud.update_order(db, order, data)

    if needs_reforecast:
        crud.set_order_status(db, order.id, status="pending", price=None)
        payload = {
            "gr_liv_area": order.gr_liv_area, "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area, "year_built": order.year_built,
            "overall_qual": order.overall_qual, "ms_zoning": order.ms_zoning,
        }
        await request.app.state.redis.enqueue_job('process_forecast', order.id, payload)

    return order

@app.delete("/orders/{oid}", status_code=204)
def delete_order(oid: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    order = crud.get_order(db, oid)
    if not order: raise HTTPException(status_code=404)
    crud.delete_order(db, order)