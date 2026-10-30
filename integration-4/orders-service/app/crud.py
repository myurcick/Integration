from sqlalchemy.orm import Session
from . import models, schemas, auth

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_order(db: Session, data: schemas.OrderCreate, status: str = "pending") -> models.Order:
    order = models.Order(**data.model_dump(), predicted_price=None, status=status)
    db.add(order)
    db.commit()
    db.refresh(order)
    return order

def get_order(db: Session, oid: int) -> models.Order | None:
    return db.get(models.Order, oid)

def list_orders(db: Session, limit: int = 20, offset: int = 0) -> list[models.Order]:
    return db.query(models.Order).order_by(models.Order.id.desc()).limit(limit).offset(offset).all()

def update_order(db: Session, order: models.Order, data: schemas.OrderUpdate) -> models.Order:
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(order, key, value)
    db.commit()
    db.refresh(order)
    return order

def set_order_status(
    db: Session,
    order_id: int,
    status: str,
    price: float | None = None,
    model_version: str | None = None,
):
    order = db.get(models.Order, order_id)
    if order:
        order.status = status
        order.predicted_price = price
        order.model_version = model_version
        db.commit()

def delete_order(db: Session, order: models.Order) -> None:
    db.delete(order)
    db.commit()