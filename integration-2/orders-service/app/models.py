from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.sql import func
from .database import Base
class Order(Base):
    """Database model representing a house pricing order."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)

    client_name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
    )

    gr_liv_area: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_bsmt_sf: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    garage_area: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    year_built: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    overall_qual: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    ms_zoning: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    predicted_price: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)