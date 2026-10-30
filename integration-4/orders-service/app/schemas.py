from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class OrderCreate(BaseModel):
    """Request schema for creating a new order."""

    client_name: str = Field(
        min_length=2,
        max_length=120,
        description="Client name.",
    )

    gr_liv_area: float = Field(
        gt=0,
        le=10_000,
        description="Above-ground living area in square feet.",
    )

    total_bsmt_sf: float = Field(
        ge=0,
        le=10_000,
        description="Total basement area in square feet.",
    )

    garage_area: float = Field(
        ge=0,
        le=2_000,
        description="Garage area in square feet.",
    )

    year_built: int = Field(
        ge=1872,
        le=2010,
        description="Original construction year.",
    )

    overall_qual: int = Field(
        ge=1,
        le=10,
        description="Overall material and finish quality, rated from 1 to 10.",
    )

    ms_zoning: str = Field(
        min_length=1,
        max_length=10,
        description="General zoning classification.",
    )



class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    is_active: bool
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class OrderCreate(BaseModel):
    client_name: str
    gr_liv_area: float
    total_bsmt_sf: float
    garage_area: float
    year_built: int
    overall_qual: int
    ms_zoning: str

class OrderUpdate(BaseModel):
    client_name: Optional[str] = None
    gr_liv_area: Optional[float] = None
    total_bsmt_sf: Optional[float] = None
    garage_area: Optional[float] = None
    year_built: Optional[int] = None
    overall_qual: Optional[int] = None
    ms_zoning: Optional[str] = None

class OrderRead(BaseModel):
    id: int
    client_name: str
    gr_liv_area: float
    total_bsmt_sf: float
    garage_area: float
    year_built: int
    overall_qual: int
    ms_zoning: str
    predicted_price: Optional[float] = None
    status: str
    model_version: Optional[str] = None  # <-- НОВЕ ПОЛЕ
    class Config:
        from_attributes = True

class BatchResultItem(BaseModel):
    index: int
    order_id: int
    status: str
    predicted_price: Optional[float] = None
    model_version: Optional[str] = None  # <-- НОВЕ ПОЛЕ
    error: Optional[str] = None

class BatchResponse(BaseModel):
    total: int
    results: List[BatchResultItem]