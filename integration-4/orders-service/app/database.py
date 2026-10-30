import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Функція для читання секретів
def read_secret(name: str) -> str:
    secret_path = Path(f"/run/secrets/{name}")
    if not secret_path.exists():
        raise RuntimeError(f"Secret {name} not found at {secret_path}")
    return secret_path.read_text().strip()

# Читаємо пароль
try:
    db_password = read_secret("db_password")
except RuntimeError as e:
    # Fallback для локального запуску без Docker (якщо потрібно)
    db_password = os.getenv("POSTGRES_PASSWORD", "orders")

# Формуємо URL програмно
DB_USER = os.getenv("POSTGRES_USER", "orders")
DB_NAME = os.getenv("POSTGRES_DB", "orders")
SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{db_password}@postgres:5432/{DB_NAME}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()