import jwt
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pwdlib import PasswordHash

# Читаємо секретний ключ для JWT
def get_jwt_secret() -> str:
    secret_path = Path("/run/secrets/jwt_secret_key")
    if not secret_path.exists():
        raise RuntimeError("Missing jwt_secret_key")
    return secret_path.read_text().strip()

SECRET_KEY = get_jwt_secret()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = PasswordHash.recommended()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt