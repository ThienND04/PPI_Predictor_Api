import os
import time
import jwt
from typing import Any, Dict
from passlib.hash import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.verify(password, hashed)


def generate_jwt(payload: Dict[str, Any], exp_seconds: int = 3600 * 60) -> str:
    secret = os.getenv('JWT_SECRET')
    if not secret:
        raise RuntimeError('JWT_SECRET is not set')
    now = int(time.time())
    payload = {**payload, 'iat': now, 'exp': now + exp_seconds}
    return jwt.encode(payload, secret, algorithm='HS256')


def verify_jwt(token: str) -> Dict[str, Any]:
    secret = os.getenv('JWT_SECRET')
    if not secret:
        raise RuntimeError('JWT_SECRET is not set')
    return jwt.decode(token, secret, algorithms=['HS256'])


