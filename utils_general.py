import logging
from typing import Dict, List, Optional

import lmdb
import redis
from PIL import Image


def get_from_cache(key: str, redis_env: redis.Redis) -> Optional[str]:
    try:
        value = redis_env.get(key.encode())
        if value:
            return value.decode()
    except Exception as e:
        logging.warning(f"Error getting from cache: {e}")
    return None


def save_to_cache(key: str, value: str, redis_env: redis.Redis):
    try:
        redis_env.set(key.encode(), value.encode())
    except Exception as e:
        logging.warning(f"Error saving to cache: {e}")
    return None
