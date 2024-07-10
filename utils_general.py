import logging
from typing import Optional

import redis
import pickle
import os


cache_llm = pickle.load(open("cache_llm.pkl", "rb"))

# def get_from_cache(key: str, redis_env: redis.Redis) -> Optional[str]:
#     try:
#         value = redis_env.get(key.encode())
#         if value:
#             cache = pickle.load(open("cache.pkl", "rb")) if os.path.exists("cache.pkl") else {}
#             cache[key.encode()] = value
#             pickle.dump(cache, open("cache.pkl", "wb"))
#             return value.decode()
#     except Exception as e:
#         logging.warning(f"Error getting from cache: {e}")
#     return None


def save_to_cache(key: str, value: str, redis_env: redis.Redis):
    # try:
    #     redis_env.set(key.encode(), value.encode())
    # except Exception as e:
    #     logging.warning(f"Error saving to cache: {e}")
    return None


def get_from_cache(key: str, redis_env: redis.Redis) -> Optional[str]:
    try:
        return cache_llm[key.encode()].decode()
    except Exception as e:
        logging.warning(f"Error getting from cache: {e}")
    return None
