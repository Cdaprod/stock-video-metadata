# scripts/redis_client.py
import redis

def get_redis():
    return redis.Redis.from_url("redis://localhost:6379/0")