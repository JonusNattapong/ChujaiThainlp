import redis
from typing import Any, Optional, Dict, List
import json
import hashlib
from functools import wraps
import time
from datetime import datetime
from prometheus_client import Counter, Histogram

CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
CACHE_LATENCY = Histogram('cache_latency_seconds', 'Cache operation latency')

class CacheConfig:
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    DEFAULT_EXPIRATION: int = 3600  # 1 hour
    MAX_MEMORY: str = "2gb"  # Maximum memory for Redis
    EVICTION_POLICY: str = "allkeys-lru"  # Least Recently Used

class CacheMetrics:
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.hits = 0
        self.misses = 0
        self.total_latency = 0
        self.operations = 0

class RedisCache:
    def __init__(self, host: str = CacheConfig.REDIS_HOST, 
                 port: int = CacheConfig.REDIS_PORT,
                 db: int = 0):
        self.redis_client = redis.Redis(
            host=host, 
            port=port, 
            db=db,
            decode_responses=True
        )
        self.metrics = CacheMetrics()
        
        # Configure Redis
        self.redis_client.config_set('maxmemory', CacheConfig.MAX_MEMORY)
        self.redis_client.config_set('maxmemory-policy', CacheConfig.EVICTION_POLICY)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with metrics tracking"""
        start_time = time.time()
        try:
            value = self.redis_client.get(key)
            duration = time.time() - start_time
            
            if value:
                self.metrics.hits += 1
                CACHE_HITS.inc()
            else:
                self.metrics.misses += 1
                CACHE_MISSES.inc()
                
            self.metrics.total_latency += duration
            self.metrics.operations += 1
            CACHE_LATENCY.observe(duration)
            
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
        
    def set(self, key: str, value: Any, 
            expiration: int = CacheConfig.DEFAULT_EXPIRATION) -> bool:
        """Set value in cache with metrics tracking"""
        start_time = time.time()
        try:
            result = self.redis_client.setex(
                key,
                expiration,
                json.dumps(value)
            )
            duration = time.time() - start_time
            self.metrics.total_latency += duration
            self.metrics.operations += 1
            CACHE_LATENCY.observe(duration)
            return bool(result)
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        return bool(self.redis_client.delete(key))
        
    def clear(self) -> bool:
        """Clear all cache"""
        return bool(self.redis_client.flushdb())

    def get_metrics(self) -> Dict:
        """Get cache performance metrics"""
        uptime = (datetime.utcnow() - self.metrics.start_time).total_seconds()
        ops = max(1, self.metrics.operations)  # Avoid division by zero
        
        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": self.metrics.hits / ops,
            "average_latency": self.metrics.total_latency / ops,
            "operations_per_second": ops / uptime,
            "memory_used": self.redis_client.info()["used_memory_human"],
            "total_operations": ops
        }

def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    key = ":".join(key_parts)
    return hashlib.md5(key.encode()).hexdigest()

def cached(expiration: int = CacheConfig.DEFAULT_EXPIRATION):
    """Decorator for caching function results with metrics"""
    def decorator(func):
        cache = RedisCache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(key)
            
            if result is None:
                # Cache miss - execute function and cache result
                result = func(*args, **kwargs)
                cache.set(key, result, expiration)
                
            return result
        return wrapper
    return decorator

class CacheCluster:
    def __init__(self, nodes: List[Dict[str, Any]]):
        """Initialize Redis cluster with multiple nodes"""
        self.nodes = [
            RedisCache(node["host"], node["port"], node.get("db", 0))
            for node in nodes
        ]
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cluster using consistent hashing"""
        node_index = hash(key) % len(self.nodes)
        return self.nodes[node_index].get(key)
        
    def set(self, key: str, value: Any,
            expiration: int = CacheConfig.DEFAULT_EXPIRATION) -> bool:
        """Set value in cluster using consistent hashing"""
        node_index = hash(key) % len(self.nodes)
        return self.nodes[node_index].set(key, value, expiration)
        
    def get_cluster_metrics(self) -> List[Dict]:
        """Get metrics from all nodes in cluster"""
        return [node.get_metrics() for node in self.nodes] 