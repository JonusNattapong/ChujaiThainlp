"""
Core base classes for ThaiNLP with advanced optimization and monitoring
"""
from typing import Any, Dict, List, Optional, Union
import time
import logging
import hashlib
import asyncio # Changed from threading
import math
import uuid
import traceback
from collections import defaultdict, deque
from datetime import datetime
from functools import lru_cache
import json

import asyncio
import enum
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

class MetricType(enum.Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricValue:
    """Container for metric values with metadata"""
    value: Union[int, float]
    type: MetricType
    timestamp: float
    labels: Dict[str, str]

class MetricsCollector:
    """Enhanced async metrics collection with structured types"""
    
    def __init__(self, max_window_seconds: int = 3600):
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger("metrics")
        self.max_window_seconds = max_window_seconds
        self._processing_task: Optional[asyncio.Task] = None
        
    async def record(self, name: str, value: Union[int, float],
                    metric_type: MetricType, labels: Dict[str, str] = None):
        """Asynchronously record a metric"""
        async with self._lock:
            metric = MetricValue(
                value=value,
                type=metric_type,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[name].append(metric)
            
            # Start background processing if not running
            if not self._processing_task or self._processing_task.done():
                self._processing_task = asyncio.create_task(self._process_metrics())
                
    async def _process_metrics(self):
        """Background task to process and prune metrics"""
        try:
            while True:
                async with self._lock:
                    current_time = time.time()
                    cutoff_time = current_time - self.max_window_seconds
                    
                    # Prune old metrics
                    for metric_name in list(self.metrics.keys()):
                        self.metrics[metric_name] = [
                            m for m in self.metrics[metric_name]
                            if m.timestamp > cutoff_time
                        ]
                        
                await asyncio.sleep(60)  # Process every minute
        except asyncio.CancelledError:
            self.logger.info("Metrics processing task cancelled")
            
    def get_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
            
        values = [m.value for m in self.metrics[metric_name]]
        if not values:
            return {}
            
        sorted_values = sorted(values)
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'median': sorted_values[len(values) // 2],
            'percentile_95': sorted_values[int(len(values) * 0.95)],
            'percentile_99': sorted_values[int(len(values) * 0.99)],
            'window_seconds': self.max_window_seconds,
            'latest': values[-1]
        }
        
    def get_metrics_by_type(self, metric_type: MetricType) -> Dict[str, List[MetricValue]]:
        """Get all metrics of a specific type"""
        return {
            name: values for name, values in self.metrics.items()
            if any(m.type == metric_type for m in values)
        }
        
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format with type information"""
        if format == 'json':
            metrics_dict = {
                name: [
                    {
                        'value': m.value,
                        'type': m.type.value,
                        'timestamp': m.timestamp,
                        'labels': m.labels
                    } for m in values
                ] for name, values in self.metrics.items()
            }
            return json.dumps(metrics_dict)
        raise ValueError(f"Unsupported format: {format}")

class SecurityManager:
    """Handles security and validation of requests"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter() # RateLimiter will use asyncio.Lock

    async def validate_request(self, client_id: str, text: str) -> bool: # Added async
        """Validate incoming request"""
        # rate_limit_check becomes async
        if not await self.rate_limit_check(client_id): # Added await
            return False
        # input_validation remains synchronous
        if not self.input_validation(text):
            return False
        return True

    async def rate_limit_check(self, client_id: str) -> bool: # Added async
        """Check rate limiting"""
        # is_allowed becomes async
        return await self.rate_limiter.is_allowed(client_id) # Added await

    def input_validation(self, text: str) -> bool:
        """Validate input text with comprehensive checks
        
        Args:
            text: Input text to validate
            
        Returns:
            bool: True if text is valid, False otherwise
            
        Checks performed:
        - Not empty or None
        - Length within limits (10000 chars)
        - Contains valid Thai characters (if applicable)
        - No suspicious patterns (SQL, XSS, etc)
        """
        if not text or not isinstance(text, str):
            return False
            
        if len(text) > 10000:
            return False
            
        # Check for valid Thai characters if text appears to be Thai
        if any(chr(c) in text for c in range(0x0E00, 0x0E7F)):
            try:
                text.encode('tis-620')  # Thai encoding
            except UnicodeEncodeError:
                return False
                
        # Basic security checks
        suspicious_patterns = [
            '<script', 'SELECT * FROM', 'DROP TABLE',
            '<?php', 'eval(', 'exec(', 'UNION SELECT'
        ]
        if any(pattern in text.upper() for pattern in suspicious_patterns):
            return False
            
        return True
        
    @staticmethod
    def hash_text(text: str) -> str:
        """Create hash of text for caching/comparison"""
        return hashlib.sha256(text.encode()).hexdigest()

class RateLimiter:
    """Efficient rate limiting using sliding window algorithm"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(lambda: deque(maxlen=max_requests))
        self._lock = asyncio.Lock() # Changed from threading.Lock

    async def is_allowed(self, client_id: str) -> bool: # Added async
        """Check if request is allowed using sliding window"""
        now = time.time()
        window_start = now - self.window_seconds

        async with self._lock: # Added async
            # Remove expired requests
            while (self.requests[client_id] and
                   self.requests[client_id][0] < window_start):
                self.requests[client_id].popleft()
                
            # Check if we have room for new request
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            # Add new request timestamp
            self.requests[client_id].append(now)
            return True
        

from typing import TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass
from collections import OrderedDict

T = TypeVar('T')

@dataclass
class CacheStats:
    """Statistics for cache performance monitoring"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

class Cache(Generic[T]):
    """Enhanced LRU caching system with performance monitoring"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = asyncio.Lock() # Changed from threading.Lock
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[T]: # Added async
        """Get value from cache with LRU tracking"""
        async with self._lock: # Added async
            if key not in self._cache:
                self.stats.misses += 1
                return None
            # Key exists, move it to end and return
            value = self._cache.pop(key)
            self._cache[key] = value
            self.stats.hits += 1
            return value
            # Removed KeyError handling

    async def set(self, key: str, value: T) -> None: # Added async
        """Set value in cache with automatic LRU eviction"""
        async with self._lock: # Added async
            if key in self._cache:
                self._cache.pop(key) # Remove to re-insert at the end
            elif len(self._cache) >= self.max_size:
                # popitem(last=False) removes the LRU item
                lru_key, _ = self._cache.popitem(last=False)
                self.stats.evictions += 1

            self._cache[key] = value # Add/update key, moves it to the end

    async def clear(self) -> None: # Added async
        """Clear cache and reset statistics"""
        async with self._lock: # Added async
            self._cache.clear()
            self.stats = CacheStats()

    async def get_stats(self) -> Dict[str, int]: # Added async
        """Get cache performance statistics"""
        async with self._lock: # Added async
            total = self.stats.hits + self.stats.misses
            hit_rate = self.stats.hits / total if total > 0 else 0.0 # Ensure float division
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'hit_rate': hit_rate
            }

class ABTesting:
    """A/B Testing system with statistical significance"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: deque(maxlen=100000))  # Fixed size for memory efficiency
        self._lock = asyncio.Lock() # Changed from threading.Lock

    # Registering experiment doesn't need lock or async usually
    def register_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]]
    ):
        """Register new A/B test experiment with weight distribution"""
        if not variants:
            raise ValueError("At least one variant required")
        self.experiments[name] = {
            'variants': variants,
            'weights': [1.0/len(variants)] * len(variants)  # Equal weights by default
        }
        
    def get_variant(self, experiment: str, user_id: str) -> Dict[str, Any]:
        """Get weighted variant for user"""
        if experiment not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment}")
            
        # Weighted random selection based on user_id hash
        rand_val = (hash(user_id) % 10000) / 10000.0
        cum_weight = 0.0
        variants = self.experiments[experiment]['variants']
        weights = self.experiments[experiment]['weights']
        
        for i, weight in enumerate(weights):
            cum_weight += weight
            if rand_val < cum_weight:
                return variants[i]
        return variants[-1]  # Fallback
    async def record_result( # Corrected definition
        self,
        experiment: str,
        variant: Dict[str, Any],
        success: bool
    ): # Removed misplaced comment
        """Record experiment result with timestamp"""
        async with self._lock: # This should now be correct
            self.results[experiment].append({
                'variant': str(variant), # Ensure variant is hashable/comparable if dict
                'success': success,
                'timestamp': time.time()
            })

    async def get_experiment_stats(self, experiment: str) -> Dict[str, Any]: # Added async
        """Get statistics with confidence intervals"""
        if experiment not in self.results:
            return {}

        stats = defaultdict(lambda: {'success': 0, 'total': 0})
        current_time = time.time()

        async with self._lock: # Added async
            # Prune old results (older than 30 days)
            # Make a temporary list to avoid modifying deque while iterating if needed
            results_copy = list(self.results[experiment])
            self.results[experiment] = deque(
                r for r in results_copy
                if current_time - r['timestamp'] < 2592000  # 30 days
            )
            
            # Calculate basic stats
            for result in self.results[experiment]:
                variant = result['variant']
                stats[variant]['total'] += 1
                if result['success']:
                    stats[variant]['success'] += 1
                    
            # Calculate statistical significance
            for variant, data in stats.items():
                if data['total'] > 0:
                    success_rate = data['success'] / data['total']
                    # Calculate 95% confidence interval using normal approximation
                    z = 1.96  # Z-score for 95% confidence
                    margin_of_error = z * math.sqrt(
                        (success_rate * (1 - success_rate)) / data['total']
                    )
                    data.update({
                        'success_rate': success_rate,
                        'confidence_interval': [
                            max(0, success_rate - margin_of_error),
                            min(1, success_rate + margin_of_error)
                        ],
                        'statistical_power': self._calculate_power(
                            data['success'],
                            data['total']
                        )
                    })
                    
        return dict(stats)
        
    def _calculate_power(self, successes: int, total: int) -> float:
        """Calculate statistical power of the experiment"""
        if total < 2:
            return 0.0
        # Simplified power calculation
        effect_size = successes / total
        return min(1.0, math.sqrt(total) * effect_size)

from typing import TypeVar, Generic, Dict, Any, Optional, Coroutine
from contextlib import asynccontextmanager

T = TypeVar('T')

class ThaiNLPBase(Generic[T]):
    """Enhanced base class with async support and strong typing"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.security = SecurityManager()
        self.cache: Cache[T] = Cache()
        self.ab_testing = ABTesting()
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure structured logging with correlation IDs"""
        self.logger = logging.getLogger("thainlp")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(correlation_id)s] '
            '%(name)s %(module)s.%(funcName)s: %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    @asynccontextmanager
    async def processing_context(self, correlation_id: str):
        """Context manager for request processing with logging context"""
        log_context = {'correlation_id': correlation_id}
        try:
            yield log_context
        finally:
            # Ensure any pending metrics are processed
            if self.metrics._processing_task:
                self.metrics._processing_task.cancel()
                try:
                    await self.metrics._processing_task
                except asyncio.CancelledError:
                    pass
                
    async def _process_with_cache(self, text: str, task: str, correlation_id: str) -> T:
        """Process text with enhanced caching and monitoring"""
        try:
            start_time = time.time() # Removed duplicated line
            text_hash = self.security.hash_text(text) # Hashing is CPU bound, okay outside lock

            # Check cache first (now async)
            cached = await self.cache.get(text_hash) # Added await
            if cached is not None:
                await self.metrics.record(
                    'cache_hit',
                    1, # Value for counter
                    MetricType.COUNTER,
                    {'task': task}
                )
                return cached
                
            # Process if not cached
            result = await self._process(text, task, correlation_id)
            process_time = time.time() - start_time
            
            # Record metrics
            await self.metrics.record(
                'process_time',
                process_time,
                MetricType.HISTOGRAM,
                {'task': task}
            )
            
            # Cache result (now async)
            await self.cache.set(text_hash, result) # Added await
            return result

        except Exception as e:
            self.logger.error(
                "Processing failed",
                extra={
                    'correlation_id': correlation_id,
                    'text_preview': text[:100],
                    'task': task,
                    'error': str(e)
                }
            )
            raise
            
    async def _process(self, text: str, task: str, correlation_id: str) -> T:
        """Abstract processing method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _process method")
        
    async def process(
        self,
        text: str,
        task: str,
        client_id: str,
        experiment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main processing pipeline with full observability"""
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        async with self.processing_context(correlation_id) as log_context:
            try:
                # Security validation (now async)
                if not await self.security.validate_request(client_id, text): # Added await
                    raise ValueError("Request validation failed")

                # A/B testing setup
                variant = None
                if experiment:
                    variant = self.ab_testing.get_variant(experiment, client_id)
                    
                # Process with caching
                result = await self._process_with_cache(text, task, correlation_id)
                process_time = time.time() - start_time
                
                # Record success metrics
                await self.metrics.record(
                    'success',
                    1,
                    MetricType.COUNTER,
                    {'task': task}
                )
                
                # Record experiment result (now async)
                if experiment and variant:
                    await self.ab_testing.record_result( # Added await
                        experiment,
                        variant,
                        success=True
                    )

                return {
                    'status': 'success',
                    'result': result,
                    'process_time': process_time,
                    'correlation_id': correlation_id
                }
                
            except Exception as e:
                # Record error metrics
                await self.metrics.record(
                    'error',
                    1,
                    MetricType.COUNTER,
                    {
                        'task': task,
                        'error_type': type(e).__name__
                    }
                )
                
                self.logger.error(
                    f"Processing error: {e}",
                    extra={**log_context, 'error_details': traceback.format_exc()}
                )
                
                # Record experiment failure (now async)
                if experiment and variant:
                    await self.ab_testing.record_result( # Added await
                        experiment,
                        variant,
                        success=False
                    )

                return {
                    'status': 'error',
                    'error': str(e),
                    'correlation_id': correlation_id,
                    'process_time': time.time() - start_time
                }
