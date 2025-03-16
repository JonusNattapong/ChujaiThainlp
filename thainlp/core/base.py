"""
Core base classes for ThaiNLP with advanced optimization and monitoring
"""
from typing import Any, Dict, List, Optional, Union
import time
import logging
import hashlib
import threading
import math
import uuid
import traceback
from collections import defaultdict, deque
from datetime import datetime
from functools import lru_cache
import json

class MetricsCollector:
    """Collect and manage system metrics with time-windowed storage"""
    
    def __init__(self, max_window_seconds: int = 3600):
        from collections import deque
        self.metrics = defaultdict(lambda: deque(maxlen=10000))  # Fixed size for memory efficiency
        self._lock = threading.Lock()
        self.logger = logging.getLogger("metrics")
        self.max_window_seconds = max_window_seconds
        
    def record(self, metric_type: str, value: Any):
        """Record a metric with timestamp, automatically pruning old data"""
        timestamp = time.time()
        with self._lock:
            # Prune old entries first
            self._prune_old_entries(metric_type, timestamp)
            self.metrics[metric_type].append({
                'value': value,
                'timestamp': timestamp
            })
            
    def _prune_old_entries(self, metric_type: str, current_time: float):
        """Remove entries older than max_window_seconds"""
        while (self.metrics[metric_type] and 
               current_time - self.metrics[metric_type][0]['timestamp'] > self.max_window_seconds):
            self.metrics[metric_type].popleft()
            
    def get_stats(self, metric_type: str) -> Dict[str, Any]:
        """Get statistical summary of a metric with time window"""
        with self._lock:
            self._prune_old_entries(metric_type, time.time())
            values = [m['value'] for m in self.metrics[metric_type]]
            
            # Initialize stats with default values
            stats = {
                'count': 0,
                'avg': 0,
                'min': 0,
                'max': 0,
                'percentile_95': 0,
                'window_seconds': self.max_window_seconds
            }
            
            if values:
                stats.update({
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'percentile_95': sorted(values)[int(len(values) * 0.95)]
                })
                
            return stats
        
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        if format == 'json':
            return json.dumps(dict(self.metrics))
        raise ValueError(f"Unsupported format: {format}")

class SecurityManager:
    """Handles security and validation of requests"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        
    def validate_request(self, client_id: str, text: str) -> bool:
        """Validate incoming request"""
        if not self.rate_limit_check(client_id):
            return False
        if not self.input_validation(text):
            return False
        return True
        
    def rate_limit_check(self, client_id: str) -> bool:
        """Check rate limiting"""
        return self.rate_limiter.is_allowed(client_id)
        
    def input_validation(self, text: str) -> bool:
        """Validate input text"""
        if not text or len(text) > 10000:  # Example limit
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
        self._lock = threading.Lock()
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed using sliding window"""
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._lock:
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
        
    def validate_request(self, client_id: str, text: str) -> bool:
        """Validate incoming request"""
        if not self.rate_limit_check(client_id):
            return False
        if not self.input_validation(text):
            return False
        return True
        
    def rate_limit_check(self, client_id: str) -> bool:
        """Check rate limiting"""
        return self.rate_limiter.is_allowed(client_id)
        
    def input_validation(self, text: str) -> bool:
        """Validate input text"""
        if not text or len(text) > 10000:  # Example limit
            return False
        return True
        
    @staticmethod
    def hash_text(text: str) -> str:
        """Create hash of text for caching/comparison"""
        return hashlib.sha256(text.encode()).hexdigest()

class Cache:
    """Advanced caching system"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
        self._access_count = defaultdict(int)
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returns None if cache is empty or key not found"""
        with self._lock:
            if not self._cache:  # Return None if cache is empty
                return None
            if key in self._cache:
                self._access_count[key] += 1
                return self._cache[key]
            return None
            
    def set(self, key: str, value: Any):
        """Set value in cache using proper LRU eviction"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove least recently used item
                lru_key = min(
                    self._access_count.items(),
                    key=lambda x: x[1]  # Find key with oldest timestamp
                )[0]
                del self._cache[lru_key]
                del self._access_count[lru_key]
                
            # Update cache and access count
            self._cache[key] = value
            self._access_count[key] = time.time()  # Update access time
            
    def clear(self):
        """Clear cache and access tracking"""
        with self._lock:
            # Clear existing dicts completely
            self._cache.clear()
            self._access_count.clear()
            # Create new empty structures to ensure complete reset
            self._cache = {}
            self._access_count = defaultdict(int)

class ABTesting:
    """A/B Testing system with statistical significance"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: deque(maxlen=100000))  # Fixed size for memory efficiency
        self._lock = threading.Lock()
        
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
        
    def record_result(
        self,
        experiment: str,
        variant: Dict[str, Any],
        success: bool
    ):
        """Record experiment result with timestamp"""
        with self._lock:
            self.results[experiment].append({
                'variant': str(variant),
                'success': success,
                'timestamp': time.time()
            })
            
    def get_experiment_stats(self, experiment: str) -> Dict[str, Any]:
        """Get statistics with confidence intervals"""
        if experiment not in self.results:
            return {}
            
        stats = defaultdict(lambda: {'success': 0, 'total': 0})
        current_time = time.time()
        
        with self._lock:
            # Prune old results (older than 30 days)
            self.results[experiment] = deque(
                r for r in self.results[experiment]
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

class ThaiNLPBase:
    """Base class with enhanced error handling and monitoring"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.security = SecurityManager()
        self.cache = Cache()
        self.ab_testing = ABTesting()
        
        # Configure structured logging
        self.logger = logging.getLogger("thainlp")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s '
            'process=%(process)d thread=%(thread)d '
            'module=%(module)s func=%(funcName)s '
            'message="%(message)s"'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    @lru_cache(maxsize=1000)
    def _process_with_cache(self, text: str, task: str) -> Any:
        """Process text with caching and error handling"""
        try:
            start_time = time.time()
            result = self._process(text, task)
            self.metrics.record('cache_hit', 1)
            self.metrics.record('cache_process_time', time.time() - start_time)
            return result
        except Exception as e:
            self.logger.error(
                "Cache processing failed",
                extra={'text': text[:100], 'task': task, 'error': str(e)}
            )
            raise
        
    def _process(self, text: str, task: str) -> Any:
        """Main processing method to be overridden"""
        raise NotImplementedError("Subclasses must implement _process method")
        
    def process(
        self,
        text: str,
        task: str,
        client_id: str,
        experiment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process text with full pipeline"""
        start_time = time.time()
        
        try:
            # Security checks
            if not self.security.validate_request(client_id, text):
                raise ValueError("Request validation failed")
                
            # Get experiment variant if applicable
            variant = None
            if experiment:
                variant = self.ab_testing.get_variant(experiment, client_id)
                
            # Process text
            text_hash = self.security.hash_text(text)
            if cached := self.cache.get(text_hash):
                result = cached
            else:
                result = self._process_with_cache(text, task)
                self.cache.set(text_hash, result)
                
            # Record metrics
            process_time = time.time() - start_time
            self.metrics.record('process_time', process_time)
            self.metrics.record('success', 1)
            
            # Record experiment result if applicable
            if experiment and variant:
                self.ab_testing.record_result(
                    experiment,
                    variant,
                    success=True
                )
                
            return {
                'status': 'success',
                'result': result,
                'process_time': process_time
            }
            
        except Exception as e:
            # Record error
            self.metrics.record('error', str(e))
            self.logger.error(f"Error processing text: {e}")
            
            # Record experiment failure if applicable
            if experiment and variant:
                self.ab_testing.record_result(
                    experiment,
                    variant,
                    success=False
                )
                
            return {
                'status': 'error',
                'error': str(e),
                'process_time': time.time() - start_time
            }
