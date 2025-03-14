"""
Core base classes for ThaiNLP with advanced optimization and monitoring
"""
from typing import Any, Dict, List, Optional, Union
import time
import logging
import hashlib
import threading
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import json

class MetricsCollector:
    """Collect and manage system metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = logging.getLogger("metrics")
        
    def record(self, metric_type: str, value: Any):
        """Record a metric with timestamp"""
        with self._lock:
            self.metrics[metric_type].append({
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
            
    def get_stats(self, metric_type: str) -> Dict[str, Any]:
        """Get statistical summary of a metric"""
        values = [m['value'] for m in self.metrics[metric_type]]
        if not values:
            return {}
            
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }
        
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        if format == 'json':
            return json.dumps(dict(self.metrics))
        raise ValueError(f"Unsupported format: {format}")

class RateLimiter:
    """Rate limiting for API protection"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self._lock = threading.Lock()
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on rate limits"""
        now = time.time()
        
        with self._lock:
            # Remove old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            # Check limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False
                
            # Record request
            self.requests[client_id].append(now)
            return True

class SecurityManager:
    """Manage security aspects"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text"""
        # Remove potential harmful characters
        return text.replace('<', '&lt;').replace('>', '&gt;')
        
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
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                self._access_count[key] += 1
                return self._cache[key]
            return None
            
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove least accessed item
                min_key = min(
                    self._access_count.items(),
                    key=lambda x: x[1]
                )[0]
                del self._cache[min_key]
                del self._access_count[min_key]
                
            self._cache[key] = value
            self._access_count[key] = 1
            
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

class ABTesting:
    """A/B Testing system"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(list)
        self._lock = threading.Lock()
        
    def register_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]]
    ):
        """Register new A/B test experiment"""
        self.experiments[name] = variants
        
    def get_variant(self, experiment: str, user_id: str) -> Dict[str, Any]:
        """Get variant for user"""
        if experiment not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment}")
            
        # Deterministic variant selection based on user_id
        variant_index = hash(user_id) % len(self.experiments[experiment])
        return self.experiments[experiment][variant_index]
        
    def record_result(
        self,
        experiment: str,
        variant: Dict[str, Any],
        success: bool
    ):
        """Record experiment result"""
        with self._lock:
            self.results[experiment].append({
                'variant': variant,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
            
    def get_experiment_stats(self, experiment: str) -> Dict[str, Any]:
        """Get statistics for experiment"""
        if experiment not in self.results:
            return {}
            
        stats = defaultdict(lambda: {'success': 0, 'total': 0})
        for result in self.results[experiment]:
            variant = str(result['variant'])
            stats[variant]['total'] += 1
            if result['success']:
                stats[variant]['success'] += 1
                
        return {
            variant: {
                'success_rate': data['success'] / data['total'],
                'total_trials': data['total']
            }
            for variant, data in stats.items()
        }

class ThaiNLPBase:
    """Base class with advanced features"""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.security = SecurityManager()
        self.cache = Cache()
        self.ab_testing = ABTesting()
        self.logger = logging.getLogger("thainlp")
        
    @lru_cache(maxsize=1000)
    def _process_with_cache(self, text: str, task: str) -> Any:
        """Process text with caching"""
        return self._process(text, task)
        
    def _process(self, text: str, task: str) -> Any:
        """Main processing method to be overridden"""
        raise NotImplementedError
        
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