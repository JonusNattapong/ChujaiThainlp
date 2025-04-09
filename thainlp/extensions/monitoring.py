"""
Simple monitoring module for Thai NLP
This is a simplified version that doesn't require external dependencies
"""
import time
import functools
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Basic health status constants
class HealthStatus:
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"

class ServiceHealth:
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.OK
        self.last_check = datetime.utcnow()
        self.details: Dict = {}

class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, total: int = 0):
        """Initialize progress tracker"""
        self.total = total
        self.current = 0
        self.start_time = None
        self.end_time = None
        
    def start_task(self, total: int = None):
        """Start tracking a task"""
        if total is not None:
            self.total = total
        self.current = 0
        self.start_time = time.time()
        return self
        
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        return self
        
    def end_task(self):
        """End the task"""
        self.end_time = time.time()
        return self
        
    @property
    def progress(self) -> float:
        """Get progress as percentage"""
        if self.total == 0:
            return 100.0
        return min(100.0, self.current / self.total * 100)
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def remaining(self) -> float:
        """Estimate remaining time in seconds"""
        if self.progress == 0:
            return float('inf')
        if self.progress >= 100:
            return 0.0
        elapsed = self.elapsed
        return (elapsed / self.progress) * (100 - self.progress)
        
    def __str__(self) -> str:
        """String representation of progress"""
        return f"{self.progress:.1f}% ({self.current}/{self.total})"

class PerformanceMonitor:
    """Simple performance monitor"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0, 
            "failed_requests": 0,
            "average_time": 0,
            "total_time": 0,
        }
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics["total_time"] += duration
        self.metrics["total_requests"] += 1
        
        # Update average
        self.metrics["average_time"] = (
            self.metrics["total_time"] / self.metrics["total_requests"]
        )
        
        if exc_type:
            self.metrics["failed_requests"] += 1
        else:
            self.metrics["successful_requests"] += 1
        
    def track_request(self, success: bool = True):
        """Track a request"""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

class ResourceMonitor:
    """Simple system resource monitor"""
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.measurements: List[Dict[str, Any]] = []
        
    def update_metrics(self):
        """Update system metrics"""
        # No dependency on psutil, could implement later
        pass
            
    def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check service health"""
        if service_name not in self.services:
            self.services[service_name] = ServiceHealth(service_name)
            
        service = self.services[service_name]
        service.last_check = datetime.utcnow()
        return service
    
    def get_system_health(self) -> Dict:
        """Get all system health metrics"""
        return {
            "services": {
                name: {
                    "status": service.status,
                    "last_check": service.last_check,
                    "details": service.details
                }
                for name, service in self.services.items()
            }
        }

def start_monitoring(port: int = 8000):
    """Start monitoring server stub"""
    logger.info(f"Simplified monitoring enabled (without server)")

def monitor_performance(func):
    """Decorator for function performance monitoring"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            status = "success"
            return result
        except Exception as e:
            status = "error"
            logger.error(f"Error in {func.__name__}: {e}")
            raise
        finally:
            duration = time.time() - start_time
            logger.debug(f"Function {func.__name__} completed in {duration:.4f}s with status: {status}")
            
    return wrapper