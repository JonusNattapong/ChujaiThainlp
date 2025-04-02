from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
import time
import functools
from typing import Dict, List, Optional
import psutil
import logging
from datetime import datetime

# Prometheus metrics
REQUESTS_TOTAL = Counter('thainlp_requests_total', 'Total requests processed')
PROCESSING_TIME = Histogram('thainlp_processing_seconds', 'Time spent processing request')
MEMORY_USAGE = Gauge('thainlp_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('thainlp_cpu_usage_percent', 'CPU usage percentage')

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
metrics.set_meter_provider(MeterProvider())
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Export traces
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Export metrics
reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
metrics.get_meter_provider().add_reader(reader)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self._request_counter = meter.create_counter(
            "requests",
            description="Number of requests"
        )
        self._duration_histogram = meter.create_histogram(
            "duration",
            description="Request duration"
        )
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        PROCESSING_TIME.observe(duration)
        self._duration_histogram.record(duration)
        
    def track_request(self):
        REQUESTS_TOTAL.inc()
        self._request_counter.add(1)

class SystemMonitor:
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        
    def update_metrics(self):
        """อัพเดทข้อมูล metrics ของระบบ"""
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent())
        
    def check_service_health(self, service_name: str) -> ServiceHealth:
        """ตรวจสอบสถานะของ service"""
        if service_name not in self.services:
            self.services[service_name] = ServiceHealth(service_name)
            
        service = self.services[service_name]
        
        try:
            # ทำการตรวจสอบ service ตามความเหมาะสม
            service.last_check = datetime.utcnow()
            service.status = HealthStatus.OK
        except Exception as e:
            service.status = HealthStatus.ERROR
            service.details["error"] = str(e)
            logger.error(f"Service {service_name} health check failed: {e}")
            
        return service
    
    def get_system_health(self) -> Dict:
        """ดึงข้อมูลสถานะของระบบทั้งหมด"""
        return {
            "memory_usage": psutil.Process().memory_info().rss,
            "cpu_usage": psutil.cpu_percent(),
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
    """เริ่มต้นระบบ monitoring"""
    start_http_server(port)
    logger.info(f"Monitoring server started on port {port}")

def monitor_performance(func):
    """Decorator สำหรับติดตามประสิทธิภาพของฟังก์ชัน"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        REQUESTS_TOTAL.inc()
        with PROCESSING_TIME.time(), tracer.start_as_current_span(func.__name__) as span:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                status = "success"
            except Exception as e:
                status = "error"
                span.set_attribute("error", str(e))
                logger.error(f"Error in {func.__name__}: {e}")
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration", duration)
                span.set_attribute("function", func.__name__)
                span.set_attribute("status", status)
            
            return result
    return wrapper 