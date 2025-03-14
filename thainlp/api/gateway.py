"""
Advanced API Gateway for ThaiNLP
"""
from typing import Any, Dict, List, Optional, Union, Callable
import time
import threading
import json
import logging
import uuid
import os
import sys
import traceback
from datetime import datetime
from functools import wraps
import asyncio
import aiohttp
from aiohttp import web
import jwt
from cryptography.fernet import Fernet

from ..core.base import ThaiNLPBase, SecurityManager, Cache
from ..monitoring.monitor import MonitoringSystem
from ..optimization.optimizer import Optimizer

class APIRateLimiter:
    """Advanced API rate limiting with tiered access"""
    
    def __init__(self):
        self.tiers = {
            'free': {'requests_per_minute': 10, 'requests_per_day': 100},
            'basic': {'requests_per_minute': 60, 'requests_per_day': 1000},
            'premium': {'requests_per_minute': 600, 'requests_per_day': 10000},
            'enterprise': {'requests_per_minute': 6000, 'requests_per_day': 100000}
        }
        self.minute_counters = {}
        self.day_counters = {}
        self._lock = threading.Lock()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
        
    def is_allowed(self, api_key: str, tier: str = 'free') -> bool:
        """Check if request is allowed based on tier limits"""
        if tier not in self.tiers:
            tier = 'free'
            
        now = time.time()
        minute_window = 60
        day_window = 86400  # 24 hours
        
        with self._lock:
            # Initialize counters if needed
            if api_key not in self.minute_counters:
                self.minute_counters[api_key] = []
            if api_key not in self.day_counters:
                self.day_counters[api_key] = []
                
            # Clean old requests
            self.minute_counters[api_key] = [
                t for t in self.minute_counters[api_key]
                if now - t < minute_window
            ]
            self.day_counters[api_key] = [
                t for t in self.day_counters[api_key]
                if now - t < day_window
            ]
            
            # Check limits
            if len(self.minute_counters[api_key]) >= self.tiers[tier]['requests_per_minute']:
                return False
            if len(self.day_counters[api_key]) >= self.tiers[tier]['requests_per_day']:
                return False
                
            # Record request
            self.minute_counters[api_key].append(now)
            self.day_counters[api_key].append(now)
            return True
            
    def _cleanup_loop(self):
        """Background thread to clean up old request records"""
        while True:
            time.sleep(300)  # Run every 5 minutes
            now = time.time()
            minute_window = 60
            day_window = 86400
            
            with self._lock:
                # Clean minute counters
                for api_key in list(self.minute_counters.keys()):
                    self.minute_counters[api_key] = [
                        t for t in self.minute_counters[api_key]
                        if now - t < minute_window
                    ]
                    
                # Clean day counters
                for api_key in list(self.day_counters.keys()):
                    self.day_counters[api_key] = [
                        t for t in self.day_counters[api_key]
                        if now - t < day_window
                    ]

class APIKeyManager:
    """Manage API keys and authentication"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.api_keys = {}
        self.user_data = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger("api_key_manager")
        
        # Create encryption key for sensitive data
        self.cipher_suite = Fernet(Fernet.generate_key())
        
    def generate_api_key(self, user_id: str, tier: str = 'free') -> str:
        """Generate new API key for user"""
        with self._lock:
            # Create payload
            payload = {
                'user_id': user_id,
                'tier': tier,
                'created': datetime.now().isoformat(),
                'jti': str(uuid.uuid4())
            }
            
            # Generate JWT token as API key
            api_key = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # Store API key info
            self.api_keys[api_key] = {
                'user_id': user_id,
                'tier': tier,
                'created': payload['created'],
                'last_used': None,
                'active': True
            }
            
            return api_key
            
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key and return user info"""
        try:
            # Decode and verify JWT
            payload = jwt.decode(
                api_key,
                self.secret_key,
                algorithms=['HS256']
            )
            
            with self._lock:
                # Check if key exists and is active
                if api_key not in self.api_keys or not self.api_keys[api_key]['active']:
                    return {'valid': False, 'reason': 'Invalid or inactive API key'}
                    
                # Update last used
                self.api_keys[api_key]['last_used'] = datetime.now().isoformat()
                
                return {
                    'valid': True,
                    'user_id': payload['user_id'],
                    'tier': payload['tier']
                }
                
        except jwt.PyJWTError as e:
            self.logger.warning(f"Invalid API key: {e}")
            return {'valid': False, 'reason': str(e)}
            
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        with self._lock:
            if api_key in self.api_keys:
                self.api_keys[api_key]['active'] = False
                return True
            return False
            
    def store_user_data(self, user_id: str, data: Dict[str, Any]):
        """Securely store user data"""
        with self._lock:
            # Encrypt sensitive data
            encrypted_data = self.cipher_suite.encrypt(
                json.dumps(data).encode()
            )
            self.user_data[user_id] = encrypted_data
            
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data"""
        with self._lock:
            if user_id not in self.user_data:
                return None
                
            # Decrypt data
            encrypted_data = self.user_data[user_id]
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data)

class RequestLogger:
    """Advanced request logging system"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up loggers
        self.access_logger = logging.getLogger("access")
        self.error_logger = logging.getLogger("error")
        
        # Set up handlers
        access_handler = logging.FileHandler(
            os.path.join(log_dir, "access.log")
        )
        error_handler = logging.FileHandler(
            os.path.join(log_dir, "error.log")
        )
        
        # Set up formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        access_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        # Add handlers to loggers
        self.access_logger.addHandler(access_handler)
        self.error_logger.addHandler(error_handler)
        
        # Set log levels
        self.access_logger.setLevel(logging.INFO)
        self.error_logger.setLevel(logging.ERROR)
        
    def log_request(
        self,
        client_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        process_time: float,
        request_size: int,
        response_size: int
    ):
        """Log API request"""
        self.access_logger.info(
            f"{client_id} - {method} {endpoint} - {status_code} - "
            f"{process_time:.4f}s - {request_size}B/{response_size}B"
        )
        
    def log_error(
        self,
        client_id: str,
        endpoint: str,
        method: str,
        error: str,
        traceback_str: str
    ):
        """Log API error"""
        self.error_logger.error(
            f"{client_id} - {method} {endpoint} - {error}\n{traceback_str}"
        )

class APIGateway:
    """Advanced API Gateway for ThaiNLP"""
    
    def __init__(
        self,
        secret_key: str,
        cache_dir: str,
        log_dir: str,
        email_config: Optional[Dict[str, str]] = None
    ):
        # Initialize components
        self.api_key_manager = APIKeyManager(secret_key)
        self.rate_limiter = APIRateLimiter()
        self.request_logger = RequestLogger(log_dir)
        self.cache = Cache(max_size=10000)
        self.security = SecurityManager()
        
        # Initialize optimization and monitoring
        self.optimizer = Optimizer(
            cache_dir=os.path.join(cache_dir, "optimizer"),
            max_workers=None  # Auto-detect
        )
        self.monitoring = MonitoringSystem(email_config=email_config)
        
        # Initialize web app
        self.app = web.Application(middlewares=[self._auth_middleware])
        self._setup_routes()
        
        # Store processors
        self.processors = {}
        
    async def _auth_middleware(
        self,
        request: web.Request,
        handler: Callable
    ) -> web.Response:
        """Authentication and rate limiting middleware"""
        # Skip auth for some endpoints
        if request.path in ['/health', '/docs', '/']:
            return await handler(request)
            
        # Get API key
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return web.json_response(
                {'error': 'Missing API key'},
                status=401
            )
            
        # Validate API key
        validation = self.api_key_manager.validate_api_key(api_key)
        if not validation['valid']:
            return web.json_response(
                {'error': validation['reason']},
                status=401
            )
            
        # Check rate limits
        if not self.rate_limiter.is_allowed(api_key, validation['tier']):
            return web.json_response(
                {'error': 'Rate limit exceeded'},
                status=429
            )
            
        # Set user info in request
        request['user_id'] = validation['user_id']
        request['tier'] = validation['tier']
        
        # Process request
        try:
            response = await handler(request)
            return response
        except Exception as e:
            # Log error
            self.request_logger.log_error(
                validation['user_id'],
                request.path,
                request.method,
                str(e),
                traceback.format_exc()
            )
            
            # Return error response
            return web.json_response(
                {'error': str(e)},
                status=500
            )
            
    def _setup_routes(self):
        """Set up API routes"""
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/docs', self.handle_docs)
        self.app.router.add_post('/api/process', self.handle_process)
        self.app.router.add_post('/api/batch', self.handle_batch)
        self.app.router.add_get('/api/status', self.handle_status)
        
    def register_processor(
        self,
        name: str,
        processor: ThaiNLPBase
    ):
        """Register text processor"""
        self.processors[name] = processor
        
    async def handle_root(self, request: web.Request) -> web.Response:
        """Handle root endpoint"""
        return web.json_response({
            'name': 'ThaiNLP API Gateway',
            'version': '1.0.0',
            'docs': '/docs'
        })
        
    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle health check endpoint"""
        # Get system metrics
        metrics = self.monitoring.system_monitor.collect_metrics()
        
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'cpu': metrics['cpu_percent'],
                'memory': metrics['memory_percent'],
                'disk': metrics['disk_usage']
            }
        })
        
    async def handle_docs(self, request: web.Request) -> web.Response:
        """Handle API documentation endpoint"""
        docs = {
            'endpoints': [
                {
                    'path': '/api/process',
                    'method': 'POST',
                    'description': 'Process text with specified processor',
                    'parameters': {
                        'processor': 'Name of processor to use',
                        'text': 'Text to process',
                        'options': 'Optional processing options'
                    },
                    'authentication': 'API key required in X-API-Key header'
                },
                {
                    'path': '/api/batch',
                    'method': 'POST',
                    'description': 'Process multiple texts in batch',
                    'parameters': {
                        'processor': 'Name of processor to use',
                        'texts': 'List of texts to process',
                        'options': 'Optional processing options'
                    },
                    'authentication': 'API key required in X-API-Key header'
                },
                {
                    'path': '/api/status',
                    'method': 'GET',
                    'description': 'Get system status and metrics',
                    'authentication': 'API key required in X-API-Key header'
                }
            ],
            'processors': list(self.processors.keys())
        }
        
        return web.json_response(docs)
        
    async def handle_process(self, request: web.Request) -> web.Response:
        """Handle text processing endpoint"""
        start_time = time.time()
        
        # Parse request
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
            
        # Validate request
        if 'processor' not in data:
            return web.json_response(
                {'error': 'Missing processor parameter'},
                status=400
            )
        if 'text' not in data:
            return web.json_response(
                {'error': 'Missing text parameter'},
                status=400
            )
            
        processor_name = data['processor']
        text = data['text']
        options = data.get('options', {})
        
        # Check if processor exists
        if processor_name not in self.processors:
            return web.json_response(
                {'error': f'Unknown processor: {processor_name}'},
                status=400
            )
            
        # Process text
        processor = self.processors[processor_name]
        
        # Try cache first
        cache_key = f"{processor_name}_{self.security.hash_text(text)}_{hash(frozenset(options.items()))}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            result = cached_result
        else:
            # Process text with optimization
            result = await asyncio.to_thread(
                self.optimizer.process_text,
                text,
                lambda t: processor.process(
                    t,
                    task=processor_name,
                    client_id=request['user_id'],
                    **options
                )['result']
            )
            
            # Cache result
            self.cache.set(cache_key, result)
            
        # Calculate metrics
        process_time = time.time() - start_time
        request_size = len(json.dumps(data))
        response_data = {'result': result, 'process_time': process_time}
        response_size = len(json.dumps(response_data))
        
        # Log request
        self.request_logger.log_request(
            request['user_id'],
            '/api/process',
            'POST',
            200,
            process_time,
            request_size,
            response_size
        )
        
        # Record performance metrics
        self.monitoring.performance_monitor.record_processing(
            processor_name,
            process_time,
            len(text),
            True
        )
        
        return web.json_response(response_data)
        
    async def handle_batch(self, request: web.Request) -> web.Response:
        """Handle batch processing endpoint"""
        start_time = time.time()
        
        # Parse request
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {'error': 'Invalid JSON'},
                status=400
            )
            
        # Validate request
        if 'processor' not in data:
            return web.json_response(
                {'error': 'Missing processor parameter'},
                status=400
            )
        if 'texts' not in data:
            return web.json_response(
                {'error': 'Missing texts parameter'},
                status=400
            )
            
        processor_name = data['processor']
        texts = data['texts']
        options = data.get('options', {})
        
        # Check if processor exists
        if processor_name not in self.processors:
            return web.json_response(
                {'error': f'Unknown processor: {processor_name}'},
                status=400
            )
            
        # Validate texts
        if not isinstance(texts, list):
            return web.json_response(
                {'error': 'Texts parameter must be a list'},
                status=400
            )
            
        # Get batch size based on tier
        batch_sizes = {
            'free': 10,
            'basic': 50,
            'premium': 200,
            'enterprise': 1000
        }
        max_batch = batch_sizes.get(request['tier'], 10)
        
        if len(texts) > max_batch:
            return web.json_response(
                {'error': f'Batch size exceeds limit for your tier ({max_batch})'},
                status=400
            )
            
        # Process texts
        processor = self.processors[processor_name]
        
        # Process batch with optimization
        results = await asyncio.to_thread(
            self.optimizer.process_batch,
            texts,
            lambda t: processor.process(
                t,
                task=processor_name,
                client_id=request['user_id'],
                **options
            )['result'],
            batch_size=100
        )
        
        # Calculate metrics
        process_time = time.time() - start_time
        request_size = len(json.dumps(data))
        response_data = {'results': results, 'process_time': process_time}
        response_size = len(json.dumps(response_data))
        
        # Log request
        self.request_logger.log_request(
            request['user_id'],
            '/api/batch',
            'POST',
            200,
            process_time,
            request_size,
            response_size
        )
        
        # Record performance metrics
        self.monitoring.performance_monitor.record_processing(
            f"{processor_name}_batch",
            process_time,
            sum(len(t) for t in texts),
            True
        )
        
        return web.json_response(response_data)
        
    async def handle_status(self, request: web.Request) -> web.Response:
        """Handle status endpoint"""
        # Get system status
        status = self.monitoring.get_system_status()
        
        # Add API usage stats for this user
        user_id = request['user_id']
        api_usage = {
            'tier': request['tier'],
            'rate_limits': self.rate_limiter.tiers[request['tier']]
        }
        
        return web.json_response({
            'system_status': status,
            'api_usage': api_usage,
            'processors': list(self.processors.keys())
        })
        
    async def start(self, host: str = '0.0.0.0', port: int = 8080):
        """Start API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        print(f"API Gateway running at http://{host}:{port}")
        
    async def stop(self):
        """Stop API server and monitoring"""
        self.monitoring.stop()
        await self.app.shutdown()
        
def create_gateway(
    secret_key: Optional[str] = None,
    cache_dir: str = "cache",
    log_dir: str = "logs",
    email_config: Optional[Dict[str, str]] = None
) -> APIGateway:
    """Create API Gateway instance"""
    if not secret_key:
        # Generate random secret key
        secret_key = Fernet.generate_key().decode()
        
    return APIGateway(
        secret_key=secret_key,
        cache_dir=cache_dir,
        log_dir=log_dir,
        email_config=email_config
    ) 