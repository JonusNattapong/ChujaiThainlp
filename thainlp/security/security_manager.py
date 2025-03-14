"""
Advanced Security System for ThaiNLP
"""
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
import time
import threading
import json
import logging
import uuid
import os
import re
import hashlib
import hmac
import base64
import secrets
from datetime import datetime, timedelta
from collections import defaultdict
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class InputValidator:
    """Validate and sanitize input"""
    
    def __init__(self):
        # Common Thai language patterns
        self.thai_pattern = re.compile(r'[\u0E00-\u0E7F]+')
        
        # Suspicious patterns
        self.script_pattern = re.compile(r'<script.*?>.*?</script>', re.IGNORECASE | re.DOTALL)
        self.sql_pattern = re.compile(r'(\b(select|insert|update|delete|drop|alter|create|exec)\b)', re.IGNORECASE)
        self.path_traversal_pattern = re.compile(r'\.\./')
        
        # Input length limits
        self.max_text_length = 100000  # 100KB
        self.max_batch_size = 1000
        
    def validate_text(self, text: str) -> Tuple[bool, str]:
        """Validate text input
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check length
        if not text:
            return False, "Empty text"
            
        if len(text) > self.max_text_length:
            return False, f"Text too long (max {self.max_text_length} chars)"
            
        # Check for suspicious patterns
        if self.script_pattern.search(text):
            return False, "Potential script injection detected"
            
        if self.sql_pattern.search(text):
            return False, "Potential SQL injection detected"
            
        if self.path_traversal_pattern.search(text):
            return False, "Potential path traversal detected"
            
        return True, ""
        
    def validate_batch(self, texts: List[str]) -> Tuple[bool, str]:
        """Validate batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check batch size
        if not texts:
            return False, "Empty batch"
            
        if len(texts) > self.max_batch_size:
            return False, f"Batch too large (max {self.max_batch_size} items)"
            
        # Validate each text
        for i, text in enumerate(texts):
            is_valid, reason = self.validate_text(text)
            if not is_valid:
                return False, f"Item {i}: {reason}"
                
        return True, ""
        
    def sanitize_text(self, text: str) -> str:
        """Sanitize text by removing potentially harmful content
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text
        """
        # Remove script tags
        text = self.script_pattern.sub('', text)
        
        # Escape HTML entities
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text
        
    def contains_thai(self, text: str) -> bool:
        """Check if text contains Thai characters
        
        Args:
            text: Input text
            
        Returns:
            Whether text contains Thai characters
        """
        return bool(self.thai_pattern.search(text))

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        # Rate limit configurations
        self.limits = {
            'global': {'requests_per_second': 1000, 'burst': 5000},
            'ip': {'requests_per_minute': 60, 'requests_per_hour': 1000},
            'user': {'requests_per_minute': 120, 'requests_per_day': 10000},
            'endpoint': {
                'default': {'requests_per_minute': 600},
                'high_load': {'requests_per_minute': 100}
            }
        }
        
        # Request counters
        self.global_counter = []
        self.ip_counters = defaultdict(list)
        self.user_counters = defaultdict(list)
        self.endpoint_counters = defaultdict(lambda: defaultdict(list))
        
        # High load endpoints
        self.high_load_endpoints = {
            '/api/batch',
            '/api/summarize',
            '/api/translate'
        }
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
        
    def is_allowed_global(self) -> bool:
        """Check global rate limit
        
        Returns:
            Whether request is allowed
        """
        now = time.time()
        second_window = 1
        
        with self._lock:
            # Clean old requests
            self.global_counter = [
                t for t in self.global_counter
                if now - t < second_window
            ]
            
            # Check limit
            if len(self.global_counter) >= self.limits['global']['requests_per_second']:
                # Check burst capacity
                if len(self.global_counter) >= self.limits['global']['burst']:
                    return False
                    
            # Record request
            self.global_counter.append(now)
            return True
            
    def is_allowed_ip(self, ip: str) -> bool:
        """Check IP-based rate limit
        
        Args:
            ip: Client IP address
            
        Returns:
            Whether request is allowed
        """
        now = time.time()
        minute_window = 60
        hour_window = 3600
        
        with self._lock:
            # Initialize counters if needed
            if ip not in self.ip_counters:
                self.ip_counters[ip] = []
                
            # Clean old requests
            requests = self.ip_counters[ip]
            minute_requests = [t for t in requests if now - t < minute_window]
            hour_requests = [t for t in requests if now - t < hour_window]
            
            # Update counters
            self.ip_counters[ip] = hour_requests
            
            # Check limits
            if len(minute_requests) >= self.limits['ip']['requests_per_minute']:
                return False
                
            if len(hour_requests) >= self.limits['ip']['requests_per_hour']:
                return False
                
            # Record request
            self.ip_counters[ip].append(now)
            return True
            
    def is_allowed_user(self, user_id: str) -> bool:
        """Check user-based rate limit
        
        Args:
            user_id: User identifier
            
        Returns:
            Whether request is allowed
        """
        now = time.time()
        minute_window = 60
        day_window = 86400
        
        with self._lock:
            # Initialize counters if needed
            if user_id not in self.user_counters:
                self.user_counters[user_id] = []
                
            # Clean old requests
            requests = self.user_counters[user_id]
            minute_requests = [t for t in requests if now - t < minute_window]
            day_requests = [t for t in requests if now - t < day_window]
            
            # Update counters
            self.user_counters[user_id] = day_requests
            
            # Check limits
            if len(minute_requests) >= self.limits['user']['requests_per_minute']:
                return False
                
            if len(day_requests) >= self.limits['user']['requests_per_day']:
                return False
                
            # Record request
            self.user_counters[user_id].append(now)
            return True
            
    def is_allowed_endpoint(self, endpoint: str) -> bool:
        """Check endpoint-based rate limit
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Whether request is allowed
        """
        now = time.time()
        minute_window = 60
        
        # Determine endpoint type
        endpoint_type = 'high_load' if endpoint in self.high_load_endpoints else 'default'
        limit = self.limits['endpoint'][endpoint_type]['requests_per_minute']
        
        with self._lock:
            # Clean old requests
            self.endpoint_counters[endpoint] = [
                t for t in self.endpoint_counters[endpoint]
                if now - t < minute_window
            ]
            
            # Check limit
            if len(self.endpoint_counters[endpoint]) >= limit:
                return False
                
            # Record request
            self.endpoint_counters[endpoint].append(now)
            return True
            
    def is_allowed(
        self,
        ip: str,
        user_id: str,
        endpoint: str
    ) -> Tuple[bool, str]:
        """Check all rate limits
        
        Args:
            ip: Client IP address
            user_id: User identifier
            endpoint: API endpoint
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Check global limit
        if not self.is_allowed_global():
            return False, "Global rate limit exceeded"
            
        # Check IP limit
        if not self.is_allowed_ip(ip):
            return False, "IP-based rate limit exceeded"
            
        # Check user limit
        if not self.is_allowed_user(user_id):
            return False, "User-based rate limit exceeded"
            
        # Check endpoint limit
        if not self.is_allowed_endpoint(endpoint):
            return False, "Endpoint-based rate limit exceeded"
            
        return True, ""
        
    def _cleanup_loop(self):
        """Background thread to clean up old request records"""
        while True:
            time.sleep(300)  # Run every 5 minutes
            
            with self._lock:
                now = time.time()
                day_window = 86400
                
                # Clean IP counters
                for ip in list(self.ip_counters.keys()):
                    self.ip_counters[ip] = [
                        t for t in self.ip_counters[ip]
                        if now - t < day_window
                    ]
                    
                # Clean user counters
                for user_id in list(self.user_counters.keys()):
                    self.user_counters[user_id] = [
                        t for t in self.user_counters[user_id]
                        if now - t < day_window
                    ]

class IPBlocklist:
    """IP address blocklist"""
    
    def __init__(self):
        self.blocked_ips = set()
        self.blocked_ranges = []
        self._lock = threading.Lock()
        
    def load_blocklist(self, file_path: str):
        """Load blocklist from file
        
        Args:
            file_path: Path to blocklist file
        """
        with self._lock:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                            
                        if '/' in line:  # CIDR notation
                            self.blocked_ranges.append(ipaddress.ip_network(line))
                        else:
                            self.blocked_ips.add(line)
                            
            except Exception as e:
                logging.error(f"Error loading IP blocklist: {e}")
                
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked
        
        Args:
            ip: IP address to check
            
        Returns:
            Whether IP is blocked
        """
        with self._lock:
            # Check exact match
            if ip in self.blocked_ips:
                return True
                
            # Check ranges
            try:
                ip_obj = ipaddress.ip_address(ip)
                for network in self.blocked_ranges:
                    if ip_obj in network:
                        return True
                        
            except ValueError:
                # Invalid IP address
                return True
                
            return False
            
    def block_ip(self, ip: str):
        """Add IP to blocklist
        
        Args:
            ip: IP address to block
        """
        with self._lock:
            self.blocked_ips.add(ip)
            
    def unblock_ip(self, ip: str) -> bool:
        """Remove IP from blocklist
        
        Args:
            ip: IP address to unblock
            
        Returns:
            Whether IP was unblocked
        """
        with self._lock:
            if ip in self.blocked_ips:
                self.blocked_ips.remove(ip)
                return True
            return False

class EncryptionManager:
    """Manage encryption and decryption"""
    
    def __init__(self, key_dir: str):
        """Initialize encryption manager
        
        Args:
            key_dir: Directory to store encryption keys
        """
        self.key_dir = key_dir
        os.makedirs(key_dir, exist_ok=True)
        
        # Generate or load Fernet key
        self.fernet_key_path = os.path.join(key_dir, 'fernet.key')
        if os.path.exists(self.fernet_key_path):
            with open(self.fernet_key_path, 'rb') as f:
                self.fernet_key = f.read()
        else:
            self.fernet_key = Fernet.generate_key()
            with open(self.fernet_key_path, 'wb') as f:
                f.write(self.fernet_key)
                
        self.fernet = Fernet(self.fernet_key)
        
        # Generate or load RSA keys
        self.rsa_private_key_path = os.path.join(key_dir, 'rsa_private.pem')
        self.rsa_public_key_path = os.path.join(key_dir, 'rsa_public.pem')
        
        if os.path.exists(self.rsa_private_key_path):
            with open(self.rsa_private_key_path, 'rb') as f:
                self.rsa_private_key = rsa.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            with open(self.rsa_public_key_path, 'rb') as f:
                self.rsa_public_key = rsa.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
        else:
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            # Save keys
            with open(self.rsa_private_key_path, 'wb') as f:
                f.write(self.rsa_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            with open(self.rsa_public_key_path, 'wb') as f:
                f.write(self.rsa_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
                
    def encrypt_symmetric(self, data: str) -> str:
        """Encrypt data using symmetric encryption
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        encrypted = self.fernet.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
        
    def decrypt_symmetric(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data
        """
        encrypted = base64.b64decode(encrypted_data)
        decrypted = self.fernet.decrypt(encrypted)
        return decrypted.decode()
        
    def encrypt_asymmetric(self, data: str) -> str:
        """Encrypt data using asymmetric encryption
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        encrypted = self.rsa_public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted).decode()
        
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """Decrypt data using asymmetric encryption
        
        Args:
            encrypted_data: Encrypted data as base64 string
            
        Returns:
            Decrypted data
        """
        encrypted = base64.b64decode(encrypted_data)
        decrypted = self.rsa_private_key.decrypt(
            encrypted,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode()
        
    def generate_hmac(self, data: str) -> str:
        """Generate HMAC for data
        
        Args:
            data: Data to sign
            
        Returns:
            HMAC signature as hex string
        """
        h = hmac.new(self.fernet_key, data.encode(), hashlib.sha256)
        return h.hexdigest()
        
    def verify_hmac(self, data: str, signature: str) -> bool:
        """Verify HMAC signature
        
        Args:
            data: Data to verify
            signature: HMAC signature as hex string
            
        Returns:
            Whether signature is valid
        """
        h = hmac.new(self.fernet_key, data.encode(), hashlib.sha256)
        return hmac.compare_digest(h.hexdigest(), signature)
        
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Derive encryption key from password
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if None)
            
        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode())
        return key, salt

class AuditLogger:
    """Advanced audit logging system"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up audit logger
        self.audit_logger = logging.getLogger("audit")
        audit_handler = logging.FileHandler(
            os.path.join(log_dir, "audit.log")
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(formatter)
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
        
    def log_security_event(
        self,
        event_type: str,
        user_id: str,
        ip: str,
        details: Dict[str, Any]
    ):
        """Log security event
        
        Args:
            event_type: Type of security event
            user_id: User identifier
            ip: Client IP address
            details: Additional event details
        """
        self.audit_logger.info(
            f"Security Event - Type: {event_type}, User: {user_id}, "
            f"IP: {ip}, Details: {json.dumps(details)}"
        )
        
    def log_data_access(
        self,
        user_id: str,
        ip: str,
        resource: str,
        action: str,
        success: bool,
        details: Dict[str, Any]
    ):
        """Log data access event
        
        Args:
            user_id: User identifier
            ip: Client IP address
            resource: Accessed resource
            action: Action performed
            success: Whether access was successful
            details: Additional access details
        """
        self.audit_logger.info(
            f"Data Access - User: {user_id}, IP: {ip}, Resource: {resource}, "
            f"Action: {action}, Success: {success}, Details: {json.dumps(details)}"
        )

class SecurityEventDetector:
    """Advanced security event detection system"""
    
    def __init__(self):
        # Event patterns
        self.patterns = {
            'brute_force': {
                'window': 300,  # 5 minutes
                'threshold': 10  # Failed attempts
            },
            'suspicious_ip': {
                'window': 3600,  # 1 hour
                'threshold': 100  # Requests
            },
            'data_exfiltration': {
                'window': 3600,  # 1 hour
                'threshold': 1000000  # Bytes
            }
        }
        
        # Event tracking
        self.failed_logins = defaultdict(list)
        self.ip_requests = defaultdict(list)
        self.data_transfers = defaultdict(list)
        self._lock = threading.Lock()
        
    def check_brute_force(self, user_id: str, ip: str) -> bool:
        """Check for potential brute force attack
        
        Args:
            user_id: User identifier
            ip: Client IP address
            
        Returns:
            Whether brute force attack is detected
        """
        now = time.time()
        window = self.patterns['brute_force']['window']
        threshold = self.patterns['brute_force']['threshold']
        
        with self._lock:
            # Clean old attempts
            key = f"{user_id}_{ip}"
            self.failed_logins[key] = [
                t for t in self.failed_logins[key]
                if now - t < window
            ]
            
            # Check threshold
            return len(self.failed_logins[key]) >= threshold
            
    def record_failed_login(self, user_id: str, ip: str):
        """Record failed login attempt
        
        Args:
            user_id: User identifier
            ip: Client IP address
        """
        with self._lock:
            key = f"{user_id}_{ip}"
            self.failed_logins[key].append(time.time())
            
    def check_suspicious_ip(self, ip: str) -> bool:
        """Check for suspicious IP activity
        
        Args:
            ip: Client IP address
            
        Returns:
            Whether suspicious activity is detected
        """
        now = time.time()
        window = self.patterns['suspicious_ip']['window']
        threshold = self.patterns['suspicious_ip']['threshold']
        
        with self._lock:
            # Clean old requests
            self.ip_requests[ip] = [
                t for t in self.ip_requests[ip]
                if now - t < window
            ]
            
            # Check threshold
            return len(self.ip_requests[ip]) >= threshold
            
    def record_ip_request(self, ip: str):
        """Record IP request
        
        Args:
            ip: Client IP address
        """
        with self._lock:
            self.ip_requests[ip].append(time.time())
            
    def check_data_exfiltration(
        self,
        user_id: str,
        bytes_transferred: int
    ) -> bool:
        """Check for potential data exfiltration
        
        Args:
            user_id: User identifier
            bytes_transferred: Number of bytes transferred
            
        Returns:
            Whether potential data exfiltration is detected
        """
        now = time.time()
        window = self.patterns['data_exfiltration']['window']
        threshold = self.patterns['data_exfiltration']['threshold']
        
        with self._lock:
            # Clean old transfers
            self.data_transfers[user_id] = [
                (t, b) for t, b in self.data_transfers[user_id]
                if now - t < window
            ]
            
            # Calculate total bytes
            total_bytes = sum(b for _, b in self.data_transfers[user_id])
            
            # Check threshold
            return total_bytes >= threshold
            
    def record_data_transfer(self, user_id: str, bytes_transferred: int):
        """Record data transfer
        
        Args:
            user_id: User identifier
            bytes_transferred: Number of bytes transferred
        """
        with self._lock:
            self.data_transfers[user_id].append(
                (time.time(), bytes_transferred)
            )

class ThreatIntelligence:
    """Advanced threat intelligence system"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize threat data
        self.known_threats = {
            'ips': set(),
            'user_agents': set(),
            'domains': set()
        }
        self.threat_scores = defaultdict(float)
        self._lock = threading.Lock()
        
        # Load threat data
        self._load_threat_data()
        
    def _load_threat_data(self):
        """Load threat data from files"""
        for category in self.known_threats:
            file_path = os.path.join(self.data_dir, f"{category}.txt")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.known_threats[category].update(
                        line.strip() for line in f
                    )
                    
    def _save_threat_data(self):
        """Save threat data to files"""
        for category, threats in self.known_threats.items():
            file_path = os.path.join(self.data_dir, f"{category}.txt")
            with open(file_path, 'w') as f:
                for threat in sorted(threats):
                    f.write(f"{threat}\n")
                    
    def check_ip_threat(self, ip: str) -> float:
        """Check IP threat score
        
        Args:
            ip: IP address to check
            
        Returns:
            Threat score (0-1)
        """
        with self._lock:
            # Check if IP is known threat
            if ip in self.known_threats['ips']:
                return 1.0
                
            # Return threat score
            return self.threat_scores.get(ip, 0.0)
            
    def check_user_agent_threat(self, user_agent: str) -> float:
        """Check User-Agent threat score
        
        Args:
            user_agent: User-Agent string to check
            
        Returns:
            Threat score (0-1)
        """
        with self._lock:
            # Check if User-Agent is known threat
            if user_agent in self.known_threats['user_agents']:
                return 1.0
                
            # Check for suspicious patterns
            suspicious_patterns = [
                'curl',
                'python',
                'wget',
                'bot',
                'scanner'
            ]
            
            score = 0.0
            user_agent_lower = user_agent.lower()
            
            for pattern in suspicious_patterns:
                if pattern in user_agent_lower:
                    score += 0.2
                    
            return min(score, 1.0)
            
    def check_domain_threat(self, domain: str) -> float:
        """Check domain threat score
        
        Args:
            domain: Domain to check
            
        Returns:
            Threat score (0-1)
        """
        with self._lock:
            # Check if domain is known threat
            if domain in self.known_threats['domains']:
                return 1.0
                
            # Return threat score
            return self.threat_scores.get(domain, 0.0)
            
    def update_threat_score(
        self,
        indicator: str,
        category: str,
        score: float
    ):
        """Update threat score for indicator
        
        Args:
            indicator: Threat indicator
            category: Indicator category
            score: New threat score
        """
        with self._lock:
            # Update score
            self.threat_scores[indicator] = score
            
            # Add to known threats if score is high
            if score >= 0.8:
                self.known_threats[category].add(indicator)
                self._save_threat_data()

class SecurityMetrics:
    """Advanced security metrics and analytics"""
    
    def __init__(self):
        # Initialize metrics
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'values': []
        })
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float):
        """Record security metric
        
        Args:
            name: Metric name
            value: Metric value
        """
        with self._lock:
            metric = self.metrics[name]
            metric['count'] += 1
            metric['sum'] += value
            metric['min'] = min(metric['min'], value)
            metric['max'] = max(metric['max'], value)
            metric['values'].append(value)
            
            # Keep only last 1000 values
            if len(metric['values']) > 1000:
                metric['values'] = metric['values'][-1000:]
                
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for metric
        
        Args:
            name: Metric name
            
        Returns:
            Dictionary of metric statistics
        """
        with self._lock:
            metric = self.metrics[name]
            
            if metric['count'] == 0:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'stddev': 0.0
                }
                
            values = metric['values']
            mean = metric['sum'] / metric['count']
            
            # Calculate standard deviation
            squared_diff_sum = sum(
                (x - mean) ** 2 for x in values
            )
            stddev = (
                (squared_diff_sum / len(values)) ** 0.5
                if len(values) > 1 else 0.0
            )
            
            return {
                'count': metric['count'],
                'mean': mean,
                'min': metric['min'],
                'max': metric['max'],
                'stddev': stddev
            }
            
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics
        
        Returns:
            Dictionary of metric names to statistics
        """
        return {
            name: self.get_metric_stats(name)
            for name in self.metrics
        }

class SecurityManager:
    """Main security manager"""
    
    def __init__(
        self,
        data_dir: str,
        blocklist_path: Optional[str] = None
    ):
        """Initialize security manager
        
        Args:
            data_dir: Directory to store security data
            blocklist_path: Path to IP blocklist file
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize components
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.ip_blocklist = IPBlocklist()
        self.encryption_manager = EncryptionManager(
            os.path.join(data_dir, 'keys')
        )
        self.audit_logger = AuditLogger(
            os.path.join(data_dir, 'audit')
        )
        self.event_detector = SecurityEventDetector()
        self.threat_intelligence = ThreatIntelligence(
            os.path.join(data_dir, 'threats')
        )
        self.security_metrics = SecurityMetrics()
        
        # Load IP blocklist
        if blocklist_path:
            self.ip_blocklist.load_blocklist(blocklist_path)
            
        # Suspicious activity tracking
        self.suspicious_activity = defaultdict(int)
        self.blocked_users = set()
        self._lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger("security")
        
    def validate_request(
        self,
        ip: str,
        user_id: str,
        endpoint: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, str]:
        """Validate API request with advanced security checks
        
        Args:
            ip: Client IP address
            user_id: User identifier
            endpoint: API endpoint
            data: Request data
            headers: Request headers
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Record metrics
        request_size = len(json.dumps(data))
        self.security_metrics.record_metric('request_size', request_size)
        
        # Check threat intelligence
        if headers:
            user_agent = headers.get('User-Agent', '')
            ip_threat = self.threat_intelligence.check_ip_threat(ip)
            ua_threat = self.threat_intelligence.check_user_agent_threat(user_agent)
            
            if ip_threat > 0.8 or ua_threat > 0.8:
                self.audit_logger.log_security_event(
                    'threat_detected',
                    user_id,
                    ip,
                    {
                        'ip_threat': ip_threat,
                        'ua_threat': ua_threat,
                        'user_agent': user_agent
                    }
                )
                return False, "Request blocked by threat detection"
                
        # Check security events
        if self.event_detector.check_brute_force(user_id, ip):
            self.audit_logger.log_security_event(
                'brute_force_detected',
                user_id,
                ip,
                {'endpoint': endpoint}
            )
            return False, "Too many failed attempts"
            
        if self.event_detector.check_suspicious_ip(ip):
            self.audit_logger.log_security_event(
                'suspicious_ip_activity',
                user_id,
                ip,
                {'endpoint': endpoint}
            )
            return False, "Suspicious IP activity detected"
            
        # Check IP blocklist
        if self.ip_blocklist.is_blocked(ip):
            self.audit_logger.log_security_event(
                'blocked_ip',
                user_id,
                ip,
                {'endpoint': endpoint}
            )
            return False, "IP address is blocked"
            
        # Check user blocklist
        if user_id in self.blocked_users:
            self.audit_logger.log_security_event(
                'blocked_user',
                user_id,
                ip,
                {'endpoint': endpoint}
            )
            return False, "User is blocked"
            
        # Check rate limits
        is_allowed, reason = self.rate_limiter.is_allowed(ip, user_id, endpoint)
        if not is_allowed:
            self.audit_logger.log_security_event(
                'rate_limit_exceeded',
                user_id,
                ip,
                {
                    'endpoint': endpoint,
                    'reason': reason
                }
            )
            return False, reason
            
        # Record IP request
        self.event_detector.record_ip_request(ip)
        
        # Validate input data
        if 'text' in data:
            is_valid, reason = self.input_validator.validate_text(data['text'])
            if not is_valid:
                self._record_suspicious_activity(ip, user_id)
                self.audit_logger.log_security_event(
                    'invalid_input',
                    user_id,
                    ip,
                    {
                        'endpoint': endpoint,
                        'reason': reason
                    }
                )
                return False, reason
                
        if 'texts' in data:
            is_valid, reason = self.input_validator.validate_batch(data['texts'])
            if not is_valid:
                self._record_suspicious_activity(ip, user_id)
                self.audit_logger.log_security_event(
                    'invalid_batch_input',
                    user_id,
                    ip,
                    {
                        'endpoint': endpoint,
                        'reason': reason
                    }
                )
                return False, reason
                
        # Log successful validation
        self.audit_logger.log_security_event(
            'request_validated',
            user_id,
            ip,
            {
                'endpoint': endpoint,
                'request_size': request_size
            }
        )
        
        return True, ""
        
    def _record_suspicious_activity(self, ip: str, user_id: str):
        """Record suspicious activity and potentially block user/IP
        
        Args:
            ip: Client IP address
            user_id: User identifier
        """
        with self._lock:
            # Increment counters
            self.suspicious_activity[ip] += 1
            self.suspicious_activity[user_id] += 1
            
            # Check thresholds
            if self.suspicious_activity[ip] >= 5:
                self.ip_blocklist.block_ip(ip)
                self.logger.warning(f"Blocked IP due to suspicious activity: {ip}")
                
            if self.suspicious_activity[user_id] >= 10:
                self.blocked_users.add(user_id)
                self.logger.warning(f"Blocked user due to suspicious activity: {user_id}")
                
    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data
        
        Args:
            data: Input data
            
        Returns:
            Sanitized data
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.input_validator.sanitize_text(value)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                sanitized[key] = [
                    self.input_validator.sanitize_text(item)
                    for item in value
                ]
            else:
                sanitized[key] = value
                
        return sanitized
        
    def encrypt_sensitive_data(
        self,
        data: Dict[str, Any],
        sensitive_fields: Set[str]
    ) -> Dict[str, Any]:
        """Encrypt sensitive fields in data
        
        Args:
            data: Data to encrypt
            sensitive_fields: Set of field names to encrypt
            
        Returns:
            Data with encrypted fields
        """
        encrypted = dict(data)
        
        for field in sensitive_fields:
            if field in encrypted and isinstance(encrypted[field], str):
                encrypted[field] = self.encryption_manager.encrypt_symmetric(
                    encrypted[field]
                )
                
        return encrypted
        
    def decrypt_sensitive_data(
        self,
        data: Dict[str, Any],
        encrypted_fields: Set[str]
    ) -> Dict[str, Any]:
        """Decrypt encrypted fields in data
        
        Args:
            data: Data with encrypted fields
            encrypted_fields: Set of field names to decrypt
            
        Returns:
            Data with decrypted fields
        """
        decrypted = dict(data)
        
        for field in encrypted_fields:
            if field in decrypted and isinstance(decrypted[field], str):
                try:
                    decrypted[field] = self.encryption_manager.decrypt_symmetric(
                        decrypted[field]
                    )
                except Exception as e:
                    self.logger.error(f"Error decrypting field {field}: {e}")
                    
        return decrypted
        
    def sign_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sign data with HMAC
        
        Args:
            data: Data to sign
            
        Returns:
            Data with signature
        """
        # Convert data to JSON string
        data_json = json.dumps(data, sort_keys=True)
        
        # Generate signature
        signature = self.encryption_manager.generate_hmac(data_json)
        
        # Add signature to data
        signed_data = dict(data)
        signed_data['_signature'] = signature
        
        return signed_data
        
    def verify_signature(self, data: Dict[str, Any]) -> bool:
        """Verify data signature
        
        Args:
            data: Data with signature
            
        Returns:
            Whether signature is valid
        """
        if '_signature' not in data:
            return False
            
        # Extract signature
        signature = data['_signature']
        data_without_signature = {
            k: v for k, v in data.items() if k != '_signature'
        }
        
        # Convert data to JSON string
        data_json = json.dumps(data_without_signature, sort_keys=True)
        
        # Verify signature
        return self.encryption_manager.verify_hmac(data_json, signature)
        
    def generate_secure_token(self, user_id: str, expiry: timedelta) -> str:
        """Generate secure token for user
        
        Args:
            user_id: User identifier
            expiry: Token expiry time
            
        Returns:
            Secure token
        """
        # Create token data
        token_data = {
            'user_id': user_id,
            'exp': int((datetime.now() + expiry).timestamp()),
            'jti': str(uuid.uuid4())
        }
        
        # Convert to JSON and encrypt
        token_json = json.dumps(token_data)
        encrypted_token = self.encryption_manager.encrypt_symmetric(token_json)
        
        return encrypted_token
        
    def validate_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """Validate secure token
        
        Args:
            token: Secure token
            
        Returns:
            Tuple of (is_valid, user_id)
        """
        try:
            # Decrypt token
            token_json = self.encryption_manager.decrypt_symmetric(token)
            token_data = json.loads(token_json)
            
            # Check expiry
            if token_data['exp'] < datetime.now().timestamp():
                return False, None
                
            return True, token_data['user_id']
            
        except Exception as e:
            self.logger.error(f"Error validating token: {e}")
            return False, None