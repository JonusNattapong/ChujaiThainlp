"""
Advanced monitoring system for ThaiNLP
"""
from typing import Any, Dict, List, Optional, Union
import time
import logging
import threading
from datetime import datetime, timedelta
import json
import smtplib
from email.message import EmailMessage
from collections import defaultdict
import psutil
import socket

class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = logging.getLogger("system_monitor")
        
    def collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections())
        }
        
        with self._lock:
            for key, value in metrics.items():
                self.metrics[key].append({
                    'value': value,
                    'timestamp': datetime.now().isoformat()
                })
                
        return metrics
        
    def get_average_metrics(
        self,
        minutes: int = 5
    ) -> Dict[str, float]:
        """Get average metrics for last n minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        averages = {}
        with self._lock:
            for key, values in self.metrics.items():
                recent_values = [
                    v['value'] for v in values
                    if datetime.fromisoformat(v['timestamp']) > cutoff
                ]
                if recent_values:
                    averages[key] = sum(recent_values) / len(recent_values)
                    
        return averages

class PerformanceMonitor:
    """Monitor processing performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_processing(
        self,
        task: str,
        process_time: float,
        input_size: int,
        success: bool
    ):
        """Record processing metrics"""
        with self._lock:
            self.metrics[task].append({
                'process_time': process_time,
                'input_size': input_size,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
            
    def get_task_stats(
        self,
        task: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get statistics for task"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            records = [
                r for r in self.metrics[task]
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
            
        if not records:
            return {}
            
        success_rate = sum(1 for r in records if r['success']) / len(records)
        process_times = [r['process_time'] for r in records]
        input_sizes = [r['input_size'] for r in records]
        
        return {
            'total_requests': len(records),
            'success_rate': success_rate,
            'avg_process_time': sum(process_times) / len(process_times),
            'avg_input_size': sum(input_sizes) / len(input_sizes),
            'p95_process_time': sorted(process_times)[int(len(process_times) * 0.95)],
            'max_process_time': max(process_times)
        }

class AlertSystem:
    """System for monitoring and alerting"""
    
    def __init__(
        self,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.alerts = []
        self.email_config = email_config
        self._lock = threading.Lock()
        self.logger = logging.getLogger("alert_system")
        
        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'error_rate': 0.1,
            'response_time': 5.0  # seconds
        }
        
    def check_system_health(
        self,
        metrics: Dict[str, float]
    ) -> List[str]:
        """Check system metrics against thresholds"""
        alerts = []
        
        if metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics['cpu_percent']}%")
            
        if metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics['memory_percent']}%")
            
        return alerts
        
    def check_performance(
        self,
        stats: Dict[str, Any]
    ) -> List[str]:
        """Check performance metrics"""
        alerts = []
        
        if stats.get('success_rate', 1.0) < (1 - self.thresholds['error_rate']):
            alerts.append(
                f"High error rate: {(1 - stats['success_rate']) * 100}%"
            )
            
        if stats.get('avg_process_time', 0) > self.thresholds['response_time']:
            alerts.append(
                f"Slow response time: {stats['avg_process_time']:.2f}s"
            )
            
        return alerts
        
    def send_alert(self, message: str, severity: str = 'warning'):
        """Record and potentially send alert"""
        with self._lock:
            alert = {
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
        self.logger.warning(f"Alert: {message}")
        
        if self.email_config and severity == 'critical':
            self._send_email_alert(message)
            
    def _send_email_alert(self, message: str):
        """Send alert via email"""
        if not self.email_config:
            return
            
        try:
            msg = EmailMessage()
            msg.set_content(message)
            msg['Subject'] = f'ThaiNLP Alert: {message[:50]}...'
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            
            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                if self.email_config.get('use_tls'):
                    server.starttls()
                if 'username' in self.email_config:
                    server.login(
                        self.email_config['username'],
                        self.email_config['password']
                    )
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

class MonitoringSystem:
    """Complete monitoring system"""
    
    def __init__(
        self,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.system_monitor = SystemMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.alert_system = AlertSystem(email_config)
        
        # Start monitoring thread
        self.should_run = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.should_run:
            try:
                # Collect system metrics
                metrics = self.system_monitor.collect_metrics()
                
                # Check for system alerts
                alerts = self.alert_system.check_system_health(metrics)
                for alert in alerts:
                    self.alert_system.send_alert(alert)
                    
                # Sleep for a bit
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.alert_system.send_alert(
                    f"Monitoring error: {e}",
                    severity='critical'
                )
                
    def stop(self):
        """Stop monitoring"""
        self.should_run = False
        self.monitor_thread.join()
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'system_metrics': self.system_monitor.get_average_metrics(),
            'performance_metrics': {
                task: self.performance_monitor.get_task_stats(task)
                for task in self.performance_monitor.metrics.keys()
            },
            'recent_alerts': self.alert_system.alerts[-10:]
        }
        
    def export_metrics(
        self,
        format: str = 'json'
    ) -> str:
        """Export all metrics"""
        data = {
            'system': dict(self.system_monitor.metrics),
            'performance': dict(self.performance_monitor.metrics),
            'alerts': self.alert_system.alerts
        }
        
        if format == 'json':
            return json.dumps(data)
        raise ValueError(f"Unsupported format: {format}") 