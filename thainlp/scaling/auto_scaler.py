"""
Advanced Auto-scaling and Load Balancing System for ThaiNLP
"""
from typing import Dict, List, Optional, Union, Callable
import time
import threading
import queue
import psutil
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

@dataclass
class ServiceMetrics:
    """Service performance metrics"""
    cpu_usage: float
    memory_usage: float
    request_count: int
    response_time: float
    error_rate: float
    timestamp: float

class LoadBalancer:
    """Advanced load balancer with multiple strategies"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.servers = []
        self.weights = {}
        self.current_index = 0
        self.server_metrics = {}
        self._lock = threading.Lock()
        
    def add_server(self, server_id: str, weight: float = 1.0):
        """Add server to pool
        
        Args:
            server_id: Server identifier
            weight: Server weight for weighted algorithms
        """
        with self._lock:
            self.servers.append(server_id)
            self.weights[server_id] = weight
            self.server_metrics[server_id] = ServiceMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                request_count=0,
                response_time=0.0,
                error_rate=0.0,
                timestamp=time.time()
            )
            
    def remove_server(self, server_id: str):
        """Remove server from pool
        
        Args:
            server_id: Server identifier
        """
        with self._lock:
            if server_id in self.servers:
                self.servers.remove(server_id)
                del self.weights[server_id]
                del self.server_metrics[server_id]
                
    def update_metrics(self, server_id: str, metrics: ServiceMetrics):
        """Update server metrics
        
        Args:
            server_id: Server identifier
            metrics: New metrics
        """
        with self._lock:
            self.server_metrics[server_id] = metrics
            
    def get_next_server(self) -> Optional[str]:
        """Get next server based on strategy
        
        Returns:
            Server identifier
        """
        with self._lock:
            if not self.servers:
                return None
                
            if self.strategy == "round_robin":
                server = self.servers[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.servers)
                return server
                
            elif self.strategy == "weighted_round_robin":
                weights = [self.weights[s] for s in self.servers]
                total_weight = sum(weights)
                r = np.random.random() * total_weight
                for i, server in enumerate(self.servers):
                    r -= weights[i]
                    if r <= 0:
                        return server
                        
            elif self.strategy == "least_connections":
                return min(
                    self.servers,
                    key=lambda s: self.server_metrics[s].request_count
                )
                
            elif self.strategy == "least_response_time":
                return min(
                    self.servers,
                    key=lambda s: self.server_metrics[s].response_time
                )
                
            return self.servers[0]

class AutoScaler:
    """Advanced auto-scaling system"""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 20.0,
        cooldown_period: int = 300,
        metric_window: int = 60
    ):
        """Initialize auto-scaler
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            scale_up_threshold: CPU/Memory threshold to scale up (%)
            scale_down_threshold: CPU/Memory threshold to scale down (%)
            cooldown_period: Cooldown period between scaling actions (seconds)
            metric_window: Window for metric averaging (seconds)
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.metric_window = metric_window
        
        self.instances = {}
        self.metrics_history = {}
        self.last_scale_time = 0
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
    def add_instance(self, instance_id: str):
        """Add instance to pool
        
        Args:
            instance_id: Instance identifier
        """
        with self._lock:
            self.instances[instance_id] = {
                'status': 'running',
                'start_time': time.time()
            }
            self.metrics_history[instance_id] = []
            
    def remove_instance(self, instance_id: str):
        """Remove instance from pool
        
        Args:
            instance_id: Instance identifier
        """
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                del self.metrics_history[instance_id]
                
    def update_metrics(self, instance_id: str, metrics: ServiceMetrics):
        """Update instance metrics
        
        Args:
            instance_id: Instance identifier
            metrics: New metrics
        """
        with self._lock:
            if instance_id in self.metrics_history:
                self.metrics_history[instance_id].append(metrics)
                
                # Keep only recent metrics
                cutoff_time = time.time() - self.metric_window
                self.metrics_history[instance_id] = [
                    m for m in self.metrics_history[instance_id]
                    if m.timestamp > cutoff_time
                ]
                
    def _get_average_metrics(self) -> ServiceMetrics:
        """Calculate average metrics across all instances
        
        Returns:
            Average metrics
        """
        with self._lock:
            if not self.instances:
                return ServiceMetrics(0.0, 0.0, 0, 0.0, 0.0, time.time())
                
            total_metrics = ServiceMetrics(0.0, 0.0, 0, 0.0, 0.0, time.time())
            instance_count = len(self.instances)
            
            for instance_id in self.instances:
                metrics = self.metrics_history.get(instance_id, [])
                if metrics:
                    latest = metrics[-1]
                    total_metrics.cpu_usage += latest.cpu_usage
                    total_metrics.memory_usage += latest.memory_usage
                    total_metrics.request_count += latest.request_count
                    total_metrics.response_time += latest.response_time
                    total_metrics.error_rate += latest.error_rate
                    
            return ServiceMetrics(
                cpu_usage=total_metrics.cpu_usage / instance_count,
                memory_usage=total_metrics.memory_usage / instance_count,
                request_count=total_metrics.request_count,
                response_time=total_metrics.response_time / instance_count,
                error_rate=total_metrics.error_rate / instance_count,
                timestamp=time.time()
            )
            
    def _should_scale(self) -> Optional[str]:
        """Determine if scaling is needed
        
        Returns:
            'up', 'down', or None
        """
        if time.time() - self.last_scale_time < self.cooldown_period:
            return None
            
        metrics = self._get_average_metrics()
        instance_count = len(self.instances)
        
        if (metrics.cpu_usage > self.scale_up_threshold or 
            metrics.memory_usage > self.scale_up_threshold) and \
           instance_count < self.max_instances:
            return 'up'
            
        if (metrics.cpu_usage < self.scale_down_threshold and
            metrics.memory_usage < self.scale_down_threshold) and \
           instance_count > self.min_instances:
            return 'down'
            
        return None
        
    def _monitor_loop(self):
        """Background monitoring thread"""
        while True:
            action = self._should_scale()
            
            if action == 'up':
                self.scale_up()
            elif action == 'down':
                self.scale_down()
                
            time.sleep(10)  # Check every 10 seconds
            
    def scale_up(self):
        """Scale up by adding new instance"""
        with self._lock:
            if len(self.instances) >= self.max_instances:
                return
                
            instance_id = f"instance_{len(self.instances) + 1}"
            self.add_instance(instance_id)
            self.last_scale_time = time.time()
            
    def scale_down(self):
        """Scale down by removing least utilized instance"""
        with self._lock:
            if len(self.instances) <= self.min_instances:
                return
                
            # Find instance with lowest utilization
            min_usage = float('inf')
            instance_to_remove = None
            
            for instance_id in self.instances:
                metrics = self.metrics_history.get(instance_id, [])
                if metrics:
                    usage = metrics[-1].cpu_usage + metrics[-1].memory_usage
                    if usage < min_usage:
                        min_usage = usage
                        instance_to_remove = instance_id
                        
            if instance_to_remove:
                self.remove_instance(instance_to_remove)
                self.last_scale_time = time.time()

class ResourceManager:
    """Manage compute resources and scaling"""
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        auto_scaler: AutoScaler,
        worker_type: str = "thread"
    ):
        """Initialize resource manager
        
        Args:
            load_balancer: Load balancer instance
            auto_scaler: Auto-scaler instance
            worker_type: Type of workers ('thread' or 'process')
        """
        self.load_balancer = load_balancer
        self.auto_scaler = auto_scaler
        self.worker_type = worker_type
        
        # Initialize worker pool
        if worker_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=auto_scaler.max_instances)
        else:
            self.executor = ProcessPoolExecutor(max_workers=auto_scaler.max_instances)
            
        # Task queue
        self.task_queue = queue.PriorityQueue()
        
        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._process_tasks,
            daemon=True
        )
        self._worker_thread.start()
        
    def submit_task(
        self,
        task: Callable,
        priority: int = 1,
        *args,
        **kwargs
    ):
        """Submit task for processing
        
        Args:
            task: Task function to execute
            priority: Task priority (lower is higher priority)
            *args: Task arguments
            **kwargs: Task keyword arguments
        """
        self.task_queue.put((priority, (task, args, kwargs)))
        
    def _process_tasks(self):
        """Background task processing thread"""
        while True:
            try:
                priority, (task, args, kwargs) = self.task_queue.get()
                
                # Get server to handle task
                server_id = self.load_balancer.get_next_server()
                if not server_id:
                    continue
                    
                # Submit task to worker pool
                start_time = time.time()
                future = self.executor.submit(task, *args, **kwargs)
                
                def callback(future):
                    # Update metrics
                    duration = time.time() - start_time
                    error = future.exception() is not None
                    
                    metrics = ServiceMetrics(
                        cpu_usage=psutil.cpu_percent(),
                        memory_usage=psutil.virtual_memory().percent,
                        request_count=1,
                        response_time=duration,
                        error_rate=1.0 if error else 0.0,
                        timestamp=time.time()
                    )
                    
                    self.load_balancer.update_metrics(server_id, metrics)
                    self.auto_scaler.update_metrics(server_id, metrics)
                    
                future.add_done_callback(callback)
                
            except Exception as e:
                print(f"Error processing task: {e}")
                
            time.sleep(0.1)  # Prevent busy waiting 