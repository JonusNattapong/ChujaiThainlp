"""
Progress monitoring and resource tracking utilities
"""
from typing import Optional
import time
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class TaskStats:
    """Statistics for a monitored task"""
    start_time: float
    end_time: Optional[float] = None
    total_items: int = 0
    processed_items: int = 0
    
    @property
    def duration(self) -> float:
        """Get task duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Get processing rate"""
        if self.processed_items == 0:
            return 0.0
        return self.processed_items / self.duration
    
    @property
    def progress(self) -> float:
        """Get progress percentage"""
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items * 100

class ProgressTracker:
    """Track progress of processing tasks"""
    
    def __init__(self, show_progress_bar: bool = True):
        """Initialize progress tracker
        
        Args:
            show_progress_bar: Whether to show progress bar
        """
        self.show_progress_bar = show_progress_bar
        self.current_task: Optional[TaskStats] = None
        self.progress_bar: Optional[tqdm] = None
        self.history: list[TaskStats] = []
        
    def start_task(self, total_items: int, desc: Optional[str] = None):
        """Start tracking a new task
        
        Args:
            total_items: Total number of items to process
            desc: Task description for progress bar
        """
        self.current_task = TaskStats(
            start_time=time.time(),
            total_items=total_items
        )
        
        if self.show_progress_bar:
            self.progress_bar = tqdm(
                total=total_items,
                desc=desc or "Processing",
                unit="items"
            )
    
    def update(self, items: int = 1):
        """Update progress
        
        Args:
            items: Number of items processed
        """
        if self.current_task is None:
            return
            
        self.current_task.processed_items += items
        
        if self.progress_bar is not None:
            self.progress_bar.update(items)
            
    def end_task(self):
        """End current task"""
        if self.current_task is None:
            return
            
        self.current_task.end_time = time.time()
        self.history.append(self.current_task)
        
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
            
        self.current_task = None
        
    def get_stats(self) -> Optional[TaskStats]:
        """Get current task statistics"""
        return self.current_task
    
    def get_history(self) -> list[TaskStats]:
        """Get history of completed tasks"""
        return self.history
    
    def clear_history(self):
        """Clear task history"""
        self.history = []
        
    def __enter__(self):
        """Context manager enter"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end_task()

class ResourceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self):
        """Initialize resource monitor"""
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
            
    def get_memory_usage(self) -> dict:
        """Get current memory usage
        
        Returns:
            Dict with memory statistics:
            - total: Total system memory (GB)
            - available: Available memory (GB)
            - used: Used memory (GB)
            - percent: Memory usage percentage
        """
        if self.psutil is None:
            return {}
            
        mem = self.psutil.virtual_memory()
        return {
            'total': mem.total / (1024**3),  # Convert to GB
            'available': mem.available / (1024**3),
            'used': mem.used / (1024**3),
            'percent': mem.percent
        }
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if self.psutil is None:
            return 0.0
        return self.psutil.cpu_percent()
    
    def get_gpu_usage(self) -> dict:
        """Get GPU usage statistics if available
        
        Returns:
            Dict with GPU statistics or empty dict if no GPU
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
                
            return {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'memory_reserved': torch.cuda.memory_reserved() / (1024**3)  # GB
            }
        except ImportError:
            return {}
            
    def get_all_stats(self) -> dict:
        """Get all resource statistics"""
        return {
            'memory': self.get_memory_usage(),
            'cpu': self.get_cpu_usage(),
            'gpu': self.get_gpu_usage()
        }