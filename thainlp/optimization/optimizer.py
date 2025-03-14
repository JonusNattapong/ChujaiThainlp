"""
Advanced optimization system for ThaiNLP
"""
from typing import Any, Dict, List, Optional, Union, Callable
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import queue
import functools
import pickle
import lru
import mmap

class MemoryOptimizer:
    """Optimize memory usage"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = lru.LRU(max_cache_size)
        self._lock = threading.Lock()
        
    def memoize(self, func: Callable) -> Callable:
        """Decorator for memoizing function results"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = pickle.dumps((args, kwargs))
            
            with self._lock:
                if key in self.cache:
                    return self.cache[key]
                    
                result = func(*args, **kwargs)
                self.cache[key] = result
                return result
                
        return wrapper
        
    def clear_cache(self):
        """Clear memoization cache"""
        with self._lock:
            self.cache.clear()

class ParallelProcessor:
    """Handle parallel processing"""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False
    ):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.use_processes = use_processes
        self._executor = None
        
    def __enter__(self):
        """Context manager entry"""
        ExecutorClass = (
            ProcessPoolExecutor if self.use_processes
            else ThreadPoolExecutor
        )
        self._executor = ExecutorClass(max_workers=self.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._executor:
            self._executor.shutdown()
            
    def map(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Process items in parallel"""
        if not items:
            return []
            
        if not chunk_size:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
            
        return list(
            self._executor.map(func, items, chunksize=chunk_size)
        )
        
    def process_batches(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int
    ) -> List[Any]:
        """Process items in batches"""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            results.extend(self.map(func, batch))
        return results

class DiskCache:
    """Efficient disk-based caching"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path for key"""
        import os
        import hashlib
        
        # Create safe filename from key
        filename = hashlib.sha256(
            key.encode()
        ).hexdigest()
        return os.path.join(self.cache_dir, filename)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            return None
            
    def set(self, key: str, value: Any):
        """Set value in cache"""
        cache_path = self._get_cache_path(key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)
            
    def clear(self):
        """Clear all cached items"""
        import os
        import shutil
        
        shutil.rmtree(self.cache_dir)
        self._ensure_cache_dir()

class BatchProcessor:
    """Process items in optimized batches"""
    
    def __init__(
        self,
        batch_size: int = 1000,
        max_queue_size: int = 10000
    ):
        self.batch_size = batch_size
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue()
        self.processor_thread = None
        self.should_run = False
        
    def start(self, process_func: Callable):
        """Start batch processor"""
        self.should_run = True
        self.processor_thread = threading.Thread(
            target=self._process_loop,
            args=(process_func,),
            daemon=True
        )
        self.processor_thread.start()
        
    def stop(self):
        """Stop batch processor"""
        self.should_run = False
        if self.processor_thread:
            self.processor_thread.join()
            
    def _process_loop(self, process_func: Callable):
        """Main processing loop"""
        current_batch = []
        
        while self.should_run:
            try:
                # Try to get item with timeout
                item = self.input_queue.get(timeout=0.1)
                current_batch.append(item)
                
                # Process batch if full
                if len(current_batch) >= self.batch_size:
                    results = process_func(current_batch)
                    for result in results:
                        self.output_queue.put(result)
                    current_batch = []
                    
            except queue.Empty:
                # Process remaining items if any
                if current_batch:
                    results = process_func(current_batch)
                    for result in results:
                        self.output_queue.put(result)
                    current_batch = []
                    
    def process(self, items: List[Any]) -> List[Any]:
        """Process items through batch processor"""
        # Add items to input queue
        for item in items:
            self.input_queue.put(item)
            
        # Get results
        results = []
        for _ in range(len(items)):
            results.append(self.output_queue.get())
            
        return results

class TextProcessor:
    """Optimized text processing"""
    
    def __init__(self):
        self.word_cache = {}
        self._lock = threading.Lock()
        
    def preprocess_text(self, text: str) -> str:
        """Optimize text for processing"""
        # Convert to lowercase for case-insensitive operations
        text = text.lower()
        
        # Remove redundant whitespace
        text = ' '.join(text.split())
        
        return text
        
    @functools.lru_cache(maxsize=10000)
    def get_word_info(self, word: str) -> Dict[str, Any]:
        """Get cached word information"""
        # This would normally do more complex processing
        return {
            'length': len(word),
            'is_thai': all('\u0E00' <= c <= '\u0E7F' for c in word)
        }
        
    def process_text_file(
        self,
        file_path: str,
        process_func: Callable
    ) -> List[Any]:
        """Process large text file efficiently"""
        results = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Memory map the file for efficient reading
            with mmap.mmap(
                f.fileno(),
                0,
                access=mmap.ACCESS_READ
            ) as mm:
                # Process file in chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                offset = 0
                
                while offset < mm.size():
                    # Read chunk
                    mm.seek(offset)
                    chunk = mm.read(chunk_size).decode('utf-8')
                    
                    # Process chunk
                    chunk_results = process_func(chunk)
                    results.extend(chunk_results)
                    
                    # Update offset
                    offset += chunk_size
                    
        return results

class Optimizer:
    """Main optimization system"""
    
    def __init__(
        self,
        cache_dir: str,
        max_workers: Optional[int] = None
    ):
        self.memory_optimizer = MemoryOptimizer()
        self.disk_cache = DiskCache(cache_dir)
        self.parallel_processor = ParallelProcessor(max_workers)
        self.batch_processor = BatchProcessor()
        self.text_processor = TextProcessor()
        
    def optimize_function(
        self,
        func: Callable,
        use_cache: bool = True
    ) -> Callable:
        """Optimize function execution"""
        if use_cache:
            func = self.memory_optimizer.memoize(func)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        return wrapper
        
    def process_batch(
        self,
        items: List[Any],
        func: Callable,
        batch_size: int = 1000
    ) -> List[Any]:
        """Process items in optimized batches"""
        with self.parallel_processor:
            return self.parallel_processor.process_batches(
                func,
                items,
                batch_size
            )
            
    def process_text(
        self,
        text: str,
        func: Callable
    ) -> Any:
        """Process text with optimizations"""
        # Preprocess
        text = self.text_processor.preprocess_text(text)
        
        # Try cache
        cache_key = f"text_process_{hash(text)}_{func.__name__}"
        if cached := self.disk_cache.get(cache_key):
            return cached
            
        # Process
        result = func(text)
        
        # Cache result
        self.disk_cache.set(cache_key, result)
        
        return result 