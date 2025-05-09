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
import pylru as lru_replacement
import mmap
import re
import unicodedata

# Create a simple adapter if needed
class LruAdapter:
    @staticmethod
    def LRU(size):
        return lru_replacement.lrucache(size)

# Replace original lru with our adapter
lru = LruAdapter

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
    """Optimize text processing operations for Thai NLP"""
    
    def __init__(self, cache_size: int = 1000):
        """Initialize text processor
        
        Args:
            cache_size: Maximum size of preprocessing cache
        """
        self.cache_size = cache_size
        self._preprocess_cache = {}
    
    @functools.lru_cache(maxsize=5000)
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for efficient processing
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Early return for empty text
        if not text:
            return ""
            
        # Check cache
        if text in self._preprocess_cache:
            return self._preprocess_cache[text]
            
        # Basic preprocessing
        processed = text
        
        # Normalize Unicode
        processed = unicodedata.normalize('NFC', processed)
        
        # Replace multiple spaces with a single space
        processed = re.sub(r'\s+', ' ', processed)
        
        # Replace zero-width characters and certain control characters
        processed = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', processed)
        
        # Normalize Thai characters (full/half width)
        # Replace Thai numerals with Arabic numerals
        thai_digits = '๐๑๒๓๔๕๖๗๘๙'
        arabic_digits = '0123456789'
        for thai, arabic in zip(thai_digits, arabic_digits):
            processed = processed.replace(thai, arabic)
            
        # Update cache (with size limit)
        if len(self._preprocess_cache) >= self.cache_size:
            # Clear half of the cache when full (the oldest entries)
            remove_count = self.cache_size // 2
            for _ in range(remove_count):
                self._preprocess_cache.popitem(last=False)
                
        self._preprocess_cache[text] = processed
        
        return processed
        
    def normalize_thai_text(self, text: str) -> str:
        """Normalize Thai text for better processing
        
        Args:
            text: Thai text to normalize
            
        Returns:
            Normalized text
        """
        # Apply preprocessing
        text = self.preprocess_text(text)
        
        # Thai-specific normalization
        # Replace various Thai forms of the same character
        replacements = {
            # Sara E variations
            '\u0e40\u0e40': '\u0e40',  # Double Sara E
            # Sara Ai variations
            '\u0e44\u0e44': '\u0e44',  # Double Sara Ai
            # Tone mark normalization
            '\u0e4d\u0e48': '\u0e48\u0e4d',  # Reorder nikkhahit and tone
            '\u0e4d\u0e49': '\u0e49\u0e4d',
            '\u0e4d\u0e4a': '\u0e4a\u0e4d',
            '\u0e4d\u0e4b': '\u0e4b\u0e4d',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def extract_thai_text(self, text: str) -> str:
        """Extract only Thai script from mixed text
        
        Args:
            text: Mixed text
            
        Returns:
            Only Thai script parts
        """
        # Match Thai Unicode range (Thai consonants, vowels, marks, digits)
        thai_chars = re.findall(r'[\u0e00-\u0e7f]+', text)
        return ' '.join(thai_chars)
    
    def is_thai_char(self, char: str) -> bool:
        """Check if a character is Thai
        
        Args:
            char: Character to check
            
        Returns:
            True if the character is Thai
        """
        if not char:
            return False
        return '\u0e00' <= char <= '\u0e7f'
    
    def count_thai_chars(self, text: str) -> int:
        """Count Thai characters in text
        
        Args:
            text: Input text
            
        Returns:
            Number of Thai characters
        """
        return sum(1 for char in text if self.is_thai_char(char))
    
    def get_thai_script_ratio(self, text: str) -> float:
        """Calculate ratio of Thai script to total characters
        
        Args:
            text: Input text
            
        Returns:
            Ratio of Thai script (0.0-1.0)
        """
        if not text:
            return 0.0
            
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return 0.0
            
        thai_chars = self.count_thai_chars(text)
        return thai_chars / total_chars
    
    def segment_by_script(self, text: str) -> List[Dict[str, Any]]:
        """Segment text by script type
        
        Args:
            text: Mixed script text
            
        Returns:
            List of segments with script type and content
        """
        if not text:
            return []
            
        segments = []
        current_type = "other"
        current_segment = ""
        
        for char in text:
            if self.is_thai_char(char):
                char_type = "thai"
            elif char.isascii() and char.isalpha():
                char_type = "latin"
            elif char.isdigit() or char in "+-.,":
                char_type = "digit"
            elif char.isspace():
                char_type = current_type  # Spaces keep the current type
            else:
                char_type = "other"
                
            # Start a new segment if script type changed
            if char_type != current_type and current_segment:
                segments.append({
                    "type": current_type,
                    "text": current_segment
                })
                current_segment = char
                current_type = char_type
            else:
                current_segment += char
                if current_type == "other":
                    current_type = char_type
        
        # Add the final segment
        if current_segment:
            segments.append({
                "type": current_type,
                "text": current_segment
            })
            
        return segments

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