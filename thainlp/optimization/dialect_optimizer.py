"""
Dialect Processing Optimization

This module provides high-performance processing for Thai dialect operations,
enabling efficient batch processing, caching, and parallel operations.
"""

import os
import time
import functools
import threading
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from pathlib import Path

from ..dialects.dialect_processor import ThaiDialectProcessor
from ..utils.thai_utils import normalize_text, clean_thai_text

class DialectOptimizer:
    """Optimize performance for Thai dialect processing"""
    
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        max_workers: int = None,
        cache_size: int = 10000
    ):
        """Initialize the dialect optimizer
        
        Args:
            cache_dir: Directory for persistence cache
            max_workers: Maximum number of parallel workers (None for auto)
            cache_size: Size of in-memory cache
        """
        # Initialize cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.expanduser("~")) / ".thainlp" / "cache" / "dialect"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Calculate optimal workers based on available CPUs
        if max_workers is None:
            import multiprocessing
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.max_workers = max_workers
            
        # Initialize processing pool
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = None  # Initialize on demand for heavier tasks
        
        # Initialize dialect processor
        self.dialect_processor = ThaiDialectProcessor(cache_size=cache_size)
        
        # Thread lock for cache access
        self._lock = threading.Lock()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread_executor.shutdown(wait=False)
        if self.process_executor:
            self.process_executor.shutdown(wait=False)
    
    def _get_process_executor(self):
        """Get or create process executor for CPU-intensive tasks"""
        if self.process_executor is None:
            self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self.process_executor
        
    def batch_detect_dialect(
        self,
        texts: List[str],
        threshold: float = 0.1,
        use_ml: bool = True,
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Detect Thai dialect for multiple texts in parallel
        
        Args:
            texts: List of Thai text strings
            threshold: Minimum confidence threshold
            use_ml: Whether to use ML-based detection
            batch_size: Size of batches for processing
            
        Returns:
            List of dialect detection results
        """
        if not texts:
            return []
        
        # Preprocess all texts
        normalized_texts = [normalize_text(text) for text in texts]
        
        # Process in batches for better memory management
        all_results = []
        for i in range(0, len(normalized_texts), batch_size):
            batch = normalized_texts[i:i+batch_size]
            
            # Process batch in parallel
            futures = [
                self.thread_executor.submit(
                    self.dialect_processor.detect_dialect, 
                    text, 
                    threshold,
                    use_ml
                ) for text in batch
            ]
            
            # Collect results in order
            batch_results = [future.result() for future in futures]
            all_results.extend(batch_results)
            
        return all_results
    
    def batch_translate(
        self,
        texts: List[str],
        source_dialect: str,
        target_dialect: str = "central",
        batch_size: int = 50
    ) -> List[str]:
        """Translate multiple texts between dialects in parallel
        
        Args:
            texts: List of texts to translate
            source_dialect: Source dialect code
            target_dialect: Target dialect code (default: central Thai)
            batch_size: Size of batches for processing
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
            
        # Determine translation function based on source and target
        if source_dialect == "central":
            translate_func = functools.partial(
                self.dialect_processor.translate_from_standard,
                target_dialect=target_dialect
            )
        elif target_dialect == "central":
            translate_func = functools.partial(
                self.dialect_processor.translate_to_standard,
                source_dialect=source_dialect
            )
        else:
            # For dialect-to-dialect, we'll translate via standard Thai
            def translate_func(text):
                standard = self.dialect_processor.translate_to_standard(text, source_dialect)
                return self.dialect_processor.translate_from_standard(standard, target_dialect)
                
        # Process in batches
        all_translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Process batch in parallel
            futures = [
                self.thread_executor.submit(translate_func, text)
                for text in batch
            ]
            
            # Collect results in order
            batch_results = [future.result() for future in futures]
            all_translations.extend(batch_results)
            
        return all_translations
                
    def batch_detect_regional_dialect(
        self,
        texts: List[str],
        primary_dialect: Optional[str] = None,
        batch_size: int = 50
    ) -> List[Dict[str, float]]:
        """Detect regional dialect variations for multiple texts in parallel
        
        Args:
            texts: List of texts to analyze
            primary_dialect: Primary dialect (if None, will detect)
            batch_size: Size of batches for processing
            
        Returns:
            List of regional dialect detection results
        """
        if not texts:
            return []
            
        # If primary_dialect is None, detect dialects first
        if primary_dialect is None:
            # First detect primary dialects
            dialect_results = self.batch_detect_dialect(texts)
            
            # Extract primary dialect for each text
            primary_dialects = [
                max(result.items(), key=lambda x: x[1])[0]
                for result in dialect_results
            ]
        else:
            # Use the same primary dialect for all texts
            primary_dialects = [primary_dialect] * len(texts)
            
        # Now detect regional dialects for each text with its primary dialect
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_dialects = primary_dialects[i:i+batch_size]
            
            # Process batch in parallel
            futures = [
                self.thread_executor.submit(
                    self.dialect_processor.detect_regional_dialect,
                    text, dialect
                )
                for text, dialect in zip(batch_texts, batch_dialects)
            ]
            
            # Collect results in order
            batch_results = [future.result() for future in futures]
            all_results.extend(batch_results)
            
        return all_results
                
    def process_large_file(
        self,
        file_path: str,
        process_func: Callable[[str], Any],
        encoding: str = 'utf-8',
        chunk_size: int = 1024 * 1024,  # 1MB chunks
        by_line: bool = True
    ) -> List[Any]:
        """Process a large text file with dialect operations efficiently
        
        Args:
            file_path: Path to the text file
            process_func: Function to apply to each chunk or line
            encoding: File encoding
            chunk_size: Size of chunks to process at once
            by_line: Whether to process line by line
            
        Returns:
            List of processing results
        """
        results = []
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                if by_line:
                    # Process line by line
                    batch = []
                    batch_size = 100  # Process 100 lines at a time
                    
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            batch.append(line)
                            
                        if len(batch) >= batch_size:
                            # Process batch
                            batch_results = []
                            futures = [
                                self.thread_executor.submit(process_func, text)
                                for text in batch
                            ]
                            batch_results = [future.result() for future in futures]
                            results.extend(batch_results)
                            batch = []
                    
                    # Process remaining lines
                    if batch:
                        futures = [
                            self.thread_executor.submit(process_func, text)
                            for text in batch
                        ]
                        batch_results = [future.result() for future in futures]
                        results.extend(batch_results)
                else:
                    # Process by chunks
                    # Memory-map the file for efficient reading
                    import mmap
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    
                    offset = 0
                    while offset < mm.size():
                        # Read chunk
                        mm.seek(offset)
                        chunk = mm.read(chunk_size).decode(encoding)
                        
                        # Process chunk
                        chunk_result = process_func(chunk)
                        results.append(chunk_result)
                        
                        # Update offset
                        offset += chunk_size
                    
                    mm.close()
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
        return results
    
    def analyze_dialect_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze dialect distribution in a corpus
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with dialect distribution statistics
        """
        # Detect dialects for all texts
        dialect_results = self.batch_detect_dialect(texts)
        
        # Calculate dialect distribution
        dialect_counts = {dialect: 0 for dialect in ["northern", "northeastern", "southern", "central", "pattani_malay"]}
        dialect_confidence = {dialect: [] for dialect in dialect_counts.keys()}
        
        for result in dialect_results:
            main_dialect = max(result.items(), key=lambda x: x[1])[0]
            confidence = result[main_dialect]
            
            dialect_counts[main_dialect] += 1
            dialect_confidence[main_dialect].append(confidence)
        
        # Calculate average confidence per dialect
        avg_confidence = {}
        for dialect, confidences in dialect_confidence.items():
            if confidences:
                avg_confidence[dialect] = sum(confidences) / len(confidences)
            else:
                avg_confidence[dialect] = 0.0
                
        # Calculate percentages
        total_texts = len(texts)
        percentages = {
            dialect: (count / total_texts) * 100 if total_texts > 0 else 0
            for dialect, count in dialect_counts.items()
        }
        
        # Detect regional dialects for texts with high confidence
        regional_data = {}
        for dialect in ["northern", "northeastern", "southern"]:
            if dialect_counts[dialect] > 0:
                # Find texts classified as this dialect with high confidence
                dialect_texts = []
                for text, result in zip(texts, dialect_results):
                    if max(result.items(), key=lambda x: x[1])[0] == dialect and result[dialect] > 0.6:
                        dialect_texts.append(text)
                
                # If we have enough texts, analyze regional distribution
                if len(dialect_texts) >= 5:
                    regional_results = self.batch_detect_regional_dialect(dialect_texts, dialect)
                    regional_counts = {}
                    
                    for result in regional_results:
                        main_region = max(result.items(), key=lambda x: x[1])[0]
                        regional_counts[main_region] = regional_counts.get(main_region, 0) + 1
                    
                    # Calculate regional percentages
                    regional_percentages = {
                        region: (count / len(dialect_texts)) * 100
                        for region, count in regional_counts.items()
                    }
                    
                    regional_data[dialect] = {
                        "counts": regional_counts,
                        "percentages": regional_percentages
                    }
        
        return {
            "counts": dialect_counts,
            "percentages": percentages,
            "avg_confidence": avg_confidence,
            "regional_data": regional_data,
            "total_texts": total_texts
        }
    
    def create_dialect_profile(self, texts: List[str], min_confidence: float = 0.7) -> Dict[str, Any]:
        """Create a dialect profile from a corpus of texts
        
        Args:
            texts: List of texts to analyze
            min_confidence: Minimum confidence for inclusion
            
        Returns:
            Dialect profile data
        """
        # Detect dialects for all texts
        dialect_results = self.batch_detect_dialect(texts)
        
        # Group texts by detected dialect
        dialect_texts = {dialect: [] for dialect in ["northern", "northeastern", "southern", "central", "pattani_malay"]}
        
        for text, result in zip(texts, dialect_results):
            main_dialect = max(result.items(), key=lambda x: x[1])[0]
            confidence = result[main_dialect]
            
            if confidence >= min_confidence:
                dialect_texts[main_dialect].append(text)
        
        # Extract dialect-specific patterns and words
        dialect_words = {}
        common_patterns = {}
        
        for dialect, d_texts in dialect_texts.items():
            if len(d_texts) < 5:  # Skip dialects with too few texts
                continue
                
            # Extract unique words
            combined_text = " ".join(d_texts)
            words = set(combined_text.split())
            
            # Find words that appear in this dialect but not in standard Thai
            dialect_specific = []
            if dialect != "central":
                standard_texts = dialect_texts["central"]
                standard_words = set(" ".join(standard_texts).split())
                
                # Find words specific to this dialect
                for word in words:
                    if word not in standard_words and len(word) > 1:
                        # Check if in our known dialect vocabulary
                        if dialect in self.dialect_processor.dialect_features:
                            vocab = self.dialect_processor.dialect_features[dialect].get("vocabulary", {})
                            for std_word, dialect_word in vocab.items():
                                if word == dialect_word:
                                    dialect_specific.append(word)
                                    break
                        
                dialect_words[dialect] = dialect_specific[:100]  # Limit to 100 words
                
            # Extract common patterns
            patterns = []
            if dialect in self.dialect_processor.dialect_features:
                particles = self.dialect_processor.dialect_features[dialect].get("particles", [])
                for particle in particles:
                    if particle in combined_text:
                        patterns.append(particle)
                        
            common_patterns[dialect] = patterns[:20]  # Limit to 20 patterns
            
        return {
            "dialect_texts_counts": {d: len(texts) for d, texts in dialect_texts.items()},
            "dialect_specific_words": dialect_words,
            "common_patterns": common_patterns
        }
    
    @staticmethod
    def get_optimizer() -> 'DialectOptimizer':
        """Get a pre-configured dialect optimizer instance"""
        return DialectOptimizer()


# Module-level functions for easy access
def batch_detect_dialect(texts: List[str], threshold: float = 0.1) -> List[Dict[str, float]]:
    """Detect Thai dialect for multiple texts in parallel
    
    Args:
        texts: List of Thai text strings
        threshold: Minimum confidence threshold
        
    Returns:
        List of dialect detection results
    """
    with DialectOptimizer() as optimizer:
        return optimizer.batch_detect_dialect(texts, threshold)

def batch_translate(
    texts: List[str],
    source_dialect: str,
    target_dialect: str = "central"
) -> List[str]:
    """Translate multiple texts between dialects in parallel
    
    Args:
        texts: List of texts to translate
        source_dialect: Source dialect code
        target_dialect: Target dialect code (default: central Thai)
        
    Returns:
        List of translated texts
    """
    with DialectOptimizer() as optimizer:
        return optimizer.batch_translate(texts, source_dialect, target_dialect)

def process_dialect_file(
    file_path: str,
    process_type: str,
    dialect: Optional[str] = None,
    target_dialect: Optional[str] = None
) -> Dict[str, Any]:
    """Process a file with dialect operations
    
    Args:
        file_path: Path to text file
        process_type: Type of processing ('detect', 'translate', 'analyze')
        dialect: Source dialect code (for translation)
        target_dialect: Target dialect code (for translation)
        
    Returns:
        Processing results
    """
    optimizer = DialectOptimizer()
    
    if process_type == "detect":
        # Detect dialects in the file
        def detect_line(line):
            return optimizer.dialect_processor.detect_dialect(line)
            
        results = optimizer.process_large_file(file_path, detect_line)
        
    elif process_type == "translate" and dialect:
        # Translate file content
        target = target_dialect or "central"
        
        def translate_line(line):
            if dialect == "central":
                return optimizer.dialect_processor.translate_from_standard(line, target)
            else:
                return optimizer.dialect_processor.translate_to_standard(line, dialect)
                
        results = optimizer.process_large_file(file_path, translate_line)
        
    elif process_type == "analyze":
        # Read file contents
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        # Analyze dialect distribution
        results = optimizer.analyze_dialect_distribution(lines)
        
    else:
        results = {"error": "Invalid process_type"}
        
    return results