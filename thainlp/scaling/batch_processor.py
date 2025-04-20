"""
Unified Batch Processing System for ThaiNLP

This module provides a unified interface for batch processing across different modalities:
- Speech processing 
- Vision processing
- Multimodal processing
- Text processing

Features:
- Automatic resource scaling
- Performance benchmarking
- Error handling and reporting
- Task prioritization
- Progress tracking
"""

from typing import Dict, List, Optional, Union, Callable, Any, TypeVar, Generic
import time
import threading
import os
import json
import numpy as np
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

from .auto_scaler import ResourceManager, LoadBalancer, AutoScaler, ServiceMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic processing
T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type

class ProcessingMode(Enum):
    """Processing modes for batch processor"""
    SEQUENTIAL = "sequential"  # Process items sequentially
    PARALLEL = "parallel"      # Process items in parallel
    STREAMING = "streaming"    # Process items as they become available
    ADAPTIVE = "adaptive"      # Dynamically choose between sequential and parallel


@dataclass
class BatchProcessResult:
    """Results from batch processing"""
    results: List[Any]                        # Processed results
    runtime: float                            # Total runtime in seconds
    success_rate: float                       # Success rate (0-1)
    errors: Dict[int, Exception] = field(default_factory=dict)  # Errors by item index
    metrics: Dict[str, Any] = field(default_factory=dict)       # Performance metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "results": [r for r in self.results],
            "runtime": self.runtime,
            "success_rate": self.success_rate,
            "errors": {str(k): str(v) for k, v in self.errors.items()},
            "metrics": self.metrics
        }
    
    def save_report(self, filepath: str):
        """Save benchmark report to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class BatchProcessor(Generic[T, U]):
    """Unified batch processor for ThaiNLP components"""
    
    def __init__(
        self,
        process_fn: Callable[[T], U],
        mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        batch_size: int = 32,
        max_workers: int = None,
        timeout: Optional[float] = None,
        resource_manager: Optional[ResourceManager] = None,
        name: str = "batch_processor"
    ):
        """Initialize batch processor
        
        Args:
            process_fn: Processing function that takes a single item
            mode: Processing mode
            batch_size: Size of batches to process together
            max_workers: Maximum number of worker threads
            timeout: Timeout for processing each item (seconds)
            resource_manager: Optional resource manager for scaling
            name: Name for this processor (for logging)
        """
        self.process_fn = process_fn
        self.mode = mode
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, os.cpu_count() * 2)
        self.timeout = timeout
        self.name = name
        
        # Create resource manager if not provided
        if resource_manager is None:
            load_balancer = LoadBalancer()
            auto_scaler = AutoScaler()
            self.resource_manager = ResourceManager(load_balancer, auto_scaler)
        else:
            self.resource_manager = resource_manager
            
        # Create executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Statistics and metrics
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self._lock = threading.Lock()
        
    def _process_item(self, item: T, item_index: int) -> dict:
        """Process a single item
        
        Args:
            item: Item to process
            item_index: Index of item in the batch
            
        Returns:
            Dictionary with results and metrics
        """
        start_time = time.time()
        success = True
        error = None
        
        try:
            result = self.process_fn(item)
        except Exception as e:
            success = False
            error = e
            result = None
            logger.error(f"Error processing item {item_index}: {str(e)}")
            traceback_str = traceback.format_exc()
            logger.debug(traceback_str)
            
        processing_time = time.time() - start_time
        
        with self._lock:
            self.processed_count += 1
            if not success:
                self.error_count += 1
            self.total_processing_time += processing_time
            
        return {
            "index": item_index,
            "result": result,
            "success": success,
            "error": error,
            "processing_time": processing_time
        }
        
    def _process_sequential(self, items: List[T]) -> BatchProcessResult:
        """Process items sequentially
        
        Args:
            items: List of items to process
            
        Returns:
            Batch processing results
        """
        results = []
        errors = {}
        total_time = 0
        
        for i, item in enumerate(items):
            result_dict = self._process_item(item, i)
            results.append(result_dict["result"])
            
            if not result_dict["success"]:
                errors[i] = result_dict["error"]
                
            total_time += result_dict["processing_time"]
            
        success_rate = 1.0 - (len(errors) / len(items) if items else 0)
        
        return BatchProcessResult(
            results=results,
            runtime=total_time,
            success_rate=success_rate,
            errors=errors,
            metrics={
                "mode": "sequential", 
                "avg_item_time": total_time / len(items) if items else 0,
                "total_items": len(items)
            }
        )
        
    def _process_parallel(self, items: List[T]) -> BatchProcessResult:
        """Process items in parallel
        
        Args:
            items: List of items to process
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        futures = []
        
        # Submit all tasks
        for i, item in enumerate(items):
            future = self.executor.submit(self._process_item, item, i)
            futures.append((i, future))
            
        # Collect results
        results = [None] * len(items)
        errors = {}
        
        for i, future in futures:
            try:
                result_dict = future.result(timeout=self.timeout)
                results[i] = result_dict["result"]
                
                if not result_dict["success"]:
                    errors[i] = result_dict["error"]
            except Exception as e:
                errors[i] = e
                logger.error(f"Exception in task {i}: {str(e)}")
                
        total_time = time.time() - start_time
        success_rate = 1.0 - (len(errors) / len(items) if items else 0)
        
        return BatchProcessResult(
            results=results,
            runtime=total_time,
            success_rate=success_rate,
            errors=errors,
            metrics={
                "mode": "parallel",
                "throughput": len(items) / total_time if total_time > 0 else 0,
                "total_items": len(items),
                "workers": self.max_workers
            }
        )
    
    def _process_adaptive(self, items: List[T]) -> BatchProcessResult:
        """Process items using adaptive mode that chooses between sequential and parallel
        
        Args:
            items: List of items to process
            
        Returns:
            Batch processing results
        """
        # For small batches, use sequential processing
        if len(items) < 4:
            return self._process_sequential(items)
            
        # For larger batches, test both methods on a subset and choose the faster one
        test_size = min(4, len(items) // 10)
        if test_size < 2:
            test_size = 2
            
        test_items = items[:test_size]
        
        # Test sequential
        seq_start = time.time()
        self._process_sequential(test_items)
        seq_time = time.time() - seq_start
        
        # Test parallel
        par_start = time.time()
        self._process_parallel(test_items)
        par_time = time.time() - par_start
        
        # Choose the faster method
        if seq_time <= par_time:
            logger.info(f"Using sequential mode (test: seq={seq_time:.4f}s, par={par_time:.4f}s)")
            return self._process_sequential(items)
        else:
            logger.info(f"Using parallel mode (test: seq={seq_time:.4f}s, par={par_time:.4f}s)")
            return self._process_parallel(items)
    
    def process_batch(self, items: List[T]) -> BatchProcessResult:
        """Process a batch of items
        
        Args:
            items: List of items to process
            
        Returns:
            Batch processing results
        """
        if not items:
            return BatchProcessResult(
                results=[],
                runtime=0.0,
                success_rate=1.0,
                errors={},
                metrics={"mode": "empty", "total_items": 0}
            )
            
        # Choose processing mode
        if self.mode == ProcessingMode.SEQUENTIAL:
            return self._process_sequential(items)
        elif self.mode == ProcessingMode.PARALLEL:
            return self._process_parallel(items)
        elif self.mode == ProcessingMode.ADAPTIVE:
            return self._process_adaptive(items)
        else:
            # Default to sequential for other modes
            return self._process_sequential(items)
            
    def process_batches(self, items: List[T]) -> BatchProcessResult:
        """Process items in multiple batches
        
        Args:
            items: List of items to process
            
        Returns:
            Combined batch processing results
        """
        start_time = time.time()
        all_results = []
        all_errors = {}
        all_metrics = {
            "batch_count": 0,
            "batch_times": [],
            "batch_success_rates": []
        }
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i+self.batch_size]
            batch_result = self.process_batch(batch_items)
            
            # Collect results
            all_results.extend(batch_result.results)
            
            # Adjust error indices to global indices
            for error_idx, error in batch_result.errors.items():
                global_idx = i + error_idx
                all_errors[global_idx] = error
                
            # Collect metrics
            all_metrics["batch_count"] += 1
            all_metrics["batch_times"].append(batch_result.runtime)
            all_metrics["batch_success_rates"].append(batch_result.success_rate)
            
        # Calculate overall metrics
        total_time = time.time() - start_time
        success_rate = 1.0 - (len(all_errors) / len(items) if items else 0)
        
        all_metrics["total_time"] = total_time
        all_metrics["average_batch_time"] = np.mean(all_metrics["batch_times"])
        all_metrics["average_success_rate"] = np.mean(all_metrics["batch_success_rates"])
        all_metrics["throughput"] = len(items) / total_time if total_time > 0 else 0
        
        return BatchProcessResult(
            results=all_results,
            runtime=total_time,
            success_rate=success_rate,
            errors=all_errors,
            metrics=all_metrics
        )
    
    def benchmark(self, items: List[T], repetitions: int = 3) -> Dict[str, Any]:
        """Benchmark processing performance
        
        Args:
            items: List of items to process
            repetitions: Number of repetitions for benchmark
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark with {len(items)} items and {repetitions} repetitions")
        
        benchmark_results = {
            "sequential": {"times": [], "success_rates": []},
            "parallel": {"times": [], "success_rates": []}
        }
        
        # Test sequential mode
        original_mode = self.mode
        self.mode = ProcessingMode.SEQUENTIAL
        
        for i in range(repetitions):
            logger.info(f"Sequential benchmark run {i+1}/{repetitions}")
            result = self.process_batches(items)
            benchmark_results["sequential"]["times"].append(result.runtime)
            benchmark_results["sequential"]["success_rates"].append(result.success_rate)
            
        # Test parallel mode
        self.mode = ProcessingMode.PARALLEL
        
        for i in range(repetitions):
            logger.info(f"Parallel benchmark run {i+1}/{repetitions}")
            result = self.process_batches(items)
            benchmark_results["parallel"]["times"].append(result.runtime)
            benchmark_results["parallel"]["success_rates"].append(result.success_rate)
            
        # Restore original mode
        self.mode = original_mode
        
        # Calculate statistics
        seq_times = benchmark_results["sequential"]["times"]
        par_times = benchmark_results["parallel"]["times"]
        
        # Calculate speedup
        avg_seq_time = np.mean(seq_times)
        avg_par_time = np.mean(par_times)
        speedup = avg_seq_time / avg_par_time if avg_par_time > 0 else 0
        
        # Format final results
        summary = {
            "sequential": {
                "avg_time": float(np.mean(seq_times)),
                "min_time": float(np.min(seq_times)),
                "max_time": float(np.max(seq_times)),
                "std_dev": float(np.std(seq_times)),
                "avg_success_rate": float(np.mean(benchmark_results["sequential"]["success_rates"]))
            },
            "parallel": {
                "avg_time": float(np.mean(par_times)),
                "min_time": float(np.min(par_times)),
                "max_time": float(np.max(par_times)),
                "std_dev": float(np.std(par_times)),
                "avg_success_rate": float(np.mean(benchmark_results["parallel"]["success_rates"]))
            },
            "speedup": float(speedup),
            "recommended_mode": "parallel" if speedup > 1 else "sequential",
            "item_count": len(items),
            "batch_size": self.batch_size,
            "max_workers": self.max_workers
        }
        
        logger.info(f"Benchmark completed. Speedup: {speedup:.2f}x")
        return summary

    def shutdown(self):
        """Shutdown the processor"""
        self.executor.shutdown(wait=True)


class MultimodalBatchProcessor(BatchProcessor):
    """Batch processor specialized for multimodal tasks"""
    
    def __init__(
        self,
        process_fn: Callable[[Any], Any],
        mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        batch_size: int = 16,
        max_workers: int = None,
        timeout: Optional[float] = None,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        memory_limit: Optional[float] = None
    ):
        """Initialize multimodal batch processor
        
        Args:
            process_fn: Processing function
            mode: Processing mode
            batch_size: Size of batches
            max_workers: Maximum workers
            timeout: Processing timeout
            device: Device to run models on
            memory_limit: Memory limit in GB
        """
        super().__init__(
            process_fn=process_fn,
            mode=mode,
            batch_size=batch_size,
            max_workers=max_workers,
            timeout=timeout,
            name="multimodal_processor"
        )
        self.device = device
        self.memory_limit = memory_limit
        
        # Adjust batch size based on device
        if device == "cuda":
            # Set smaller batch size for GPU to avoid OOM
            import torch
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            if self.memory_limit:
                # Calculate batch size based on memory limit
                memory_per_item = self.memory_limit / self.batch_size
                safe_batch_size = int(self.gpu_memory * 0.7 / memory_per_item)
                self.batch_size = min(self.batch_size, max(1, safe_batch_size))
                logger.info(f"Adjusted batch size to {self.batch_size} based on GPU memory ({self.gpu_memory:.1f}GB)")


class SpeechBatchProcessor(BatchProcessor):
    """Batch processor specialized for speech processing tasks"""
    
    def process_audio_batch(self, 
                           file_paths: List[str], 
                           process_type: str = "transcribe", 
                           **kwargs) -> BatchProcessResult:
        """Process a batch of audio files
        
        Args:
            file_paths: List of audio file paths
            process_type: Type of processing (transcribe, translate, etc.)
            **kwargs: Additional processing parameters
            
        Returns:
            Batch processing results
        """
        # Define processing function based on process_type
        def process_item(file_path):
            from ..speech import ThaiSpeechProcessor
            processor = ThaiSpeechProcessor()
            
            if process_type == "transcribe":
                return processor.speech_to_text(file_path, **kwargs)
            elif process_type == "translate":
                transcription = processor.speech_to_text(file_path)
                # Integrate with translation module
                from ..translation import translate_text
                target_lang = kwargs.get("target_lang", "en")
                return translate_text(transcription, target_lang=target_lang)
            elif process_type == "vad":
                return processor.voice_activity_detection(file_path, **kwargs)
            elif process_type == "diarization":
                return processor.speaker_diarization(file_path)
            elif process_type == "emotion":
                return processor.detect_emotion_from_speech(file_path)
            else:
                raise ValueError(f"Unknown process type: {process_type}")
                
        # Override process_fn with our audio processing function
        self.process_fn = process_item
        
        # Process batches
        return self.process_batches(file_paths)


class VisionBatchProcessor(BatchProcessor):
    """Batch processor specialized for vision processing tasks"""
    
    def process_image_batch(self, 
                           image_paths: List[str], 
                           process_type: str = "classify", 
                           **kwargs) -> BatchProcessResult:
        """Process a batch of image files
        
        Args:
            image_paths: List of image file paths
            process_type: Type of processing (classify, caption, etc.)
            **kwargs: Additional processing parameters
            
        Returns:
            Batch processing results
        """
        # Define processing function based on process_type
        def process_item(image_path):
            if process_type == "classify":
                from ..vision import classify_image
                return classify_image(image_path, **kwargs)
            elif process_type == "detect":
                from ..vision import detect_objects
                return detect_objects(image_path, **kwargs)
            elif process_type == "caption":
                from ..multimodal import caption_image
                return caption_image(image_path, **kwargs)
            elif process_type == "ocr":
                from ..multimodal import extract_text_from_image
                return extract_text_from_image(image_path, **kwargs)
            else:
                raise ValueError(f"Unknown process type: {process_type}")
                
        # Override process_fn with our image processing function
        self.process_fn = process_item
        
        # Process batches
        return self.process_batches(image_paths)


# Integrated batch processing function for easy access
def batch_process(
    items: List[Any],
    process_type: str,
    mode: str = "adaptive",
    batch_size: int = 32,
    max_workers: int = None,
    **kwargs
) -> Dict[str, Any]:
    """Integrated batch processing for multiple modalities
    
    Args:
        items: List of items to process
        process_type: Type of processing 
                    (audio.transcribe, audio.translate, 
                     vision.classify, vision.detect,
                     multimodal.vqa, etc.)
        mode: Processing mode (sequential, parallel, adaptive)
        batch_size: Size of batches
        max_workers: Maximum workers
        **kwargs: Additional processing parameters
        
    Returns:
        Dictionary with processing results and metrics
    """
    try:
        # Determine processor type from process_type
        mode_enum = ProcessingMode(mode)
        processor_type, operation = process_type.split(".", 1)
        
        if processor_type == "audio" or processor_type == "speech":
            processor = SpeechBatchProcessor(
                process_fn=lambda x: x,  # Placeholder, will be overridden
                mode=mode_enum,
                batch_size=batch_size,
                max_workers=max_workers
            )
            result = processor.process_audio_batch(items, operation, **kwargs)
            
        elif processor_type == "vision" or processor_type == "image":
            processor = VisionBatchProcessor(
                process_fn=lambda x: x,  # Placeholder, will be overridden
                mode=mode_enum,
                batch_size=batch_size,
                max_workers=max_workers
            )
            result = processor.process_image_batch(items, operation, **kwargs)
            
        elif processor_type == "multimodal":
            processor = MultimodalBatchProcessor(
                process_fn=lambda x: x,  # Will be defined inside
                mode=mode_enum,
                batch_size=batch_size,
                max_workers=max_workers
            )
            # Handle different multimodal operations
            if operation == "vqa":
                from ..multimodal import answer_visual_question
                # For VQA, items should be tuples of (image_path, question)
                processor.process_fn = lambda item: answer_visual_question(item[0], item[1], **kwargs)
            elif operation == "document_qa":
                from ..multimodal import answer_document_question
                # For document QA, items should be tuples of (document_path, question)
                processor.process_fn = lambda item: answer_document_question(item[0], item[1], **kwargs)
                
            result = processor.process_batches(items)
            
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
            
        # Return as dictionary for API compatibility
        return {
            "results": result.results,
            "success_rate": result.success_rate,
            "runtime": result.runtime,
            "errors": {str(k): str(v) for k, v in result.errors.items()},
            "metrics": result.metrics
        }
        
    except Exception as e:
        logger.error(f"Error in batch_process: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.debug(traceback_str)
        return {
            "results": [],
            "success_rate": 0.0,
            "runtime": 0.0,
            "errors": {"global": str(e)},
            "metrics": {"status": "failed", "error": str(e)}
        }