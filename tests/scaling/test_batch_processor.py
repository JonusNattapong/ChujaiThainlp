"""
Tests for the batch processing system
"""
import unittest
import os
import time
import tempfile
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from thainlp.scaling.batch_processor import (
    BatchProcessor, 
    SpeechBatchProcessor, 
    VisionBatchProcessor, 
    MultimodalBatchProcessor,
    ProcessingMode,
    batch_process,
    BatchProcessResult
)


class TestBatchProcessor(unittest.TestCase):
    """Test the batch processor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple processing function for testing
        def simple_process(x):
            time.sleep(0.01)  # Simulate processing time
            return x * 2
            
        self.process_fn = simple_process
        self.test_data = list(range(100))
        
    def test_sequential_processing(self):
        """Test sequential processing mode"""
        processor = BatchProcessor(
            process_fn=self.process_fn,
            mode=ProcessingMode.SEQUENTIAL,
            batch_size=10
        )
        
        result = processor.process_batches(self.test_data)
        
        self.assertEqual(len(result.results), len(self.test_data))
        self.assertEqual(result.results[0], 0)
        self.assertEqual(result.results[-1], 198)
        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(len(result.errors), 0)
        
    def test_parallel_processing(self):
        """Test parallel processing mode"""
        processor = BatchProcessor(
            process_fn=self.process_fn,
            mode=ProcessingMode.PARALLEL,
            batch_size=10,
            max_workers=4
        )
        
        result = processor.process_batches(self.test_data)
        
        self.assertEqual(len(result.results), len(self.test_data))
        self.assertEqual(result.results[0], 0)
        self.assertEqual(result.results[-1], 198)
        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(len(result.errors), 0)
        
    def test_adaptive_processing(self):
        """Test adaptive processing mode"""
        processor = BatchProcessor(
            process_fn=self.process_fn,
            mode=ProcessingMode.ADAPTIVE,
            batch_size=10
        )
        
        result = processor.process_batches(self.test_data)
        
        self.assertEqual(len(result.results), len(self.test_data))
        self.assertEqual(result.success_rate, 1.0)
        
    def test_error_handling(self):
        """Test error handling"""
        def process_with_errors(x):
            if x % 10 == 0:
                raise ValueError(f"Error on item {x}")
            return x * 2
            
        processor = BatchProcessor(
            process_fn=process_with_errors,
            mode=ProcessingMode.SEQUENTIAL,
            batch_size=10
        )
        
        result = processor.process_batches(self.test_data)
        
        self.assertEqual(len(result.results), len(self.test_data))
        self.assertLess(result.success_rate, 1.0)
        self.assertGreater(len(result.errors), 0)
        
    def test_benchmarking(self):
        """Test benchmarking functionality"""
        processor = BatchProcessor(
            process_fn=self.process_fn,
            batch_size=10
        )
        
        benchmark = processor.benchmark(self.test_data[:20], repetitions=2)
        
        self.assertIn("sequential", benchmark)
        self.assertIn("parallel", benchmark)
        self.assertIn("speedup", benchmark)
        self.assertIn("recommended_mode", benchmark)


class TestMultimodalBenchmark(unittest.TestCase):
    """Integration tests for multimodal batch processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Skip integration tests if no data available
        self.test_dir = os.environ.get("THAINLP_TEST_DATA", None)
        if not self.test_dir or not os.path.exists(self.test_dir):
            self.skipTest("No test data directory specified in THAINLP_TEST_DATA")
            
    def test_vision_batch_processing(self):
        """Test vision batch processing if data available"""
        image_dir = os.path.join(self.test_dir, "images")
        if not os.path.exists(image_dir):
            self.skipTest("No image test data available")
            
        # Get some test images
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)[:5]]
        if not image_files:
            self.skipTest("No image files found")
            
        # Test batch processing
        processor = VisionBatchProcessor(
            process_fn=lambda x: x,  # Placeholder
            batch_size=2
        )
        
        # Test with a mock function to avoid dependency on models
        def mock_classify(image_path):
            time.sleep(0.1)  # Simulate processing
            return {"image": image_path, "class": "test", "score": 0.9}
            
        processor.process_fn = mock_classify
        result = processor.process_batches(image_files)
        
        self.assertEqual(len(result.results), len(image_files))
        self.assertEqual(result.success_rate, 1.0)
        
    def test_speech_batch_processing(self):
        """Test speech batch processing if data available"""
        audio_dir = os.path.join(self.test_dir, "audio")
        if not os.path.exists(audio_dir):
            self.skipTest("No audio test data available")
            
        # Get some test audio files
        audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)[:3]]
        if not audio_files:
            self.skipTest("No audio files found")
            
        # Test with a mock function to avoid dependency on models
        def mock_transcribe(audio_path):
            time.sleep(0.2)  # Simulate processing
            return f"Transcription for {os.path.basename(audio_path)}"
            
        processor = SpeechBatchProcessor(
            process_fn=mock_transcribe,
            batch_size=2
        )
        
        result = processor.process_batches(audio_files)
        
        self.assertEqual(len(result.results), len(audio_files))
        self.assertEqual(result.success_rate, 1.0)


def generate_benchmark_plots(results_dir):
    """Generate benchmark plots from results"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Create some dummy benchmark data for demonstration
    tasks = ['Speech → Text', 'Image → Caption', 'QA', 'Translation', 'OCR']
    sequential_times = [3.2, 4.5, 2.8, 1.9, 5.6]
    parallel_times = [1.8, 2.9, 1.5, 1.1, 3.2]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Task': tasks,
        'Sequential': sequential_times,
        'Parallel': parallel_times
    })
    
    # Calculate speedup
    df['Speedup'] = df['Sequential'] / df['Parallel']
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bar_width = 0.35
    x = np.arange(len(tasks))
    plt.bar(x - bar_width/2, sequential_times, bar_width, label='Sequential')
    plt.bar(x + bar_width/2, parallel_times, bar_width, label='Parallel')
    
    plt.xlabel('Task')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Sequential vs Parallel Processing Time by Task')
    plt.xticks(x, tasks)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(results_dir, 'processing_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Speedup plot
    plt.figure(figsize=(10, 5))
    plt.bar(tasks, df['Speedup'], color='green')
    plt.axhline(y=1.0, linestyle='--', color='red')
    plt.xlabel('Task')
    plt.ylabel('Speedup (Sequential / Parallel)')
    plt.title('Processing Speedup by Task')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(df['Speedup']):
        plt.text(i, v + 0.05, f'{v:.2f}x', ha='center')
        
    plt.savefig(os.path.join(results_dir, 'speedup_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Save the data
    df.to_csv(os.path.join(results_dir, 'benchmark_results.csv'), index=False)
    
    return df


def run_real_benchmarks(data_dirs: Dict[str, str] = None) -> Dict[str, Any]:
    """Run real benchmarks on available data
    
    Args:
        data_dirs: Dictionary mapping data types to directories
        
    Returns:
        Benchmark results
    """
    results = {}
    
    # Use default test data directory if none provided
    if not data_dirs:
        base_dir = os.environ.get("THAINLP_TEST_DATA", None)
        if not base_dir or not os.path.exists(base_dir):
            print("No test data available. Using mock data.")
            # Return mock results
            return {
                "speech.transcribe": {
                    "sequential": {"avg_time": 2.5, "success_rate": 0.98},
                    "parallel": {"avg_time": 1.2, "success_rate": 0.97},
                    "speedup": 2.08,
                    "recommended_mode": "parallel"
                },
                "vision.classify": {
                    "sequential": {"avg_time": 3.1, "success_rate": 1.0},
                    "parallel": {"avg_time": 1.8, "success_rate": 1.0},
                    "speedup": 1.72,
                    "recommended_mode": "parallel"
                }
            }
            
        data_dirs = {
            "speech": os.path.join(base_dir, "audio"),
            "vision": os.path.join(base_dir, "images"),
            "text": os.path.join(base_dir, "text"),
            "multimodal": os.path.join(base_dir, "multimodal")
        }
        
    # Run speech benchmarks
    if os.path.exists(data_dirs.get("speech", "")):
        speech_files = [os.path.join(data_dirs["speech"], f) for f in os.listdir(data_dirs["speech"])[:5]]
        if speech_files:
            # Mock function for testing without loading models
            def mock_speech_process(file_path):
                time.sleep(0.1)
                return f"Processed {os.path.basename(file_path)}"
                
            speech_processor = SpeechBatchProcessor(
                process_fn=mock_speech_process,
                batch_size=2
            )
            
            results["speech.transcribe"] = speech_processor.benchmark(speech_files, repetitions=2)
            
    # Run vision benchmarks
    if os.path.exists(data_dirs.get("vision", "")):
        vision_files = [os.path.join(data_dirs["vision"], f) for f in os.listdir(data_dirs["vision"])[:5]]
        if vision_files:
            # Mock function for testing without loading models
            def mock_vision_process(file_path):
                time.sleep(0.1)
                return {"image": file_path, "class": "test", "score": 0.9}
                
            vision_processor = VisionBatchProcessor(
                process_fn=mock_vision_process,
                batch_size=2
            )
            
            results["vision.classify"] = vision_processor.benchmark(vision_files, repetitions=2)
    
    return results


if __name__ == "__main__":
    # Example of running benchmarks and generating plots
    results_dir = os.path.join(tempfile.gettempdir(), "thainlp_benchmarks")
    
    print(f"Running benchmarks and generating plots in {results_dir}")
    benchmarks = run_real_benchmarks()
    
    # Print summary
    for process_type, result in benchmarks.items():
        print(f"\n{process_type} Benchmark:")
        print(f"  Sequential: {result['sequential']['avg_time']:.3f}s")
        print(f"  Parallel: {result['parallel']['avg_time']:.3f}s")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Recommended: {result['recommended_mode']}")
    
    # Generate plots
    df = generate_benchmark_plots(results_dir)
    print(f"\nBenchmark plots saved to {results_dir}")
    print(df)