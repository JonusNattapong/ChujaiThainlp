"""
Examples of batch processing with ChujaiThaiNLP

This example demonstrates how to use the batch processing system
across different modalities (speech, vision, and multimodal) to
improve performance and accuracy.
"""

import os
import time
import argparse
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from thainlp.scaling.batch_processor import (
    batch_process,
    SpeechBatchProcessor,
    VisionBatchProcessor,
    MultimodalBatchProcessor,
    ProcessingMode
)
from thainlp.speech import ThaiSpeechProcessor
from thainlp.vision import classify_image, detect_objects
from thainlp.multimodal import caption_image, answer_visual_question


def speech_batch_example(audio_dir: str, output_dir: str = None):
    """Example of batch processing audio files
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory for saving results
    """
    print("\n=== Speech Batch Processing Example ===")
    
    # Find audio files
    audio_files = []
    if os.path.exists(audio_dir):
        audio_files = [
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))
        ]
    
    if not audio_files:
        print("No audio files found. Using mock data.")
        # Create mock data for demonstration
        audio_files = [f"mock_audio_{i}.wav" for i in range(10)]
    
    # Process audio files with SpeechBatchProcessor
    print(f"Processing {len(audio_files)} audio files...")
    
    # Choose whether to use sequential or parallel processing
    for mode in ["sequential", "parallel"]:
        print(f"\nUsing {mode} processing mode:")
        start_time = time.time()
        
        # Use the batch_process helper function
        result = batch_process(
            items=audio_files,
            process_type="speech.transcribe",
            mode=mode,
            batch_size=4
        )
        
        elapsed = time.time() - start_time
        print(f"  Processed {len(audio_files)} files in {elapsed:.2f} seconds")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Average processing time: {result['runtime']/len(audio_files):.3f}s per file")
        
        # You can also access the raw results
        for i, (file, transcription) in enumerate(zip(audio_files[:3], result['results'][:3])):
            print(f"  Sample {i+1}: {os.path.basename(file)} -> {transcription[:50]}...")
        
        if i < len(result['results']) - 1:
            print(f"  ...and {len(result['results']) - 3} more")
        
        # If there were any errors
        if result['errors']:
            print(f"  Encountered {len(result['errors'])} errors")
        
    # Alternative: create a custom processor for more control
    print("\nRunning with custom speech processor:")
    processor = SpeechBatchProcessor(
        process_fn=lambda x: x,  # Will be overridden
        mode=ProcessingMode.ADAPTIVE,
        batch_size=4,
        max_workers=8
    )
    
    # Define a custom processing function
    def process_audio_with_emotion(file_path):
        # Create a ThaiSpeechProcessor instance
        speech_proc = ThaiSpeechProcessor()
        
        # Get both transcription and emotion
        transcription = speech_proc.speech_to_text(file_path)
        try:
            emotion = speech_proc.detect_emotion_from_speech(file_path)
            primary_emotion = max(emotion.items(), key=lambda x: x[1])[0]
        except:
            primary_emotion = "unknown"
        
        return {
            "file": os.path.basename(file_path),
            "transcription": transcription,
            "emotion": primary_emotion
        }
    
    # Set the custom processing function and process
    processor.process_fn = process_audio_with_emotion
    
    # Only run on a few files for demonstration
    demo_files = audio_files[:3]
    result = processor.process_batches(demo_files)
    
    print(f"Processed with custom function: {len(result.results)} files")
    print(f"Success rate: {result.success_rate:.1%}")
    print(f"Runtime: {result.runtime:.2f}s")
    
    # Save results if output directory provided
    if output_dir and os.path.exists(output_dir):
        result_path = os.path.join(output_dir, "speech_batch_results.txt")
        with open(result_path, "w") as f:
            for r in result.results:
                if r:  # Skip None results from errors
                    f.write(f"File: {r['file']}\n")
                    f.write(f"Transcription: {r['transcription']}\n")
                    f.write(f"Emotion: {r['emotion']}\n")
                    f.write("\n")
        print(f"Results saved to {result_path}")


def vision_batch_example(image_dir: str, output_dir: str = None):
    """Example of batch processing images
    
    Args:
        image_dir: Directory containing image files
        output_dir: Directory for saving results
    """
    print("\n=== Vision Batch Processing Example ===")
    
    # Find image files
    image_files = []
    if os.path.exists(image_dir):
        image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
    
    if not image_files:
        print("No image files found. Using mock data.")
        # Create mock data for demonstration
        image_files = [f"mock_image_{i}.jpg" for i in range(10)]
    
    # Process a batch of images for classification
    print(f"Processing {len(image_files)} images for classification...")
    
    result = batch_process(
        items=image_files,
        process_type="vision.classify", 
        mode="adaptive",  # Automatically choose between sequential and parallel
        batch_size=4
    )
    
    print(f"Classification completed in {result['runtime']:.2f} seconds")
    print(f"Success rate: {result['success_rate']:.1%}")
    
    # Process a batch of images for object detection
    print(f"\nProcessing {len(image_files)} images for object detection...")
    
    result = batch_process(
        items=image_files[:5],  # Use fewer images for object detection
        process_type="vision.detect",
        mode="parallel",
        batch_size=2
    )
    
    print(f"Object detection completed in {result['runtime']:.2f} seconds")
    print(f"Success rate: {result['success_rate']:.1%}")
    
    # Alternative: use VisionBatchProcessor directly for more control
    print("\nRunning image captioning with custom processor:")
    
    processor = VisionBatchProcessor(
        process_fn=lambda x: caption_image(x),
        mode=ProcessingMode.PARALLEL,
        batch_size=4
    )
    
    # Only process a few images for demonstration
    demo_files = image_files[:3]
    result = processor.process_batches(demo_files)
    
    print(f"Captioned {len(result.results)} images")
    for i, (img, caption) in enumerate(zip(demo_files, result.results)):
        print(f"  Image {i+1}: {os.path.basename(img)} -> {caption}")
    
    # Save results if output directory provided
    if output_dir and os.path.exists(output_dir):
        result_path = os.path.join(output_dir, "vision_batch_results.txt")
        with open(result_path, "w") as f:
            for i, (img, caption) in enumerate(zip(demo_files, result.results)):
                f.write(f"Image: {os.path.basename(img)}\n")
                f.write(f"Caption: {caption}\n\n")
        print(f"Results saved to {result_path}")


def multimodal_batch_example(data_dir: str, output_dir: str = None):
    """Example of batch processing multimodal data
    
    Args:
        data_dir: Directory containing multimodal data
        output_dir: Directory for saving results
    """
    print("\n=== Multimodal Batch Processing Example ===")
    
    # For visual question answering, we need both images and questions
    image_dir = os.path.join(data_dir, "images") if os.path.exists(os.path.join(data_dir, "images")) else data_dir
    
    # Find image files
    image_files = []
    if os.path.exists(image_dir):
        image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ][:5]  # Limit to 5 images for demonstration
    
    if not image_files:
        print("No image files found. Using mock data.")
        image_files = [f"mock_image_{i}.jpg" for i in range(5)]
    
    # Create sample questions for each image
    questions = [
        "What can you see in this image?",
        "What is the main subject of this image?",
        "Describe the scene in this image.",
        "What objects are visible in this image?",
        "Is there a person in this image?"
    ]
    
    # Create pairs of (image, question)
    vqa_pairs = list(zip(image_files, questions))
    
    print(f"Processing {len(vqa_pairs)} visual QA tasks...")
    
    # Use MultimodalBatchProcessor
    processor = MultimodalBatchProcessor(
        process_fn=lambda pair: answer_visual_question(pair[0], pair[1]),
        mode=ProcessingMode.PARALLEL,
        batch_size=2
    )
    
    # Process VQA pairs
    start_time = time.time()
    result = processor.process_batches(vqa_pairs)
    elapsed = time.time() - start_time
    
    print(f"VQA completed in {elapsed:.2f} seconds")
    print(f"Success rate: {result.success_rate:.1%}")
    
    # Display sample results
    for i, ((img, question), answer) in enumerate(zip(vqa_pairs[:3], result.results[:3])):
        print(f"\nSample {i+1}:")
        print(f"  Image: {os.path.basename(img)}")
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")
    
    # Save results if output directory provided
    if output_dir and os.path.exists(output_dir):
        result_path = os.path.join(output_dir, "multimodal_batch_results.txt")
        with open(result_path, "w") as f:
            for (img, question), answer in zip(vqa_pairs, result.results):
                f.write(f"Image: {os.path.basename(img)}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {answer}\n\n")
        print(f"Results saved to {result_path}")


def benchmark_and_visualize(data_dir: str, output_dir: str = None):
    """Benchmark different batch processing methods and visualize results
    
    Args:
        data_dir: Directory with test data
        output_dir: Directory for saving results
    """
    print("\n=== Benchmark and Visualization ===")
    
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), "batch_benchmarks")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data paths
    speech_dir = os.path.join(data_dir, "audio") if os.path.exists(os.path.join(data_dir, "audio")) else data_dir
    vision_dir = os.path.join(data_dir, "images") if os.path.exists(os.path.join(data_dir, "images")) else data_dir
    
    # Define benchmarks to run
    benchmarks = [
        {
            "name": "Speech → Text",
            "processor_type": "speech",
            "operation": "transcribe",
            "data_dir": speech_dir,
            "extension": ".wav"
        },
        {
            "name": "Image → Caption",
            "processor_type": "vision",
            "operation": "caption",
            "data_dir": vision_dir,
            "extension": ".jpg"
        },
        {
            "name": "Image → Classification",
            "processor_type": "vision",
            "operation": "classify",
            "data_dir": vision_dir,
            "extension": ".jpg"
        }
    ]
    
    # Results storage
    results = {
        "task": [],
        "sequential_time": [],
        "parallel_time": [],
        "speedup": [],
        "success_rate_seq": [],
        "success_rate_par": []
    }
    
    # Run benchmarks
    for benchmark in benchmarks:
        print(f"\nBenchmarking: {benchmark['name']}...")
        
        # Find files
        files = []
        if os.path.exists(benchmark["data_dir"]):
            files = [
                os.path.join(benchmark["data_dir"], f) 
                for f in os.listdir(benchmark["data_dir"])
                if f.lower().endswith(benchmark["extension"])
            ][:10]  # Limit to 10 files
        
        if not files:
            print(f"No suitable files found for {benchmark['name']}. Skipping.")
            continue
        
        # Process with both sequential and parallel modes
        process_type = f"{benchmark['processor_type']}.{benchmark['operation']}"
        
        print(f"  Running sequential mode with {len(files)} files...")
        seq_result = batch_process(
            items=files,
            process_type=process_type,
            mode="sequential",
            batch_size=2
        )
        
        print(f"  Running parallel mode with {len(files)} files...")
        par_result = batch_process(
            items=files,
            process_type=process_type,
            mode="parallel",
            batch_size=2
        )
        
        # Calculate speedup
        seq_time = seq_result["runtime"]
        par_time = par_result["runtime"]
        speedup = seq_time / par_time if par_time > 0 else 0
        
        # Store results
        results["task"].append(benchmark["name"])
        results["sequential_time"].append(seq_time)
        results["parallel_time"].append(par_time)
        results["speedup"].append(speedup)
        results["success_rate_seq"].append(seq_result["success_rate"])
        results["success_rate_par"].append(par_result["success_rate"])
        
        print(f"  Sequential: {seq_time:.3f}s, Parallel: {par_time:.3f}s, Speedup: {speedup:.2f}x")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate visualizations
    if len(df) > 0:
        # Bar chart comparing sequential vs parallel
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        x = np.arange(len(df["task"]))
        plt.bar(x - bar_width/2, df["sequential_time"], bar_width, label="Sequential")
        plt.bar(x + bar_width/2, df["parallel_time"], bar_width, label="Parallel")
        plt.xlabel("Task")
        plt.ylabel("Processing Time (seconds)")
        plt.title("Sequential vs Parallel Processing Time by Task")
        plt.xticks(x, df["task"])
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "batch_processing_comparison.png"), dpi=300, bbox_inches="tight")
        
        # Speedup chart
        plt.figure(figsize=(10, 5))
        plt.bar(df["task"], df["speedup"], color="green")
        plt.axhline(y=1.0, linestyle="--", color="red")
        plt.xlabel("Task")
        plt.ylabel("Speedup (Sequential / Parallel)")
        plt.title("Processing Speedup by Task")
        plt.grid(axis="y", alpha=0.3)
        
        for i, v in enumerate(df["speedup"]):
            plt.text(i, v + 0.05, f"{v:.2f}x", ha="center")
        
        plt.savefig(os.path.join(output_dir, "batch_processing_speedup.png"), dpi=300, bbox_inches="tight")
        
        # Save results
        df.to_csv(os.path.join(output_dir, "batch_processing_benchmarks.csv"), index=False)
        
        print(f"\nBenchmark results and visualizations saved to {output_dir}")
        print("\nSummary:")
        print(df)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch processing examples")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory with test data")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory for saving results")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["speech", "vision", "multimodal", "benchmark", "all"],
                        help="Which examples to run")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run examples based on mode
    if args.mode in ["speech", "all"]:
        speech_batch_example(
            audio_dir=os.path.join(args.data_dir, "audio"),
            output_dir=args.output_dir
        )
    
    if args.mode in ["vision", "all"]:
        vision_batch_example(
            image_dir=os.path.join(args.data_dir, "images"),
            output_dir=args.output_dir
        )
    
    if args.mode in ["multimodal", "all"]:
        multimodal_batch_example(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    
    if args.mode in ["benchmark", "all"]:
        benchmark_and_visualize(
            data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "benchmarks")
        )
    
    print("\nExamples completed. Check the output directory for results.")


if __name__ == "__main__":
    main()