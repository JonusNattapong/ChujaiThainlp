"""
Benchmark for Thai Fill-Mask task

This script benchmarks the Fill-Mask task performance on Thai language texts.
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
import torch

# Add parent directory to path to run this script directly
sys.path.append(os.path.abspath(".."))

from benchmark_utils import ThaiNLPBenchmark
from thainlp.generation.fill_mask import ThaiFillMask

def prepare_dataset(json_path):
    """Load and prepare dataset from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data['data'])
    return df

def evaluate_model(model_name, dataset_path, mask_token="<mask>", top_k=5, device=None):
    """Evaluate Fill-Mask model on benchmark dataset"""
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    fill_mask = ThaiFillMask(model_name_or_path=model_name)
    
    # Load dataset
    df = prepare_dataset(dataset_path)
    
    # Set up benchmark
    benchmark = ThaiNLPBenchmark(
        task_name="fill_mask",
        output_dir=Path(__file__).parent / "results" / "fill_mask"
    )
    
    # Define evaluation function
    def predict_mask(texts):
        results = []
        for text in texts:
            predictions = fill_mask.fill_mask(text, top_k=top_k)
            # Return top prediction token
            if predictions:
                results.append(predictions[0]['token'])
            else:
                results.append("")
        return results
    
    # Run benchmark
    result = benchmark.run_benchmark(
        model_func=predict_mask,
        dataset=df,
        input_cols="masked_text",
        truth_col="correct_token",
        model_name=model_name.split('/')[-1],
        batch_size=1  # Process one at a time due to ThaiFillMask API design
    )
    
    print(f"Benchmark completed for {model_name}")
    print(f"Results saved to {benchmark.output_dir}")
    
    return result

def compare_models(models, dataset_path, mask_token="<mask>", top_k=5):
    """Compare multiple Fill-Mask models"""
    results = []
    
    for model in models:
        print(f"\nEvaluating model: {model}")
        result = evaluate_model(
            model_name=model,
            dataset_path=dataset_path,
            mask_token=mask_token,
            top_k=top_k
        )
        results.append(result['id'])
    
    # Set up benchmark for comparison
    benchmark = ThaiNLPBenchmark(
        task_name="fill_mask",
        output_dir=Path(__file__).parent / "results" / "fill_mask"
    )
    
    # Compare models
    benchmark.compare_models(results, metric="f1")

if __name__ == "__main__":
    # Define models to evaluate
    models = [
        "airesearch/wangchanberta-base-att-spm-uncased",
        "google-bert/bert-base-multilingual-cased"
    ]
    
    # Path to benchmark dataset
    dataset_path = Path(__file__).parent / "datasets" / "fill_mask_benchmark.json"
    
    # Run comparison
    compare_models(models, dataset_path)