"""
Benchmarking utilities for Thai NLP tasks

This module provides tools for benchmarking and evaluating various NLP tasks
on Thai language datasets.
"""
import os
import time
import json
import csv
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Union, Callable, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_squared_error,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ThaiNLPBenchmark:
    """Benchmark evaluator for Thai NLP tasks"""
    
    def __init__(
        self,
        task_name: str,
        output_dir: str = None,
        verbose: bool = True
    ):
        """Initialize benchmark evaluator
        
        Args:
            task_name: Name of the task to benchmark
            output_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.task_name = task_name
        self.verbose = verbose
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "results" / task_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.metrics = {}
        self.timing = {}
        self.current_dataset = None
        
        if self.verbose:
            print(f"Initialized benchmark for {task_name}")
            print(f"Results will be saved to {self.output_dir}")

    def load_dataset(self, dataset_path: Union[str, Path], dataset_format: str = "json") -> pd.DataFrame:
        """Load benchmark dataset
        
        Args:
            dataset_path: Path to dataset
            dataset_format: Format of dataset (json, csv, tsv)
            
        Returns:
            Pandas dataframe with dataset
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        if self.verbose:
            print(f"Loading dataset from {dataset_path}")
            
        if dataset_format.lower() == "json":
            dataset = pd.read_json(dataset_path)
        elif dataset_format.lower() == "csv":
            dataset = pd.read_csv(dataset_path)
        elif dataset_format.lower() == "tsv":
            dataset = pd.read_csv(dataset_path, sep="\t")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
            
        self.current_dataset = dataset_path.stem
        
        if self.verbose:
            print(f"Loaded dataset with {len(dataset)} samples")
            
        return dataset
        
    def run_benchmark(
        self,
        model_func: Callable,
        dataset: pd.DataFrame,
        input_cols: Union[str, List[str]],
        truth_col: str,
        model_name: str,
        batch_size: int = 16,
        **kwargs
    ) -> Dict:
        """Run benchmark on dataset
        
        Args:
            model_func: Function to evaluate
            dataset: Benchmark dataset
            input_cols: Column(s) with input data
            truth_col: Column with ground truth
            model_name: Name of model
            batch_size: Batch size for processing
            **kwargs: Additional arguments to model_func
            
        Returns:
            Dictionary with benchmark results
        """
        if isinstance(input_cols, str):
            input_cols = [input_cols]
            
        # Prepare inputs
        inputs = []
        for _, row in dataset.iterrows():
            if len(input_cols) == 1:
                inputs.append(row[input_cols[0]])
            else:
                inputs.append([row[col] for col in input_cols])
        
        # Prepare ground truth
        ground_truth = dataset[truth_col].tolist()
        
        # Process in batches
        predictions = []
        start_time = time.time()
        
        for i in tqdm(range(0, len(inputs), batch_size), disable=not self.verbose):
            batch_inputs = inputs[i:i + batch_size]
            batch_predictions = model_func(batch_inputs, **kwargs)
            predictions.extend(batch_predictions)
            
        elapsed_time = time.time() - start_time
        
        # Calculate metrics based on task type
        metrics = self._calculate_metrics(predictions, ground_truth)
        
        # Add timing information
        metrics["timing"] = {
            "total_time": elapsed_time,
            "average_time": elapsed_time / len(dataset),
            "samples_per_second": len(dataset) / elapsed_time
        }
        
        # Store results
        benchmark_id = f"{self.current_dataset}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.results[benchmark_id] = {
            "dataset": self.current_dataset,
            "model": model_name,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics[benchmark_id] = metrics
        self.timing[benchmark_id] = metrics["timing"]
        
        if self.verbose:
            self._print_metrics(metrics, model_name)
            
        # Save results
        self._save_results(benchmark_id)
        
        return {
            "metrics": metrics,
            "id": benchmark_id
        }
        
    def _calculate_metrics(self, predictions: List, ground_truth: List) -> Dict:
        """Calculate metrics based on task type"""
        metrics = {}
        
        if self.task_name.lower() in ["classification", "sentiment", "text_classification"]:
            # Classification metrics
            accuracy = accuracy_score(ground_truth, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth,
                predictions,
                average="weighted"
            )
            
            metrics["accuracy"] = accuracy
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            
            # Confusion matrix
            labels = sorted(set(ground_truth))
            matrix = confusion_matrix(
                ground_truth,
                predictions,
                labels=labels
            )
            
            metrics["confusion_matrix"] = {
                "matrix": matrix.tolist(),
                "labels": labels
            }
            
        elif self.task_name.lower() in ["token_classification", "ner", "pos_tagging"]:
            # Token classification metrics
            flat_true = []
            flat_pred = []
            
            for true_seq, pred_seq in zip(ground_truth, predictions):
                for true_token, pred_token in zip(true_seq, pred_seq):
                    if true_token != "O":  # Ignore outside tokens
                        flat_true.append(true_token)
                        flat_pred.append(pred_token)
            
            # Get unique labels
            labels = sorted(set(flat_true) | set(flat_pred))
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                flat_true,
                flat_pred,
                average="weighted",
                labels=labels
            )
            
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            metrics["labels"] = labels
            
        elif self.task_name.lower() in ["qa", "question_answering", "table_qa"]:
            # QA metrics (exact match and F1)
            exact_match = 0
            f1_scores = []
            
            for pred, true in zip(predictions, ground_truth):
                if pred.lower() == true.lower():
                    exact_match += 1
                    
                pred_tokens = set(pred.lower().split())
                true_tokens = set(true.lower().split())
                
                if not true_tokens:
                    continue
                    
                common_tokens = pred_tokens.intersection(true_tokens)
                
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(true_tokens) if true_tokens else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                    
                f1_scores.append(f1)
                
            metrics["exact_match"] = exact_match / len(ground_truth)
            metrics["f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            
        elif self.task_name.lower() in ["summarization", "translation"]:
            # Text generation metrics
            # BLEU, ROUGE scores would be calculated here
            # These require additional libraries
            metrics["placeholder"] = "Text generation metrics require additional evaluation"
            
        return metrics
        
    def _print_metrics(self, metrics: Dict, model_name: str):
        """Print benchmark metrics"""
        print(f"\nBenchmark results for {model_name}:")
        
        if "accuracy" in metrics:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            
        if "precision" in metrics:
            print(f"Precision: {metrics['precision']:.4f}")
            
        if "recall" in metrics:
            print(f"Recall: {metrics['recall']:.4f}")
            
        if "f1" in metrics:
            print(f"F1 Score: {metrics['f1']:.4f}")
            
        if "exact_match" in metrics:
            print(f"Exact Match: {metrics['exact_match']:.4f}")
            
        # Print timing
        timing = metrics["timing"]
        print(f"\nTiming:")
        print(f"- Total time: {timing['total_time']:.2f} seconds")
        print(f"- Average time: {timing['average_time'] * 1000:.2f} ms per sample")
        print(f"- Processing speed: {timing['samples_per_second']:.2f} samples/second")
        
    def _save_results(self, benchmark_id: str):
        """Save benchmark results to files"""
        # Create result directory
        result_dir = self.output_dir / benchmark_id
        result_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics[benchmark_id], f, indent=2)
            
        # Save predictions
        predictions_df = pd.DataFrame({
            "ground_truth": self.results[benchmark_id]["ground_truth"],
            "prediction": self.results[benchmark_id]["predictions"]
        })
        
        predictions_df.to_csv(
            result_dir / "predictions.csv",
            index=False,
            encoding="utf-8"
        )
        
        # Save metadata
        metadata = {
            "dataset": self.current_dataset,
            "model": self.results[benchmark_id]["model"],
            "timestamp": self.results[benchmark_id]["timestamp"],
            "task": self.task_name
        }
        
        with open(result_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        # Generate visualization if applicable
        if "confusion_matrix" in self.metrics[benchmark_id]:
            self._plot_confusion_matrix(benchmark_id, result_dir)
            
    def _plot_confusion_matrix(self, benchmark_id: str, output_dir: Path):
        """Plot confusion matrix and save to file"""
        cm_data = self.metrics[benchmark_id]["confusion_matrix"]
        matrix = np.array(cm_data["matrix"])
        labels = cm_data["labels"]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title(f"Confusion Matrix: {self.results[benchmark_id]['model']}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close()
        
    def compare_models(self, model_ids: List[str], metric: str = "f1"):
        """Compare multiple models on the same dataset
        
        Args:
            model_ids: List of benchmark IDs to compare
            metric: Metric to use for comparison
        """
        if not model_ids:
            raise ValueError("No model IDs provided for comparison")
            
        comparison = []
        
        for model_id in model_ids:
            if model_id not in self.metrics:
                print(f"Warning: {model_id} not found in metrics")
                continue
                
            model_metrics = self.metrics[model_id]
            
            if metric not in model_metrics:
                print(f"Warning: {metric} not found in metrics for {model_id}")
                continue
                
            comparison.append({
                "model": self.results[model_id]["model"],
                "dataset": self.results[model_id]["dataset"],
                metric: model_metrics[metric],
                "samples_per_second": model_metrics["timing"]["samples_per_second"]
            })
            
        # Convert to dataframe for easier analysis
        comparison_df = pd.DataFrame(comparison)
        
        if not comparison_df.empty:
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(comparison_df))
            bar_width = 0.35
            
            # Plot metric scores
            metric_bars = ax.bar(
                x - bar_width/2,
                comparison_df[metric],
                bar_width,
                label=metric.capitalize()
            )
            
            # Normalize speed for better visualization
            max_speed = comparison_df["samples_per_second"].max()
            normalized_speed = comparison_df["samples_per_second"] / max_speed
            
            # Plot speed
            speed_bars = ax.bar(
                x + bar_width/2,
                normalized_speed,
                bar_width,
                label="Relative Speed"
            )
            
            # Add labels and legend
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.set_title(f"Model Comparison: {metric.capitalize()} vs Speed")
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df["model"], rotation=45, ha="right")
            ax.legend()
            
            # Add value labels
            for bar in metric_bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom"
                )
                
            # Print comparison table
            print("\nModel Comparison:")
            print(comparison_df.to_string(index=False))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"model_comparison_{metric}.png", dpi=300)
            plt.close()
            
            return comparison_df
            
        return None