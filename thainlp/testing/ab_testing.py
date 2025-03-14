"""
Advanced A/B Testing System for ThaiNLP
"""
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
import threading
import json
import logging
import uuid
import os
import random
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

class Experiment:
    """Represents a single A/B test experiment"""
    
    def __init__(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        success_metric: str,
        min_sample_size: int = 100,
        significance_level: float = 0.05
    ):
        """Initialize experiment
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variant configurations
            success_metric: Metric to use for success evaluation
            min_sample_size: Minimum sample size for statistical significance
            significance_level: Statistical significance level (alpha)
        """
        self.name = name
        self.description = description
        self.variants = variants
        self.success_metric = success_metric
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.start_time = datetime.now()
        self.end_time = None
        self.is_active = True
        self.results = defaultdict(list)
        self._lock = threading.Lock()
        
    def record_result(
        self,
        variant_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record experiment result
        
        Args:
            variant_id: Variant identifier
            metrics: Metrics from the experiment run
            metadata: Additional metadata about the run
        """
        with self._lock:
            self.results[variant_id].append({
                'metrics': metrics,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            })
            
    def get_variant_results(
        self,
        variant_id: str
    ) -> List[Dict[str, Any]]:
        """Get results for a specific variant
        
        Args:
            variant_id: Variant identifier
            
        Returns:
            List of result records for the variant
        """
        with self._lock:
            return self.results.get(variant_id, [])
            
    def get_variant_metrics(
        self,
        variant_id: str,
        metric: Optional[str] = None
    ) -> List[float]:
        """Get metric values for a specific variant
        
        Args:
            variant_id: Variant identifier
            metric: Specific metric to retrieve (default: success_metric)
            
        Returns:
            List of metric values
        """
        metric = metric or self.success_metric
        results = self.get_variant_results(variant_id)
        return [r['metrics'].get(metric, 0) for r in results]
        
    def get_sample_size(self, variant_id: str) -> int:
        """Get sample size for a variant
        
        Args:
            variant_id: Variant identifier
            
        Returns:
            Number of samples for the variant
        """
        return len(self.get_variant_results(variant_id))
        
    def has_sufficient_data(self) -> bool:
        """Check if experiment has sufficient data for analysis
        
        Returns:
            Whether all variants have sufficient data
        """
        return all(
            self.get_sample_size(v['id']) >= self.min_sample_size
            for v in self.variants
        )
        
    def end_experiment(self):
        """End the experiment"""
        self.is_active = False
        self.end_time = datetime.now()
        
    def get_experiment_duration(self) -> timedelta:
        """Get experiment duration
        
        Returns:
            Duration of the experiment
        """
        end = self.end_time or datetime.now()
        return end - self.start_time
        
    def get_variant_stats(
        self,
        variant_id: str,
        metric: Optional[str] = None
    ) -> Dict[str, float]:
        """Get statistics for a variant's metric
        
        Args:
            variant_id: Variant identifier
            metric: Specific metric to analyze (default: success_metric)
            
        Returns:
            Dictionary of statistics
        """
        values = self.get_variant_metrics(variant_id, metric)
        
        if not values:
            return {}
            
        return {
            'count': len(values),
            'mean': statistics.mean(values) if values else 0,
            'median': statistics.median(values) if values else 0,
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values) if values else 0,
            'max': max(values) if values else 0
        }
        
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of experiment results
        
        Returns:
            Dictionary with experiment summary
        """
        variant_stats = {}
        for variant in self.variants:
            variant_id = variant['id']
            stats = self.get_variant_stats(variant_id)
            variant_stats[variant_id] = {
                'config': variant,
                'stats': stats,
                'sample_size': self.get_sample_size(variant_id)
            }
            
        return {
            'name': self.name,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': str(self.get_experiment_duration()),
            'is_active': self.is_active,
            'variants': variant_stats,
            'has_sufficient_data': self.has_sufficient_data(),
            'statistical_analysis': self.perform_statistical_analysis()
        }
        
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results
        
        Returns:
            Dictionary with statistical analysis results
        """
        if len(self.variants) < 2:
            return {'error': 'Need at least 2 variants for comparison'}
            
        # Get control variant (first one)
        control_id = self.variants[0]['id']
        control_values = self.get_variant_metrics(control_id)
        
        if not control_values:
            return {'error': 'No data for control variant'}
            
        results = {}
        
        # Compare each treatment variant to control
        for variant in self.variants[1:]:
            variant_id = variant['id']
            variant_values = self.get_variant_metrics(variant_id)
            
            if not variant_values:
                results[variant_id] = {'error': 'No data for variant'}
                continue
                
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(
                control_values,
                variant_values,
                equal_var=False  # Welch's t-test
            )
            
            # Calculate effect size (Cohen's d)
            control_mean = statistics.mean(control_values)
            variant_mean = statistics.mean(variant_values)
            pooled_std = np.sqrt(
                (
                    (len(control_values) - 1) * statistics.variance(control_values) +
                    (len(variant_values) - 1) * statistics.variance(variant_values)
                ) / (len(control_values) + len(variant_values) - 2)
            )
            
            effect_size = (variant_mean - control_mean) / pooled_std if pooled_std else 0
            
            # Determine if result is statistically significant
            is_significant = p_value < self.significance_level
            
            # Calculate relative improvement
            relative_improvement = (
                (variant_mean - control_mean) / control_mean * 100
                if control_mean else 0
            )
            
            results[variant_id] = {
                'control_mean': control_mean,
                'variant_mean': variant_mean,
                'difference': variant_mean - control_mean,
                'relative_improvement': relative_improvement,
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'is_significant': is_significant,
                'sample_sizes': {
                    'control': len(control_values),
                    'variant': len(variant_values)
                }
            }
            
        return results
        
    def generate_report(self, output_dir: str) -> str:
        """Generate detailed report of experiment results
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Path to generated report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"{self.name.replace(' ', '_')}_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        # Get experiment summary
        summary = self.get_experiment_summary()
        
        # Create pandas DataFrames for analysis
        variant_data = []
        for variant in self.variants:
            variant_id = variant['id']
            metrics = self.get_variant_metrics(variant_id)
            for i, value in enumerate(metrics):
                variant_data.append({
                    'variant_id': variant_id,
                    'variant_name': variant.get('name', variant_id),
                    'sample_index': i,
                    'metric_value': value
                })
                
        if not variant_data:
            return "No data available for report"
            
        df = pd.DataFrame(variant_data)
        
        # Generate HTML report
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Experiment Report: {self.name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".significant { color: green; font-weight: bold; }",
            ".not-significant { color: red; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Experiment Report: {self.name}</h1>",
            f"<p><strong>Description:</strong> {self.description}</p>",
            f"<p><strong>Start Time:</strong> {self.start_time.isoformat()}</p>",
            f"<p><strong>End Time:</strong> {self.end_time.isoformat() if self.end_time else 'Ongoing'}</p>",
            f"<p><strong>Duration:</strong> {self.get_experiment_duration()}</p>",
            f"<p><strong>Status:</strong> {'Active' if self.is_active else 'Completed'}</p>",
            "<h2>Variants</h2>",
            "<table>",
            "<tr><th>Variant ID</th><th>Name</th><th>Sample Size</th><th>Mean</th><th>Median</th><th>StdDev</th></tr>"
        ]
        
        # Add variant rows
        for variant in self.variants:
            variant_id = variant['id']
            stats = self.get_variant_stats(variant_id)
            html.append(
                f"<tr><td>{variant_id}</td><td>{variant.get('name', variant_id)}</td>"
                f"<td>{stats.get('count', 0)}</td><td>{stats.get('mean', 0):.4f}</td>"
                f"<td>{stats.get('median', 0):.4f}</td><td>{stats.get('stdev', 0):.4f}</td></tr>"
            )
            
        html.append("</table>")
        
        # Add statistical analysis
        html.append("<h2>Statistical Analysis</h2>")
        
        analysis = summary.get('statistical_analysis', {})
        if 'error' in analysis:
            html.append(f"<p>Error: {analysis['error']}</p>")
        else:
            html.append("<table>")
            html.append("<tr><th>Variant</th><th>Control Mean</th><th>Variant Mean</th><th>Difference</th><th>Improvement</th><th>p-value</th><th>Significant?</th></tr>")
            
            for variant_id, result in analysis.items():
                if 'error' in result:
                    continue
                    
                significant_class = "significant" if result['is_significant'] else "not-significant"
                significant_text = "Yes" if result['is_significant'] else "No"
                
                html.append(
                    f"<tr><td>{variant_id}</td>"
                    f"<td>{result['control_mean']:.4f}</td>"
                    f"<td>{result['variant_mean']:.4f}</td>"
                    f"<td>{result['difference']:.4f}</td>"
                    f"<td>{result['relative_improvement']:.2f}%</td>"
                    f"<td>{result['p_value']:.4f}</td>"
                    f"<td class='{significant_class}'>{significant_text}</td></tr>"
                )
                
            html.append("</table>")
            
        # Generate plots
        if len(df) > 0:
            # Distribution plot
            plt.figure(figsize=(10, 6))
            for variant in self.variants:
                variant_id = variant['id']
                variant_name = variant.get('name', variant_id)
                variant_metrics = self.get_variant_metrics(variant_id)
                if variant_metrics:
                    plt.hist(
                        variant_metrics,
                        alpha=0.5,
                        bins=20,
                        label=variant_name
                    )
                    
            plt.title(f"Distribution of {self.success_metric}")
            plt.xlabel(self.success_metric)
            plt.ylabel("Frequency")
            plt.legend()
            
            # Save plot
            plot_filename = f"{self.name.replace(' ', '_')}_{timestamp}_dist.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            
            # Add plot to report
            html.append("<h2>Metric Distribution</h2>")
            html.append(f"<img src='{plot_filename}' alt='Metric Distribution' style='max-width:100%;'>")
            
            # Box plot
            plt.figure(figsize=(10, 6))
            variant_data = []
            variant_names = []
            
            for variant in self.variants:
                variant_id = variant['id']
                variant_name = variant.get('name', variant_id)
                variant_metrics = self.get_variant_metrics(variant_id)
                if variant_metrics:
                    variant_data.append(variant_metrics)
                    variant_names.append(variant_name)
                    
            if variant_data:
                plt.boxplot(variant_data, labels=variant_names)
                plt.title(f"Box Plot of {self.success_metric}")
                plt.ylabel(self.success_metric)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Save plot
                boxplot_filename = f"{self.name.replace(' ', '_')}_{timestamp}_box.png"
                boxplot_path = os.path.join(output_dir, boxplot_filename)
                plt.savefig(boxplot_path)
                plt.close()
                
                # Add plot to report
                html.append("<h2>Box Plot Comparison</h2>")
                html.append(f"<img src='{boxplot_filename}' alt='Box Plot Comparison' style='max-width:100%;'>")
                
        # Finish HTML
        html.append("</body>")
        html.append("</html>")
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write('\n'.join(html))
            
        return report_path

class ExperimentManager:
    """Manage multiple A/B test experiments"""
    
    def __init__(self, data_dir: str):
        """Initialize experiment manager
        
        Args:
            data_dir: Directory to store experiment data
        """
        self.data_dir = data_dir
        self.experiments = {}
        self.user_assignments = {}
        self._lock = threading.Lock()
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'reports'), exist_ok=True)
        
        # Load existing experiments
        self._load_experiments()
        
    def _load_experiments(self):
        """Load experiments from disk"""
        experiments_dir = os.path.join(self.data_dir, 'experiments')
        os.makedirs(experiments_dir, exist_ok=True)
        
        for filename in os.listdir(experiments_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(experiments_dir, filename), 'r') as f:
                        data = json.load(f)
                        
                    # Create experiment
                    experiment = Experiment(
                        name=data['name'],
                        description=data['description'],
                        variants=data['variants'],
                        success_metric=data['success_metric'],
                        min_sample_size=data.get('min_sample_size', 100),
                        significance_level=data.get('significance_level', 0.05)
                    )
                    
                    # Restore experiment state
                    experiment.start_time = datetime.fromisoformat(data['start_time'])
                    if data.get('end_time'):
                        experiment.end_time = datetime.fromisoformat(data['end_time'])
                    experiment.is_active = data.get('is_active', True)
                    
                    # Restore results
                    for variant_id, results in data.get('results', {}).items():
                        for result in results:
                            experiment.record_result(
                                variant_id,
                                result['metrics'],
                                result.get('metadata')
                            )
                            
                    # Add to experiments
                    self.experiments[experiment.name] = experiment
                    
                except Exception as e:
                    logging.error(f"Error loading experiment {filename}: {e}")
                    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk
        
        Args:
            experiment: Experiment to save
        """
        experiments_dir = os.path.join(self.data_dir, 'experiments')
        os.makedirs(experiments_dir, exist_ok=True)
        
        # Create experiment data
        data = {
            'name': experiment.name,
            'description': experiment.description,
            'variants': experiment.variants,
            'success_metric': experiment.success_metric,
            'min_sample_size': experiment.min_sample_size,
            'significance_level': experiment.significance_level,
            'start_time': experiment.start_time.isoformat(),
            'end_time': experiment.end_time.isoformat() if experiment.end_time else None,
            'is_active': experiment.is_active,
            'results': {
                variant_id: [
                    {
                        'metrics': result['metrics'],
                        'metadata': result.get('metadata', {}),
                        'timestamp': result['timestamp']
                    }
                    for result in results
                ]
                for variant_id, results in experiment.results.items()
            }
        }
        
        # Save to file
        filename = f"{experiment.name.replace(' ', '_')}.json"
        with open(os.path.join(experiments_dir, filename), 'w') as f:
            json.dump(data, f, indent=2)
            
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        success_metric: str,
        min_sample_size: int = 100,
        significance_level: float = 0.05
    ) -> Experiment:
        """Create new experiment
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variant configurations
            success_metric: Metric to use for success evaluation
            min_sample_size: Minimum sample size for statistical significance
            significance_level: Statistical significance level (alpha)
            
        Returns:
            Created experiment
        """
        with self._lock:
            if name in self.experiments:
                raise ValueError(f"Experiment '{name}' already exists")
                
            # Ensure each variant has an ID
            for i, variant in enumerate(variants):
                if 'id' not in variant:
                    variant['id'] = f"variant_{i}"
                    
            # Create experiment
            experiment = Experiment(
                name=name,
                description=description,
                variants=variants,
                success_metric=success_metric,
                min_sample_size=min_sample_size,
                significance_level=significance_level
            )
            
            # Add to experiments
            self.experiments[name] = experiment
            
            # Save experiment
            self._save_experiment(experiment)
            
            return experiment
            
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment if found, None otherwise
        """
        return self.experiments.get(name)
        
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments
        
        Returns:
            List of experiment summaries
        """
        return [
            {
                'name': exp.name,
                'description': exp.description,
                'start_time': exp.start_time.isoformat(),
                'end_time': exp.end_time.isoformat() if exp.end_time else None,
                'is_active': exp.is_active,
                'variants': len(exp.variants),
                'total_samples': sum(
                    len(results) for results in exp.results.values()
                )
            }
            for exp in self.experiments.values()
        ]
        
    def end_experiment(self, name: str) -> bool:
        """End experiment
        
        Args:
            name: Experiment name
            
        Returns:
            Whether experiment was ended
        """
        with self._lock:
            experiment = self.get_experiment(name)
            if not experiment:
                return False
                
            experiment.end_experiment()
            self._save_experiment(experiment)
            return True
            
    def delete_experiment(self, name: str) -> bool:
        """Delete experiment
        
        Args:
            name: Experiment name
            
        Returns:
            Whether experiment was deleted
        """
        with self._lock:
            if name not in self.experiments:
                return False
                
            # Remove from experiments
            del self.experiments[name]
            
            # Delete experiment file
            filename = f"{name.replace(' ', '_')}.json"
            file_path = os.path.join(self.data_dir, 'experiments', filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return True
            
    def get_variant_for_user(
        self,
        experiment_name: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get variant for user
        
        Args:
            experiment_name: Experiment name
            user_id: User identifier
            
        Returns:
            Variant configuration if found, None otherwise
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment or not experiment.is_active:
            return None
            
        # Check if user already assigned
        assignment_key = f"{experiment_name}_{user_id}"
        if assignment_key in self.user_assignments:
            variant_id = self.user_assignments[assignment_key]
            for variant in experiment.variants:
                if variant['id'] == variant_id:
                    return variant
                    
        # Assign user to variant
        with self._lock:
            # Deterministic assignment based on user_id
            variant_index = hash(user_id) % len(experiment.variants)
            variant = experiment.variants[variant_index]
            
            # Store assignment
            self.user_assignments[assignment_key] = variant['id']
            
            return variant
            
    def record_experiment_result(
        self,
        experiment_name: str,
        user_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record experiment result
        
        Args:
            experiment_name: Experiment name
            user_id: User identifier
            metrics: Metrics from the experiment run
            metadata: Additional metadata about the run
            
        Returns:
            Whether result was recorded
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            return False
            
        # Get variant for user
        assignment_key = f"{experiment_name}_{user_id}"
        variant_id = self.user_assignments.get(assignment_key)
        
        if not variant_id:
            # Try to assign variant
            variant = self.get_variant_for_user(experiment_name, user_id)
            if not variant:
                return False
            variant_id = variant['id']
            
        # Record result
        experiment.record_result(variant_id, metrics, metadata)
        
        # Save experiment
        self._save_experiment(experiment)
        
        return True
        
    def generate_experiment_report(
        self,
        experiment_name: str
    ) -> Optional[str]:
        """Generate report for experiment
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Path to generated report if successful, None otherwise
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            return None
            
        reports_dir = os.path.join(self.data_dir, 'reports')
        return experiment.generate_report(reports_dir)
        
    def get_experiment_summary(
        self,
        experiment_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary of experiment results
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Dictionary with experiment summary if found, None otherwise
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            return None
            
        return experiment.get_experiment_summary() 