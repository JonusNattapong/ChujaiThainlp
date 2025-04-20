"""
System management and automation agent for Thai NLP system
"""
from typing import Dict, List, Any, Optional
import os
import psutil
import logging
from datetime import datetime
from .thai_agent import ThaiLanguageAgent
from .learning_engine import ContinuousLearningEngine

class ThaiSystemAgent:
    """Agent for managing and optimizing Thai NLP system"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.language_agent = ThaiLanguageAgent()
        self.learning_engine = ContinuousLearningEngine()
        
        # System monitoring
        self.performance_logs = []
        self.error_logs = []
        self.resource_usage = {}
        
        # System optimization rules
        self.optimization_rules = {
            'memory_threshold': 0.8,  # 80% memory usage
            'cpu_threshold': 0.9,     # 90% CPU usage
            'storage_threshold': 0.85, # 85% storage usage
            'batch_size_adjust': True,
            'model_caching': True
        }
        
        # Initialize logging
        logging.basicConfig(
            filename='thai_system_agent.log',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
    def monitor_system(self) -> Dict[str, Any]:
        """Monitor system health and performance"""
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage(self.base_path)
            
            # Record metrics
            metrics = {
                'memory_usage': memory.percent / 100,
                'cpu_usage': cpu / 100,
                'storage_usage': disk.percent / 100,
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_logs.append(metrics)
            
            # Check for issues
            issues = self._check_system_issues(metrics)
            if issues:
                self._handle_system_issues(issues)
                
            return {
                'status': 'healthy' if not issues else 'issues_detected',
                'metrics': metrics,
                'issues': issues
            }
            
        except Exception as e:
            error = f"System monitoring error: {str(e)}"
            logging.error(error)
            self.error_logs.append({
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            return {'status': 'error', 'message': error}
            
    def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance"""
        optimizations = {
            'actions_taken': [],
            'improvements': {}
        }
        
        # Check resource usage trends
        if len(self.performance_logs) >= 2:
            recent_metrics = self.performance_logs[-1]
            
            # Optimize memory usage
            if recent_metrics['memory_usage'] > self.optimization_rules['memory_threshold']:
                if self._optimize_memory_usage():
                    optimizations['actions_taken'].append('memory_optimization')
                    
            # Optimize CPU usage
            if recent_metrics['cpu_usage'] > self.optimization_rules['cpu_threshold']:
                if self._optimize_cpu_usage():
                    optimizations['actions_taken'].append('cpu_optimization')
                    
            # Optimize storage
            if recent_metrics['storage_usage'] > self.optimization_rules['storage_threshold']:
                if self._optimize_storage():
                    optimizations['actions_taken'].append('storage_optimization')
                    
        # Optimize model performance
        model_optimizations = self._optimize_models()
        optimizations['actions_taken'].extend(model_optimizations)
        
        # Calculate improvements
        if self.performance_logs:
            before = self.performance_logs[-2] if len(self.performance_logs) > 1 else None
            after = self.performance_logs[-1]
            
            if before:
                optimizations['improvements'] = {
                    'memory': before['memory_usage'] - after['memory_usage'],
                    'cpu': before['cpu_usage'] - after['cpu_usage']
                }
                
        return optimizations
        
    def manage_resources(self) -> Dict[str, Any]:
        """Manage system resources"""
        resources = {
            'allocated': {},
            'available': {},
            'recommendations': []
        }
        
        try:
            # Get memory information
            memory = psutil.virtual_memory()
            resources['allocated']['memory'] = memory.used
            resources['available']['memory'] = memory.available
            
            # Get CPU information
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            resources['allocated']['cpu'] = sum(cpu_percent) / len(cpu_percent)
            resources['available']['cpu'] = 100 - resources['allocated']['cpu']
            
            # Get storage information
            disk = psutil.disk_usage(self.base_path)
            resources['allocated']['storage'] = disk.used
            resources['available']['storage'] = disk.free
            
            # Make recommendations
            if memory.percent > 75:
                resources['recommendations'].append({
                    'type': 'memory',
                    'action': 'Consider increasing batch size or enabling model offloading'
                })
                
            if resources['allocated']['cpu'] > 80:
                resources['recommendations'].append({
                    'type': 'cpu',
                    'action': 'Consider enabling model quantization or reducing concurrent processes'
                })
                
            if disk.percent > 80:
                resources['recommendations'].append({
                    'type': 'storage',
                    'action': 'Consider cleaning cached models or implementing model pruning'
                })
                
            return resources
            
        except Exception as e:
            error = f"Resource management error: {str(e)}"
            logging.error(error)
            return {'status': 'error', 'message': error}
            
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system performance and generate insights"""
        analysis = {
            'performance_metrics': {},
            'trends': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        if not self.performance_logs:
            return analysis
            
        # Calculate average metrics
        metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'storage_usage': []
        }
        
        for log in self.performance_logs:
            for key in metrics:
                metrics[key].append(log[key])
                
        analysis['performance_metrics'] = {
            key: sum(values) / len(values)
            for key, values in metrics.items()
        }
        
        # Analyze trends
        for key, values in metrics.items():
            if len(values) > 1:
                trend = values[-1] - values[0]
                analysis['trends'][key] = {
                    'direction': 'increasing' if trend > 0 else 'decreasing',
                    'magnitude': abs(trend)
                }
                
        # Identify bottlenecks
        recent_metrics = self.performance_logs[-1]
        if recent_metrics['memory_usage'] > 0.8:
            analysis['bottlenecks'].append('high_memory_usage')
        if recent_metrics['cpu_usage'] > 0.8:
            analysis['bottlenecks'].append('high_cpu_usage')
        if recent_metrics['storage_usage'] > 0.8:
            analysis['bottlenecks'].append('high_storage_usage')
            
        # Generate recommendations
        self._generate_recommendations(analysis)
        
        return analysis
        
    def _check_system_issues(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for system issues"""
        issues = []
        
        # Check memory usage
        if metrics['memory_usage'] > self.optimization_rules['memory_threshold']:
            issues.append({
                'type': 'memory',
                'severity': 'high' if metrics['memory_usage'] > 0.9 else 'medium',
                'message': 'High memory usage detected'
            })
            
        # Check CPU usage
        if metrics['cpu_usage'] > self.optimization_rules['cpu_threshold']:
            issues.append({
                'type': 'cpu',
                'severity': 'high' if metrics['cpu_usage'] > 0.95 else 'medium',
                'message': 'High CPU usage detected'
            })
            
        # Check storage usage
        if metrics['storage_usage'] > self.optimization_rules['storage_threshold']:
            issues.append({
                'type': 'storage',
                'severity': 'high' if metrics['storage_usage'] > 0.95 else 'medium',
                'message': 'High storage usage detected'
            })
            
        return issues
        
    def _handle_system_issues(self, issues: List[Dict[str, Any]]):
        """Handle detected system issues"""
        for issue in issues:
            if issue['severity'] == 'high':
                if issue['type'] == 'memory':
                    self._optimize_memory_usage()
                elif issue['type'] == 'cpu':
                    self._optimize_cpu_usage()
                elif issue['type'] == 'storage':
                    self._optimize_storage()
                    
            logging.warning(f"System issue detected: {issue['message']}")
            
    def _optimize_memory_usage(self) -> bool:
        """Optimize memory usage"""
        try:
            # Implement memory optimization strategies
            if self.optimization_rules['model_caching']:
                # Clear model cache
                torch.cuda.empty_cache()
                
            if self.optimization_rules['batch_size_adjust']:
                # Reduce batch size if needed
                current_batch_size = self.language_agent.batch_size
                self.language_agent.batch_size = max(1, current_batch_size // 2)
                
            return True
            
        except Exception as e:
            logging.error(f"Memory optimization error: {str(e)}")
            return False
            
    def _optimize_cpu_usage(self) -> bool:
        """Optimize CPU usage"""
        try:
            # Implement CPU optimization strategies
            if hasattr(self.language_agent, 'model'):
                # Enable dynamic quantization
                self.language_agent.model = torch.quantization.quantize_dynamic(
                    self.language_agent.model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
            return True
            
        except Exception as e:
            logging.error(f"CPU optimization error: {str(e)}")
            return False
            
    def _optimize_storage(self) -> bool:
        """Optimize storage usage"""
        try:
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Remove old log files
            self._cleanup_old_logs()
            
            return True
            
        except Exception as e:
            logging.error(f"Storage optimization error: {str(e)}")
            return False
            
    def _optimize_models(self) -> List[str]:
        """Optimize model performance"""
        optimizations = []
        
        try:
            # Get model performance insights
            insights = self.learning_engine.get_learning_insights()
            
            # Check if models need adaptation
            for task_type, metrics in insights['performance_trends'].items():
                if metrics.get('trend', 0) < 0:  # Declining performance
                    self.learning_engine.update_adaptation_rules(
                        task_type,
                        {'trigger_adaptation': True}
                    )
                    optimizations.append(f'{task_type}_model_adaptation')
                    
            return optimizations
            
        except Exception as e:
            logging.error(f"Model optimization error: {str(e)}")
            return optimizations
            
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = os.path.join(self.base_path, 'temp')
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {str(e)}")
                    
    def _cleanup_old_logs(self):
        """Clean up old log files"""
        log_dir = os.path.join(self.base_path, 'logs')
        if os.path.exists(log_dir):
            current_time = datetime.now().timestamp()
            for file in os.listdir(log_dir):
                file_path = os.path.join(log_dir, file)
                try:
                    if os.path.isfile(file_path):
                        # Remove logs older than 30 days
                        if current_time - os.path.getctime(file_path) > 30 * 86400:
                            os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error deleting log {file_path}: {str(e)}")
                    
    def _generate_recommendations(self, analysis: Dict[str, Any]):
        """Generate system recommendations"""
        recommendations = []
        
        # Memory recommendations
        if 'high_memory_usage' in analysis['bottlenecks']:
            recommendations.append({
                'type': 'memory',
                'action': 'Enable model offloading or reduce batch size',
                'priority': 'high'
            })
            
        # CPU recommendations
        if 'high_cpu_usage' in analysis['bottlenecks']:
            recommendations.append({
                'type': 'cpu',
                'action': 'Enable model quantization or limit concurrent processes',
                'priority': 'high'
            })
            
        # Storage recommendations
        if 'high_storage_usage' in analysis['bottlenecks']:
            recommendations.append({
                'type': 'storage',
                'action': 'Implement model pruning or clean cached files',
                'priority': 'high'
            })
            
        # Performance trend recommendations
        for metric, trend in analysis['trends'].items():
            if trend['direction'] == 'increasing' and trend['magnitude'] > 0.1:
                recommendations.append({
                    'type': metric,
                    'action': f'Investigate increasing {metric.replace("_", " ")}',
                    'priority': 'medium'
                })
                
        analysis['recommendations'] = recommendations