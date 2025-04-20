"""
Continuous learning engine for Thai Language Agent
"""
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
from ..evaluation.metrics import ThaiEvaluationMetrics

class ContinuousLearningEngine:
    """Engine for continuous learning and adaptation"""
    
    def __init__(self):
        self.memory_buffer = []
        self.learning_stats = defaultdict(list)
        self.adaptation_rules = {}
        self.confidence_thresholds = {
            'classification': 0.8,
            'qa': 0.7,
            'pos': 0.9
        }
        
    def store_experience(self,
                        task_type: str,
                        input_data: Any,
                        output: Any,
                        feedback: Optional[Dict[str, Any]] = None):
        """
        Store learning experience
        
        Args:
            task_type: Type of task
            input_data: Input data
            output: Model output
            feedback: Optional user feedback
        """
        experience = {
            'task_type': task_type,
            'input': input_data,
            'output': output,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self._calculate_performance(task_type, output, feedback)
        }
        
        self.memory_buffer.append(experience)
        self._update_learning_stats(experience)
        
        # Trigger adaptation if needed
        if self._should_adapt(task_type):
            self._adapt_models(task_type)
            
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning experiences"""
        insights = {
            'performance_trends': self._analyze_performance_trends(),
            'error_patterns': self._analyze_error_patterns(),
            'adaptation_history': self._get_adaptation_history(),
            'confidence_metrics': self._calculate_confidence_metrics()
        }
        return insights
        
    def update_adaptation_rules(self,
                              task_type: str,
                              new_rules: Dict[str, Any]):
        """
        Update adaptation rules for a task
        
        Args:
            task_type: Type of task
            new_rules: New adaptation rules
        """
        if task_type in self.adaptation_rules:
            self.adaptation_rules[task_type].update(new_rules)
        else:
            self.adaptation_rules[task_type] = new_rules
            
    def _calculate_performance(self,
                             task_type: str,
                             output: Any,
                             feedback: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        if feedback and 'correct_answer' in feedback:
            evaluator = ThaiEvaluationMetrics()
            
            if task_type == 'classification':
                result = evaluator.text_classification_metrics(
                    [feedback['correct_answer']],
                    [output['label']]
                )
                metrics['accuracy'] = result['accuracy']
                metrics['f1'] = result['f1']
                
            elif task_type == 'qa':
                result = evaluator.qa_metrics(
                    [{'answer': feedback['correct_answer']}],
                    [{'answer': output['answer']}]
                )
                metrics['exact_match'] = result['exact_match']
                metrics['f1'] = result['f1']
                
            elif task_type == 'pos':
                if 'correct_tags' in feedback:
                    result = evaluator.token_classification_metrics(
                        [feedback['correct_tags']],
                        [output['tags']]
                    )
                    metrics.update(result['token_level'])
                    
        return metrics
        
    def _update_learning_stats(self, experience: Dict[str, Any]):
        """Update learning statistics"""
        task_type = experience['task_type']
        
        if 'performance_metrics' in experience:
            for metric, value in experience['performance_metrics'].items():
                self.learning_stats[f'{task_type}_{metric}'].append(value)
                
        # Keep only recent statistics
        max_history = 1000
        for key in self.learning_stats:
            if len(self.learning_stats[key]) > max_history:
                self.learning_stats[key] = self.learning_stats[key][-max_history:]
                
    def _should_adapt(self, task_type: str) -> bool:
        """Determine if model adaptation is needed"""
        if task_type not in self.confidence_thresholds:
            return False
            
        # Check recent performance
        recent_experiences = [
            exp for exp in self.memory_buffer[-100:]
            if exp['task_type'] == task_type and 'performance_metrics' in exp
        ]
        
        if not recent_experiences:
            return False
            
        # Calculate average performance
        avg_performance = np.mean([
            exp['performance_metrics'].get('accuracy', 0)
            for exp in recent_experiences
        ])
        
        return avg_performance < self.confidence_thresholds[task_type]
        
    def _adapt_models(self, task_type: str):
        """Adapt models based on learning experiences"""
        # Get relevant experiences
        experiences = [
            exp for exp in self.memory_buffer
            if exp['task_type'] == task_type and 
            exp.get('feedback', {}).get('correct_answer')
        ]
        
        if not experiences:
            return
            
        # Prepare training data
        training_data = []
        for exp in experiences:
            if task_type == 'classification':
                training_data.append({
                    'text': exp['input'],
                    'label': exp['feedback']['correct_answer']
                })
            elif task_type == 'qa':
                training_data.append({
                    'question': exp['input'],
                    'answer': exp['feedback']['correct_answer'],
                    'context': exp['feedback'].get('context', '')
                })
            elif task_type == 'pos':
                if 'correct_tags' in exp['feedback']:
                    training_data.append({
                        'tokens': exp['input'].split(),
                        'tags': exp['feedback']['correct_tags']
                    })
                    
        # Update adaptation rules
        if task_type in self.adaptation_rules:
            rules = self.adaptation_rules[task_type]
            if 'min_samples' in rules and len(training_data) < rules['min_samples']:
                return
                
            if 'max_samples' in rules:
                training_data = training_data[-rules['max_samples']:]
                
        return training_data
        
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        trends = {}
        
        for key, values in self.learning_stats.items():
            if len(values) > 1:
                # Calculate moving average
                window_size = min(10, len(values))
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                
                # Calculate trend
                trend = np.polyfit(range(len(moving_avg)), moving_avg, 1)[0]
                
                trends[key] = {
                    'current': values[-1],
                    'trend': trend,
                    'improvement': trend > 0
                }
                
        return trends
        
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze common error patterns"""
        error_patterns = defaultdict(int)
        
        for experience in self.memory_buffer:
            if experience.get('feedback', {}).get('is_error'):
                error_type = experience['feedback'].get('error_type', 'unknown')
                error_patterns[error_type] += 1
                
        return dict(error_patterns)
        
    def _get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get history of model adaptations"""
        adaptations = []
        
        current_task = None
        current_samples = []
        
        for experience in self.memory_buffer:
            if experience.get('triggered_adaptation'):
                if current_task == experience['task_type']:
                    current_samples.append(experience)
                else:
                    if current_task and current_samples:
                        adaptations.append({
                            'task_type': current_task,
                            'samples': len(current_samples),
                            'timestamp': current_samples[-1]['timestamp']
                        })
                    current_task = experience['task_type']
                    current_samples = [experience]
                    
        return adaptations
        
    def _calculate_confidence_metrics(self) -> Dict[str, float]:
        """Calculate confidence metrics for each task"""
        confidence_metrics = {}
        
        for task_type in set(exp['task_type'] for exp in self.memory_buffer):
            task_experiences = [
                exp for exp in self.memory_buffer
                if exp['task_type'] == task_type and 'performance_metrics' in exp
            ]
            
            if task_experiences:
                avg_performance = np.mean([
                    exp['performance_metrics'].get('accuracy', 0)
                    for exp in task_experiences
                ])
                confidence_metrics[task_type] = avg_performance
                
        return confidence_metrics