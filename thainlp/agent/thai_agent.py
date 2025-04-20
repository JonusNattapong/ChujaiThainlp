"""
Thai Language Processing Agent with autonomous capabilities
"""
from typing import List, Dict, Any, Optional, Callable
import torch
from thainlp.thai_preprocessor import ThaiTextPreprocessor
from .error_correction_agent import ThaiErrorCorrectionAgent

class ThaiLanguageAgent:
    """Autonomous agent for Thai language processing tasks"""
    
    def __init__(self,
                 task_handlers: Optional[Dict[str, Callable]] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Thai language agent
        
        Args:
            task_handlers: Custom task handling functions
            device: Device to run models on
        """
        # Initialize components
        self.preprocessor = ThaiTextPreprocessor()
        self.error_corrector = ThaiErrorCorrectionAgent()
        
        # Task handlers
        self.task_handlers = task_handlers or {}
        
        # Learning memory
        self.interaction_memory = []
        self.performance_history = []
        
        # Task routing rules
        self.task_rules = {
            'correction': lambda x: any(word in x.lower() for word in 
                ['แก้ไข', 'ผิด', 'พิมพ์']),
            'analysis': lambda x: any(word in x.lower() for word in 
                ['วิเคราะห์', 'ตรวจสอบ'])
        }
    
    def process_text(self, text: str, task_hint: Optional[str] = None) -> Dict[str, Any]:
        """Process Thai text based on task type"""
        task_type = task_hint or self._determine_task(text)
        
        if task_type == 'correction':
            return self.error_corrector.correct_text(text)
        elif task_type == 'analysis':
            return self.error_corrector.analyze_errors(text)
        elif task_type in self.task_handlers:
            return self.task_handlers[task_type](text)
        else:
            return {
                'status': 'error',
                'message': f'Unknown task type: {task_type}'
            }
    
    def _determine_task(self, text: str) -> str:
        """Determine appropriate task type from text"""
        for task_type, rule in self.task_rules.items():
            if rule(text):
                return task_type
        return 'correction'  # Default task