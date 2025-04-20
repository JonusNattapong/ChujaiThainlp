"""
Testing utilities for Thai NLP components
"""
from typing import List, Dict, Any, Callable
import torch
import numpy as np
from ..validation.validators import ThaiTextValidator, DatasetValidator
from ..evaluation.metrics import ThaiEvaluationMetrics

class ThaiNLPTester:
    """Test suite for Thai NLP components"""
    
    def __init__(self):
        self.text_validator = ThaiTextValidator()
        self.dataset_validator = DatasetValidator()
        self.evaluator = ThaiEvaluationMetrics()
        
    def test_model_robustness(self,
                             model: Callable,
                             test_cases: List[Dict[str, Any]],
                             task_type: str) -> Dict[str, Any]:
        """
        Test model robustness on various input variations
        
        Args:
            model: Model function to test
            test_cases: List of test cases
            task_type: Type of task ('classification', 'token', 'qa')
            
        Returns:
            Dict containing test results
        """
        results = {
            'basic_tests': [],
            'noise_tests': [],
            'edge_cases': [],
            'overall_metrics': {}
        }
        
        # Run basic tests
        for test in test_cases:
            try:
                input_text = test['input']
                expected = test['expected']
                
                # Validate input
                validation = self.text_validator.validate_input_text(input_text)
                if not validation.is_valid:
                    results['basic_tests'].append({
                        'input': input_text,
                        'status': 'invalid_input',
                        'errors': validation.errors
                    })
                    continue
                
                # Get model prediction
                prediction = model(input_text)
                
                # Evaluate result
                if task_type == 'classification':
                    metrics = self.evaluator.text_classification_metrics(
                        [expected],
                        [prediction]
                    )
                elif task_type == 'token':
                    metrics = self.evaluator.token_classification_metrics(
                        [expected],
                        [prediction]
                    )
                else:  # qa
                    metrics = self.evaluator.qa_metrics(
                        [{'answer': expected}],
                        [{'answer': prediction}]
                    )
                
                results['basic_tests'].append({
                    'input': input_text,
                    'expected': expected,
                    'predicted': prediction,
                    'metrics': metrics,
                    'status': 'success' if metrics.get('accuracy', 0) > 0.8 else 'failed'
                })
                
            except Exception as e:
                results['basic_tests'].append({
                    'input': input_text,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Test with noisy input
        noisy_cases = self._generate_noisy_cases(test_cases)
        for test in noisy_cases:
            try:
                prediction = model(test['input'])
                results['noise_tests'].append({
                    'input': test['input'],
                    'expected': test['expected'],
                    'predicted': prediction,
                    'noise_type': test['noise_type']
                })
            except Exception as e:
                results['noise_tests'].append({
                    'input': test['input'],
                    'status': 'error',
                    'error': str(e)
                })
        
        # Test edge cases
        edge_cases = self._generate_edge_cases(task_type)
        for test in edge_cases:
            try:
                prediction = model(test['input'])
                results['edge_cases'].append({
                    'input': test['input'],
                    'predicted': prediction,
                    'case_type': test['case_type']
                })
            except Exception as e:
                results['edge_cases'].append({
                    'input': test['input'],
                    'status': 'error',
                    'error': str(e)
                })
        
        # Calculate overall metrics
        successful_tests = sum(1 for test in results['basic_tests']
                             if test.get('status') == 'success')
        total_tests = len(test_cases)
        
        results['overall_metrics'] = {
            'accuracy': successful_tests / total_tests if total_tests > 0 else 0,
            'robustness_score': self._calculate_robustness_score(results),
            'error_rate': sum(1 for test in results['basic_tests']
                            if test.get('status') == 'error') / total_tests
        }
        
        return results
    
    def _generate_noisy_cases(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate test cases with various types of noise"""
        noisy_cases = []
        
        for test in test_cases:
            input_text = test['input']
            expected = test['expected']
            
            # Add character noise
            noisy_text = self._add_character_noise(input_text)
            noisy_cases.append({
                'input': noisy_text,
                'expected': expected,
                'noise_type': 'character'
            })
            
            # Add spacing noise
            noisy_text = self._add_spacing_noise(input_text)
            noisy_cases.append({
                'input': noisy_text,
                'expected': expected,
                'noise_type': 'spacing'
            })
            
            # Add tone mark variation
            noisy_text = self._add_tone_variation(input_text)
            noisy_cases.append({
                'input': noisy_text,
                'expected': expected,
                'noise_type': 'tone'
            })
            
        return noisy_cases
    
    def _add_character_noise(self, text: str) -> str:
        """Add character-level noise to Thai text"""
        chars = list(text)
        for i in range(len(chars)):
            if np.random.random() < 0.1:  # 10% chance to modify each character
                if chars[i] in 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ':
                    # Replace with similar-looking character
                    similar_chars = {
                        'ก': 'ถ', 'ข': 'ฃ', 'ค': 'ด', 'ง': 'ว',
                        'จ': 'ใ', 'ช': 'ซ', 'ด': 'ต', 'ต': 'ศ',
                        'บ': 'ป', 'ป': 'บ', 'พ': 'ฟ', 'ฟ': 'พ'
                    }
                    chars[i] = similar_chars.get(chars[i], chars[i])
                    
        return ''.join(chars)
    
    def _add_spacing_noise(self, text: str) -> str:
        """Add spacing noise to Thai text"""
        words = text.split()
        noisy_words = []
        
        for word in words:
            if np.random.random() < 0.2:  # 20% chance to add extra space
                word = ' '.join(list(word))
            noisy_words.append(word)
            
        return ' '.join(noisy_words)
    
    def _add_tone_variation(self, text: str) -> str:
        """Add tone mark variations to Thai text"""
        tone_marks = {'่': '้', '้': '๊', '๊': '๋', '๋': '่'}
        chars = list(text)
        
        for i in range(len(chars)):
            if chars[i] in tone_marks and np.random.random() < 0.3:
                chars[i] = tone_marks[chars[i]]
                
        return ''.join(chars)
    
    def _generate_edge_cases(self, task_type: str) -> List[Dict[str, Any]]:
        """Generate edge cases for testing"""
        edge_cases = []
        
        # Empty input
        edge_cases.append({
            'input': '',
            'case_type': 'empty_input'
        })
        
        # Very long input
        edge_cases.append({
            'input': 'ทดสอบ ' * 1000,
            'case_type': 'long_input'
        })
        
        # Mixed script input
        edge_cases.append({
            'input': 'ทดสอบ test การ mix ภาษา',
            'case_type': 'mixed_script'
        })
        
        # Special characters
        edge_cases.append({
            'input': 'ทดสอบ!@#$%^&*()_+',
            'case_type': 'special_chars'
        })
        
        # Repeated characters
        edge_cases.append({
            'input': 'ทดสอบบบบบบ',
            'case_type': 'repeated_chars'
        })
        
        # Task-specific edge cases
        if task_type == 'classification':
            edge_cases.extend([
                {'input': 'อออออ', 'case_type': 'single_char_repeat'},
                {'input': 'ก.ข.ค.', 'case_type': 'abbreviated'}
            ])
        elif task_type == 'token':
            edge_cases.extend([
                {'input': 'กขคงจ', 'case_type': 'no_spaces'},
                {'input': 'ก ข ค ง จ', 'case_type': 'all_spaces'}
            ])
        elif task_type == 'qa':
            edge_cases.extend([
                {'input': 'ทำไม?', 'case_type': 'question_only'},
                {'input': 'ก: ข\nค: ง', 'case_type': 'structured_text'}
            ])
            
        return edge_cases
    
    def _calculate_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall robustness score"""
        # Weight different aspects of robustness
        weights = {
            'basic_accuracy': 0.4,
            'noise_handling': 0.3,
            'edge_case_handling': 0.3
        }
        
        # Calculate basic accuracy
        basic_acc = results['overall_metrics']['accuracy']
        
        # Calculate noise handling score
        noise_tests = results['noise_tests']
        if noise_tests:
            noise_success = sum(1 for test in noise_tests
                              if test.get('status') != 'error')
            noise_score = noise_success / len(noise_tests)
        else:
            noise_score = 0
            
        # Calculate edge case handling score
        edge_tests = results['edge_cases']
        if edge_tests:
            edge_success = sum(1 for test in edge_tests
                             if test.get('status') != 'error')
            edge_score = edge_success / len(edge_tests)
        else:
            edge_score = 0
            
        # Calculate weighted score
        robustness_score = (
            weights['basic_accuracy'] * basic_acc +
            weights['noise_handling'] * noise_score +
            weights['edge_case_handling'] * edge_score
        )
        
        return robustness_score