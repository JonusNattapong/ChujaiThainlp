"""
Input and output validation for Thai NLP tasks
"""
from typing import List, Dict, Any, Union
import re
from dataclasses import dataclass
import torch
from ..tokenization import word_tokenize

@dataclass
class ValidationResult:
    """Validation result container"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ThaiTextValidator:
    """Validator for Thai text input and output"""
    
    @staticmethod
    def validate_input_text(text: str) -> ValidationResult:
        """
        Validate Thai text input
        
        Args:
            text: Input text to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check if text is empty or None
        if not text:
            errors.append("Input text cannot be empty")
            return ValidationResult(False, errors, warnings)
            
        # Check if text is string
        if not isinstance(text, str):
            errors.append(f"Input must be string, got {type(text)}")
            return ValidationResult(False, errors, warnings)
            
        # Check for Thai characters
        thai_pattern = re.compile(r'[\u0E00-\u0E7F]')
        if not thai_pattern.search(text):
            errors.append("Text contains no Thai characters")
            
        # Check text length
        if len(text) < 2:
            warnings.append("Text is very short")
        elif len(text) > 10000:
            warnings.append("Text is very long, may impact performance")
            
        # Check for common input issues
        if '\u200b' in text:  # Zero-width space
            warnings.append("Text contains zero-width spaces")
            
        if '๏' in text:  # Special Thai character
            warnings.append("Text contains special Thai character '๏'")
            
        # Validate character combinations
        prev_char = ''
        for char in text:
            # Check for invalid Thai character combinations
            if prev_char and prev_char.isspace() and char.isspace():
                warnings.append("Text contains consecutive whitespace")
            if ThaiTextValidator._is_followup_vowel(prev_char) and ThaiTextValidator._is_followup_vowel(char):
                errors.append(f"Invalid vowel combination: {prev_char}{char}")
            prev_char = char
            
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_classification_result(result: Dict[str, Any]) -> ValidationResult:
        """
        Validate classification result
        
        Args:
            result: Classification result to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        required_fields = ['label']
        
        # Check required fields
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")
                
        # Check confidence score if present
        if 'confidence' in result:
            confidence = result['confidence']
            if not isinstance(confidence, (int, float)):
                errors.append("Confidence must be numeric")
            elif not 0 <= confidence <= 1:
                errors.append("Confidence must be between 0 and 1")
            elif confidence < 0.5:
                warnings.append("Low confidence prediction")
                
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_token_sequence(tokens: List[Tuple[str, str]]) -> ValidationResult:
        """
        Validate token-tag sequence
        
        Args:
            tokens: List of (token, tag) pairs
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        if not tokens:
            errors.append("Token sequence cannot be empty")
            return ValidationResult(False, errors, warnings)
            
        # Validate format
        for i, (token, tag) in enumerate(tokens):
            if not isinstance(token, str) or not isinstance(tag, str):
                errors.append(f"Invalid token-tag pair at position {i}")
                
        # Validate tag sequence
        prev_tag = ''
        for i, (_, tag) in enumerate(tokens):
            if tag.startswith('I-') and (not prev_tag or not prev_tag.endswith(tag[2:])):
                errors.append(f"Invalid BIO sequence at position {i}")
            prev_tag = tag
            
        # Check for potential issues
        if len(tokens) < 2:
            warnings.append("Very short token sequence")
            
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_qa_result(result: Dict[str, Any]) -> ValidationResult:
        """
        Validate question answering result
        
        Args:
            result: QA result to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        required_fields = ['answer', 'context']
        
        # Check required fields
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")
                
        if 'answer' in result:
            # Validate answer
            answer = result['answer']
            if not isinstance(answer, str):
                errors.append("Answer must be string")
            elif not answer.strip():
                errors.append("Answer cannot be empty")
            elif len(answer) > 1000:
                warnings.append("Answer is unusually long")
                
            # Check answer position if available
            if 'start' in result and 'end' in result:
                if result['start'] > result['end']:
                    errors.append("Invalid answer position: start > end")
                if 'context' in result:
                    context_len = len(result['context'])
                    if result['end'] > context_len:
                        errors.append("Answer position exceeds context length")
                        
            # Validate confidence
            if 'confidence' in result:
                confidence = result['confidence']
                if not isinstance(confidence, (int, float)):
                    errors.append("Confidence must be numeric")
                elif not 0 <= confidence <= 1:
                    errors.append("Confidence must be between 0 and 1")
                elif confidence < 0.3:
                    warnings.append("Very low confidence answer")
                    
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_model_input(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None
    ) -> ValidationResult:
        """
        Validate model input tensors
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Validate input_ids
        if not isinstance(input_ids, torch.Tensor):
            errors.append("input_ids must be torch.Tensor")
        else:
            if input_ids.dim() != 2:
                errors.append("input_ids must be 2-dimensional")
            if input_ids.max() > 100000:  # Arbitrary large vocab size
                warnings.append("Unusually large token IDs detected")
                
        # Validate attention_mask if provided
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                errors.append("attention_mask must be torch.Tensor")
            elif attention_mask.shape != input_ids.shape:
                errors.append("attention_mask shape must match input_ids")
            elif not ((attention_mask == 0) | (attention_mask == 1)).all():
                errors.append("attention_mask must contain only 0s and 1s")
                
        # Validate token_type_ids if provided
        if token_type_ids is not None:
            if not isinstance(token_type_ids, torch.Tensor):
                errors.append("token_type_ids must be torch.Tensor")
            elif token_type_ids.shape != input_ids.shape:
                errors.append("token_type_ids shape must match input_ids")
                
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def _is_followup_vowel(char: str) -> bool:
        """Check if character is a Thai followup vowel"""
        followup_vowels = {'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ'}
        return char in followup_vowels

class DatasetValidator:
    """Validator for training and evaluation datasets"""
    
    @staticmethod
    def validate_classification_dataset(
        data: List[Dict[str, Any]],
        labels: List[str] = None
    ) -> ValidationResult:
        """
        Validate classification dataset
        
        Args:
            data: List of examples
            labels: Valid label list
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        if not data:
            errors.append("Dataset cannot be empty")
            return ValidationResult(False, errors, warnings)
            
        # Check required fields
        required_fields = ['text', 'label']
        for i, example in enumerate(data):
            for field in required_fields:
                if field not in example:
                    errors.append(f"Missing {field} in example {i}")
                    
            # Validate text
            if 'text' in example:
                text_validation = ThaiTextValidator.validate_input_text(example['text'])
                errors.extend(text_validation.errors)
                warnings.extend(text_validation.warnings)
                
            # Validate label
            if 'label' in example and labels:
                if example['label'] not in labels:
                    errors.append(f"Invalid label in example {i}: {example['label']}")
                    
        # Check class distribution
        if labels:
            label_counts = {}
            for example in data:
                if 'label' in example:
                    label = example['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
                    
            # Check for class imbalance
            min_count = min(label_counts.values())
            max_count = max(label_counts.values())
            if max_count > 10 * min_count:
                warnings.append("Severe class imbalance detected")
                
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_token_classification_dataset(
        data: List[Dict[str, Any]],
        tag_scheme: str = 'BIO'
    ) -> ValidationResult:
        """
        Validate token classification dataset
        
        Args:
            data: List of examples
            tag_scheme: Tagging scheme
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        if not data:
            errors.append("Dataset cannot be empty")
            return ValidationResult(False, errors, warnings)
            
        # Check required fields
        required_fields = ['tokens', 'tags']
        for i, example in enumerate(data):
            for field in required_fields:
                if field not in example:
                    errors.append(f"Missing {field} in example {i}")
                    continue
                    
            if len(example['tokens']) != len(example['tags']):
                errors.append(f"Mismatched tokens and tags in example {i}")
                
            # Validate tag sequence
            if tag_scheme == 'BIO':
                prev_tag = 'O'
                for tag in example['tags']:
                    if tag.startswith('I-') and not prev_tag.endswith(tag[2:]):
                        errors.append(f"Invalid BIO sequence in example {i}")
                    prev_tag = tag
                    
            # Check sequence length
            if len(example['tokens']) > 512:  # Common model limit
                warnings.append(f"Long sequence in example {i}")
                
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    @staticmethod
    def validate_qa_dataset(data: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate question answering dataset
        
        Args:
            data: List of examples
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        if not data:
            errors.append("Dataset cannot be empty")
            return ValidationResult(False, errors, warnings)
            
        required_fields = ['question', 'context', 'answer']
        
        for i, example in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in example:
                    errors.append(f"Missing {field} in example {i}")
                    continue
                    
            # Validate text fields
            for field in ['question', 'context', 'answer']:
                if field in example:
                    text_validation = ThaiTextValidator.validate_input_text(example[field])
                    errors.extend(text_validation.errors)
                    warnings.extend(text_validation.warnings)
                    
            # Validate answer
            if all(field in example for field in ['answer', 'context']):
                answer = example['answer']
                context = example['context']
                
                # Check if answer is in context
                if answer not in context:
                    errors.append(f"Answer not found in context in example {i}")
                    
                # Check answer position if provided
                if 'answer_start' in example:
                    start = example['answer_start']
                    if start < 0 or start >= len(context):
                        errors.append(f"Invalid answer_start in example {i}")
                    elif context[start:start + len(answer)] != answer:
                        errors.append(f"Mismatched answer position in example {i}")
                        
            # Check lengths
            if 'context' in example and len(example['context']) > 5000:
                warnings.append(f"Very long context in example {i}")
                
        return ValidationResult(len(errors) == 0, errors, warnings)