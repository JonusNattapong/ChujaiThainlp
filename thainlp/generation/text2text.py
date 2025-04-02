"""
Text-to-text generation for Thai
"""
from typing import List, Dict, Optional, Union
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize

class Text2Text(TransformerBase):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        
    def generate(
        self,
        source_text: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """Generate text output from input text
        
        Args:
            source_text: Input text
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text sequences
        """
        # Basic validation
        if not source_text:
            return []
            
        try:
            # Tokenize input
            tokens = word_tokenize(source_text)
            
            # Use model if available
            if self.model:
                return self.model.generate(
                    text=''.join(tokens),
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    **kwargs
                )
                
            # Fall back to basic generation
            return self._basic_generation(
                tokens,
                max_length,
                num_return_sequences
            )
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return []
            
    def _basic_generation(
        self,
        input_tokens: List[str],
        max_length: int,
        num_sequences: int
    ) -> List[str]:
        """Basic text continuation using simple rules"""
        import random
        
        # Common Thai sentence patterns
        patterns = {
            'statement': ['[SUBJ]', '[VERB]', '[OBJ]'],
            'question': ['[QWORD]', '[VERB]', '[OBJ]', 'ไหม'],
            'description': ['[SUBJ]', '[ADJ]', 'มาก']
        }
        
        # Basic vocabulary for filling patterns
        vocab = {
            'SUBJ': ['ฉัน', 'เขา', 'เธอ', 'พวกเรา', 'มัน'],
            'VERB': ['กิน', 'เดิน', 'วิ่ง', 'นอน', 'พูด'],
            'OBJ': ['ข้าว', 'น้ำ', 'หนังสือ', 'รถ', 'บ้าน'],
            'ADJ': ['ดี', 'สวย', 'เร็ว', 'ช้า', 'ใหญ่'],
            'QWORD': ['ใคร', 'อะไร', 'ที่ไหน', 'เมื่อไร', 'ทำไม']
        }
        
        generations = []
        for _ in range(num_sequences):
            # Select random pattern
            pattern = random.choice(list(patterns.values()))
            
            # Fill pattern with vocabulary
            output_tokens = []
            for token in pattern:
                if token.startswith('[') and token.endswith(']'):
                    category = token[1:-1]
                    word = random.choice(vocab[category])
                    output_tokens.append(word)
                else:
                    output_tokens.append(token)
                    
            # Combine with input
            result = input_tokens + output_tokens
            
            # Truncate to max length
            result = result[:max_length]
            
            generations.append(''.join(result))
            
        return generations
        
    def batch_generate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[str]]:
        """Generate outputs for multiple input texts"""
        results = []
        for text in texts:
            result = self.generate(text, **kwargs)
            results.append(result)
        return results