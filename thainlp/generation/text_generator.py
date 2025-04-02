"""
Thai text generation utilities
"""
from typing import List, Dict, Optional
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize

class TextGenerator(TransformerBase):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> List[str]:
        """Generate Thai text continuations
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            List of generated text sequences
        """
        # Basic validation
        if not prompt:
            return []
        
        # Set up generation parameters
        generation_config = {
            'max_length': max_length,
            'num_return_sequences': num_return_sequences, 
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'pad_token_id': 1,  # [PAD]
            'bos_token_id': 0,  # [BOS]
            'eos_token_id': 2,  # [EOS]
            **kwargs
        }
        
        try:
            # Generate text
            tokens = word_tokenize(prompt)
            seed_text = ''.join(tokens)
            
            # Simple n-gram based generation when no model available
            if not self.model:
                return [self._generate_ngrams(seed_text, max_length) 
                        for _ in range(num_return_sequences)]
                
            # Use transformer model if available
            outputs = self.model.generate(
                text=seed_text,
                **generation_config
            )
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            return []
            
    def _generate_ngrams(self, seed_text: str, max_length: int) -> str:
        """Simple n-gram based generation when no model available"""
        import random
        
        # Use trigrams
        n = 3
        
        # Build n-gram model from seed text
        ngrams = {}
        tokens = word_tokenize(seed_text)
        
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i+n])
            next_token = tokens[i+n] if i+n < len(tokens) else None
            
            if gram not in ngrams:
                ngrams[gram] = []
            if next_token:
                ngrams[gram].append(next_token)
                
        # Generate text
        result = list(tokens[:n])  # Start with seed
        current_len = len(''.join(result))
        
        while current_len < max_length:
            # Get current n-gram
            current_gram = tuple(result[-n:])
            
            # Find possible next tokens
            next_tokens = ngrams.get(current_gram, [])
            
            if not next_tokens:
                break
                
            # Select random next token
            next_token = random.choice(next_tokens)
            result.append(next_token)
            current_len += len(next_token)
            
        return ''.join(result)
        
    def generate_stream(
        self,
        prompt: str,
        max_length: int = 100,
        **kwargs
    ) -> str:
        """Generate text in streaming mode, yielding tokens one at a time"""
        tokens = word_tokenize(prompt)
        seed_text = ''.join(tokens)
        
        if not self.model:
            # Fall back to basic generation
            text = self._generate_ngrams(seed_text, max_length)
            for token in word_tokenize(text):
                yield token
            return
            
        # Stream from model
        for token in self.model.generate_stream(
            text=seed_text,
            max_length=max_length,
            **kwargs
        ):
            yield token