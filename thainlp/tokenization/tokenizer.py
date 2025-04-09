"""
Unified tokenization for Thai and English text
"""
from typing import List, Optional, Union
import re
from transformers import AutoTokenizer
from .maximum_matching import MaximumMatchingTokenizer
from ..core.transformers import TransformerBase
from ..utils.thai_utils import contains_thai, separate_thai_english
import torch
class ThaiTokenizer(TransformerBase):
    """Unified tokenizer supporting both Thai and English"""
    
    def __init__(self,
                 model_name: str = "xlm-roberta-base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_thai_dict: bool = True):
        """Initialize tokenizer
        
        Args:
            model_name: Pretrained model name for subword tokenization
            device: Device to run model on
            use_thai_dict: Whether to use Thai dictionary for word segmentation
        """
        super().__init__(model_name)
        self.device = device
        
        # Load transformer tokenizer for subword tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load Thai dictionary tokenizer
        self.thai_tokenizer = MaximumMatchingTokenizer() if use_thai_dict else None
        
        # Special tokens
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.mask_token = self.tokenizer.mask_token
        
        # Regex for English word boundaries
        self.eng_word_pattern = re.compile(r'\b\w+\b')
        
    def tokenize(self,
                text: Union[str, List[str]],
                mode: str = "word",
                return_tensors: Optional[str] = None) -> Union[List[str], List[List[str]]]:
        """Tokenize text into words or subwords
        
        Args:
            text: Input text or texts
            mode: Tokenization mode ('word' or 'subword')
            return_tensors: Return format for transformer tokenizer
            
        Returns:
            List of tokens or list of token lists
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
            
        all_tokens = []
        for t in text:
            if mode == "word":
                tokens = self._word_tokenize(t)
            else:
                tokens = self._subword_tokenize(t, return_tensors)
            all_tokens.append(tokens)
            
        return all_tokens[0] if single_input else all_tokens
    
    def _word_tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
            
        # Separate Thai and English segments
        segments = separate_thai_english(text)
        tokens = []
        
        for segment in segments:
            if contains_thai(segment):
                # Use Thai word segmentation
                if self.thai_tokenizer:
                    tokens.extend(self.thai_tokenizer.tokenize(segment))
                else:
                    # Fallback to character tokenization for Thai
                    tokens.extend(list(segment))
            else:
                # Use regex for English word boundaries
                tokens.extend(self.eng_word_pattern.findall(segment))
                
        return tokens
    
    def _subword_tokenize(self,
                         text: str,
                         return_tensors: Optional[str] = None) -> Union[List[str], dict]:
        """Tokenize text into subword tokens"""
        # Use transformer tokenizer
        if return_tensors:
            return self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                truncation=True
            )
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to vocabulary indices
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token ids
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert vocabulary indices to tokens
        
        Args:
            ids: List of token ids
            
        Returns:
            List of tokens
        """
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def encode(self,
              text: Union[str, List[str]],
              padding: bool = True,
              truncation: bool = True,
              max_length: Optional[int] = None,
              return_tensors: Optional[str] = None) -> Union[List[int], dict]:
        """Encode text to model inputs
        
        Args:
            text: Input text or texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return format ('pt' for PyTorch)
            
        Returns:
            Token ids or model inputs dict
        """
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors
        )
    
    def decode(self,
              token_ids: List[int],
              skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text
        
        Args:
            token_ids: List of token ids
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def tokenize_and_align(self,
                          text: str,
                          annotations: List[dict],
                          annotation_key: str = "label") -> tuple:
        """Tokenize text and align with token-level annotations
        
        Args:
            text: Input text
            annotations: List of dicts with character offsets
            annotation_key: Key containing annotation value
            
        Returns:
            Tuple of (tokens, aligned_annotations)
        """
        # Get word tokens
        tokens = self._word_tokenize(text)
        
        # Track character offsets
        offset = 0
        token_starts = []
        token_ends = []
        
        for token in tokens:
            # Find token in remaining text
            start = text.find(token, offset)
            if start == -1:
                continue
                
            end = start + len(token)
            token_starts.append(start)
            token_ends.append(end)
            offset = end
            
        # Align annotations with tokens
        aligned_annotations = []
        
        for token_start, token_end in zip(token_starts, token_ends):
            # Find overlapping annotations
            token_labels = []
            
            for ann in annotations:
                ann_start = ann.get("start", 0)
                ann_end = ann.get("end", 0)
                
                # Check for overlap
                if (ann_start <= token_start < ann_end or
                    ann_start < token_end <= ann_end or
                    token_start <= ann_start < token_end):
                    token_labels.append(ann[annotation_key])
                    
            aligned_annotations.append(token_labels)
            
        return tokens, aligned_annotations
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)
    
    @property 
    def all_special_tokens(self) -> List[str]:
        """Get list of all special tokens"""
        return self.tokenizer.all_special_tokens
    
    def save_pretrained(self, path: str):
        """Save tokenizer to directory
        
        Args:
            path: Directory to save to
        """
        self.tokenizer.save_pretrained(path)
        
    @classmethod
    def from_pretrained(cls, path: str) -> 'ThaiTokenizer':
        """Load tokenizer from saved directory
        
        Args:
            path: Directory containing saved tokenizer
            
        Returns:
            Loaded tokenizer
        """
        tokenizer = cls(model_name=path)
        tokenizer.tokenizer = AutoTokenizer.from_pretrained(path)
        return tokenizer