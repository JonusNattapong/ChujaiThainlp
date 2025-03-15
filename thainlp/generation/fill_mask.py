"""
Advanced Masked Language Modeling for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
import re
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    BertForMaskedLM,
    BertTokenizer
)
from pythainlp.tokenize import word_tokenize
from ..core.transformers import TransformerBase

class ThaiFillMask(TransformerBase):
    """Advanced masked language modeling for Thai language"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_length: int = 512,
        top_k: int = 5,
        use_cuda: bool = True,
        **kwargs
    ):
        """Initialize mask filler
        
        Args:
            model_name_or_path: Name or path of the model
            max_length: Maximum sequence length
            top_k: Number of top predictions to return
            use_cuda: Whether to use GPU if available
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "airesearch/wangchanberta-base-att-spm-uncased"
            
        # Special handling for Gemma models
        self.is_gemma = "google/gemma" in str(model_name_or_path)
        if self.is_gemma:
            kwargs['trust_remote_code'] = True
            
        self.max_length = max_length
        self.top_k = top_k
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="fill-mask",
            **kwargs
        )
        
        # Initialize fill-mask pipeline
        self.fill_mask_pipeline = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if use_cuda and torch.cuda.is_available() else -1
        )
        
        # Get mask token - Gemma uses <mask> instead of [MASK]
        self.mask_token = "<mask>" if self.is_gemma else self.tokenizer.mask_token
        
    def _validate_input(self, text: str) -> bool:
        """Validate input text contains mask token
        
        Args:
            text: Input text
            
        Returns:
            Whether input is valid
        """
        return self.mask_token in text
        
    def _prepare_input(
        self,
        text: str,
        mask_token: Optional[str] = None
    ) -> str:
        """Prepare input text by replacing custom mask token
        
        Args:
            text: Input text
            mask_token: Custom mask token
            
        Returns:
            Prepared text
        """
        if mask_token:
            text = text.replace(mask_token, self.mask_token)
            
        return text
        
    def _format_predictions(
        self,
        predictions: List[Dict[str, Any]],
        include_scores: bool = True,
        min_score: float = 0.0
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Format model predictions
        
        Args:
            predictions: Raw model predictions
            include_scores: Whether to include prediction scores
            min_score: Minimum score threshold
            
        Returns:
            Formatted predictions
        """
        # Filter by score
        filtered = [
            p for p in predictions
            if p['score'] >= min_score
        ]
        
        if include_scores:
            return [
                {
                    'token': p['token_str'],
                    'score': p['score']
                }
                for p in filtered
            ]
            
        return [p['token_str'] for p in filtered]
        
    def fill_mask(
        self,
        text: str,
        mask_token: Optional[str] = None,
        top_k: Optional[int] = None,
        include_scores: bool = True,
        min_score: float = 0.0,
        **kwargs
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """Fill masked token in text
        
        Args:
            text: Input text with mask token
            mask_token: Custom mask token
            top_k: Number of predictions to return
            include_scores: Whether to include prediction scores
            min_score: Minimum score threshold
            **kwargs: Additional arguments for pipeline
            
        Returns:
            List of predictions or dictionaries with predictions and scores
        """
        # Prepare input
        text = self._prepare_input(text, mask_token)
        
        # Validate input
        if not self._validate_input(text):
            raise ValueError(f"Input text must contain mask token: {self.mask_token}")
            
        # Get predictions
        predictions = self.fill_mask_pipeline(
            text,
            top_k=top_k or self.top_k,
            **kwargs
        )
        
        return self._format_predictions(
            predictions,
            include_scores=include_scores,
            min_score=min_score
        )
        
    def fill_multiple_masks(
        self,
        text: str,
        mask_token: Optional[str] = None,
        strategy: str = "sequential",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fill multiple masks in text
        
        Args:
            text: Input text with multiple masks
            mask_token: Custom mask token
            strategy: Filling strategy (sequential or parallel)
            **kwargs: Additional arguments for fill_mask
            
        Returns:
            List of predictions for each mask
        """
        # Prepare input
        text = self._prepare_input(text, mask_token)
        
        # Count masks
        mask_count = text.count(self.mask_token)
        if mask_count == 0:
            raise ValueError(f"Input text must contain at least one mask token: {self.mask_token}")
            
        if strategy == "sequential":
            # Fill masks one by one
            results = []
            current_text = text
            
            for _ in range(mask_count):
                # Get predictions for first mask
                predictions = self.fill_mask(current_text, **kwargs)
                results.append(predictions)
                
                # Replace first mask with top prediction
                if isinstance(predictions[0], dict):
                    top_prediction = predictions[0]['token']
                else:
                    top_prediction = predictions[0]
                    
                current_text = current_text.replace(self.mask_token, top_prediction, 1)
                
        else:  # parallel strategy
            # Get predictions for all masks independently
            results = []
            mask_positions = [
                m.start()
                for m in re.finditer(re.escape(self.mask_token), text)
            ]
            
            for pos in mask_positions:
                # Create text with single mask
                masked_text = (
                    text[:pos] +
                    self.mask_token +
                    text[pos + len(self.mask_token):]
                ).replace(self.mask_token, "[FILLED]")
                
                # Get predictions
                predictions = self.fill_mask(masked_text, **kwargs)
                results.append(predictions)
                
        return results
        
    def suggest_completions(
        self,
        text: str,
        num_tokens: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Suggest completions for text
        
        Args:
            text: Input text to complete
            num_tokens: Number of tokens to predict
            **kwargs: Additional arguments for fill_mask
            
        Returns:
            List of completion predictions
        """
        # Add mask tokens for completion
        completion_text = text
        if not completion_text.endswith(" "):
            completion_text += " "
            
        completion_text += " ".join([self.mask_token] * num_tokens)
        
        return self.fill_multiple_masks(
            completion_text,
            strategy="sequential",
            **kwargs
        )
        
    def fill_template(
        self,
        template: str,
        mask_token: str = "_____",
        **kwargs
    ) -> Dict[str, Any]:
        """Fill template with masked tokens
        
        Args:
            template: Text template with mask placeholders
            mask_token: Custom mask token
            **kwargs: Additional arguments for fill_mask
            
        Returns:
            Dictionary with filled template and predictions
        """
        # Replace template mask token with model mask token
        text = template.replace(mask_token, self.mask_token)
        
        # Get predictions for all masks
        predictions = self.fill_multiple_masks(
            text,
            strategy="parallel",
            **kwargs
        )
        
        # Create filled versions
        filled_versions = []
        
        # Get all combinations of top predictions
        import itertools
        prediction_lists = []
        for mask_preds in predictions:
            if isinstance(mask_preds[0], dict):
                pred_tokens = [p['token'] for p in mask_preds]
            else:
                pred_tokens = mask_preds
            prediction_lists.append(pred_tokens)
            
        for combination in itertools.product(*prediction_lists):
            filled = template
            for token in combination:
                filled = filled.replace(mask_token, token, 1)
            filled_versions.append(filled)
            
        return {
            'template': template,
            'predictions': predictions,
            'filled_versions': filled_versions
        }
        
    def analyze_context(
        self,
        text: str,
        target_word: str,
        window_size: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Analyze word predictions in different contexts
        
        Args:
            text: Input text
            target_word: Word to analyze
            window_size: Context window size
            **kwargs: Additional arguments for fill_mask
            
        Returns:
            List of context analyses
        """
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Find target word positions
        target_positions = [
            i for i, token in enumerate(tokens)
            if token == target_word
        ]
        
        results = []
        
        for pos in target_positions:
            # Get context window
            start = max(0, pos - window_size)
            end = min(len(tokens), pos + window_size + 1)
            
            # Create masked text
            context_tokens = tokens[start:end]
            context_tokens[pos - start] = self.mask_token
            context_text = ' '.join(context_tokens)
            
            # Get predictions
            predictions = self.fill_mask(context_text, **kwargs)
            
            results.append({
                'position': pos,
                'context': context_text,
                'predictions': predictions
            })
            
        return results
