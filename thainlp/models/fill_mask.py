"""
Fill-Mask Module for Thai Language
Supports masked language modeling for Thai text
"""

from typing import List, Dict, Union, Optional
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from ..core.transformers import TransformerBase

class ThaiFillMask(TransformerBase):
    """Fill-Mask model for Thai language"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_cuda: bool = True,
        **kwargs
    ):
        """Initialize fill-mask model
        
        Args:
            model_name_or_path: Name or path of the model
            use_cuda: Whether to use GPU if available
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "airesearch/wangchanberta-base-att-spm-uncased"
            
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="fill-mask",
            **kwargs
        )
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        if use_cuda and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            
    def fill_mask(
        self,
        text: str,
        top_k: int = 5,
        include_scores: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """Fill in masked token(s) in text
        
        Args:
            text: Input text with [MASK] token(s)
            top_k: Number of predictions to return per mask
            include_scores: Whether to include prediction scores
            
        Returns:
            List of predictions with filled text and scores
        """
        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Get mask token indices
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs.input_ids == mask_token_id).nonzero()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = []
        for pos in mask_positions:
            batch_idx, token_idx = pos
            
            # Get predictions for this position
            logits = outputs.logits[batch_idx, token_idx, :]
            probs = torch.softmax(logits, dim=0)
            
            # Get top k predictions
            values, indices = torch.topk(probs, k=top_k)
            
            # Convert to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(indices)
            
            # Create predictions
            position_preds = []
            for token, score in zip(tokens, values):
                # Create filled text by replacing mask at current position
                filled_tokens = list(self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
                filled_tokens[token_idx] = token
                filled_text = self.tokenizer.convert_tokens_to_string(filled_tokens)
                
                pred = {'text': filled_text}
                if include_scores:
                    pred['score'] = float(score)
                position_preds.append(pred)
                
            predictions.append(position_preds)
            
        return predictions
        
    def fill_mask_batch(
        self,
        texts: List[str],
        top_k: int = 5,
        include_scores: bool = True,
        batch_size: int = 8,
        return_only_top: bool = False
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Fill in masked tokens for multiple texts
        
        Args:
            texts: List of input texts with [MASK] token(s)
            top_k: Number of predictions to return per mask
            include_scores: Whether to include prediction scores
            batch_size: Number of texts to process in each batch
            return_only_top: Whether to return only the top prediction
            
        Returns:
            List of predictions for each input text
        """
        if not texts:
            return []
            
        all_results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize all texts at once
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            # Find mask positions for each text in batch
            mask_token_id = self.tokenizer.mask_token_id
            batch_results = []
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process each text in the batch
            for batch_idx in range(len(batch_texts)):
                # Find mask positions in this text
                mask_positions = (inputs.input_ids[batch_idx] == mask_token_id).nonzero(as_tuple=True)[0]
                
                if len(mask_positions) == 0:
                    # No masks found in this text
                    batch_results.append([])
                    continue
                
                text_predictions = []
                
                # Process each mask in this text
                for token_idx in mask_positions:
                    # Get predictions for this position
                    logits = outputs.logits[batch_idx, token_idx, :]
                    probs = torch.softmax(logits, dim=0)
                    
                    # Get top k predictions
                    values, indices = torch.topk(probs, k=top_k)
                    
                    # Convert to tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(indices)
                    
                    # Create predictions
                    position_preds = []
                    for token, score in zip(tokens, values):
                        # Create filled text by replacing mask
                        filled_tokens = list(self.tokenizer.convert_ids_to_tokens(inputs.input_ids[batch_idx]))
                        filled_tokens[token_idx] = token
                        filled_text = self.tokenizer.convert_tokens_to_string(filled_tokens)
                        
                        pred = {'text': filled_text}
                        if include_scores:
                            pred['score'] = float(score)
                        position_preds.append(pred)
                    
                    if return_only_top:
                        text_predictions.append(position_preds[0])
                    else:
                        text_predictions.append(position_preds)
                
                batch_results.append(text_predictions)
            
            all_results.extend(batch_results)
            
        return all_results
        
    def generate_masks(
        self,
        text: str,
        mask_ratio: float = 0.15,
        ensure_mask: bool = True
    ) -> str:
        """Generate masked version of input text
        
        Args:
            text: Input text
            mask_ratio: Ratio of tokens to mask
            ensure_mask: Ensure at least one token is masked
            
        Returns:
            Text with masked tokens
        """
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Calculate number of tokens to mask
        n_masks = max(1 if ensure_mask else 0,
                     int(len(tokens) * mask_ratio))
        
        # Randomly select positions to mask
        mask_positions = torch.randperm(len(tokens))[:n_masks]
        
        # Apply masks
        for pos in mask_positions:
            tokens[pos] = self.tokenizer.mask_token
            
        # Convert back to text
        return self.tokenizer.convert_tokens_to_string(tokens)