"""
Advanced masked language modeling for Thai and English
"""
from typing import List, Dict, Tuple, Optional, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline
)
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..extensions.monitoring import ProgressTracker

class FillMask(TransformerBase):
    """Masked language model supporting Thai and English text"""
    
    def __init__(self,
                 model_name: str = "xlm-roberta-large",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32):
        """Initialize mask filler
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up masking tokens
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
    def fill_mask(self,
                  text: Union[str, List[str]],
                  top_k: int = 5,
                  threshold: float = 0.0,
                  use_context: bool = True,
                  return_full_sequences: bool = True) -> Union[List[Dict], List[List[Dict]]]:
        """Fill masked tokens in text(s)
        
        Args:
            text: Input text(s) with mask token(s)
            top_k: Number of predictions per mask
            threshold: Minimum confidence threshold
            use_context: Whether to use surrounding context
            return_full_sequences: Whether to return complete sequences
            
        Returns:
            List of predictions per mask or list of lists for multiple inputs
            Each prediction is a dict containing:
            - token: Predicted token
            - score: Confidence score
            - sequence: Full text with prediction (if return_full_sequences=True)
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
            
        all_results = []
        self.progress.start_task(len(text))
        
        # Process in batches
        for i in range(0, len(text), self.batch_size):
            batch_texts = text[i:i + self.batch_size]
            batch_results = self._process_batch(
                batch_texts,
                top_k=top_k,
                threshold=threshold,
                use_context=use_context,
                return_full_sequences=return_full_sequences
            )
            all_results.extend(batch_results)
            self.progress.update(len(batch_texts))
            
        self.progress.end_task()
        
        return all_results[0] if single_input else all_results
    
    def _process_batch(self,
                      texts: List[str],
                      **kwargs) -> List[List[Dict]]:
        """Process a batch of texts"""
        batch_results = []
        
        # Prepare inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Find mask positions
        mask_positions = []
        for input_ids in inputs.input_ids:
            positions = torch.where(input_ids == self.mask_token_id)[0]
            mask_positions.append(positions)
            
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Process each text
        for text_idx, text in enumerate(texts):
            text_results = []
            
            # Handle each mask in the text
            for mask_idx in mask_positions[text_idx]:
                mask_logits = logits[text_idx, mask_idx]
                probs = torch.softmax(mask_logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = probs.topk(kwargs.get('top_k', 5))
                
                predictions = []
                for token_idx, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    score = prob.item()
                    
                    # Apply threshold
                    if score < kwargs.get('threshold', 0.0):
                        continue
                        
                    token = self.tokenizer.decode([idx])
                    pred = {'token': token, 'score': score}
                    
                    # Add full sequence if requested
                    if kwargs.get('return_full_sequences'):
                        input_ids = inputs.input_ids[text_idx].clone()
                        input_ids[mask_idx] = idx
                        sequence = self.tokenizer.decode(
                            input_ids,
                            skip_special_tokens=True
                        )
                        pred['sequence'] = sequence
                        
                    predictions.append(pred)
                    
                text_results.append(predictions)
                
            # Single list if only one mask
            if len(text_results) == 1:
                text_results = text_results[0]
                
            batch_results.append(text_results)
            
        return batch_results
    
    def fine_tune(self,
                 train_data: List[Dict[str, str]],
                 val_data: Optional[List[Dict[str, str]]] = None,
                 epochs: int = 3,
                 learning_rate: float = 5e-5):
        """Fine-tune the masked language model
        
        Args:
            train_data: List of dicts with 'text' (masked) and 'target' keys
            val_data: Optional validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                
                # Prepare inputs
                inputs = self.tokenizer(
                    [d['text'] for d in batch_data],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Prepare targets
                labels = self.tokenizer(
                    [d['target'] for d in batch_data],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Validation
            if val_data:
                self.model.eval()
                val_loss = 0
                correct_preds = 0
                total_preds = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_data), self.batch_size):
                        batch_data = val_data[i:i + self.batch_size]
                        
                        # Prepare inputs
                        inputs = self.tokenizer(
                            [d['text'] for d in batch_data],
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        # Prepare targets
                        labels = self.tokenizer(
                            [d['target'] for d in batch_data],
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        ).input_ids.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(**inputs, labels=labels)
                        val_loss += outputs.loss.item()
                        
                        # Calculate accuracy
                        mask_positions = (inputs.input_ids == self.mask_token_id)
                        predictions = outputs.logits[mask_positions].argmax(dim=-1)
                        targets = labels[mask_positions]
                        correct_preds += (predictions == targets).sum().item()
                        total_preds += len(predictions)
                        
                avg_val_loss = val_loss / len(val_data)
                accuracy = correct_preds / total_preds
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"Validation Accuracy: {accuracy:.4f}")
                
                self.model.train()
    
    def evaluate(self,
                eval_data: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            eval_data: List of dicts with 'text' (masked) and 'target' keys
            
        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for i in range(0, len(eval_data), self.batch_size):
                batch_data = eval_data[i:i + self.batch_size]
                
                inputs = self.tokenizer(
                    [d['text'] for d in batch_data],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                labels = self.tokenizer(
                    [d['target'] for d in batch_data],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(self.device)
                
                outputs = self.model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                
                mask_positions = (inputs.input_ids == self.mask_token_id)
                predictions = outputs.logits[mask_positions].argmax(dim=-1)
                targets = labels[mask_positions]
                correct_preds += (predictions == targets).sum().item()
                total_preds += len(predictions)
                
        return {
            'loss': total_loss / len(eval_data),
            'accuracy': correct_preds / total_preds
        }
