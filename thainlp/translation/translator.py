"""
Advanced multilingual translation system with focus on Thai
"""
from typing import List, Dict, Optional, Union, Tuple
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqGeneration,
    M2M100Tokenizer,
    M2M100ForConditionalGeneration
)
from sacrebleu.metrics import BLEU
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..extensions.monitoring import ProgressTracker

class Translator(TransformerBase):
    """Neural machine translation with support for multiple languages"""
    
    def __init__(self, 
                 model_name: str = "facebook/m2m100_418M",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 32):
        """Initialize translator
        
        Args:
            model_name: Pretrained model name/path
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer
        if "m2m100" in model_name:
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        # Set up progress tracking
        self.progress = ProgressTracker()
        
        # BLEU score calculator
        self.bleu = BLEU()
        
    def translate(self,
                 text: Union[str, List[str]],
                 source_lang: str = "th",
                 target_lang: str = "en",
                 num_beams: int = 5,
                 num_return_sequences: int = 1,
                 max_length: int = 128,
                 return_scores: bool = False,
                 **kwargs) -> Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]:
        """Translate text between languages
        
        Args:
            text: Input text or texts
            source_lang: Source language code
            target_lang: Target language code 
            num_beams: Number of beams for beam search
            num_return_sequences: Number of translation variants to return
            max_length: Maximum length of generated translation
            return_scores: Whether to return confidence scores
            **kwargs: Additional generation parameters
            
        Returns:
            Translated text(s) or tuples of (text, score) if return_scores=True
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
                source_lang=source_lang,
                target_lang=target_lang,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_length=max_length,
                return_scores=return_scores,
                **kwargs
            )
            all_results.extend(batch_results)
            self.progress.update(len(batch_texts))
            
        self.progress.end_task()
        
        # Return appropriate format
        if single_input:
            if num_return_sequences == 1:
                return all_results[0]
            return all_results[0:num_return_sequences]
        return all_results
    
    def _process_batch(self,
                      texts: List[str],
                      source_lang: str,
                      target_lang: str,
                      **kwargs) -> List[Union[str, Tuple[str, float]]]:
        """Process a batch of texts"""
        # Set language codes for tokenizer
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = source_lang
        if hasattr(self.tokenizer, 'tgt_lang'):    
            self.tokenizer.tgt_lang = target_lang
            
        # Prepare inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=kwargs.get('max_length', 128),
            return_tensors="pt"
        ).to(self.device)
        
        # Force target language token for m2m100
        if isinstance(self.tokenizer, M2M100Tokenizer):
            forced_tokens = self.tokenizer.get_lang_id(target_lang)
            inputs['forced_bos_token_id'] = forced_tokens
            
        # Generate translations
        outputs = self.model.generate(
            **inputs,
            num_beams=kwargs.get('num_beams', 5),
            num_return_sequences=kwargs.get('num_return_sequences', 1),
            max_length=kwargs.get('max_length', 128),
            early_stopping=True,
            output_scores=kwargs.get('return_scores', False),
            return_dict_in_generate=True,
        )
        
        # Decode outputs
        translations = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )
        
        # Calculate scores if requested
        results = []
        if kwargs.get('return_scores'):
            scores = self._calculate_sequence_scores(outputs)
            
            # Group by input text
            for i in range(len(texts)):
                text_translations = []
                start_idx = i * kwargs.get('num_return_sequences', 1)
                end_idx = start_idx + kwargs.get('num_return_sequences', 1)
                
                for trans, score in zip(
                    translations[start_idx:end_idx],
                    scores[start_idx:end_idx]
                ):
                    text_translations.append((trans, score))
                    
                if kwargs.get('num_return_sequences', 1) == 1:
                    results.append(text_translations[0])
                else:
                    results.append(text_translations)
        else:
            # Group by input text
            for i in range(len(texts)):
                start_idx = i * kwargs.get('num_return_sequences', 1)
                end_idx = start_idx + kwargs.get('num_return_sequences', 1)
                text_translations = translations[start_idx:end_idx]
                
                if kwargs.get('num_return_sequences', 1) == 1:
                    results.append(text_translations[0])
                else:
                    results.append(text_translations)
                    
        return results
    
    def _calculate_sequence_scores(self, outputs) -> List[float]:
        """Calculate sequence scores from generation outputs"""
        scores = []
        
        if hasattr(outputs, 'sequences_scores'):
            # Direct sequence scores
            return outputs.sequences_scores.tolist()
        elif hasattr(outputs, 'scores'):
            # Calculate from step scores
            for i in range(len(outputs.sequences)):
                step_scores = [score[i] for score in outputs.scores]
                log_probs = torch.log_softmax(torch.stack(step_scores), dim=-1)
                
                # Get token probabilities
                token_probs = log_probs[
                    torch.arange(len(step_scores)),
                    outputs.sequences[i][1:]  # Skip start token
                ]
                
                # Average log prob
                score = torch.mean(token_probs).exp().item()
                scores.append(score)
        else:
            # Fallback scores
            scores = [1.0] * len(outputs.sequences)
            
        return scores
    
    def evaluate(self,
                sources: List[str],
                references: List[str],
                source_lang: str,
                target_lang: str) -> Dict[str, float]:
        """Evaluate translation quality
        
        Args:
            sources: Source texts
            references: Reference translations
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dict with quality metrics
        """
        # Generate translations
        translations = self.translate(sources, source_lang, target_lang)
        
        # Calculate BLEU score
        bleu_score = self.bleu.corpus_score(
            translations,
            [references]
        ).score
        
        return {
            'bleu': bleu_score / 100.0  # Normalize to 0-1
        }
    
    def fine_tune(self,
                 train_data: List[Dict[str, str]],
                 val_data: Optional[List[Dict[str, str]]] = None,
                 source_lang: str = "th",
                 target_lang: str = "en",
                 epochs: int = 3,
                 learning_rate: float = 5e-5):
        """Fine-tune the model on parallel data
        
        Args:
            train_data: List of dicts with 'source' and 'target' keys
            val_data: Optional validation data
            source_lang: Source language code
            target_lang: Target language code
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Set up training
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                
                # Prepare inputs
                source_texts = [d['source'] for d in batch_data]
                target_texts = [d['target'] for d in batch_data]
                
                # Tokenize source
                inputs = self.tokenizer(
                    source_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Tokenize target
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        target_texts,
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
                
                with torch.no_grad():
                    for i in range(0, len(val_data), self.batch_size):
                        batch_data = val_data[i:i + self.batch_size]
                        
                        source_texts = [d['source'] for d in batch_data]
                        target_texts = [d['target'] for d in batch_data]
                        
                        inputs = self.tokenizer(
                            source_texts,
                            padding=True,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        with self.tokenizer.as_target_tokenizer():
                            labels = self.tokenizer(
                                target_texts,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                            ).input_ids.to(self.device)
                            
                        outputs = self.model(**inputs, labels=labels)
                        val_loss += outputs.loss.item()
                        
                avg_val_loss = val_loss / len(val_data)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                
                # Evaluate on validation set
                val_metrics = self.evaluate(
                    [d['source'] for d in val_data[:100]],  # Sample for speed
                    [d['target'] for d in val_data[:100]],
                    source_lang,
                    target_lang
                )
                print(f"Validation BLEU: {val_metrics['bleu']:.4f}")
                
                self.model.train()