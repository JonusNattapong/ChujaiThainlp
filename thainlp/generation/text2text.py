"""
Text-to-text generation supporting multiple tasks and languages
"""
from typing import List, Dict, Optional, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..extensions.monitoring import ProgressTracker

class Text2Text(TransformerBase):
    """Text-to-text generation model supporting multiple tasks"""
    
    SUPPORTED_TASKS = {
        'translation': 'Helsinki-NLP/opus-mt-th-en',
        'summarization': 'facebook/mbart-large-cc25',
        'style_transfer': 'facebook/bart-large',
        'paraphrase': 'facebook/bart-large-paraphrase'
    }
    
    def __init__(self, 
                 model_name: str = "facebook/mbart-large-cc25",
                 task: str = "translation",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 16):
        """Initialize text2text model
        
        Args:
            model_name: Pretrained model name/path
            task: Generation task (translation, summarization, etc)
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        self.task = task
        
        # Load model and tokenizer
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
    def generate(self,
                source_text: Union[str, List[str]],
                max_length: int = 128,
                num_return_sequences: int = 1,
                min_length: Optional[int] = None,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9,
                repetition_penalty: float = 1.0,
                do_sample: bool = True,
                return_scores: bool = False,
                source_lang: Optional[str] = None,
                target_lang: Optional[str] = None,
                **kwargs) -> Union[List[str], List[Tuple[str, float]]]:
        """Generate text based on input
        
        Args:
            source_text: Input text or texts
            max_length: Maximum output length
            num_return_sequences: Number of sequences to generate
            min_length: Minimum output length
            temperature: Sampling temperature
            top_k: Top k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to sample or use greedy decoding
            return_scores: Whether to return generation scores
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional generation parameters
            
        Returns:
            Generated sequences or (sequence, score) tuples if return_scores=True
        """
        # Handle single string input
        if isinstance(source_text, str):
            source_text = [source_text]
            
        all_results = []
        
        # Process in batches
        self.progress.start_task(len(source_text))
        for i in range(0, len(source_text), self.batch_size):
            batch_texts = source_text[i:i + self.batch_size]
            batch_results = self._process_batch(
                batch_texts,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                return_scores=return_scores,
                source_lang=source_lang,
                target_lang=target_lang,
                **kwargs
            )
            all_results.extend(batch_results)
            self.progress.update(len(batch_texts))
            
        self.progress.end_task()
        
        # Return single result for single input
        if len(source_text) == 1:
            return all_results[0]
        return all_results
    
    def _process_batch(self,
                      texts: List[str],
                      **kwargs) -> Union[List[str], List[Tuple[str, float]]]:
        """Process a batch of texts"""
        # Prepare inputs
        inputs = self._prepare_inputs(texts, kwargs.get('source_lang'))
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=kwargs.get('max_length', 128),
                min_length=kwargs.get('min_length'),
                num_return_sequences=kwargs.get('num_return_sequences', 1),
                temperature=kwargs.get('temperature', 1.0),
                top_k=kwargs.get('top_k', 50),
                top_p=kwargs.get('top_p', 0.9),
                repetition_penalty=kwargs.get('repetition_penalty', 1.0),
                do_sample=kwargs.get('do_sample', True),
                return_dict_in_generate=True,
                output_scores=kwargs.get('return_scores', False)
            )
            
        # Decode outputs
        generated_sequences = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        
        # Calculate scores if requested
        if kwargs.get('return_scores'):
            scores = self._calculate_sequence_scores(outputs)
            results = list(zip(generated_sequences, scores))
        else:
            results = generated_sequences
            
        return results
    
    def _prepare_inputs(self,
                       texts: List[str],
                       source_lang: Optional[str] = None) -> Dict:
        """Prepare model inputs"""
        # Handle language codes for multilingual models
        if hasattr(self.tokenizer, 'src_lang') and source_lang:
            self.tokenizer.src_lang = source_lang
            
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
    
    def _calculate_sequence_scores(self, outputs) -> List[float]:
        """Calculate normalized log probabilities for sequences"""
        scores = []
        
        for i in range(len(outputs.sequences)):
            if hasattr(outputs, 'scores'):
                # Get log probs for each step
                step_scores = [score[i] for score in outputs.scores]
                log_probs = torch.log_softmax(torch.stack(step_scores), dim=-1)
                
                # Get token probabilities
                token_probs = log_probs[torch.arange(len(step_scores)), 
                                      outputs.sequences[i][1:]]  # Skip start token
                                      
                # Average log prob
                score = torch.mean(token_probs).exp().item()
                scores.append(score)
            else:
                scores.append(1.0)
                
        return scores
    
    def fine_tune(self,
                 train_texts: List[str],
                 train_targets: List[str],
                 val_texts: Optional[List[str]] = None,
                 val_targets: Optional[List[str]] = None,
                 epochs: int = 3,
                 learning_rate: float = 5e-5):
        """Fine-tune the model on domain data
        
        Args:
            train_texts: Training source texts
            train_targets: Training target texts
            val_texts: Validation source texts
            val_targets: Validation target texts
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Prepare data
        train_inputs = self._prepare_inputs(train_texts)
        train_targets = self.tokenizer(
            train_targets,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        if val_texts and val_targets:
            val_inputs = self._prepare_inputs(val_texts)
            val_targets = self.tokenizer(
                val_targets,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(train_texts), self.batch_size):
                # Get batch
                batch_inputs = {k: v[i:i + self.batch_size] 
                              for k, v in train_inputs.items()}
                batch_targets = train_targets[i:i + self.batch_size]
                
                # Forward pass
                outputs = self.model(**batch_inputs, labels=batch_targets)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            avg_loss = total_loss / len(train_texts)
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_loss:.4f}")
            
            # Validation
            if val_texts and val_targets:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for i in range(0, len(val_texts), self.batch_size):
                        batch_inputs = {k: v[i:i + self.batch_size]
                                      for k, v in val_inputs.items()}
                        batch_targets = val_targets[i:i + self.batch_size]
                        
                        outputs = self.model(**batch_inputs, labels=batch_targets)
                        val_loss += outputs.loss.item()
                        
                avg_val_loss = val_loss / len(val_texts)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                self.model.train()