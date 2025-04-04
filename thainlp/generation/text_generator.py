"""
Advanced text generation with transformer models supporting Thai and English
"""
from typing import List, Dict, Optional, Union, Iterator, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..extensions.monitoring import ProgressTracker

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation"""
    
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = set(stop_token_ids)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token = input_ids[0][-1]
        return last_token.item() in self.stop_token_ids

class TextGenerator(TransformerBase):
    """Advanced text generation using transformer models"""
    
    def __init__(self,
                 model_name: str = "facebook/xglm-7.5B",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 4,
                 max_context_length: int = 2048):
        """Initialize text generator
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
            max_context_length: Maximum context length
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up generation settings
        self.stop_tokens = ["</s>", "<|endoftext|>"]
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(t) 
            for t in self.stop_tokens
        ]
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
    def generate(self,
                prompt: Union[str, List[str]],
                max_length: int = 100,
                min_length: Optional[int] = None,
                num_return_sequences: int = 1,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.95,
                repetition_penalty: float = 1.2,
                do_sample: bool = True,
                early_stopping: bool = True,
                return_full_text: bool = True,
                clean_up_tokenization_spaces: bool = True,
                **kwargs) -> Union[str, List[str], List[List[str]]]:
        """Generate text from prompt(s)
        
        Args:
            prompt: Input prompt text or texts
            max_length: Maximum generation length
            min_length: Minimum generation length
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling vs greedy decoding
            early_stopping: Whether to stop on EOS token
            return_full_text: Whether to include prompt in output
            clean_up_tokenization_spaces: Whether to clean up spaces
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s)
        """
        # Handle single string input
        if isinstance(prompt, str):
            prompt = [prompt]
            single_prompt = True
        else:
            single_prompt = False
            
        all_results = []
        self.progress.start_task(len(prompt))
        
        # Process in batches
        for i in range(0, len(prompt), self.batch_size):
            batch_prompts = prompt[i:i + self.batch_size]
            batch_results = self._process_batch(
                batch_prompts,
                max_length=max_length,
                min_length=min_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                early_stopping=early_stopping,
                return_full_text=return_full_text,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs
            )
            all_results.extend(batch_results)
            self.progress.update(len(batch_prompts))
            
        self.progress.end_task()
        
        # Return appropriate format
        if single_prompt:
            if num_return_sequences == 1:
                return all_results[0][0]
            return all_results[0]
        return all_results
    
    def _process_batch(self,
                      prompts: List[str],
                      **kwargs) -> List[List[str]]:
        """Process a batch of prompts"""
        # Prepare inputs
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_context_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Set up stopping criteria
        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.stop_token_ids)
        ])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=kwargs.get('max_length', 100),
                min_length=kwargs.get('min_length'),
                num_return_sequences=kwargs.get('num_return_sequences', 1),
                temperature=kwargs.get('temperature', 1.0),
                top_k=kwargs.get('top_k', 50),
                top_p=kwargs.get('top_p', 0.95),
                repetition_penalty=kwargs.get('repetition_penalty', 1.2),
                do_sample=kwargs.get('do_sample', True),
                early_stopping=kwargs.get('early_stopping', True),
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True
            )
            
        # Decode outputs
        generated_sequences = []
        
        for i, prompt in enumerate(prompts):
            sequences = []
            start_idx = i * kwargs.get('num_return_sequences', 1)
            end_idx = start_idx + kwargs.get('num_return_sequences', 1)
            
            for j in range(start_idx, end_idx):
                sequence = self.tokenizer.decode(
                    outputs.sequences[j],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=kwargs.get(
                        'clean_up_tokenization_spaces',
                        True
                    )
                )
                
                # Remove prompt if not returning full text
                if not kwargs.get('return_full_text', True):
                    sequence = sequence[len(prompt):].strip()
                    
                sequences.append(sequence)
                
            generated_sequences.append(sequences)
            
        return generated_sequences
    
    def generate_stream(self,
                       prompt: str,
                       max_length: int = 100,
                       temperature: float = 1.0,
                       **kwargs) -> Iterator[str]:
        """Generate text in streaming mode
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Yields:
            Generated tokens one at a time
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Set up stopping criteria
        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(self.stop_token_ids)
        ])
        
        # Generate tokens one at a time
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 1,
                    temperature=temperature,
                    do_sample=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                
            next_token = outputs.sequences[0][-1:]
            
            # Stop if we hit a stop token
            if next_token.item() in self.stop_token_ids:
                break
                
            # Decode and yield token
            token = self.tokenizer.decode(next_token, skip_special_tokens=True)
            if token:
                yield token
                
            # Update input ids
            input_ids = outputs.sequences
    
    def apply_prompt_template(self,
                            template: str,
                            **kwargs) -> str:
        """Apply template to format prompt
        
        Args:
            template: Prompt template with {variables}
            **kwargs: Values to fill template variables
            
        Returns:
            Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
    
    def fine_tune(self,
                 train_texts: List[str],
                 val_texts: Optional[List[str]] = None,
                 epochs: int = 3,
                 learning_rate: float = 5e-5,
                 warmup_steps: int = 100):
        """Fine-tune the model on custom texts
        
        Args:
            train_texts: Training texts
            val_texts: Optional validation texts
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        from transformers import Trainer, TrainingArguments
        
        # Prepare datasets
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.max_context_length,
            return_tensors="pt"
        )
        
        if val_texts:
            val_encodings = self.tokenizer(
                val_texts,
                truncation=True,
                padding=True,
                max_length=self.max_context_length,
                return_tensors="pt"
            )
            
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_dir="./logs",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_encodings,
            eval_dataset=val_encodings if val_texts else None
        )
        
        # Train model
        trainer.train()
        
        # Evaluate if validation set provided
        if val_texts:
            eval_results = trainer.evaluate()
            print("Validation Results:")
            for metric, value in eval_results.items():
                print(f"{metric}: {value:.4f}")