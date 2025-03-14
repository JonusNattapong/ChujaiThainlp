"""
Advanced Text Generation for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from pythainlp.tokenize import word_tokenize
from ..core.transformers import TransformerBase

class ThaiTextGenerator(TransformerBase):
    """Advanced text generation for Thai language"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_cuda: bool = True,
        **kwargs
    ):
        """Initialize text generator
        
        Args:
            model_name_or_path: Name or path of the model
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_cuda: Whether to use GPU if available
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "airesearch/wangchanberta-base-att-spm-uncased"
            
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="text-generation",
            **kwargs
        )
        
        # Initialize generation pipeline
        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if use_cuda and torch.cuda.is_available() else -1
        )
        
    def _prepare_prompt(
        self,
        prompt: str,
        style: Optional[str] = None,
        topic: Optional[str] = None,
        format_template: Optional[str] = None
    ) -> str:
        """Prepare prompt with control tokens and formatting
        
        Args:
            prompt: Base prompt text
            style: Writing style
            topic: Topic or theme
            format_template: Text format template
            
        Returns:
            Formatted prompt
        """
        control_tokens = []
        
        if style:
            control_tokens.append(f"<style={style}>")
            
        if topic:
            control_tokens.append(f"<topic={topic}>")
            
        if format_template:
            # Replace placeholders in template
            formatted_prompt = format_template.replace("{prompt}", prompt)
            prompt = formatted_prompt
            
        # Combine control tokens and prompt
        if control_tokens:
            prompt = ' '.join(control_tokens + [prompt])
            
        return prompt
        
    def _filter_generated_text(
        self,
        text: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        remove_prompt: bool = True,
        clean_special_tokens: bool = True
    ) -> str:
        """Filter and clean generated text
        
        Args:
            text: Generated text
            min_length: Minimum length in words
            max_length: Maximum length in words
            remove_prompt: Whether to remove the prompt
            clean_special_tokens: Whether to clean special tokens
            
        Returns:
            Filtered text
        """
        # Remove prompt if needed
        if remove_prompt and "<|endoftext|>" in text:
            text = text.split("<|endoftext|>")[1]
            
        # Clean special tokens
        if clean_special_tokens:
            special_tokens = ["<style=", "<topic=", "<|endoftext|>"]
            for token in special_tokens:
                text = text.replace(token, "")
                
        # Apply length constraints
        tokens = word_tokenize(text)
        
        if min_length and len(tokens) < min_length:
            return ""
            
        if max_length and len(tokens) > max_length:
            text = ' '.join(tokens[:max_length])
            
        return text.strip()
        
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_return_sequences: int = 1,
        style: Optional[str] = None,
        topic: Optional[str] = None,
        format_template: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            num_return_sequences: Number of sequences to generate
            style: Writing style
            topic: Topic or theme
            format_template: Text format template
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text or list of texts
        """
        # Prepare prompt
        formatted_prompt = self._prepare_prompt(
            prompt,
            style=style,
            topic=topic,
            format_template=format_template
        )
        
        # Generate text
        outputs = self.generation_pipeline(
            formatted_prompt,
            max_length=max_length or self.max_length,
            min_length=min_length,
            num_return_sequences=num_return_sequences,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            **kwargs
        )
        
        # Process outputs
        generated_texts = [
            self._filter_generated_text(
                output['generated_text'],
                min_length=min_length,
                max_length=max_length,
                remove_prompt=True
            )
            for output in outputs
        ]
        
        return generated_texts[0] if num_return_sequences == 1 else generated_texts
        
    def generate_with_keywords(
        self,
        keywords: List[str],
        max_length: Optional[int] = None,
        style: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text incorporating given keywords
        
        Args:
            keywords: List of keywords to include
            max_length: Maximum length of generated text
            style: Writing style
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        # Create prompt with keywords
        prompt = f"สร้างข้อความที่มีคำสำคัญต่อไปนี้: {', '.join(keywords)}\n\nข้อความ:"
        
        # Generate text
        text = self.generate(
            prompt,
            max_length=max_length,
            style=style,
            **kwargs
        )
        
        return text
        
    def continue_text(
        self,
        text: str,
        min_new_length: int = 50,
        max_new_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Continue existing text
        
        Args:
            text: Existing text to continue
            min_new_length: Minimum length of new text
            max_new_length: Maximum length of new text
            **kwargs: Additional arguments for generation
            
        Returns:
            Continued text
        """
        # Generate continuation
        continuation = self.generate(
            text,
            min_length=len(word_tokenize(text)) + min_new_length,
            max_length=len(word_tokenize(text)) + (max_new_length or self.max_length),
            **kwargs
        )
        
        # Ensure smooth continuation
        if not continuation.startswith(text):
            continuation = text + " " + continuation
            
        return continuation
        
    def generate_with_control(
        self,
        prompt: str,
        control_codes: Dict[str, Any],
        **kwargs
    ) -> str:
        """Generate text with fine-grained control
        
        Args:
            prompt: Input prompt
            control_codes: Dictionary of control parameters
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated text
        """
        # Process control codes
        style = control_codes.get('style')
        topic = control_codes.get('topic')
        format_template = control_codes.get('format')
        
        # Additional control parameters
        length = control_codes.get('length', self.max_length)
        temperature = control_codes.get('temperature', self.temperature)
        top_p = control_codes.get('top_p', self.top_p)
        
        # Generate text with controls
        text = self.generate(
            prompt,
            max_length=length,
            style=style,
            topic=topic,
            format_template=format_template,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return text
        
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **kwargs: Additional arguments for generation
            
        Returns:
            List of generated texts
        """
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Generate for batch
            batch_outputs = [
                self.generate(prompt, **kwargs)
                for prompt in batch_prompts
            ]
            
            results.extend(batch_outputs)
            
        return results 