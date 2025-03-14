"""
Advanced Text-to-Text Generation for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from pythainlp.tokenize import word_tokenize
from ..core.transformers import TransformerBase

class ThaiText2Text(TransformerBase):
    """Advanced text-to-text generation for Thai language"""
    
    TASK_PREFIXES = {
        'summarize': 'สรุป: ',
        'translate': 'แปล: ',
        'simplify': 'ทำให้เข้าใจง่าย: ',
        'elaborate': 'อธิบายเพิ่มเติม: ',
        'formalize': 'ทำให้เป็นทางการ: ',
        'correct': 'แก้ไขไวยากรณ์: ',
        'question': 'สร้างคำถาม: ',
        'answer': 'ตอบคำถาม: ',
        'paraphrase': 'เขียนใหม่: '
    }
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_length: int = 512,
        min_length: int = 10,
        use_cuda: bool = True,
        **kwargs
    ):
        """Initialize text-to-text generator
        
        Args:
            model_name_or_path: Name or path of the model
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            use_cuda: Whether to use GPU if available
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "wannaphong/mt5-small-thai-qa"
            
        self.max_length = max_length
        self.min_length = min_length
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="text2text-generation",
            **kwargs
        )
        
        # Initialize generation pipeline
        self.generation_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if use_cuda and torch.cuda.is_available() else -1
        )
        
    def _prepare_input(
        self,
        text: str,
        task: str,
        context: Optional[str] = None,
        instructions: Optional[str] = None
    ) -> str:
        """Prepare input text with task prefix and context
        
        Args:
            text: Input text
            task: Task type
            context: Optional context
            instructions: Optional specific instructions
            
        Returns:
            Formatted input text
        """
        # Get task prefix
        prefix = self.TASK_PREFIXES.get(task, '')
        
        # Combine components
        components = []
        
        if context:
            components.append(f"บริบท: {context}")
            
        if instructions:
            components.append(f"คำแนะนำ: {instructions}")
            
        components.append(f"{prefix}{text}")
        
        return '\n'.join(components)
        
    def _postprocess_output(
        self,
        text: str,
        clean_special_tokens: bool = True,
        ensure_thai_ending: bool = True
    ) -> str:
        """Clean and format generated text
        
        Args:
            text: Generated text
            clean_special_tokens: Whether to remove special tokens
            ensure_thai_ending: Whether to ensure proper Thai sentence ending
            
        Returns:
            Cleaned text
        """
        # Remove special tokens
        if clean_special_tokens:
            special_tokens = ["<pad>", "</s>", "<s>", "<unk>"]
            for token in special_tokens:
                text = text.replace(token, "")
                
        # Clean whitespace
        text = ' '.join(text.split())
        
        # Ensure proper ending
        if ensure_thai_ending and text and not text[-1] in ['ๆ', 'ฯ', '.', '?', '!']:
            text = text.rstrip() + ' ครับ/ค่ะ'
            
        return text.strip()
        
    def transform(
        self,
        text: str,
        task: str,
        context: Optional[str] = None,
        instructions: Optional[str] = None,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Transform text according to specified task
        
        Args:
            text: Input text
            task: Transformation task
            context: Optional context
            instructions: Optional specific instructions
            num_return_sequences: Number of outputs to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            Transformed text or list of texts
        """
        # Prepare input
        input_text = self._prepare_input(
            text,
            task,
            context=context,
            instructions=instructions
        )
        
        # Generate outputs
        outputs = self.generation_pipeline(
            input_text,
            max_length=self.max_length,
            min_length=self.min_length,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        # Process outputs
        results = [
            self._postprocess_output(output['generated_text'])
            for output in outputs
        ]
        
        return results[0] if num_return_sequences == 1 else results
        
    def batch_transform(
        self,
        texts: List[str],
        task: str,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """Transform multiple texts
        
        Args:
            texts: List of input texts
            task: Transformation task
            batch_size: Batch size for processing
            **kwargs: Additional arguments for transformation
            
        Returns:
            List of transformed texts
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Transform batch
            batch_results = [
                self.transform(text, task, **kwargs)
                for text in batch_texts
            ]
            
            results.extend(batch_results)
            
        return results
        
    def paraphrase(
        self,
        text: str,
        style: Optional[str] = None,
        num_variations: int = 3,
        **kwargs
    ) -> List[str]:
        """Generate multiple paraphrases of text
        
        Args:
            text: Input text
            style: Optional style specification
            num_variations: Number of variations to generate
            **kwargs: Additional arguments for generation
            
        Returns:
            List of paraphrased texts
        """
        instructions = f"สร้างประโยคที่มีความหมายเหมือนกัน {num_variations} แบบ"
        if style:
            instructions += f" ในรูปแบบ{style}"
            
        return self.transform(
            text,
            task="paraphrase",
            instructions=instructions,
            num_return_sequences=num_variations,
            **kwargs
        )
        
    def simplify(
        self,
        text: str,
        target_level: str = "ทั่วไป",
        **kwargs
    ) -> str:
        """Simplify text to target complexity level
        
        Args:
            text: Input text
            target_level: Target complexity level
            **kwargs: Additional arguments for generation
            
        Returns:
            Simplified text
        """
        instructions = f"ทำให้เข้าใจง่ายในระดับ{target_level}"
        
        return self.transform(
            text,
            task="simplify",
            instructions=instructions,
            **kwargs
        )
        
    def formalize(
        self,
        text: str,
        level: str = "ทางการ",
        **kwargs
    ) -> str:
        """Convert text to formal style
        
        Args:
            text: Input text
            level: Formality level
            **kwargs: Additional arguments for generation
            
        Returns:
            Formalized text
        """
        instructions = f"แปลงเป็นภาษา{level}"
        
        return self.transform(
            text,
            task="formalize",
            instructions=instructions,
            **kwargs
        )
        
    def elaborate(
        self,
        text: str,
        aspects: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Elaborate on text with additional details
        
        Args:
            text: Input text
            aspects: Aspects to elaborate on
            **kwargs: Additional arguments for generation
            
        Returns:
            Elaborated text
        """
        if aspects:
            instructions = f"อธิบายเพิ่มเติมในประเด็น: {', '.join(aspects)}"
        else:
            instructions = "อธิบายเพิ่มเติมให้ละเอียด"
            
        return self.transform(
            text,
            task="elaborate",
            instructions=instructions,
            **kwargs
        )
        
    def correct_grammar(
        self,
        text: str,
        explain: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, str]]:
        """Correct grammatical errors in text
        
        Args:
            text: Input text
            explain: Whether to include explanation of corrections
            **kwargs: Additional arguments for generation
            
        Returns:
            Corrected text or dictionary with text and explanations
        """
        instructions = "แก้ไขไวยากรณ์"
        if explain:
            instructions += "พร้อมคำอธิบาย"
            
        result = self.transform(
            text,
            task="correct",
            instructions=instructions,
            **kwargs
        )
        
        if explain:
            # Split result into correction and explanation
            parts = result.split("\nคำอธิบาย:")
            return {
                'corrected': parts[0].strip(),
                'explanation': parts[1].strip() if len(parts) > 1 else ""
            }
            
        return result 