"""
Advanced Neural Machine Translation for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    MarianTokenizer,
    MarianMTModel
)
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from ..core.transformers import TransformerBase

class ThaiTranslator(TransformerBase):
    """Advanced neural machine translation for Thai language"""
    
    SUPPORTED_LANGUAGES = {
        'th': 'Thai',
        'en': 'English',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'vi': 'Vietnamese',
        'ms': 'Malay',
        'id': 'Indonesian'
    }
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        source_lang: str = 'th',
        target_lang: str = 'en',
        use_cuda: bool = True,
        max_length: int = 512,
        use_context: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ):
        """Initialize translator
        
        Args:
            model_name_or_path: Name or path of the model
            source_lang: Source language code
            target_lang: Target language code
            use_cuda: Whether to use GPU if available
            max_length: Maximum sequence length
            use_context: Whether to use context for translation
            few_shot_examples: List of few-shot example pairs
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            # Select appropriate model based on language pair
            model_name_or_path = self._get_default_model(source_lang, target_lang)
            
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.use_context = use_context
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="translation",
            **kwargs
        )
        
        # Initialize translation pipeline
        self.translation_pipeline = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if use_cuda and torch.cuda.is_available() else -1
        )
        
        # Initialize sentence transformer for context matching
        if use_context:
            self.sentence_encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
        # Store few-shot examples
        self.few_shot_examples = few_shot_examples or []
        
    def _get_default_model(self, source_lang: str, target_lang: str) -> str:
        """Get default model for language pair
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Model name
        """
        # Use M2M100 for all language pairs as a fallback
        return "facebook/m2m100_418M"

    def _encode_context(self, text: str) -> torch.Tensor:
        """Encode text for context matching
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        return self.sentence_encoder.encode(text, convert_to_tensor=True)
        
    def _get_relevant_examples(
        self,
        text: str,
        n_examples: int = 3
    ) -> List[Dict[str, str]]:
        """Get most relevant few-shot examples
        
        Args:
            text: Input text
            n_examples: Number of examples to retrieve
            
        Returns:
            List of relevant example pairs
        """
        if not self.few_shot_examples:
            return []
            
        # Encode input text
        text_embedding = self._encode_context(text)
        
        # Encode all examples
        example_embeddings = self.sentence_encoder.encode(
            [ex['source'] for ex in self.few_shot_examples],
            convert_to_tensor=True
        )
        
        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            text_embedding.unsqueeze(0),
            example_embeddings
        )
        
        # Get top-k examples
        top_k = torch.topk(similarities, min(n_examples, len(self.few_shot_examples)))
        relevant_examples = [
            self.few_shot_examples[idx]
            for idx in top_k.indices
        ]
        
        return relevant_examples
        
    def _get_context_window(
        self,
        text: str,
        context: Union[str, List[str]],
        window_size: int = 3
    ) -> str:
        """Extract relevant context window
        
        Args:
            text: Input text
            context: Context text or list of texts
            window_size: Size of context window
            
        Returns:
            Relevant context window
        """
        if isinstance(context, list):
            context = ' '.join(context)
            
        # Tokenize text and context
        text_tokens = word_tokenize(text)
        context_tokens = word_tokenize(context)
        
        # Find position of text in context
        text_str = ''.join(text_tokens)
        context_str = ''.join(context_tokens)
        pos = context_str.find(text_str)
        
        if pos == -1:
            return context
            
        # Extract window around text
        start = max(0, pos - window_size)
        end = min(len(context_tokens), pos + len(text_tokens) + window_size)
        window_tokens = context_tokens[start:end]
        
        return ''.join(window_tokens)
        
    def add_few_shot_example(self, source: str, target: str):
        """Add new few-shot example
        
        Args:
            source: Source text
            target: Target text
        """
        self.few_shot_examples.append({
            'source': source,
            'target': target
        })
        
    def translate(
        self,
        text: Union[str, List[str]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        context: Optional[Union[str, List[str]]] = None,
        use_few_shot: bool = True,
        return_alternatives: bool = False,
        num_beams: int = 5,
        **kwargs
    ) -> Union[str, List[str], Dict[str, Any]]:
        """Translate text
        
        Args:
            text: Input text or list of texts
            source_lang: Source language code (overrides default)
            target_lang: Target language code (overrides default)
            context: Optional context for translation
            use_few_shot: Whether to use few-shot examples
            return_alternatives: Whether to return alternative translations
            num_beams: Number of beams for beam search
            **kwargs: Additional arguments for translation pipeline
            
        Returns:
            Translated text(s) or dictionary with translations and metadata
        """
        # Handle language overrides
        src_lang = source_lang or self.source_lang
        tgt_lang = target_lang or self.target_lang
        
        # Validate languages
        if src_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {src_lang}")
        if tgt_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported target language: {tgt_lang}")
            
        # Handle single text input
        single_text = isinstance(text, str)
        if single_text:
            text = [text]
            
        all_results = []
        
        for txt in text:
            result = {'source': txt}
            
            # Get relevant context
            if context and self.use_context:
                context_window = self._get_context_window(txt, context)
                result['context'] = context_window
            else:
                context_window = None
                
            # Get relevant few-shot examples
            if use_few_shot:
                examples = self._get_relevant_examples(txt)
                result['examples'] = examples
                
                if examples:
                    # Prepare prompt with examples
                    prompt = ""
                    for ex in examples:
                        prompt += f"{ex['source']} => {ex['target']}\n"
                    prompt += f"{txt} =>"
                    
                    result['prompt'] = prompt
                    txt = prompt
                    
            # Translate
            translations = self.translation_pipeline(
                txt,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                num_beams=num_beams,
                num_return_sequences=num_beams if return_alternatives else 1,
                **kwargs
            )
            
            if return_alternatives:
                result['translations'] = [
                    {
                        'text': t['translation_text'],
                        'score': t['score']
                    }
                    for t in translations
                ]
                result['best_translation'] = translations[0]['translation_text']
            else:
                result['translation'] = translations[0]['translation_text']
            
            all_results.append(result)
            
        if single_text:
            return all_results[0]
        return all_results
        
    def batch_translate(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Translate multiple texts
        
        Args:
            texts: List of texts to translate
            batch_size: Batch size for processing
            **kwargs: Additional arguments for translation
            
        Returns:
            List of translation results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.translate(batch_texts, **kwargs)
            results.extend(batch_results)
            
        return results
        
    def zero_shot_translate(
        self,
        text: str,
        target_lang: str,
        pivot_lang: str = 'en',
        **kwargs
    ) -> Dict[str, Any]:
        """Perform zero-shot translation through pivot language
    
    Args:
            text: Input text
            target_lang: Target language code
            pivot_lang: Pivot language code
            **kwargs: Additional arguments for translation
        
    Returns:
            Dictionary with translation results
        """
        # First translate to pivot language
        pivot_result = self.translate(
            text,
            target_lang=pivot_lang,
            **kwargs
        )
        
        # Then translate to target language
        final_result = self.translate(
            pivot_result['translation'],
            source_lang=pivot_lang,
            target_lang=target_lang,
            **kwargs
        )
        
        return {
            'source': text,
            'pivot': pivot_result['translation'],
            'translation': final_result['translation']
        } 