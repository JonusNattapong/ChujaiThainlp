"""
Advanced Text Summarization for Thai Language
"""

from typing import List, Dict, Union, Optional, Any
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)
from pythainlp.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..core.transformers import TransformerBase

class ThaiSummarizer(TransformerBase):
    """Advanced text summarization for Thai language"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        mode: str = "abstractive",
        max_length: int = 1024,
        min_length: int = 50,
        use_cuda: bool = True,
        **kwargs
    ):
        """Initialize summarizer
        
        Args:
            model_name_or_path: Name or path of the model
            mode: Summarization mode (abstractive or extractive)
            max_length: Maximum input sequence length
            min_length: Minimum summary length
            use_cuda: Whether to use GPU if available
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "wangchanberta-base-att-spm-uncased"
            
        self.mode = mode
        self.max_length = max_length
        self.min_length = min_length
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="summarization",
            **kwargs
        )
        
        # Initialize summarization pipeline for abstractive mode
        if mode == "abstractive":
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if use_cuda and torch.cuda.is_available() else -1
            )
            
    def _get_sentence_scores(
        self,
        sentences: List[str],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Score sentences for extractive summarization
        
        Args:
            sentences: List of sentences
            top_n: Number of top sentences to return
            
        Returns:
            List of sentence scores and metadata
        """
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sentences)
        
        # Calculate sentence similarities
        similarities = cosine_similarity(vectors)
        
        # Calculate sentence scores
        scores = []
        for i, sentence in enumerate(sentences):
            # Score based on similarity with other sentences
            similarity_score = np.mean(similarities[i])
            
            # Score based on position (earlier sentences get higher scores)
            position_score = 1.0 / (i + 1)
            
            # Score based on sentence length
            length_score = len(word_tokenize(sentence)) / 100
            
            # Combine scores
            total_score = (
                0.4 * similarity_score +
                0.3 * position_score +
                0.3 * (1 - min(1.0, length_score))
            )
            
            scores.append({
                'sentence': sentence,
                'score': total_score,
                'position': i,
                'similarity': similarity_score
            })
            
        # Sort by score and return top_n
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_n]
        
    def _merge_sentences(
        self,
        sentences: List[str],
        max_length: Optional[int] = None
    ) -> str:
        """Merge sentences into coherent text
        
        Args:
            sentences: List of sentences
            max_length: Maximum length in words
            
        Returns:
            Merged text
        """
        if not max_length:
            return ' '.join(sentences)
            
        result = []
        current_length = 0
        
        for sent in sentences:
            words = word_tokenize(sent)
            if current_length + len(words) <= max_length:
                result.append(sent)
                current_length += len(words)
            else:
                break
                
        return ' '.join(result)
        
    def _clean_generated_summary(self, summary: str) -> str:
        """Clean and format generated summary
        
        Args:
            summary: Generated summary text
            
        Returns:
            Cleaned summary
        """
        # Remove redundant spaces
        summary = ' '.join(summary.split())
        
        # Ensure proper sentence endings
        if not summary.endswith(('.', '?', '!')):
            summary += '.'
            
        return summary
        
    def summarize(
        self,
        text: Union[str, List[str]],
        ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        return_metadata: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Generate summary of text
        
        Args:
            text: Input text or list of texts
            ratio: Target summary ratio for extractive mode
            max_length: Maximum summary length
            min_length: Minimum summary length
            return_metadata: Whether to return additional metadata
            **kwargs: Additional arguments for summarization
            
        Returns:
            Generated summary or dictionary with summary and metadata
        """
        # Handle list input
        if isinstance(text, list):
            text = ' '.join(text)
            
        # Set length constraints
        max_len = max_length or int(len(word_tokenize(text)) * ratio)
        min_len = min_length or self.min_length
        
        result = {'source_text': text}
        
        if self.mode == "abstractive":
            # Generate abstractive summary
            summary = self.summarization_pipeline(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                **kwargs
            )[0]['summary_text']
            
            summary = self._clean_generated_summary(summary)
            result['summary'] = summary
            result['mode'] = 'abstractive'
            
        else:
            # Generate extractive summary
            sentences = sent_tokenize(text)
            scored_sentences = self._get_sentence_scores(
                sentences,
                top_n=max(1, int(len(sentences) * ratio))
            )
            
            # Sort selected sentences by original position
            selected_sentences = sorted(
                scored_sentences,
                key=lambda x: x['position']
            )
            
            summary = self._merge_sentences(
                [s['sentence'] for s in selected_sentences],
                max_length=max_len
            )
            
            result['summary'] = summary
            result['mode'] = 'extractive'
            result['selected_sentences'] = selected_sentences
            
        if return_metadata:
            result['statistics'] = {
                'source_length': len(word_tokenize(text)),
                'summary_length': len(word_tokenize(result['summary'])),
                'compression_ratio': len(word_tokenize(result['summary'])) / len(word_tokenize(text))
            }
            return result
            
        return result['summary']
        
    def multi_document_summarize(
        self,
        documents: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate summary from multiple documents
        
        Args:
            documents: List of document texts
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with summary and metadata
        """
        # First summarize each document
        summaries = []
        for doc in documents:
            summary = self.summarize(doc, return_metadata=True, **kwargs)
            summaries.append(summary)
            
        # Then summarize the combined summaries
        combined_summary = self.summarize(
            [s['summary'] for s in summaries],
            return_metadata=True,
            **kwargs
        )
        
        return {
            'final_summary': combined_summary['summary'],
            'document_summaries': summaries,
            'statistics': {
                'num_documents': len(documents),
                'avg_document_length': np.mean([
                    s['statistics']['source_length']
                    for s in summaries
                ]),
                'final_compression_ratio': combined_summary['statistics']['compression_ratio']
            }
        }
        
    def controlled_summarize(
        self,
        text: str,
        target_length: int,
        style: Optional[str] = None,
        focus_points: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate summary with controlled length and style
        
        Args:
            text: Input text
            target_length: Target summary length in words
            style: Summary style (formal, casual, etc.)
            focus_points: List of points to focus on
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with summary and metadata
        """
        # Prepare control tokens
        control_tokens = []
        
        if style:
            control_tokens.append(f"<style={style}>")
            
        if focus_points:
            control_tokens.extend([f"<focus={point}>" for point in focus_points])
            
        # Add control tokens to input
        controlled_input = ' '.join(control_tokens + [text])
        
        # Generate summary
        summary = self.summarize(
            controlled_input,
            max_length=target_length,
            min_length=max(10, target_length - 20),
            return_metadata=True,
            **kwargs
        )
        
        # Add control metadata
        summary['control'] = {
            'target_length': target_length,
            'style': style,
            'focus_points': focus_points
        }
        
        return summary 