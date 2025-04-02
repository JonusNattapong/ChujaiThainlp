"""
Thai question answering system
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..similarity.sentence_similarity import SentenceSimilarity

class QuestionAnswering(TransformerBase):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.sentence_similarity = SentenceSimilarity()
        
    def answer_question(
        self,
        question: str,
        context: str,
        max_answer_len: int = 100
    ) -> Dict[str, any]:
        """Answer question based on context
        
        Args:
            question: Question text
            context: Context text to find answer in
            max_answer_len: Maximum length of answer
            
        Returns:
            Dict containing:
            - answer: Extracted answer text
            - score: Confidence score
            - start: Start position in context
            - end: End position in context
        """
        # Split context into sentences
        sentences = self._split_into_sentences(context)
        if not sentences:
            return {
                'answer': '',
                'score': 0.0,
                'start': -1,
                'end': -1
            }
            
        # Find most relevant sentence
        similar_sents = self.sentence_similarity.find_most_similar(
            question,
            sentences,
            method='transformer',
            top_k=3
        )
        
        if not similar_sents:
            return {
                'answer': '',
                'score': 0.0,
                'start': -1,
                'end': -1
            }
            
        # Get best matching sentence
        best_sent, score = similar_sents[0]
        
        # Find answer span in sentence
        answer_span = self._find_answer_span(
            question,
            best_sent,
            max_answer_len
        )
        
        if not answer_span:
            return {
                'answer': best_sent[:max_answer_len],
                'score': score * 0.5,
                'start': context.find(best_sent),
                'end': context.find(best_sent) + len(best_sent)
            }
            
        answer, start, end = answer_span
        context_start = context.find(best_sent)
        
        return {
            'answer': answer,
            'score': score,
            'start': context_start + start,
            'end': context_start + end
        }
        
    def answer_multiple(
        self,
        questions: List[str],
        context: str
    ) -> List[Dict[str, any]]:
        """Answer multiple questions on same context"""
        return [
            self.answer_question(q, context)
            for q in questions
        ]
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in {'。', '！', '？', '।', '។', '။', '၏', '?', '!', '.', '\n'}:
                if current:
                    sentences.append(''.join(current).strip())
                    current = []
                    
        if current:
            sentences.append(''.join(current).strip())
            
        return [s for s in sentences if s]
        
    def _find_answer_span(
        self,
        question: str,
        sentence: str,
        max_len: int
    ) -> Optional[Tuple[str, int, int]]:
        """Find best answer span in sentence"""
        # Tokenize
        q_tokens = word_tokenize(question)
        s_tokens = word_tokenize(sentence)
        
        # Find matching tokens
        matches = []
        for i, token in enumerate(s_tokens):
            if token in q_tokens:
                matches.append(i)
                
        if not matches:
            return None
            
        # Find best span around matches
        best_span = None
        best_score = -1
        
        for start in range(len(s_tokens)):
            for end in range(start + 1, len(s_tokens) + 1):
                span = s_tokens[start:end]
                span_text = ''.join(span)
                
                if len(span_text) > max_len:
                    break
                    
                # Score span based on:
                # - Number of question tokens
                # - Length of span
                # - Distance to matches
                q_overlap = len(set(span) & set(q_tokens))
                closest_match = min(abs(i - start) + abs(i - end) 
                                  for i in matches)
                
                score = (q_overlap / len(q_tokens)) * \
                       (1 - len(span) / len(s_tokens)) * \
                       (1 / (1 + closest_match))
                       
                if score > best_score:
                    best_score = score
                    best_span = (span_text, start, end)
                    
        if best_span is None:
            return None
            
        return best_span