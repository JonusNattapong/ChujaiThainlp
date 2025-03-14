"""
Question Answering for Thai Text using Advanced Transformer Models
"""

from typing import List, Dict, Union, Optional, Any, Tuple
import torch
import numpy as np
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline,
    QuestionAnsweringPipeline
)
from pythainlp.tokenize import word_tokenize
from pythainlp.translate import translate
from ..core.transformers import TransformerBase

class ThaiQuestionAnswering(TransformerBase):
    """Advanced question answering for Thai text with multi-source and inferential capabilities"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_seq_length: int = 512,
        doc_stride: int = 128,
        max_answer_length: int = 100,
        min_confidence: float = 0.1,
        use_cuda: bool = True,
        **kwargs
    ):
        """Initialize question answering model
        
        Args:
            model_name_or_path: Name or path of the model
            max_seq_length: Maximum sequence length
            doc_stride: Stride for splitting long documents
            max_answer_length: Maximum answer length
            min_confidence: Minimum confidence score for answers
            use_cuda: Whether to use GPU if available
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "wangchanberta-base-att-spm-uncased"
            
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.min_confidence = min_confidence
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="question-answering",
            **kwargs
        )
        
        # Initialize QA pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if use_cuda and torch.cuda.is_available() else -1
        )
        
    def _split_context(self, context: str) -> List[str]:
        """Split long context into overlapping chunks
        
        Args:
            context: Input context text
            
        Returns:
            List of context chunks
        """
        # Tokenize context
        tokens = word_tokenize(context)
        
        # Calculate number of tokens per chunk
        chunk_size = self.max_seq_length - 50  # Reserve space for question and special tokens
        
        # Split into chunks with overlap
        chunks = []
        for i in range(0, len(tokens), chunk_size - self.doc_stride):
            chunk = tokens[i:i + chunk_size]
            chunks.append(''.join(chunk))
            
        return chunks
        
    def _merge_answers(
        self,
        answers: List[Dict[str, Any]],
        remove_duplicates: bool = True
    ) -> List[Dict[str, Any]]:
        """Merge answers from multiple chunks
        
        Args:
            answers: List of answer dictionaries
            remove_duplicates: Whether to remove duplicate answers
            
        Returns:
            Merged and sorted answer list
        """
        # Filter by confidence
        answers = [a for a in answers if a['score'] >= self.min_confidence]
        
        if remove_duplicates:
            # Remove duplicates based on answer text
            seen = set()
            unique_answers = []
            for ans in answers:
                if ans['answer'] not in seen:
                    seen.add(ans['answer'])
                    unique_answers.append(ans)
            answers = unique_answers
            
        # Sort by confidence score
        answers.sort(key=lambda x: x['score'], reverse=True)
        
        return answers
        
    def _get_supporting_facts(
        self,
        question: str,
        context: str,
        answer: str,
        k: int = 3
    ) -> List[str]:
        """Extract supporting facts for answer
        
        Args:
            question: Input question
            context: Input context
            answer: Generated answer
            k: Number of supporting facts to extract
            
        Returns:
            List of supporting fact sentences
        """
        from pythainlp.tokenize import sent_tokenize
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Split context into sentences
        sentences = sent_tokenize(context)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([question + " " + answer] + sentences)
        
        # Calculate similarity between question+answer and each sentence
        similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        
        # Get top-k most similar sentences
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        supporting_facts = [sentences[i] for i in top_k_idx]
        
        return supporting_facts
        
    def _evaluate_answer_reliability(
        self,
        question: str,
        answer: str,
        supporting_facts: List[str]
    ) -> Dict[str, float]:
        """Evaluate reliability of answer
        
        Args:
            question: Input question
            answer: Generated answer
            supporting_facts: Supporting fact sentences
            
        Returns:
            Dictionary of reliability metrics
        """
        # Calculate answer consistency
        consistency_scores = []
        for fact in supporting_facts:
            inputs = self.tokenizer(
                question,
                fact,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            start_probs = torch.softmax(outputs.start_logits, dim=-1)
            end_probs = torch.softmax(outputs.end_logits, dim=-1)
            consistency = float(start_probs.max() * end_probs.max())
            consistency_scores.append(consistency)
            
        # Calculate overall metrics
        metrics = {
            'consistency': np.mean(consistency_scores),
            'support_count': len(supporting_facts),
            'confidence_variance': np.var(consistency_scores)
        }
        
        # Calculate overall reliability score
        metrics['reliability_score'] = (
            0.4 * metrics['consistency'] +
            0.4 * min(1.0, metrics['support_count'] / 3) +
            0.2 * (1 - min(1.0, metrics['confidence_variance']))
        )
        
        return metrics
        
    def answer_question(
        self,
        question: str,
        context: Union[str, List[str]],
        return_all_answers: bool = False,
        max_answers: int = 3,
        include_supporting_facts: bool = True,
        evaluate_reliability: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Answer question using context
        
        Args:
            question: Question in Thai
            context: Context text or list of texts
            return_all_answers: Whether to return multiple answers
            max_answers: Maximum number of answers to return
            include_supporting_facts: Whether to include supporting facts
            evaluate_reliability: Whether to evaluate answer reliability
            **kwargs: Additional arguments for QA pipeline
            
        Returns:
            Dictionary containing answer and metadata, or list of answers
        """
        # Handle multiple contexts
        if isinstance(context, list):
            all_chunks = []
            for ctx in context:
                chunks = self._split_context(ctx)
                all_chunks.extend(chunks)
        else:
            all_chunks = self._split_context(context)
            
        # Get answers from all chunks
        answers = []
        for chunk in all_chunks:
            result = self.qa_pipeline(
                question=question,
                context=chunk,
                handle_impossible_answer=True,
                max_answer_len=self.max_answer_length,
                **kwargs
            )
            
            if isinstance(result, dict):
                result = [result]
                
            answers.extend(result)
            
        # Merge and filter answers
        answers = self._merge_answers(answers)
        
        if not answers:
            return {
                'answer': None,
                'score': 0.0,
                'message': 'No answer found'
            }
            
        # Process each answer
        processed_answers = []
        for ans in answers[:max_answers]:
            result = {
                'answer': ans['answer'],
                'score': ans['score'],
                'context': ans['context']
            }
            
            # Get supporting facts
            if include_supporting_facts:
                result['supporting_facts'] = self._get_supporting_facts(
                    question,
                    ans['context'],
                    ans['answer']
                )
                
            # Evaluate reliability
            if evaluate_reliability:
                result['reliability'] = self._evaluate_answer_reliability(
                    question,
                    ans['answer'],
                    result.get('supporting_facts', [ans['context']])
                )
                
            processed_answers.append(result)
            
        return processed_answers if return_all_answers else processed_answers[0]
        
    def batch_answer_questions(
        self,
        questions: List[str],
        contexts: Union[List[str], List[List[str]]],
        batch_size: int = 8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions
        
        Args:
            questions: List of questions in Thai
            contexts: List of contexts or list of context lists
            batch_size: Batch size for processing
            **kwargs: Additional arguments for answering
            
        Returns:
            List of answer dictionaries
        """
        if len(questions) != len(contexts):
            raise ValueError("Number of questions must match number of contexts")
            
        answers = []
        
        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            batch_answers = [
                self.answer_question(q, c, **kwargs)
                for q, c in zip(batch_questions, batch_contexts)
            ]
            
            answers.extend(batch_answers)
            
        return answers
        
    def answer_with_inference(
        self,
        question: str,
        context: Union[str, List[str]],
        max_inference_steps: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Answer question with multi-step inference
        
        Args:
            question: Question in Thai
            context: Context text or list of texts
            max_inference_steps: Maximum number of inference steps
            **kwargs: Additional arguments for answering
            
        Returns:
            Dictionary containing answer and inference chain
        """
        # Initialize inference chain
        inference_chain = []
        current_question = question
        
        for step in range(max_inference_steps):
            # Get answer for current question
            result = self.answer_question(
                current_question,
                context,
                include_supporting_facts=True,
                evaluate_reliability=True,
                **kwargs
            )
            
            if not result['answer']:
                break
                
            inference_chain.append({
                'step': step + 1,
                'question': current_question,
                'answer': result['answer'],
                'supporting_facts': result.get('supporting_facts', []),
                'reliability': result.get('reliability', {})
            })
            
            # Generate follow-up question if needed
            if step < max_inference_steps - 1:
                current_question = f"Based on the fact that {result['answer']}, {question}"
            
        # Combine inference results
        final_answer = inference_chain[-1]['answer'] if inference_chain else None
        reliability = np.mean([
            step['reliability']['reliability_score']
            for step in inference_chain
            if 'reliability' in step
        ]) if inference_chain else 0.0
        
        return {
            'answer': final_answer,
            'inference_chain': inference_chain,
            'inference_reliability': reliability
        } 