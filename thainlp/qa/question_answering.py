"""
Advanced question answering system supporting both text and table QA
"""
from typing import List, Dict, Tuple, Optional, Union
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)
from sentence_transformers import SentenceTransformer, util
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..similarity.sentence_similarity import SentenceSimilarity

class QuestionAnswering(TransformerBase):
    """Question answering model supporting text and table inputs"""
    
    def __init__(self, 
                 model_name: str = "xlm-roberta-large-squad2",
                 retriever_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8,
                 max_seq_length: int = 384,
                 doc_stride: int = 128):
        """Initialize QA model
        
        Args:
            model_name: Name of QA model
            retriever_model: Model for passage retrieval
            device: Device to run model on
            batch_size: Batch size for processing
            max_seq_length: Maximum sequence length
            doc_stride: Stride for document processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        
        # Load models
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load retriever model
        self.retriever = SentenceTransformer(retriever_model).to(device)
        
        # Table QA model
        self.table_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/tapas-large-finetuned-wtq"
        ).to(device)
        self.table_tokenizer = AutoTokenizer.from_pretrained(
            "google/tapas-large-finetuned-wtq"
        )
        
    def answer_question(self,
                       question: Union[str, List[str]],
                       context: Union[str, pd.DataFrame],
                       max_answer_len: int = 100,
                       return_scores: bool = True,
                       num_answers: int = 1) -> Union[Dict, List[Dict]]:
        """Answer questions using either text or table context
        
        Args:
            question: Question or list of questions
            context: Text passage or pandas DataFrame
            max_answer_len: Maximum answer length
            return_scores: Whether to return confidence scores
            num_answers: Number of answers to return per question
            
        Returns:
            Dict or list of dicts containing:
            - answer: Extracted answer text
            - score: Confidence score (if return_scores=True)
            - start: Start position in context (text QA only)
            - end: End position in context (text QA only)
        """
        # Handle single question
        if isinstance(question, str):
            question = [question]
            single_query = True
        else:
            single_query = False
            
        # Select QA approach based on context type
        if isinstance(context, pd.DataFrame):
            results = self._table_qa(question, context, num_answers)
        else:
            results = self._text_qa(
                question, 
                context,
                max_answer_len,
                return_scores,
                num_answers
            )
            
        return results[0] if single_query else results
    
    def _text_qa(self,
                 questions: List[str],
                 context: str,
                 max_answer_len: int,
                 return_scores: bool,
                 num_answers: int) -> List[Dict]:
        """Process text-based questions"""
        # Split context into passages
        passages = self._split_into_passages(context)
        
        # Get passage embeddings
        passage_embeddings = self.retriever.encode(
            passages, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        results = []
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            
            # Get question embeddings
            question_embeddings = self.retriever.encode(
                batch_questions,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Find relevant passages
            scores = util.cos_sim(question_embeddings, passage_embeddings)
            top_k = min(3, len(passages))  # Use top 3 passages
            top_passages = {}
            
            for q_idx, q_scores in enumerate(scores):
                top_indices = q_scores.topk(top_k).indices
                q_passages = [passages[idx] for idx in top_indices]
                top_passages[q_idx] = q_passages
                
            # Get answers from relevant passages
            batch_results = []
            for q_idx, q in enumerate(batch_questions):
                q_results = []
                
                for passage in top_passages[q_idx]:
                    # Prepare inputs
                    inputs = self.tokenizer(
                        q,
                        passage,
                        max_length=self.max_seq_length,
                        truncation=True,
                        stride=self.doc_stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        
                    # Process outputs
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    
                    # Get top answer spans
                    start_end_pairs = self._get_best_spans(
                        start_logits[0],
                        end_logits[0],
                        max_answer_len,
                        num_answers
                    )
                    
                    for start_idx, end_idx, score in start_end_pairs:
                        answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
                        answer = self.tokenizer.decode(answer_tokens)
                        
                        result = {
                            'answer': answer,
                            'context': passage,
                        }
                        
                        if return_scores:
                            result['score'] = score.item()
                            
                        q_results.append(result)
                
                # Sort by score
                if return_scores:
                    q_results = sorted(
                        q_results, 
                        key=lambda x: x['score'],
                        reverse=True
                    )[:num_answers]
                else:
                    q_results = q_results[:num_answers]
                    
                if num_answers == 1:
                    batch_results.append(q_results[0])
                else:
                    batch_results.append(q_results)
                    
            results.extend(batch_results)
            
        return results
    
    def _table_qa(self,
                  questions: List[str],
                  table: pd.DataFrame,
                  num_answers: int) -> List[Dict]:
        """Process table-based questions"""
        results = []
        
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            
            # Prepare inputs
            inputs = self.table_tokenizer(
                table=table,
                queries=batch_questions,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.table_model.generate(
                    **inputs,
                    max_length=64,
                    num_return_sequences=num_answers
                )
                
            # Process outputs
            batch_results = []
            for q_idx in range(len(batch_questions)):
                q_outputs = outputs[q_idx * num_answers:(q_idx + 1) * num_answers]
                answers = self.table_tokenizer.batch_decode(
                    q_outputs, 
                    skip_special_tokens=True
                )
                
                if num_answers == 1:
                    result = {'answer': answers[0]}
                else:
                    result = {'answers': answers}
                    
                batch_results.append(result)
                
            results.extend(batch_results)
            
        return results
    
    def _get_best_spans(self,
                       start_logits: torch.Tensor,
                       end_logits: torch.Tensor,
                       max_len: int,
                       num_spans: int) -> List[Tuple[int, int, float]]:
        """Get best answer spans from logits"""
        # Get top start/end scores
        top_k = min(10, len(start_logits))  # Consider top 10 positions
        start_top = start_logits.topk(top_k)
        end_top = end_logits.topk(top_k)
        
        # Find best spans
        best_spans = []
        for start_idx in start_top.indices:
            for end_idx in end_top.indices:
                if end_idx < start_idx or end_idx - start_idx + 1 > max_len:
                    continue
                    
                score = start_logits[start_idx] + end_logits[end_idx]
                best_spans.append((start_idx, end_idx, score))
                
        # Sort by score and return top spans
        best_spans = sorted(best_spans, key=lambda x: x[2], reverse=True)
        return best_spans[:num_spans]
    
    def _split_into_passages(self, text: str, max_len: int = 500) -> List[str]:
        """Split text into overlapping passages"""
        sentences = []
        current = []
        current_len = 0
        
        # Split into sentences
        for char in text:
            current.append(char)
            if char in {'。', '！', '？', '।', '។', '။', '၏', '?', '!', '.', '\n'}:
                if current:
                    sentence = ''.join(current).strip()
                    sentences.append(sentence)
                    current = []
                    
        if current:
            sentences.append(''.join(current).strip())
            
        # Combine into passages
        passages = []
        current_passage = []
        
        for sentence in sentences:
            if current_len + len(sentence) > max_len and current_passage:
                passages.append(' '.join(current_passage))
                current_passage = []
                current_len = 0
                
            current_passage.append(sentence)
            current_len += len(sentence)
            
        if current_passage:
            passages.append(' '.join(current_passage))
            
        return [p for p in passages if p]
    
    def fine_tune(self,
                 train_data: List[Dict],
                 val_data: Optional[List[Dict]] = None,
                 epochs: int = 3,
                 learning_rate: float = 3e-5):
        """Fine-tune the QA model
        
        Args:
            train_data: List of dicts with 'question', 'context', 'answer' keys
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
                    [d['question'] for d in batch_data],
                    [d['context'] for d in batch_data],
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Prepare labels
                start_positions = []
                end_positions = []
                
                for data in batch_data:
                    answer_start = data['context'].find(data['answer'])
                    answer_end = answer_start + len(data['answer'])
                    
                    # Convert to token positions
                    char_to_token = inputs.char_to_token(len(start_positions), answer_start)
                    if char_to_token is None:
                        char_to_token = inputs.char_to_token(len(start_positions), answer_start - 1)
                    start_positions.append(char_to_token)
                    
                    char_to_token = inputs.char_to_token(len(end_positions), answer_end)
                    if char_to_token is None:
                        char_to_token = inputs.char_to_token(len(end_positions), answer_end - 1)
                    end_positions.append(char_to_token)
                
                start_positions = torch.tensor(start_positions).to(self.device)
                end_positions = torch.tensor(end_positions).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    **inputs,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
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
                        
                        inputs = self.tokenizer(
                            [d['question'] for d in batch_data],
                            [d['context'] for d in batch_data],
                            max_length=self.max_seq_length,
                            truncation=True,
                            padding=True,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        start_positions = []
                        end_positions = []
                        for data in batch_data:
                            answer_start = data['context'].find(data['answer'])
                            answer_end = answer_start + len(data['answer'])
                            
                            char_to_token = inputs.char_to_token(
                                len(start_positions),
                                answer_start
                            )
                            start_positions.append(char_to_token)
                            
                            char_to_token = inputs.char_to_token(
                                len(end_positions),
                                answer_end
                            )
                            end_positions.append(char_to_token)
                            
                        start_positions = torch.tensor(start_positions).to(self.device)
                        end_positions = torch.tensor(end_positions).to(self.device)
                        
                        outputs = self.model(
                            **inputs,
                            start_positions=start_positions,
                            end_positions=end_positions
                        )
                        
                        val_loss += outputs.loss.item()
                        
                avg_val_loss = val_loss / len(val_data)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                self.model.train()

# Define ThaiQuestionAnswering class as a wrapper around QuestionAnswering
class ThaiQuestionAnswering(QuestionAnswering):
    """
    Thai Question Answering class, specialized for Thai language text
    """
    
    def __init__(self, 
                 model_name: str = "monsoon-nlp/bert-base-thai-squad",  # Thai-specific default
                 retriever_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8,
                 max_seq_length: int = 384,
                 doc_stride: int = 128):
        """Initialize Thai QA model
        
        Args:
            model_name: Name of QA model optimized for Thai
            retriever_model: Model for passage retrieval
            device: Device to run model on
            batch_size: Batch size for processing
            max_seq_length: Maximum sequence length
            doc_stride: Stride for document processing
        """
        super().__init__(
            model_name=model_name,
            retriever_model=retriever_model,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride
        )
        
    def preprocess_thai_text(self, text: str) -> str:
        """Preprocess Thai text for better QA performance
        
        Args:
            text: Thai text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Simple preprocessing for now - can be expanded with Thai-specific logic
        return text.strip()
        
    def answer_question(self,
                       question: Union[str, List[str]],
                       context: Union[str, pd.DataFrame],
                       max_answer_len: int = 100,
                       return_scores: bool = True,
                       num_answers: int = 1) -> Union[Dict, List[Dict]]:
        """Answer questions using either text or table context, with Thai preprocessing
        
        Args:
            question: Question or list of questions
            context: Text passage or pandas DataFrame
            max_answer_len: Maximum answer length
            return_scores: Whether to return confidence scores
            num_answers: Number of answers to return per question
            
        Returns:
            Dict or list of dicts containing:
            - answer: Extracted answer text
            - score: Confidence score (if return_scores=True)
            - start: Start position in context (text QA only)
            - end: End position in context (text QA only)
        """
        # Preprocess Thai text
        if isinstance(question, str):
            question = self.preprocess_thai_text(question)
        else:
            question = [self.preprocess_thai_text(q) for q in question]
            
        if isinstance(context, str):
            context = self.preprocess_thai_text(context)
            
        # Call parent implementation
        return super().answer_question(
            question=question,
            context=context,
            max_answer_len=max_answer_len,
            return_scores=return_scores,
            num_answers=num_answers
        )

# Module level function
def answer_question(question: str, context: str, **kwargs) -> Dict:
    """Answer a question based on context
    
    Args:
        question: Question to answer
        context: Text or table to find answer in
        **kwargs: Additional arguments to pass to ThaiQuestionAnswering
        
    Returns:
        Dictionary with answer and metadata
    """
    qa = ThaiQuestionAnswering()
    return qa.answer_question(question, context, **kwargs)