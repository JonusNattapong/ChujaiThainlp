"""
Advanced text-based question answering for Thai and English
"""
from typing import List, Dict, Tuple, Optional, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)
from sentence_transformers import SentenceTransformer, util
from ...core.transformers import TransformerBase
from ...utils.monitoring import ProgressTracker

class TextQA(TransformerBase):
    """Question answering model for text passages"""
    
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
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
    def answer_question(self,
                       question: Union[str, List[str]],
                       context: str,
                       max_answer_len: int = 100,
                       return_scores: bool = True,
                       num_answers: int = 1) -> Union[Dict, List[Dict]]:
        """Answer questions using text context
        
        Args:
            question: Question or list of questions
            context: Text passage for context
            max_answer_len: Maximum answer length
            return_scores: Whether to return confidence scores
            num_answers: Number of answers to return per question
            
        Returns:
            Dict or list of dicts containing:
            - answer: Extracted answer text
            - score: Confidence score (if return_scores=True)
            - start: Start position in context
            - end: End position in context
        """
        # Handle single question
        if isinstance(question, str):
            question = [question]
            single_query = True
        else:
            single_query = False
            
        all_results = []
        self.progress.start_task(len(question))
        
        # Process in batches
        for i in range(0, len(question), self.batch_size):
            batch_questions = question[i:i + self.batch_size]
            batch_results = self._process_batch(
                batch_questions,
                context,
                max_answer_len=max_answer_len,
                return_scores=return_scores,
                num_answers=num_answers
            )
            all_results.extend(batch_results)
            self.progress.update(len(batch_questions))
            
        self.progress.end_task()
        
        return all_results[0] if single_query else all_results
    
    def _process_batch(self,
                      questions: List[str],
                      context: str,
                      **kwargs) -> List[Dict]:
        """Process a batch of questions"""
        # Prepare inputs
        inputs = self.tokenizer(
            questions,
            context,
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
            
        # Process outputs for each question
        batch_results = []
        for q_idx, question in enumerate(questions):
            # Get start/end logits
            start_logits = outputs.start_logits[q_idx]
            end_logits = outputs.end_logits[q_idx]
            
            # Get best spans
            start_end_pairs = self._get_best_spans(
                start_logits,
                end_logits,
                kwargs.get('max_answer_len', 100),
                kwargs.get('num_answers', 1)
            )
            
            # Convert to answer text
            results = []
            for start_idx, end_idx, score in start_end_pairs:
                answer_tokens = inputs.input_ids[q_idx][start_idx:end_idx + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                result = {
                    'answer': answer,
                    'start': start_idx,
                    'end': end_idx
                }
                
                if kwargs.get('return_scores'):
                    result['score'] = score.item()
                    
                results.append(result)
                
            # Return single result if num_answers=1
            if kwargs.get('num_answers', 1) == 1:
                batch_results.append(results[0])
            else:
                batch_results.append(results)
                
        return batch_results
    
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