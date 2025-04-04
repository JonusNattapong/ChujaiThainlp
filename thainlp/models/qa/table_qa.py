"""
Advanced table-based question answering
"""
from typing import List, Dict, Optional, Union
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTableQuestionAnswering,
    BatchEncoding
)
from ...core.transformers import TransformerBase
from ...utils.monitoring import ProgressTracker

class TableQA(TransformerBase):
    """Question answering model for structured table data"""
    
    def __init__(self,
                 model_name: str = "google/tapas-large-finetuned-wtq",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        """Initialize table QA model
        
        Args:
            model_name: Name of table QA model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.model = AutoModelForTableQuestionAnswering.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
    def answer_question(self,
                       question: Union[str, List[str]],
                       table: pd.DataFrame,
                       return_scores: bool = True,
                       num_answers: int = 1) -> Union[Dict, List[Dict]]:
        """Answer questions using table data
        
        Args:
            question: Question or list of questions
            table: Pandas DataFrame containing table data
            return_scores: Whether to return confidence scores
            num_answers: Number of answers to return per question
            
        Returns:
            Dict or list of dicts containing:
            - answer: Extracted answer
            - score: Confidence score (if return_scores=True)
            - cells: List of cells used in answer
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
                table,
                return_scores=return_scores,
                num_answers=num_answers
            )
            all_results.extend(batch_results)
            self.progress.update(len(batch_questions))
            
        self.progress.end_task()
        
        return all_results[0] if single_query else all_results
    
    def _process_batch(self,
                      questions: List[str],
                      table: pd.DataFrame,
                      **kwargs) -> List[Dict]:
        """Process a batch of questions"""
        # Prepare inputs
        inputs = self._prepare_inputs(questions, table)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Process outputs
        batch_results = []
        for q_idx in range(len(questions)):
            # Get logits for this question
            logits = outputs.logits[q_idx]
            
            # Get cell selection probabilities
            cell_probs = torch.sigmoid(logits)
            
            # Get top cells
            top_cells = []
            scores = []
            
            # Find cells above threshold
            thresh = 0.5  # Confidence threshold
            cell_mask = cell_probs > thresh
            
            if cell_mask.any():
                # Get selected cells and their probabilities
                selected_cells = torch.nonzero(cell_mask)
                for row, col in selected_cells:
                    cell_value = table.iloc[row.item(), col.item()]
                    prob = cell_probs[row, col].item()
                    top_cells.append(str(cell_value))
                    scores.append(prob)
                    
                # Sort by probability
                cells_and_scores = sorted(
                    zip(top_cells, scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Take top k answers
                num_answers = kwargs.get('num_answers', 1)
                cells_and_scores = cells_and_scores[:num_answers]
                
                # Format results
                if kwargs.get('return_scores'):
                    if num_answers == 1:
                        result = {
                            'answer': cells_and_scores[0][0],
                            'score': cells_and_scores[0][1],
                            'cells': [cells_and_scores[0][0]]
                        }
                    else:
                        result = [
                            {
                                'answer': cell,
                                'score': score,
                                'cells': [cell]
                            }
                            for cell, score in cells_and_scores
                        ]
                else:
                    if num_answers == 1:
                        result = {
                            'answer': cells_and_scores[0][0],
                            'cells': [cells_and_scores[0][0]]
                        }
                    else:
                        result = [
                            {
                                'answer': cell,
                                'cells': [cell]
                            }
                            for cell, _ in cells_and_scores
                        ]
            else:
                # No cells above threshold
                result = {
                    'answer': '',
                    'cells': []
                }
                if kwargs.get('return_scores'):
                    result['score'] = 0.0
                    
            batch_results.append(result)
            
        return batch_results
    
    def _prepare_inputs(self,
                       questions: List[str],
                       table: pd.DataFrame) -> BatchEncoding:
        """Prepare model inputs from questions and table"""
        # Convert table to string format
        table_data = []
        for _, row in table.iterrows():
            table_data.extend([str(cell) for cell in row])
            
        # Tokenize inputs
        inputs = self.tokenizer(
            questions,
            table_data,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Add table structure information
        table_shape = (len(table), len(table.columns))
        inputs['table_shape'] = torch.tensor([table_shape] * len(questions))
        
        return inputs.to(self.device)
    
    def fine_tune(self,
                 train_data: List[Dict],
                 val_data: Optional[List[Dict]] = None,
                 epochs: int = 3,
                 learning_rate: float = 3e-5):
        """Fine-tune the table QA model
        
        Args:
            train_data: List of dicts with 'question', 'table', 'answer' keys
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
                questions = [d['question'] for d in batch_data]
                tables = [d['table'] for d in batch_data]
                answers = [d['answer'] for d in batch_data]
                
                # Process each table-question pair
                for table, question, answer in zip(tables, questions, answers):
                    inputs = self._prepare_inputs([question], table)
                    
                    # Find answer cells in table
                    answer_coords = []
                    for idx, row in table.iterrows():
                        for col, cell in enumerate(row):
                            if str(cell) == answer:
                                answer_coords.append((idx, col))
                                
                    if not answer_coords:
                        continue  # Skip if answer not found in table
                        
                    # Create answer matrix
                    answer_matrix = torch.zeros(
                        (len(table), len(table.columns)),
                        device=self.device
                    )
                    for row, col in answer_coords:
                        answer_matrix[row, col] = 1
                        
                    # Forward pass
                    outputs = self.model(**inputs, labels=answer_matrix)
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
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data in val_data:
                        question = data['question']
                        table = data['table']
                        true_answer = data['answer']
                        
                        # Get model prediction
                        pred = self.answer_question(question, table)
                        pred_answer = pred['answer']
                        
                        # Update metrics
                        if pred_answer == true_answer:
                            correct += 1
                        total += 1
                        
                accuracy = correct / total
                print(f"Validation Accuracy: {accuracy:.4f}")
                
                self.model.train()