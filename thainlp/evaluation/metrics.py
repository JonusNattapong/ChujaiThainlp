"""
Comprehensive evaluation metrics for Thai NLP tasks
"""
from typing import List, Dict, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from ..tokenization import word_tokenize

class ThaiEvaluationMetrics:
    """Evaluation metrics specialized for Thai language tasks"""
    
    @staticmethod
    def text_classification_metrics(y_true: List[str],
                                 y_pred: List[str],
                                 labels: List[str] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for text classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            
        Returns:
            Dict containing various metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average='weighted'
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate per-class metrics
        class_metrics = {}
        if labels:
            for i, label in enumerate(labels):
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    labels=[label],
                    average='binary'
                )
                class_metrics[label] = {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1
                }
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_class': class_metrics,
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist()
        }
    
    @staticmethod
    def token_classification_metrics(true_sequences: List[List[Tuple[str, str]]],
                                  pred_sequences: List[List[Tuple[str, str]]],
                                  scheme: str = 'BIO') -> Dict[str, float]:
        """
        Calculate metrics for token classification tasks
        
        Args:
            true_sequences: List of true token-tag sequences
            pred_sequences: List of predicted token-tag sequences
            scheme: Tagging scheme ('BIO', 'BIOES', etc.)
            
        Returns:
            Dict containing token-level and chunk-level metrics
        """
        token_true = []
        token_pred = []
        chunk_true = []
        chunk_pred = []
        
        for true_seq, pred_seq in zip(true_sequences, pred_sequences):
            # Token level evaluation
            token_true.extend([tag for _, tag in true_seq])
            token_pred.extend([tag for _, tag in pred_seq])
            
            # Chunk level evaluation
            true_chunks = ThaiEvaluationMetrics._get_chunks(true_seq, scheme)
            pred_chunks = ThaiEvaluationMetrics._get_chunks(pred_seq, scheme)
            
            chunk_true.extend(true_chunks)
            chunk_pred.extend(pred_chunks)
        
        # Calculate token-level metrics
        token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
            token_true,
            token_pred,
            average='weighted'
        )
        
        # Calculate chunk-level metrics
        chunk_scores = ThaiEvaluationMetrics._evaluate_chunks(chunk_true, chunk_pred)
        
        return {
            'token_level': {
                'accuracy': accuracy_score(token_true, token_pred),
                'precision': token_precision,
                'recall': token_recall,
                'f1': token_f1
            },
            'chunk_level': chunk_scores
        }
    
    @staticmethod
    def qa_metrics(true_answers: List[Dict],
                  pred_answers: List[Dict],
                  thai_specific: bool = True) -> Dict[str, float]:
        """
        Calculate metrics for question answering tasks
        
        Args:
            true_answers: List of ground truth answers
            pred_answers: List of predicted answers
            thai_specific: Whether to use Thai-specific matching
            
        Returns:
            Dict containing exact match and F1 scores
        """
        exact_matches = []
        f1_scores = []
        
        for true, pred in zip(true_answers, pred_answers):
            # Get answer texts
            true_text = true['answer']
            pred_text = pred['answer']
            
            # Calculate exact match
            if thai_specific:
                exact_match = ThaiEvaluationMetrics._thai_exact_match(true_text, pred_text)
            else:
                exact_match = true_text.strip() == pred_text.strip()
            
            exact_matches.append(exact_match)
            
            # Calculate F1 score
            f1 = ThaiEvaluationMetrics._thai_token_f1(true_text, pred_text)
            f1_scores.append(f1)
        
        results = {
            'exact_match': np.mean(exact_matches),
            'f1': np.mean(f1_scores)
        }
        
        # Add position-based metrics for extractive QA
        if all('start' in ans and 'end' in ans for ans in true_answers + pred_answers):
            position_scores = ThaiEvaluationMetrics._position_based_metrics(
                true_answers,
                pred_answers
            )
            results.update(position_scores)
        
        return results
    
    @staticmethod
    def _thai_exact_match(text1: str, text2: str) -> bool:
        """Thai-specific exact matching considering variations"""
        # Normalize texts
        text1 = ThaiEvaluationMetrics._normalize_thai_text(text1)
        text2 = ThaiEvaluationMetrics._normalize_thai_text(text2)
        
        return text1 == text2
    
    @staticmethod
    def _thai_token_f1(text1: str, text2: str) -> float:
        """Calculate F1 score between two Thai texts"""
        # Tokenize texts
        tokens1 = set(word_tokenize(text1))
        tokens2 = set(word_tokenize(text2))
        
        # Calculate overlap
        common = tokens1 & tokens2
        
        # Handle empty sets
        if not tokens1 or not tokens2:
            return int(tokens1 == tokens2)
        
        precision = len(common) / len(tokens2)
        recall = len(common) / len(tokens1)
        
        if precision + recall == 0:
            return 0
            
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def _normalize_thai_text(text: str) -> str:
        """Normalize Thai text for comparison"""
        # Remove whitespace and convert to lowercase
        text = ''.join(text.lower().split())
        
        # Normalize Thai characters
        char_map = {
            'ะ': '',  # Remove redundant sara a
            'ั': '',  # Remove mai han akat
            'าา': 'า',  # Normalize repeated sara aa
            'ๆ': '',  # Remove mai yamok
            '์': ''   # Remove karan
        }
        
        for old, new in char_map.items():
            text = text.replace(old, new)
            
        return text
    
    @staticmethod
    def _get_chunks(sequence: List[Tuple[str, str]], scheme: str) -> List[Tuple[str, int, int]]:
        """Extract chunks from a tagged sequence"""
        chunks = []
        current_chunk = None
        
        for i, (token, tag) in enumerate(sequence):
            if scheme == 'BIO':
                if tag.startswith('B-'):
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = (tag[2:], i, i)
                elif tag.startswith('I-'):
                    if current_chunk and current_chunk[0] == tag[2:]:
                        current_chunk = (current_chunk[0], current_chunk[1], i)
                else:  # 'O' tag
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = None
                        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    @staticmethod
    def _evaluate_chunks(true_chunks: List[Tuple[str, int, int]],
                        pred_chunks: List[Tuple[str, int, int]]) -> Dict[str, float]:
        """Evaluate chunk-level predictions"""
        correct = len(set(true_chunks) & set(pred_chunks))
        
        if not true_chunks:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
            
        precision = correct / len(pred_chunks) if pred_chunks else 0
        recall = correct / len(true_chunks)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def _position_based_metrics(true_answers: List[Dict],
                              pred_answers: List[Dict]) -> Dict[str, float]:
        """Calculate metrics based on answer positions"""
        position_matches = []
        position_scores = []
        
        for true, pred in zip(true_answers, pred_answers):
            # Check exact position match
            position_match = (true['start'] == pred['start'] and 
                            true['end'] == pred['end'])
            position_matches.append(position_match)
            
            # Calculate position overlap score
            true_range = set(range(true['start'], true['end'] + 1))
            pred_range = set(range(pred['start'], pred['end'] + 1))
            
            overlap = len(true_range & pred_range)
            union = len(true_range | pred_range)
            
            position_scores.append(overlap / union if union > 0 else 0)
        
        return {
            'position_match': np.mean(position_matches),
            'position_overlap': np.mean(position_scores)
        }