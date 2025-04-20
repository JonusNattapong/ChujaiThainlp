"""
Enhanced benchmarking utilities for Thai NLP tasks
"""
import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from thainlp.tokenization import word_tokenize

class ThaiNLPEvaluator:
    """Enhanced evaluator for Thai language tasks"""
    
    def __init__(self, task: str, dataset_path: Union[str, Path] = None):
        self.task = task
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.thai_metrics = {
            'word_seg_accuracy': None,
            'thai_char_accuracy': None,
            'thai_oov_rate': None
        }
        
    def load_thai_dataset(self, dataset_name: str = None):
        """Load Thai benchmark dataset"""
        if dataset_name:
            return load_dataset(dataset_name)
        elif self.dataset_path:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        raise ValueError("No dataset provided")

    def calculate_thai_metrics(self, predictions: List, references: List):
        """Calculate Thai-specific metrics"""
        # Word segmentation accuracy
        correct_seg = 0
        total_tokens = 0
        
        # Character-level accuracy
        correct_chars = 0
        total_chars = 0
        
        # OOV rate calculation
        known_vocab = self._load_thai_vocab()
        oov_count = 0
        
        for pred, ref in zip(predictions, references):
            # Word segmentation evaluation
            pred_tokens = word_tokenize(pred)
            ref_tokens = word_tokenize(ref)
            total_tokens += len(ref_tokens)
            correct_seg += sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
            
            # Character-level evaluation
            for p_char, r_char in zip(pred, ref):
                if p_char == r_char and '\u0E00' <= r_char <= '\u0E7F':
                    correct_chars += 1
                if '\u0E00' <= r_char <= '\u0E7F':
                    total_chars += 1
            
            # OOV calculation
            for token in ref_tokens:
                if token not in known_vocab and any('\u0E00' <= c <= '\u0E7F' for c in token):
                    oov_count += 1
        
        self.thai_metrics = {
            'word_seg_accuracy': correct_seg / total_tokens if total_tokens else 0,
            'thai_char_accuracy': correct_chars / total_chars if total_chars else 0,
            'thai_oov_rate': oov_count / total_tokens if total_tokens else 0
        }
        return self.thai_metrics
    
    def _load_thai_vocab(self) -> Set[str]:
        """Load Thai vocabulary for OOV calculation"""
        # TODO: Replace with actual Thai vocabulary
        return set(word_tokenize(" ".join([
            "ครับ", "ค่ะ", "ไทย", "ภาษา", "ครับผม", "ขอบคุณ", "สวัสดี"
        ])))
    
    def evaluate_task(self, predictions: List, references: List) -> Dict:
        """Run comprehensive evaluation for Thai NLP task"""
        # Standard metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average='weighted'
        )
        
        # Thai-specific metrics
        thai_metrics = self.calculate_thai_metrics(predictions, references)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'thai_metrics': thai_metrics
        }

def create_thai_benchmark(task: str, output_dir: Path):
    """Initialize benchmark directory for Thai NLP task"""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'task': task,
        'thai_specific': True,
        'metrics': ['precision', 'recall', 'f1', 'word_seg_accuracy', 'thai_char_accuracy']
    }
    
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)