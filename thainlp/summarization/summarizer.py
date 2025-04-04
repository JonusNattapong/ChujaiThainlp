"""
Advanced text summarization supporting both extractive and abstractive methods
"""
from typing import List, Dict, Set, Union, Optional, Tuple
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqGeneration,
    PreTrainedTokenizer
)
from rouge_score import rouge_scorer
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..extensions.monitoring import ProgressTracker

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation rules"""
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

class Summarizer(TransformerBase):
    """Text summarization with support for extractive and abstractive methods"""
    
    def __init__(self,
                 model_name: str = "facebook/bart-large-cnn",
                 method: str = "abstractive",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8):
        """Initialize summarizer
        
        Args:
            model_name: Name of pretrained model (for abstractive)
            method: Summarization method ('extractive' or 'abstractive')
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name)
        self.method = method
        self.device = device
        self.batch_size = batch_size
        
        # Set up progress tracking
        self.progress = ProgressTracker()
        
        # Initialize components based on method
        if method == "abstractive":
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.vectorizer = TfidfVectorizer(
                tokenizer=word_tokenize,
                stop_words=[]  # Stopwords handled by tokenizer
            )
            
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
    def summarize(self,
                 text: Union[str, List[str]],
                 ratio: float = 0.3,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 num_beams: int = 4,
                 length_penalty: float = 2.0,
                 num_return_sequences: int = 1,
                 return_scores: bool = False,
                 **kwargs) -> Union[str, List[str], Tuple[str, float], List[Tuple[str, float]]]:
        """Generate summary of input text(s)
        
        Args:
            text: Input text or texts
            ratio: Target length ratio for extractive summary
            min_length: Minimum summary length
            max_length: Maximum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            num_return_sequences: Number of summaries to return
            return_scores: Whether to return confidence scores
            **kwargs: Additional generation parameters
            
        Returns:
            Generated summary/summaries or tuples of (summary, score)
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False
            
        all_results = []
        self.progress.start_task(len(text))
        
        # Process in batches
        for i in range(0, len(text), self.batch_size):
            batch_texts = text[i:i + self.batch_size]
            
            if self.method == "abstractive":
                batch_results = self._abstractive_summarize(
                    batch_texts,
                    min_length=min_length,
                    max_length=max_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_sequences,
                    return_scores=return_scores,
                    **kwargs
                )
            else:
                batch_results = self._extractive_summarize(
                    batch_texts,
                    ratio=ratio,
                    min_length=min_length,
                    max_length=max_length,
                    return_scores=return_scores
                )
                
            all_results.extend(batch_results)
            self.progress.update(len(batch_texts))
            
        self.progress.end_task()
        
        # Return appropriate format
        if single_input:
            if num_return_sequences == 1:
                return all_results[0]
            return all_results[0:num_return_sequences]
        return all_results
    
    def _abstractive_summarize(self,
                             texts: List[str],
                             **kwargs) -> List[Union[str, Tuple[str, float]]]:
        """Generate abstractive summaries"""
        # Prepare inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": kwargs.get("max_length", 130),
            "min_length": kwargs.get("min_length", 30),
            "num_beams": kwargs.get("num_beams", 4),
            "length_penalty": kwargs.get("length_penalty", 2.0),
            "num_return_sequences": kwargs.get("num_return_sequences", 1),
            "early_stopping": True,
        }
        
        # Generate summaries
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs,
            output_scores=kwargs.get("return_scores", False),
            return_dict_in_generate=True
        )
        
        # Decode outputs
        summaries = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )
        
        # Calculate scores if requested
        if kwargs.get("return_scores"):
            scores = self._calculate_sequence_scores(outputs)
            results = []
            
            for i in range(len(texts)):
                text_summaries = []
                start_idx = i * kwargs.get("num_return_sequences", 1)
                end_idx = start_idx + kwargs.get("num_return_sequences", 1)
                
                for summary, score in zip(
                    summaries[start_idx:end_idx],
                    scores[start_idx:end_idx]
                ):
                    text_summaries.append((summary, score))
                    
                if kwargs.get("num_return_sequences", 1) == 1:
                    results.append(text_summaries[0])
                else:
                    results.append(text_summaries)
        else:
            results = []
            for i in range(len(texts)):
                start_idx = i * kwargs.get("num_return_sequences", 1)
                end_idx = start_idx + kwargs.get("num_return_sequences", 1)
                text_summaries = summaries[start_idx:end_idx]
                
                if kwargs.get("num_return_sequences", 1) == 1:
                    results.append(text_summaries[0])
                else:
                    results.append(text_summaries)
                    
        return results
    
    def _extractive_summarize(self,
                            texts: List[str],
                            **kwargs) -> List[Union[str, Tuple[str, float]]]:
        """Generate extractive summaries"""
        results = []
        
        for text in texts:
            # Split into sentences
            sentences = split_sentences(text)
            if len(sentences) <= 1:
                results.append((text, 1.0) if kwargs.get("return_scores") else text)
                continue
                
            # Convert to TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores
            similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
            scores = self._score_sentences(similarity_matrix)
            
            # Select top sentences
            ratio = kwargs.get("ratio", 0.3)
            num_sentences = max(1, int(len(sentences) * ratio))
            selected_indices = np.argsort(scores)[-num_sentences:]
            selected_indices.sort()
            
            # Build summary
            summary_sentences = [sentences[i] for i in selected_indices]
            summary = ' '.join(summary_sentences)
            
            # Apply length constraints
            max_length = kwargs.get("max_length")
            min_length = kwargs.get("min_length")
            
            if min_length and len(summary) < min_length and len(text) > min_length:
                summary = text[:max_length] if max_length else text
            elif max_length and len(summary) > max_length:
                summary = summary[:max_length]
                
            if kwargs.get("return_scores"):
                mean_score = float(np.mean(scores[selected_indices]))
                results.append((summary, mean_score))
            else:
                results.append(summary)
                
        return results
    
    def _score_sentences(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Score sentences using TextRank algorithm"""
        damping = 0.85
        epsilon = 1e-8
        max_iter = 100
        
        n_sentences = len(similarity_matrix)
        
        # Normalize similarity matrix
        norm = similarity_matrix.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        transition_matrix = similarity_matrix / norm
        
        # Initialize scores
        scores = np.ones(n_sentences) / n_sentences
        
        # Power iteration
        for _ in range(max_iter):
            prev_scores = scores
            scores = (1 - damping) + damping * (transition_matrix.T @ scores)
            
            # Check convergence
            if np.abs(scores - prev_scores).sum() < epsilon:
                break
                
        return scores
    
    def _calculate_sequence_scores(self, outputs) -> List[float]:
        """Calculate sequence scores from generation outputs"""
        scores = []
        
        if hasattr(outputs, 'sequences_scores'):
            return outputs.sequences_scores.tolist()
        elif hasattr(outputs, 'scores'):
            for i in range(len(outputs.sequences)):
                step_scores = [score[i] for score in outputs.scores]
                log_probs = torch.log_softmax(torch.stack(step_scores), dim=-1)
                
                token_probs = log_probs[
                    torch.arange(len(step_scores)),
                    outputs.sequences[i][1:]  # Skip start token
                ]
                
                score = torch.mean(token_probs).exp().item()
                scores.append(score)
        else:
            scores = [1.0] * len(outputs.sequences)
            
        return scores
    
    def evaluate(self,
                texts: List[str],
                reference_summaries: List[str]) -> Dict[str, float]:
        """Evaluate summary quality using ROUGE metrics
        
        Args:
            texts: Input texts
            reference_summaries: Reference summaries
            
        Returns:
            Dict with ROUGE scores
        """
        # Generate summaries
        generated_summaries = self.summarize(texts)
        
        # Calculate ROUGE scores
        rouge_scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        for gen, ref in zip(generated_summaries, reference_summaries):
            scores = self.rouge_scorer.score(gen, ref)
            for metric in rouge_scores:
                rouge_scores[metric] += scores[metric].fmeasure
                
        # Average scores
        for metric in rouge_scores:
            rouge_scores[metric] /= len(texts)
            
        return rouge_scores
    
    def fine_tune(self,
                 train_data: List[Dict[str, str]],
                 val_data: Optional[List[Dict[str, str]]] = None,
                 epochs: int = 3,
                 learning_rate: float = 5e-5):
        """Fine-tune the summarization model
        
        Args:
            train_data: List of dicts with 'text' and 'summary' keys
            val_data: Optional validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        if self.method != "abstractive":
            raise ValueError("Fine-tuning only supported for abstractive method")
            
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                
                # Prepare inputs
                inputs = self.tokenizer(
                    [d['text'] for d in batch_data],
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt"
                ).to(self.device)
                
                # Prepare targets
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        [d['summary'] for d in batch_data],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    ).input_ids.to(self.device)
                    
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
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
                            [d['text'] for d in batch_data],
                            padding=True,
                            truncation=True,
                            max_length=1024,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        with self.tokenizer.as_target_tokenizer():
                            labels = self.tokenizer(
                                [d['summary'] for d in batch_data],
                                padding=True,
                                truncation=True,
                                max_length=128,
                                return_tensors="pt"
                            ).input_ids.to(self.device)
                            
                        outputs = self.model(**inputs, labels=labels)
                        val_loss += outputs.loss.item()
                        
                avg_val_loss = val_loss / len(val_data)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                
                # Evaluate ROUGE scores
                val_metrics = self.evaluate(
                    [d['text'] for d in val_data[:100]],  # Sample for speed
                    [d['summary'] for d in val_data[:100]]
                )
                print("Validation ROUGE scores:")
                for metric, score in val_metrics.items():
                    print(f"{metric}: {score:.4f}")
                    
                self.model.train()