"""
Text Classification for Thai Text using Transformer models
"""

from typing import List, Dict, Union, Optional
import torch
import torch.nn.functional as F
from ..core.transformers import TransformerBase

class ThaiTextClassifier(TransformerBase):
    """Text classifier for Thai text using transformer models"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_labels: Optional[int] = None,
        **kwargs
    ):
        """Initialize text classifier
        
        Args:
            model_name_or_path: Name or path of the model
            num_labels: Number of labels for classification
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = self.get_default_model()
            
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="text-classification",
            num_labels=num_labels,
            **kwargs
        )
        
    @staticmethod
    def get_default_model() -> str:
        """Get default model for Thai text classification"""
        return "airesearch/wangchanberta-base-att-spm-uncased"
    
    def classify(
        self,
        texts: Union[str, List[str]],
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, float]]:
        """Classify text into predefined categories
        
        Args:
            texts: Input text or list of texts
            labels: Optional list of label names
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of dictionaries containing classification probabilities
        """
        # Encode texts
        inputs = self.encode(texts, **kwargs)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        # Convert to list of dictionaries
        results = []
        for prob in probs:
            if labels:
                result = {label: float(p) for label, p in zip(labels, prob)}
            else:
                result = {str(i): float(p) for i, p in enumerate(prob)}
            results.append(result)
            
        return results[0] if isinstance(texts, str) else results
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[Union[int, str]],
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[List[Union[int, str]]] = None,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        **kwargs
    ):
        """Fine-tune the model on custom data
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            validation_texts: Validation texts
            validation_labels: Validation labels
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            max_length: Maximum sequence length
            **kwargs: Additional training arguments
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        from tqdm.auto import tqdm
        
        # Prepare training data
        train_encodings = self.encode(train_texts, max_length=max_length)
        train_labels = torch.tensor(train_labels)
        train_dataset = TensorDataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            train_labels
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        if validation_texts and validation_labels:
            val_encodings = self.encode(validation_texts, max_length=max_length)
            val_labels = torch.tensor(validation_labels)
            val_dataset = TensorDataset(
                val_encodings["input_ids"],
                val_encodings["attention_mask"],
                val_labels
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss / len(train_loader)})
            
            # Validation
            if validation_texts and validation_labels:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        val_loss += outputs.loss.item()
                        
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                
                val_loss = val_loss / len(val_loader)
                accuracy = correct / total
                print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                self.model.train()
    
    def zero_shot_classify(
        self,
        texts: Union[str, List[str]],
        candidate_labels: List[str],
        hypothesis_template: str = "นี่คือเรื่องเกี่ยวกับ{}",
        multi_label: bool = False,
        **kwargs
    ) -> List[Dict[str, float]]:
        """Zero-shot classification using natural language inference
        
        Args:
            texts: Input text or list of texts
            candidate_labels: List of possible labels
            hypothesis_template: Template for hypothesis generation
            multi_label: Whether to allow multiple labels
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of dictionaries containing classification probabilities
        """
        from transformers import pipeline
        
        classifier = pipeline(
            "zero-shot-classification",
            model="airesearch/wangchanberta-base-att-spm-uncased",
            device=0 if torch.cuda.is_available() else -1
        )
        
        results = classifier(
            texts,
            candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label
        )
        
        if isinstance(texts, str):
            return {
                "labels": results["labels"],
                "scores": results["scores"]
            }
        return [
            {
                "labels": r["labels"],
                "scores": r["scores"]
            }
            for r in results
        ] 