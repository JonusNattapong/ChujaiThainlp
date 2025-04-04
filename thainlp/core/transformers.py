"""
Base transformer model functionality
"""
from typing import Optional, Any
from logging import Logger
import torch

class TransformerBase:
    """Base class for transformer-based models"""
    
    def __init__(self, model_name: str = ""):
        """Initialize base transformer
        
        Args:
            model_name: Name/path of pretrained model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = Logger(__name__)
        
    def load_model(self):
        """Load model and tokenizer if not already loaded"""
        raise NotImplementedError(
            "Subclasses must implement model loading"
        )
        
    def save_model(self, path: str):
        """Save model and tokenizer to path
        
        Args:
            path: Directory to save model
        """
        if self.model is None:
            raise ValueError("No model loaded")
            
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load_from_pretrained(self, path: str):
        """Load model from saved directory
        
        Args:
            path: Directory containing saved model
        """
        raise NotImplementedError(
            "Subclasses must implement loading from pretrained"
        )
        
    def to(self, device: str):
        """Move model to specified device
        
        Args:
            device: Device to move model to
        """
        if self.model is not None:
            self.device = device
            self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode"""
        if self.model is not None:
            self.model.train()
        return self
    
    def state_dict(self):
        """Get model state dict"""
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        """Load model state dict
        
        Args:
            state_dict: State dict to load
        """
        if self.model is None:
            raise ValueError("No model loaded")
        self.model.load_state_dict(state_dict)
        
    @property
    def name(self) -> str:
        """Get model name"""
        return self.model_name
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"