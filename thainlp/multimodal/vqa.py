"""
Visual Question Answering (VQA) for answering questions about images
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForQuestionAnswering,
    ViltProcessor,
    ViltForQuestionAnswering
)
from .base import MultimodalBase

class VisualQA(MultimodalBase):
    """Visual Question Answering model"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize Visual QA model
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load model depending on architecture
        if "blip" in model_name.lower():
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
            self.model_type = "blip"
        elif "vilt" in model_name.lower():
            self.processor = ViltProcessor.from_pretrained(model_name)
            self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(device)
            self.model_type = "vilt"
        else:
            raise ValueError(f"Unsupported VQA model: {model_name}")
    
    def answer(
        self,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        question: Union[str, List[str]],
        max_length: int = 50,
        num_answers: int = 1,
        return_scores: bool = False
    ) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Answer questions about an image
        
        Args:
            image: Image or list of images (as paths or PIL Images)
            question: Question or list of questions
            max_length: Maximum length of generated answers
            num_answers: Number of answers to generate per question
            return_scores: Whether to return confidence scores
            
        Returns:
            Answers or dictionaries with answers and metadata
        """
        # Ensure image is a list
        if isinstance(image, (str, Image.Image)):
            images = [image]
            single_image = True
        else:
            images = image
            single_image = False
        
        # Ensure question is a list matching the image count
        if isinstance(question, str):
            if single_image:
                questions = [question]
            else:
                questions = [question] * len(images)
        else:
            questions = question
            if len(questions) < len(images):
                questions = questions + [questions[-1]] * (len(images) - len(questions))
            elif len(questions) > len(images) and len(images) == 1:
                images = images * len(questions)
                single_image = False
        
        def process_batch(batch_data):
            batch_results = []
            
            for img, q in batch_data:
                # Preprocess image if it's a path
                if isinstance(img, str):
                    img = self.load_image(img)
                
                # Process based on model type
                if self.model_type == "blip":
                    # BLIP model processing
                    inputs = self.processor(img, q, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            num_return_sequences=num_answers,
                            do_sample=(num_answers > 1),
                            top_k=50 if num_answers > 1 else None,
                            top_p=0.95 if num_answers > 1 else None
                        )
                    
                    answers = self.processor.decode_batch(outputs, skip_special_tokens=True)
                    
                    if num_answers == 1:
                        if return_scores:
                            batch_results.append({
                                "answer": answers[0],
                                "score": 1.0  # BLIP doesn't provide scores directly
                            })
                        else:
                            batch_results.append(answers[0])
                    else:
                        if return_scores:
                            # Assign decreasing scores as a heuristic
                            scored_answers = [
                                {"answer": a, "score": 1.0 - 0.1 * i} 
                                for i, a in enumerate(answers)
                            ]
                            batch_results.append(scored_answers)
                        else:
                            batch_results.append(answers)
                            
                elif self.model_type == "vilt":
                    # ViLT model processing
                    inputs = self.processor(img, q, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    top_k_probs, top_k_indices = probs.topk(num_answers, dim=1)
                    
                    top_answers = []
                    for i in range(num_answers):
                        answer_idx = top_k_indices[0, i].item()
                        answer = self.model.config.id2label[answer_idx]
                        score = top_k_probs[0, i].item()
                        top_answers.append((answer, score))
                    
                    if num_answers == 1:
                        if return_scores:
                            batch_results.append({
                                "answer": top_answers[0][0],
                                "score": top_answers[0][1]
                            })
                        else:
                            batch_results.append(top_answers[0][0])
                    else:
                        if return_scores:
                            scored_answers = [
                                {"answer": a, "score": s} for a, s in top_answers
                            ]
                            batch_results.append(scored_answers)
                        else:
                            batch_results.append([a for a, _ in top_answers])
            
            return batch_results
            
        # Create list of (image, question) pairs for batch processing
        image_question_pairs = list(zip(images, questions))
        
        results = self.batch_process(
            image_question_pairs,
            process_batch
        )
        
        return results[0] if single_image and len(questions) == 1 else results
    
    def batch_answer(
        self,
        images: List[Union[str, Image.Image]],
        questions: List[str],
        max_length: int = 50,
        num_answers: int = 1,
        return_scores: bool = False
    ) -> List[Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]]:
        """Answer multiple questions about multiple images
        
        Args:
            images: List of images
            questions: List of questions
            max_length: Maximum length of generated answers
            num_answers: Number of answers to generate per question
            return_scores: Whether to return confidence scores
            
        Returns:
            List of answers or dictionaries with answers and metadata
        """
        # Ensure questions and images have the same length
        if len(questions) != len(images):
            raise ValueError(
                f"Number of questions ({len(questions)}) must match number of images ({len(images)})"
            )
        
        return self.answer(
            images,
            questions,
            max_length=max_length,
            num_answers=num_answers,
            return_scores=return_scores
        )
