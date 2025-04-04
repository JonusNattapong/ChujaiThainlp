"""
Document Question Answering for extracting information from documents
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    LayoutLMv3Processor,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3TokenizerFast
)
from .base import MultimodalBase
from .image_text import OCRProcessor

class DocumentQA(MultimodalBase):
    """Question answering model for documents including layout information"""
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base-finetuned-docvqa",
        text_qa_model: str = "deepset/roberta-base-squad2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize document QA model
        
        Args:
            model_name: Name of pretrained document QA model
            text_qa_model: Fallback text QA model if layout model fails
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Try to load LayoutLM-based model for document understanding
        try:
            self.processor = LayoutLMv3Processor.from_pretrained(model_name)
            self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name).to(device)
            self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name)
            self.model_type = "layoutlm"
        except (ImportError, OSError):
            # Fallback to text-only QA model
            print(f"Warning: Could not load {model_name}. Falling back to text-only QA model.")
            self.tokenizer = AutoTokenizer.from_pretrained(text_qa_model)
            self.model = AutoModelForQuestionAnswering.from_pretrained(text_qa_model).to(device)
            self.model_type = "text"
            
        # Initialize OCR processor for text extraction if needed
        self.ocr_processor = None
    
    def _load_ocr_processor(self):
        """Load OCR processor on demand"""
        if self.ocr_processor is None:
            self.ocr_processor = OCRProcessor(device=self.device)
    
    def answer(
        self,
        document: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
        question: Union[str, List[str]],
        max_answer_len: int = 100,
        max_seq_len: int = 512,
        return_context: bool = False,
        num_answers: int = 1
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Answer questions about documents
        
        Args:
            document: Document file path, or pre-processed document dict, or list of documents
            question: Question or list of questions
            max_answer_len: Maximum length of extracted answers
            max_seq_len: Maximum sequence length for the model
            return_context: Whether to return the context around the answer
            num_answers: Number of answers to return
            
        Returns:
            Answer text, dictionary with answer details, or list of answers/dictionaries
        """
        # Handle single document input
        if isinstance(document, (str, dict)):
            documents = [document]
            single_document = True
        else:
            documents = document
            single_document = False
            
        # Handle single question input and ensure question-document match
        if isinstance(question, str):
            if single_document:
                questions = [question]
            else:
                questions = [question] * len(documents)
        else:
            questions = question
            if len(questions) < len(documents):
                questions = questions + [questions[-1]] * (len(documents) - len(questions))
            elif len(questions) > len(documents) and single_document:
                documents = documents * len(questions)
                single_document = False
                
        # Process each document-question pair
        all_results = []
        
        for doc, q in zip(documents, questions):
            # Process document if it's a file path
            if isinstance(doc, str):
                doc_data = self.process(doc)
            else:
                doc_data = doc
                
            # Use appropriate QA approach based on model type
            if self.model_type == "layoutlm" and "images" in doc_data and doc_data["images"]:
                # Process with LayoutLM for document with images/layout
                result = self._process_document_with_layout(doc_data, q, max_answer_len, max_seq_len, num_answers)
            else:
                # Process with text-only QA
                result = self._process_document_with_text(doc_data, q, max_answer_len, max_seq_len, num_answers)
                
            # Add context if requested
            if return_context and "text" in doc_data:
                if isinstance(result, dict):
                    result["context"] = doc_data["text"]
                elif isinstance(result, list):
                    for r in result:
                        if isinstance(r, dict):
                            r["context"] = doc_data["text"]
                            
            all_results.append(result)
            
        # Return single result for single document input
        if single_document and len(questions) == 1:
            return all_results[0]
        
        return all_results
    
    def _process_document_with_layout(
        self,
        document: Dict[str, Any],
        question: str,
        max_answer_len: int,
        max_seq_len: int,
        num_answers: int
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Process document using LayoutLM model considering layout"""
        results = []
        
        # Get document image
        if isinstance(document["images"], list) and document["images"]:
            # Use first page image
            image = document["images"][0]["image"] if isinstance(document["images"][0], dict) else document["images"][0]
        else:
            # Fallback to text-only processing if no images
            return self._process_document_with_text(document, question, max_answer_len, max_seq_len, num_answers)
            
        # Get document text if not already extracted
        if "text" not in document or not document["text"]:
            self._load_ocr_processor()
            document["text"] = self.ocr_processor.extract_text(image)
            
        # Prepare inputs for LayoutLM
        encoding = self.processor(
            image,
            question,
            document["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model(**encoding)
            
        # Get start and end scores
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Get the top-k answers
        start_indices = torch.topk(start_logits[0], num_answers).indices
        end_indices = torch.topk(end_logits[0], num_answers).indices
        
        for start_idx in start_indices:
            for end_idx in end_indices:
                # Skip invalid spans
                if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_len:
                    continue
                    
                # Extract answer tokens
                input_ids = encoding.input_ids[0]
                answer_tokens = input_ids[start_idx:end_idx+1]
                answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Calculate answer score
                score = (start_logits[0, start_idx] + end_logits[0, end_idx]).item()
                
                results.append({
                    "answer": answer_text,
                    "score": score,
                    "start": start_idx.item(),
                    "end": end_idx.item()
                })
                
                # Break after finding valid span
                break
                
        # Sort by score and take top-k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:num_answers]
        
        if num_answers == 1:
            return results[0] if results else {"answer": "", "score": 0.0}
        
        return results if results else [{"answer": "", "score": 0.0}]
    
    def _process_document_with_text(
        self,
        document: Dict[str, Any],
        question: str,
        max_answer_len: int,
        max_seq_len: int,
        num_answers: int
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Process document using text-only QA model"""
        # Get document text
        if isinstance(document, dict) and "text" in document:
            if isinstance(document["text"], list):
                # Join text from multiple pages/sections
                text = " ".join(document["text"])
            else:
                text = document["text"]
        elif isinstance(document, str):
            # Assume it's a text string
            text = document
        else:
            # Extract text if we have a document without text but with images
            if "images" in document and document["images"]:
                self._load_ocr_processor()
                image = document["images"][0]["image"] if isinstance(document["images"][0], dict) else document["images"][0]
                text = self.ocr_processor.extract_text(image)
            else:
                return {"answer": "", "score": 0.0, "error": "No text found in document"}
        
        # Prepare inputs for QA model
        inputs = self.tokenizer(
            question,
            text,
            max_length=max_seq_len,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        # Get offset mapping and overflow mapping
        offset_mapping = inputs.pop("offset_mapping").cpu().numpy()
        overflow_to_sample_mapping = inputs.pop("overflow_to_sample_mapping").cpu().numpy()
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get start and end scores
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        
        # Process all chunks and get the best answers
        results = []
        
        for i in range(len(start_logits)):
            # Get top-k answers for each chunk
            for start_idx in np.argsort(start_logits[i])[-num_answers:]:
                for end_idx in np.argsort(end_logits[i])[-num_answers:]:
                    # Skip invalid spans
                    if end_idx < start_idx or end_idx - start_idx + 1 > max_answer_len:
                        continue
                        
                    # Get the offset for this chunk
                    offset = offset_mapping[i]
                    
                    # Convert to text span in original document
                    if offset[start_idx][0] == offset[start_idx][1] or offset[end_idx][0] == offset[end_idx][1]:
                        # Skip if it points to a special token
                        continue
                        
                    start_char = offset[start_idx][0]
                    end_char = offset[end_idx][1]
                    
                    answer_text = text[start_char:end_char]
                    
                    # Calculate answer score
                    score = (float(start_logits[i][start_idx]) + float(end_logits[i][end_idx]))
                    
                    results.append({
                        "answer": answer_text,
                        "score": score,
                        "start_char": int(start_char),
                        "end_char": int(end_char)
                    })
                    
                    # Break after finding valid span
                    break
                    
        # Sort by score and take top-k
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:num_answers]
        
        if num_answers == 1:
            return results[0] if results else {"answer": "", "score": 0.0}
        
        return results if results else [{"answer": "", "score": 0.0}]
    
    def process(
        self,
        document_path: str,
        extract_text: bool = True,
        extract_tables: bool = True,
        extract_layout: bool = True
    ) -> Dict[str, Any]:
        """Process a document for later analysis
        
        Args:
            document_path: Path to document file
            extract_text: Whether to extract text
            extract_tables: Whether to extract tables
            extract_layout: Whether to extract layout information
            
        Returns:
            Dictionary with document content and metadata
        """
        # Load document based on file type
        document = self.load_document(document_path)
        
        # Explicitly extract text if requested but not already present
        if extract_text and "text" not in document:
            self._load_ocr_processor()
            
            # Extract text from all document images
            if "images" in document and document["images"]:
                texts = []
                for img_data in document["images"]:
                    image = img_data["image"] if isinstance(img_data, dict) else img_data
                    texts.append(self.ocr_processor.extract_text(image))
                
                document["text"] = texts
        
        # Extract tables if requested
        if extract_tables and "tables" not in document:
            document["tables"] = self._extract_tables(document)
            
        # Extract layout if requested
        if extract_layout and "layout" not in document:
            document["layout"] = self._extract_layout(document)
            
        return document
    
    def _extract_tables(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        tables = []
        
        # This is a placeholder implementation
        # In a real implementation, this would use a table extraction model
        # Such as TableTransformer or similar
        
        # Check if document has images to extract tables from
        if "images" in document and document["images"]:
            for i, img_data in enumerate(document["images"]):
                # Create a dummy table for demonstration
                tables.append({
                    "page": i,
                    "content": "Table extraction requires specialized models",
                    "rows": [],
                    "columns": []
                })
                
        return tables
    
    def _extract_layout(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract layout information from document"""
        layout = []
        
        # This is a placeholder implementation
        # In a real implementation, this would use a layout detection model
        # Such as LayoutLMv3 or similar
        
        # Check if document has images to extract layout from
        if "images" in document and document["images"]:
            for i, img_data in enumerate(document["images"]):
                # Create a dummy layout for demonstration
                layout.append({
                    "page": i,
                    "sections": [
                        {"type": "text", "bbox": [0, 0, 100, 100], "text": "Sample text"}
                    ]
                })
                
        return layout

class DocumentProcessor(DocumentQA):
    """Specialized class for document processing without QA"""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        super().__init__(None, None, device, batch_size)
        
        # Initialize OCR processor
        self._load_ocr_processor()
        
    def __call__(
        self,
        document_path: str,
        extract_text: bool = True,
        extract_tables: bool = True,
        extract_layout: bool = True
    ) -> Dict[str, Any]:
        """Process a document for analysis"""
        return self.process(
            document_path,
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_layout=extract_layout
        )
