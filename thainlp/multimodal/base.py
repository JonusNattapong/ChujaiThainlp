"""
Base classes for multimodal processing
"""
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from ..core.transformers import TransformerBase
from ..extensions.monitoring import ProgressTracker
from ..vision.base import VisionBase
from ..speech.audio_utils import AudioUtils

@dataclass
class MultimodalConfig:
    """Configuration for multimodal models"""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    max_text_length: int = 512
    max_audio_length: int = 30  # seconds
    max_video_length: int = 60  # seconds
    supported_image_formats: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    supported_audio_formats: Tuple[str] = ('.wav', '.mp3', '.flac', '.ogg')
    supported_video_formats: Tuple[str] = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    supported_document_formats: Tuple[str] = ('.pdf', '.docx', '.txt', '.md', '.html')
    cache_dir: str = "~/.cache/thainlp/multimodal"

class MultimodalBase:
    """Base class for multimodal tasks"""
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4
    ):
        """Initialize multimodal processor
        
        Args:
            model_name: Name of pretrained model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        self.config = MultimodalConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.progress = ProgressTracker()
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        return AudioUtils.load_audio(audio_path)
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        return Image.open(image_path).convert("RGB")
    
    def load_video(self, video_path: str, num_frames: int = 16) -> List[np.ndarray]:
        """Load video file
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            import decord
        except ImportError:
            raise ImportError("Video loading requires decord library.")
            
        # Load video and extract frames
        video = decord.VideoReader(video_path)
        total_frames = len(video)
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = video.get_batch(indices).asnumpy()
        
        return frames
    
    def load_document(self, document_path: str) -> Dict[str, Any]:
        """Load document file
        
        Args:
            document_path: Path to document file
            
        Returns:
            Dictionary with document content and metadata
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document file not found: {document_path}")
            
        ext = os.path.splitext(document_path)[1].lower()
        
        if ext == '.pdf':
            return self._load_pdf(document_path)
        elif ext == '.docx':
            return self._load_docx(document_path)
        elif ext in ('.txt', '.md'):
            return self._load_text(document_path)
        elif ext == '.html':
            return self._load_html(document_path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")
    
    def _load_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Load PDF document"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PDF loading requires PyMuPDF library.")
            
        doc = fitz.open(pdf_path)
        content = []
        images = []
        
        for i, page in enumerate(doc):
            # Extract text
            content.append(page.get_text())
            
            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_img = doc.extract_image(xref)
                image_data = base_img["image"]
                # Convert to PIL Image
                from io import BytesIO
                image = Image.open(BytesIO(image_data))
                images.append({
                    "page": i,
                    "index": img_index,
                    "image": image
                })
        
        return {
            "type": "pdf",
            "content": content,
            "images": images,
            "metadata": {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "pages": len(doc)
            }
        }
    
    def _load_docx(self, docx_path: str) -> Dict[str, Any]:
        """Load DOCX document"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("DOCX loading requires python-docx library.")
            
        doc = Document(docx_path)
        content = []
        
        for para in doc.paragraphs:
            content.append(para.text)
            
        return {
            "type": "docx",
            "content": content,
            "metadata": {
                "title": os.path.basename(docx_path),
                "paragraphs": len(content)
            }
        }
    
    def _load_text(self, text_path: str) -> Dict[str, Any]:
        """Load plain text document"""
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        
        return {
            "type": "text",
            "content": content,
            "lines": lines,
            "metadata": {
                "title": os.path.basename(text_path),
                "line_count": len(lines)
            }
        }
    
    def _load_html(self, html_path: str) -> Dict[str, Any]:
        """Load HTML document"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("HTML loading requires BeautifulSoup4 library.")
            
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract text content
        text_content = soup.get_text()
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            images.append({
                "src": src,
                "alt": alt
            })
            
        return {
            "type": "html",
            "content": text_content,
            "html": html_content,
            "images": images,
            "metadata": {
                "title": soup.title.string if soup.title else os.path.basename(html_path)
            }
        }
    
    def determine_input_type(self, input_path: str) -> str:
        """Determine the type of input file
        
        Args:
            input_path: Path to input file
            
        Returns:
            Input type as string ('image', 'audio', 'video', 'document', 'text')
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        ext = os.path.splitext(input_path)[1].lower()
        
        if ext in self.config.supported_image_formats:
            return "image"
        elif ext in self.config.supported_audio_formats:
            return "audio"
        elif ext in self.config.supported_video_formats:
            return "video"
        elif ext in self.config.supported_document_formats:
            return "document"
        else:
            # Default to text if extension is not recognized
            return "text"
    
    def batch_process(self, 
                     items: List[Any], 
                     process_fn: callable, 
                     preprocess_fn: Optional[callable] = None,
                     postprocess_fn: Optional[callable] = None) -> List[Any]:
        """Process items in batches
        
        Args:
            items: List of items to process
            process_fn: Function to process each batch
            preprocess_fn: Optional function to preprocess each item
            postprocess_fn: Optional function to postprocess results
            
        Returns:
            List of processed results
        """
        all_results = []
        self.progress.start_task(len(items))
        
        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i + self.batch_size]
            
            # Preprocess if needed
            if preprocess_fn:
                batch_items = [preprocess_fn(item) for item in batch_items]
                
            # Process batch
            batch_results = process_fn(batch_items)
            
            # Postprocess if needed
            if postprocess_fn:
                batch_results = [postprocess_fn(res) for res in batch_results]
                
            all_results.extend(batch_results)
            self.progress.update(len(batch_items))
            
        self.progress.end_task()
        return all_results
