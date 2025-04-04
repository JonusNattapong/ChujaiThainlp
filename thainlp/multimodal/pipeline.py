"""
Multimodal pipeline for seamless processing across modalities
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import os
from PIL import Image
import numpy as np
from .base import MultimodalBase
from .audio_text import AudioTextProcessor
from .image_text import ImageTextProcessor, OCRProcessor
from .vqa import VisualQA
from .document_qa import DocumentQA
from .video_text import VideoTextProcessor
from .converters import ModalityConverter

class MultimodalPipeline(MultimodalBase):
    """Unified pipeline for multimodal processing tasks"""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
        load_all_models: bool = False
    ):
        """Initialize multimodal pipeline
        
        Args:
            device: Device to run model on
            batch_size: Batch size for processing
            load_all_models: Whether to preload all models
        """
        super().__init__(None, device, batch_size)
        
        # Initialize component processors
        self.processors = {}
        
        # Load core processors if requested
        if load_all_models:
            self.processors["audio"] = AudioTextProcessor(device=device, batch_size=batch_size)
            self.processors["image"] = ImageTextProcessor(device=device, batch_size=batch_size)
            self.processors["ocr"] = OCRProcessor(device=device, batch_size=batch_size)
            self.processors["vqa"] = VisualQA(device=device, batch_size=batch_size)
            self.processors["document"] = DocumentQA(device=device, batch_size=batch_size)
            self.processors["video"] = VideoTextProcessor(device=device, batch_size=batch_size)
            self.processors["converter"] = ModalityConverter(device=device, batch_size=batch_size)
    
    def _get_processor(self, processor_type: str):
        """Get or load a processor on demand"""
        if processor_type not in self.processors:
            if processor_type == "audio":
                self.processors[processor_type] = AudioTextProcessor(device=self.device, batch_size=self.batch_size)
            elif processor_type == "image":
                self.processors[processor_type] = ImageTextProcessor(device=self.device, batch_size=self.batch_size)
            elif processor_type == "ocr":
                self.processors[processor_type] = OCRProcessor(device=self.device, batch_size=self.batch_size)
            elif processor_type == "vqa":
                self.processors[processor_type] = VisualQA(device=self.device, batch_size=self.batch_size)
            elif processor_type == "document":
                self.processors[processor_type] = DocumentQA(device=self.device, batch_size=self.batch_size)
            elif processor_type == "video":
                self.processors[processor_type] = VideoTextProcessor(device=self.device, batch_size=self.batch_size)
            elif processor_type == "converter":
                self.processors[processor_type] = ModalityConverter(device=self.device, batch_size=self.batch_size)
            else:
                raise ValueError(f"Unknown processor type: {processor_type}")
                
        return self.processors[processor_type]
    
    def process(
        self,
        inputs: Union[str, Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        return_intermediate: bool = False
    ) -> Union[Any, Dict[str, Any]]:
        """Process multimodal inputs with a series of tasks
        
        Args:
            inputs: Input data (file path or dictionary with typed inputs)
            tasks: List of task specifications
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Final result or dictionary with all results
        """
        # Prepare input data
        if isinstance(inputs, str):
            # Determine input type from file path
            input_type = self.determine_input_type(inputs)
            data = {"type": input_type, "path": inputs}
        else:
            data = inputs
            
        # Initialize results storage
        results = {"input": data}
        
        # Process each task in sequence
        for task_idx, task in enumerate(tasks):
            task_type = task.get("type")
            task_params = task.get("params", {})
            task_name = task.get("name", f"task_{task_idx}")
            
            # Execute task based on type
            if task_type == "audio_transcribe":
                audio_input = task.get("input", "input")
                processor = self._get_processor("audio")
                audio_path = results[audio_input]["path"] if audio_input in results else data["path"]
                results[task_name] = processor.transcribe(audio_path, **task_params)
                
            elif task_type == "audio_translate":
                audio_input = task.get("input", "input")
                processor = self._get_processor("audio")
                audio_path = results[audio_input]["path"] if audio_input in results else data["path"]
                results[task_name] = processor.translate(audio_path, **task_params)
                
            elif task_type == "image_caption":
                image_input = task.get("input", "input")
                processor = self._get_processor("image")
                image_path = results[image_input]["path"] if image_input in results else data["path"]
                results[task_name] = processor.caption(image_path, **task_params)
                
            elif task_type == "image_ocr":
                image_input = task.get("input", "input")
                processor = self._get_processor("ocr")
                image_path = results[image_input]["path"] if image_input in results else data["path"]
                results[task_name] = processor.extract_text(image_path, **task_params)
                
            elif task_type == "image_analyze":
                image_input = task.get("input", "input")
                processor = self._get_processor("image")
                image_path = results[image_input]["path"] if image_input in results else data["path"]
                results[task_name] = processor.analyze(image_path, **task_params)
                
            elif task_type == "vqa":
                image_input = task.get("input", "input")
                processor = self._get_processor("vqa")
                image_path = results[image_input]["path"] if image_input in results else data["path"]
                question = task_params.pop("question")
                results[task_name] = processor.answer(image_path, question, **task_params)
                
            elif task_type == "document_qa":
                document_input = task.get("input", "input")
                processor = self._get_processor("document")
                document_path = results[document_input]["path"] if document_input in results else data["path"]
                question = task_params.pop("question")
                results[task_name] = processor.answer(document_path, question, **task_params)
                
            elif task_type == "document_process":
                document_input = task.get("input", "input")
                processor = self._get_processor("document")
                document_path = results[document_input]["path"] if document_input in results else data["path"]
                results[task_name] = processor.process(document_path, **task_params)
                
            elif task_type == "video_transcribe":
                video_input = task.get("input", "input")
                processor = self._get_processor("video")
                video_path = results[video_input]["path"] if video_input in results else data["path"]
                results[task_name] = processor.transcribe(video_path, **task_params)
                
            elif task_type == "video_summarize":
                video_input = task.get("input", "input")
                processor = self._get_processor("video")
                video_path = results[video_input]["path"] if video_input in results else data["path"]
                results[task_name] = processor.summarize(video_path, **task_params)
                
            elif task_type == "convert_modality":
                source_input = task.get("input", "input")
                processor = self._get_processor("converter")
                source = results[source_input] if source_input in results else data
                source_type = task_params.pop("source_type", source.get("type", "text"))
                target_type = task_params.pop("target_type")
                results[task_name] = processor.convert(source, source_type, target_type, **task_params)
                
            else:
                results[task_name] = {"error": f"Unknown task type: {task_type}"}
        
        # Return results
        if return_intermediate:
            return results
        else:
            # Return the result of the last task
            return results[task_name] if task_name in results else results
