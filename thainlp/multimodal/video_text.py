"""
Video-Text processing for transcription and summarization
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import os
import torch
import numpy as np
from PIL import Image
import tempfile
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoProcessor,
    AutoModelForSequenceClassification
)
from .base import MultimodalBase
from .audio_text import AudioTextProcessor
from .image_text import ImageTextProcessor

class VideoTextProcessor(MultimodalBase):
    """Process videos for transcription and summarization"""
    
    def __init__(
        self,
        transcription_model: str = "openai/whisper-large-v2",
        summarization_model: str = "facebook/bart-large-cnn",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        """Initialize video text processor
        
        Args:
            transcription_model: Name of pretrained ASR model
            summarization_model: Name of pretrained summarization model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(None, device, batch_size)
        
        # Store model names for lazy loading
        self.transcription_model_name = transcription_model
        self.summarization_model_name = summarization_model
        
        # Initialize component processors
        self.audio_processor = None
        self.image_processor = None
        self.caption_model = None
        self.summary_model = None
        self.summary_tokenizer = None
        
    def _load_audio_processor(self):
        """Load audio processor on demand"""
        if self.audio_processor is None:
            self.audio_processor = AudioTextProcessor(
                model_name=self.transcription_model_name,
                device=self.device
            )
            
    def _load_image_processor(self):
        """Load image processor on demand"""
        if self.image_processor is None:
            self.image_processor = ImageTextProcessor(device=self.device)
            
    def _load_summarization_model(self):
        """Load summarization model on demand"""
        if self.summary_model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self.summary_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
                self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.summarization_model_name
                ).to(self.device)
            except (ImportError, OSError):
                print(f"Warning: Could not load {self.summarization_model_name}. Summarization will be limited.")
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        try:
            import ffmpeg
        except ImportError:
            raise ImportError("Audio extraction requires ffmpeg-python library.")
            
        # Create temporary file for audio
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"extracted_audio_{os.path.basename(video_path)}.wav")
        
        # Extract audio using ffmpeg
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .run(quiet=True, overwrite_output=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e}")
            raise
            
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 10,
        uniform: bool = True
    ) -> List[Image.Image]:
        """Extract frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            uniform: Whether to extract frames uniformly
            
        Returns:
            List of extracted frames as PIL Images
        """
        frames = self.load_video(video_path, num_frames=num_frames)
        return [Image.fromarray(frame) for frame in frames]
    
    def transcribe(
        self,
        video: Union[str, Dict[str, Any]],
        language: Optional[str] = None,
        return_timestamps: bool = True,
        extract_audio: bool = True
    ) -> Dict[str, Any]:
        """Transcribe speech from video
        
        Args:
            video: Video file path or pre-processed video dict
            language: Language code for transcription (e.g., "th", "en")
            return_timestamps: Whether to return timestamps
            extract_audio: Whether to extract audio from video
            
        Returns:
            Dictionary with transcription and metadata
        """
        # Load audio processor
        self._load_audio_processor()
        
        # Extract or get audio path
        if isinstance(video, str):
            if extract_audio:
                audio_path = self.extract_audio(video)
            else:
                audio_path = video  # Assume the path is already an audio file
            video_info = {"path": video}
        else:
            if "audio_path" in video:
                audio_path = video["audio_path"]
            elif "path" in video and extract_audio:
                audio_path = self.extract_audio(video["path"])
            else:
                raise ValueError("Cannot find audio in provided video data")
            video_info = video
            
        # Transcribe audio
        transcription = self.audio_processor.transcribe(
            audio_path,
            language=language,
            return_timestamps=return_timestamps
        )
        
        # Add video info to result
        if isinstance(transcription, dict):
            transcription["video_info"] = video_info
        else:
            transcription = {
                "text": transcription,
                "video_info": video_info
            }
            
        return transcription
    
    def summarize(
        self,
        video: Union[str, Dict[str, Any]],
        max_length: int = 150,
        min_length: int = 50,
        use_transcription: bool = True,
        use_visual: bool = True,
        language: Optional[str] = None,
        num_frames: int = 5
    ) -> Dict[str, Any]:
        """Generate a summary of video content
        
        Args:
            video: Video file path or pre-processed video dict
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            use_transcription: Whether to use transcription for summarization
            use_visual: Whether to use visual information for summarization
            language: Language code for transcription if needed
            num_frames: Number of frames to extract if using visual
            
        Returns:
            Dictionary with summary and metadata
        """
        # Initialize result dict
        result = {"video_info": {"path": video} if isinstance(video, str) else video}
        
        # Get transcription if requested
        if use_transcription:
            transcription = self.transcribe(
                video,
                language=language,
                return_timestamps=False
            )
            if isinstance(transcription, dict) and "text" in transcription:
                transcript_text = transcription["text"]
            else:
                transcript_text = transcription
                
            result["transcript"] = transcript_text
        else:
            transcript_text = ""
            
        # Get visual information if requested
        if use_visual and (isinstance(video, str) or "path" in video):
            visual_text = self._extract_visual_information(
                video if isinstance(video, str) else video["path"],
                num_frames=num_frames
            )
            result["visual_description"] = visual_text
        else:
            visual_text = ""
            
        # Combine information for summarization
        if transcript_text and visual_text:
            source_text = f"Video transcript: {transcript_text} Visual content: {visual_text}"
        elif transcript_text:
            source_text = transcript_text
        elif visual_text:
            source_text = visual_text
        else:
            return {"summary": "", "error": "No content available for summarization"}
            
        # Load summarization model
        self._load_summarization_model()
        
        # Check if we have a summarization model
        if self.summary_model is None:
            # Fallback to extractive summarization
            summary = self._extractive_summarize(source_text, max_length)
        else:
            # Use abstractive summarization
            inputs = self.summary_tokenizer(
                source_text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.summary_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True
                )
                
            summary = self.summary_tokenizer.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
            
        result["summary"] = summary
        return result
    
    def _extract_visual_information(
        self,
        video_path: str,
        num_frames: int = 5
    ) -> str:
        """Extract visual information from video frames"""
        # Load image processor
        self._load_image_processor()
        
        # Extract frames
        frames = self.extract_frames(video_path, num_frames=num_frames)
        
        # Caption each frame
        captions = []
        for frame in frames:
            caption = self.image_processor.caption(frame)
            captions.append(caption)
            
        # Combine captions
        return " ".join(captions)
    
    def _extractive_summarize(
        self,
        text: str,
        max_length: int = 150
    ) -> str:
        """Simple extractive summarization as fallback"""
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 3:
            return text
            
        # Simple algorithm: take first sentence and a few distributed throughout
        summary_sentences = [sentences[0]]
        
        # Add sentences distributed throughout the text
        step = max(1, len(sentences) // min(5, len(sentences) - 1))
        for i in range(step, len(sentences), step):
            summary_sentences.append(sentences[i])
            if len(" ".join(summary_sentences)) >= max_length:
                break
                
        return " ".join(summary_sentences)
        
class VideoTranscriber(VideoTextProcessor):
    """Specialized class for video transcription"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(transcription_model=model_name, device=device, batch_size=batch_size)
        
    def __call__(
        self,
        video: Union[str, Dict[str, Any]],
        language: Optional[str] = None,
        return_timestamps: bool = True
    ) -> Dict[str, Any]:
        """Transcribe speech from video"""
        return self.transcribe(video, language, return_timestamps)
        
class VideoSummarizer(VideoTextProcessor):
    """Specialized class for video summarization"""
    
    def __init__(
        self,
        transcription_model: str = "openai/whisper-large-v2",
        summarization_model: str = "facebook/bart-large-cnn",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(transcription_model, summarization_model, device, batch_size)
        
    def __call__(
        self,
        video: Union[str, Dict[str, Any]],
        max_length: int = 150,
        min_length: int = 50,
        use_transcription: bool = True,
        use_visual: bool = True
    ) -> Dict[str, Any]:
        """Generate a summary of video content"""
        return self.summarize(
            video,
            max_length=max_length,
            min_length=min_length,
            use_transcription=use_transcription,
            use_visual=use_visual
        )
