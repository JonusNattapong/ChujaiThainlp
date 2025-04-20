"""
Thai Automatic Speech Recognition Module
"""
from typing import Optional, Union, List
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from ..model_hub import ModelManager
from thainlp.spellcheck.spell_checker import ThaiSpellChecker  # Fixed import

class ThaiASR:
    """Thai automatic speech recognition"""
    
    def __init__(self, 
                model_name: str = "openai/whisper-large-v3-turbo",  # Use OpenAI's Whisper model
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize ASR model
        
        Args:
            model_name: Name of ASR model
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        """Load ASR model and processor"""
        manager = ModelManager()
        model_info = manager.get_model_info(self.model_name)
        
        if not model_info or model_info['task'] != 'asr':
            raise ValueError(f"Model {self.model_name} not found or invalid for ASR")
            
        self.processor = WhisperProcessor.from_pretrained(model_info['hf_id'])
        self.model = WhisperForConditionalGeneration.from_pretrained(model_info['hf_id']).to(self.device)
        self.sample_rate = model_info.get('sample_rate', 16000)
        
    def transcribe(self,
                  audio: Union[np.ndarray, str, bytes],
                  language: Optional[str] = "th",
                  validate_thai: bool = True,
                  **kwargs) -> Union[str, dict]:
        """
        Transcribe audio to text
        
        Args:
            audio: Input audio as:
                  - numpy array (waveform)
                  - file path (str)
                  - bytes (raw audio data)
            language: Language code ('th' for Thai)
            
        Returns:
            Transcribed text
        """
        # Load audio if needed
        if isinstance(audio, str):
            audio = self._load_audio_file(audio)
        elif isinstance(audio, bytes):
            audio = self._bytes_to_array(audio)
            
        # Process audio
        # Process audio for Whisper with attention mask
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Add attention mask if not present
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs.input_features)
            
        inputs = inputs.to(self.device)
        
        # Generate transcription with Whisper
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
                language="thai",
                do_sample=False,
                num_beams=5
            )
            
        # Decode the generated IDs to text
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Post-process and validate text
        text = self._postprocess_text(text)
        
        if validate_thai:
            # Validate Thai text
            spell_checker = ThaiSpellChecker()
            is_valid, corrected = spell_checker.validate_label(text)
            
            result = {
                "text": text,
                "valid_thai": is_valid,
            }
            if corrected:
                result["corrected_text"] = corrected
                
            return result
        else:
            return text
    
    def transcribe_batch(self,
                       audio_list: List[Union[np.ndarray, str, bytes]],
                       **kwargs) -> List[str]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_list: List of audio inputs
            **kwargs: Additional arguments for transcribe()
            
        Returns:
            List of transcriptions
        """
        return [self.transcribe(audio, **kwargs) for audio in audio_list]
        
    def _load_audio_file(self, path: str) -> np.ndarray:
        """Load audio file into numpy array"""
        import soundfile as sf
        audio, sr = sf.read(path)
        
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        return audio
        
    def _bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw bytes to numpy array"""
        import soundfile as sf
        import io
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f)
            
        # Resample if needed
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
        return audio
        
    def _postprocess_text(self, text: str) -> str:
        """Post-process transcribed text"""
        # TODO: Add Thai-specific text normalization
        text = text.replace("<unk>", "").strip()
        text = " ".join(text.split())  # Remove extra spaces
        return text
