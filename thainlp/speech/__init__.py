"""
Speech Processing Module for Thai Language

Includes:
- Text-to-speech synthesis
- Automatic speech recognition
- Voice activity detection
- Voice processing/conversion
- Audio utilities
"""
import numpy as np
from typing import Union
from .tts import ThaiTTS
from .asr import ThaiASR
from .voice_activity import VoiceActivityDetector
from .voice_processing import VoiceProcessor
from .audio_utils import AudioUtils
from .summarization import SpeechSummarizer
from .audio_utils import AudioUtils

# Default instances for convenience
_default_tts = None
_default_asr = None
_default_vad = None
_default_summarizer = None

def get_tts(model_name: str = "facebook/mms-tts-tha") -> ThaiTTS:
    """Get or create default TTS instance"""
    global _default_tts
    if _default_tts is None:
        _default_tts = ThaiTTS(model_name=model_name)
    return _default_tts

def get_asr(model_name: str = "openai/whisper-large-v3-turbo") -> ThaiASR:
    """Get or create default ASR instance"""
    global _default_asr
    if _default_asr is None:
        _default_asr = ThaiASR(model_name=model_name)
    return _default_asr

def get_vad(model_name: str = None) -> VoiceActivityDetector:
    """Get or create default VAD instance"""
    global _default_vad
    if _default_vad is None:
        _default_vad = VoiceActivityDetector(model_name=model_name)
    return _default_vad

def get_summarizer(model_name: str = "openai/whisper-large-v3") -> SpeechSummarizer:
    """Get or create default speech summarizer instance"""
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = SpeechSummarizer(model_name=model_name)
    return _default_summarizer

def summarize_speech(audio: Union[np.ndarray, str, bytes], **kwargs) -> str:
    """Summarize speech content (uses default summarizer)"""
    return get_summarizer().summarize(audio, **kwargs)

def synthesize(text: str, **kwargs) -> np.ndarray:
    """Synthesize speech from text (uses default TTS)"""
    return get_tts().synthesize(text, **kwargs)

def transcribe(audio: Union[np.ndarray, str, bytes], **kwargs) -> str:
    """Transcribe audio to text (uses default ASR)"""
    return get_asr().transcribe(audio, **kwargs)

def detect_voice_activity(audio: Union[np.ndarray, str, bytes], **kwargs) -> list:
    """Detect speech segments in audio (uses default VAD)"""
    return get_vad().detect(audio, **kwargs)

__all__ = [
    # Core classes
    'ThaiTTS',
    'ThaiASR',
    'VoiceActivityDetector',
    'VoiceProcessor',
    'AudioUtils',
    'SpeechSummarizer',
    
    # Helper functions
    'get_tts',
    'get_asr',
    'get_vad',
    'get_summarizer',
    'synthesize',
    'transcribe',
    'detect_voice_activity',
    'summarize_speech'
]
