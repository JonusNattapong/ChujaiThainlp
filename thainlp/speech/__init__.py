"""
Speech Processing Module for Thai Language

Includes:
- Text-to-speech synthesis
- Automatic speech recognition
- Voice activity detection
- Voice processing/conversion
- Audio utilities
"""
from .tts import ThaiTTS
from .asr import ThaiASR
from .voice_activity import VoiceActivityDetector
from .voice_processing import VoiceProcessor
from .audio_utils import AudioUtils

# Default instances for convenience
_default_tts = None
_default_asr = None
_default_vad = None

def get_tts(model_name: str = None) -> ThaiTTS:
    """Get or create default TTS instance"""
    global _default_tts
    if _default_tts is None:
        _default_tts = ThaiTTS(model_name=model_name)
    return _default_tts

def get_asr(model_name: str = None) -> ThaiASR:
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
    
    # Helper functions
    'get_tts',
    'get_asr',
    'get_vad',
    'synthesize',
    'transcribe',
    'detect_voice_activity'
]