"""
Speech Dialect Adapter for Thai dialects

This module provides functionality to adapt speech synthesis to different Thai dialects,
allowing TTS systems to produce speech with regional accents.
"""

from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
from pathlib import Path
import torch

from ..dialects.dialect_processor import ThaiDialectProcessor, get_speech_characteristics

class ThaiDialectSpeechAdapter:
    """Adapt speech synthesis for Thai dialects"""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize dialect adapter for speech synthesis
        
        Args:
            data_dir: Directory containing dialect speech data
            device: Computation device (cuda/cpu)
        """
        self.device = device
        self.dialect_processor = ThaiDialectProcessor()
        
        # Configure data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Default to module directory/data
            self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Load accent parameters for each dialect
        self.accent_parameters = self._load_accent_parameters()
        
    def _load_accent_parameters(self) -> Dict[str, Any]:
        """Load dialect accent parameters from files"""
        accent_file = self.data_dir / "dialect_accents.json"
        
        if accent_file.exists():
            try:
                with open(accent_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading accent parameters: {e}")
                
        # Default accent parameters if file doesn't exist
        return {
            "northern": {
                "pitch_shift": 1.2,     # Higher pitch
                "formant_shift": 0.97,  # Slight formant shift
                "speech_rate": 0.9,     # Slower speech rate
                "tone_emphasis": 1.2,   # Emphasize tones more
                "final_particle_emphasis": ["เจ้า", "กา", "แล้วกา"],
                "phoneme_mapping": {
                    # Maps standard phonemes to dialect phonemes
                    "r": "h", # Replace some 'r' sounds with 'h'
                    "ch": "s", # Some 'ch' sounds become 's'
                }
            },
            "northeastern": {
                "pitch_shift": 1.1,
                "formant_shift": 0.96,
                "speech_rate": 1.05,    # Slightly faster
                "tone_emphasis": 1.25,  # More exaggerated tones
                "final_particle_emphasis": ["เด้อ", "สิ", "กะ"],
                "phoneme_mapping": {
                    "r": "l", # Replace some 'r' sounds with 'l'
                }
            },
            "southern": {
                "pitch_shift": 0.95,    # Lower pitch
                "formant_shift": 1.05,  # Different formant shift
                "speech_rate": 1.15,    # Faster speech
                "tone_emphasis": 1.3,   # Strong tone emphasis
                "final_particle_emphasis": ["หนิ", "โหล", "แอ"],
                "phoneme_mapping": {
                    "ch": "s", # Some 'ch' sounds become 's'
                }
            },
            "pattani_malay": {
                "pitch_shift": 0.9,
                "formant_shift": 1.1,
                "speech_rate": 1.0,
                "tone_emphasis": 0.9,   # Less tonal
                "final_particle_emphasis": ["มะ", "เลอ", "ยอ"],
                "phoneme_mapping": {
                    "r": "",  # Often drop 'r' sounds
                    "l": "w", # Some 'l' sounds become 'w'
                }
            },
            # Standard Thai is the baseline
            "central": {
                "pitch_shift": 1.0,
                "formant_shift": 1.0,
                "speech_rate": 1.0,
                "tone_emphasis": 1.0,
                "final_particle_emphasis": [],
                "phoneme_mapping": {}
            }
        }
    
    def get_accent_parameters(
        self,
        dialect: str,
        region: Optional[str] = None,
        strength: float = 1.0
    ) -> Dict[str, Any]:
        """Get accent parameters for a dialect, optionally for a specific region
        
        Args:
            dialect: Dialect code (northern, northeastern, southern, etc.)
            region: Optional specific region within the dialect
            strength: Accent strength factor (0.0-1.0)
            
        Returns:
            Dictionary of accent parameters
        """
        # Get base parameters for the dialect
        if dialect not in self.accent_parameters:
            return self.accent_parameters["central"].copy()
        
        params = self.accent_parameters[dialect].copy()
        
        # Apply region-specific adjustments if available
        if region:
            # Get more specific characteristics from dialect processor
            speech_chars = self.dialect_processor.get_speech_characteristics(dialect, region)
            
            if speech_chars:
                # Apply regional adjustments
                params["pitch_shift"] = speech_chars.get("pitch_shift", params["pitch_shift"])
                params["speech_rate"] = speech_chars.get("speech_rate", params["speech_rate"])
                
                # Add region-specific particles if available
                if "characteristic_sounds" in speech_chars:
                    region_particles = speech_chars["characteristic_sounds"].get("final_particles", [])
                    params["final_particle_emphasis"].extend(region_particles)
        
        # Apply strength factor (interpolate between standard Thai and full dialect)
        if strength < 1.0:
            standard_params = self.accent_parameters["central"]
            for key in ["pitch_shift", "formant_shift", "speech_rate", "tone_emphasis"]:
                # Interpolate between standard (strength=0) and dialect (strength=1)
                params[key] = standard_params[key] + strength * (params[key] - standard_params[key])
        
        return params
    
    def adapt_speech_parameters(
        self,
        text: str,
        tts_params: Dict[str, Any],
        dialect: Optional[str] = None,
        region: Optional[str] = None,
        strength: float = 1.0,
        auto_detect: bool = False
    ) -> Dict[str, Any]:
        """Adapt speech synthesis parameters for dialect-specific pronunciation
        
        Args:
            text: Text to synthesize
            tts_params: Base TTS parameters
            dialect: Dialect code (if None and auto_detect, will be detected)
            region: Optional specific region within the dialect
            strength: Accent strength factor (0.0-1.0)
            auto_detect: Whether to auto-detect dialect from text
            
        Returns:
            Modified TTS parameters for dialect-specific speech
        """
        # Auto-detect dialect if needed
        if auto_detect and not dialect:
            dialect_scores = self.dialect_processor.detect_dialect(text)
            dialect = max(dialect_scores, key=lambda k: dialect_scores[k])
            
            # Auto-detect region if we successfully detected a dialect
            if dialect in self.dialect_processor.dialect_variations:
                region_scores = self.dialect_processor.detect_regional_dialect(text, dialect)
                if region_scores:
                    region = max(region_scores, key=lambda k: region_scores[k])
        
        # Default to central Thai if no dialect specified or detected
        if not dialect or dialect not in self.accent_parameters:
            dialect = "central"
            
        # Get accent parameters
        accent_params = self.get_accent_parameters(dialect, region, strength)
        
        # Clone the original TTS parameters
        adapted_params = tts_params.copy()
        
        # Apply accent adjustments to TTS parameters
        # Pitch shift
        if "pitch_factor" in adapted_params:
            adapted_params["pitch_factor"] *= accent_params["pitch_shift"]
        else:
            adapted_params["pitch_factor"] = accent_params["pitch_shift"]
            
        # Speech rate
        if "speed_factor" in adapted_params:
            adapted_params["speed_factor"] *= accent_params["speech_rate"]
        else:
            adapted_params["speed_factor"] = accent_params["speech_rate"]
            
        # Add any other TTS-specific parameters
        adapted_params["dialect"] = dialect
        if region:
            adapted_params["region"] = region
            
        # Store accent parameters for post-processing
        adapted_params["accent_params"] = accent_params
        
        return adapted_params
    
    def post_process_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        accent_params: Dict[str, Any]
    ) -> Tuple[torch.Tensor, int]:
        """Apply dialect-specific post-processing to synthesized audio
        
        Args:
            audio: Audio tensor
            sample_rate: Audio sample rate
            accent_params: Accent parameters for post-processing
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Nothing to process if we have standard Thai or no parameters
        if not accent_params or accent_params.get("pitch_shift", 1.0) == 1.0:
            return audio, sample_rate
            
        # Apply formant shifting if needed (more complex, requires additional libraries)
        if "formant_shift" in accent_params and accent_params["formant_shift"] != 1.0:
            try:
                # This would require a specialized formant shifting implementation
                # For simplicity, we're providing a placeholder that could be implemented later
                pass
            except Exception as e:
                print(f"Formant shifting not applied: {e}")
        
        return audio, sample_rate
    
    def detect_dialect_from_speech(
        self,
        audio_path: str
    ) -> Dict[str, float]:
        """Detect Thai dialect from speech audio (if supported)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary mapping dialect codes to confidence scores
        """
        # This is a placeholder for future implementation
        # In a real implementation, this would use acoustic models to detect
        # dialect features from speech audio
        
        print("Speech dialect detection not yet implemented")
        return {"central": 1.0}  # Default to central Thai
    
    def create_dialect_voice_profile(
        self,
        dialect: str,
        region: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None
    ) -> str:
        """Create a reusable voice profile for a dialect
        
        Args:
            dialect: Dialect code
            region: Optional specific region
            custom_params: Custom adjustment parameters
            profile_name: Optional name for the profile
            
        Returns:
            Profile identifier
        """
        # Generate a name if not provided
        if not profile_name:
            profile_name = f"{dialect}"
            if region:
                profile_name += f"_{region}"
                
        # Get base parameters
        params = self.get_accent_parameters(dialect, region)
        
        # Apply any custom adjustments
        if custom_params:
            for key, value in custom_params.items():
                if key in params:
                    params[key] = value
        
        # Save the profile
        profiles_file = self.data_dir / "voice_profiles.json"
        
        profiles = {}
        if profiles_file.exists():
            try:
                with open(profiles_file, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except:
                pass
                
        profiles[profile_name] = {
            "dialect": dialect,
            "region": region,
            "parameters": params
        }
        
        with open(profiles_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
            
        return profile_name
    
    def get_voice_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        """Get a saved voice profile
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Profile parameters or None if not found
        """
        profiles_file = self.data_dir / "voice_profiles.json"
        
        if profiles_file.exists():
            try:
                with open(profiles_file, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                    if profile_name in profiles:
                        return profiles[profile_name]
            except Exception as e:
                print(f"Error loading voice profile: {e}")
                
        return None


def adapt_tts_for_dialect(
    text: str,
    tts_params: Dict[str, Any],
    dialect: str,
    region: Optional[str] = None,
    strength: float = 1.0
) -> Dict[str, Any]:
    """Adapt TTS parameters for a specific Thai dialect
    
    Args:
        text: Text to synthesize
        tts_params: Base TTS parameters
        dialect: Dialect code
        region: Optional specific region
        strength: Accent strength factor (0.0-1.0)
        
    Returns:
        Modified TTS parameters
    """
    adapter = ThaiDialectSpeechAdapter()
    return adapter.adapt_speech_parameters(text, tts_params, dialect, region, strength)

def detect_dialect_from_text(text: str) -> Tuple[str, Optional[str]]:
    """Detect dialect and region from text for speech synthesis
    
    Args:
        text: Thai text
        
    Returns:
        Tuple of (dialect_code, region_code)
    """
    processor = ThaiDialectProcessor()
    dialect_scores = processor.detect_dialect(text)
    dialect = max(dialect_scores, key=lambda k: dialect_scores[k])
    
    region = None
    if dialect in processor.dialect_variations:
        region_scores = processor.detect_regional_dialect(text, dialect)
        if region_scores:
            region = max(region_scores, key=lambda k: region_scores[k])
            
    return dialect, region

def get_available_dialects() -> List[Dict[str, Any]]:
    """Get information about all available dialects for speech
    
    Returns:
        List of dialect information dictionaries
    """
    processor = ThaiDialectProcessor()
    dialects = []
    
    for code, info in processor.get_all_dialects().items():
        dialect_info = {
            "code": code,
            "name": info["name"],
            "thai_name": info["thai_name"],
            "regions": []
        }
        
        # Add regions if available
        if code in processor.dialect_variations:
            for region, details in processor.dialect_variations[code].items():
                region_info = {
                    "code": region,
                    "description": details.get("description", "")
                }
                dialect_info["regions"].append(region_info)
                
        dialects.append(dialect_info)
        
    return dialects