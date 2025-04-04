"""
Audio-Text processing for transcription and translation
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import os
import torch
import numpy as np
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    MBartForConditionalGeneration,
    MBart50TokenizerFast
)
from .base import MultimodalBase
from ..speech.audio_utils import AudioUtils

class AudioTextProcessor(MultimodalBase):
    """Process audio for transcription and translation"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        translation_model: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        """Initialize audio text processor
        
        Args:
            model_name: Name of pretrained ASR model
            translation_model: Name of pretrained translation model
            device: Device to run model on
            batch_size: Batch size for processing
        """
        super().__init__(model_name, device, batch_size)
        
        # Load ASR model and processor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Initialize translation components
        self.translation_model_name = translation_model
        self.translation_model = None
        self.translation_tokenizer = None
        
    def _load_translation_model(self):
        """Load translation model on demand"""
        if self.translation_model is None:
            self.translation_tokenizer = MBart50TokenizerFast.from_pretrained(
                self.translation_model_name
            )
            self.translation_model = MBartForConditionalGeneration.from_pretrained(
                self.translation_model_name
            ).to(self.device)
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        chunk_length_s: int = 30
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Transcribe audio to text
        
        Args:
            audio: Audio file path, numpy array, or list of audio inputs
            language: Language code for transcription (e.g., "th", "en")
            task: Task type ("transcribe" or "translate")
            return_timestamps: Whether to return timestamps
            chunk_length_s: Length of audio chunks in seconds for processing
            
        Returns:
            Transcription text or dictionary with text and metadata
        """
        # Handle single audio input
        if isinstance(audio, (str, np.ndarray)):
            audio = [audio]
            single_input = True
        else:
            single_input = False
            
        def process_batch(batch_audio):
            batch_results = []
            
            for audio_input in batch_audio:
                # Load audio if path is provided
                if isinstance(audio_input, str):
                    audio_array, sample_rate = self.load_audio(audio_input)
                else:
                    audio_array = audio_input
                    sample_rate = 16000  # Default sample rate
                
                # Prepare inputs
                inputs = self.processor(
                    audio_array, 
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to(self.device)
                
                # Set forced decoder ids for language and task
                forced_decoder_ids = None
                if language is not None:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=language, task=task
                    )
                
                # Generate transcription
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        forced_decoder_ids=forced_decoder_ids,
                        return_timestamps=return_timestamps
                    )
                
                # Decode transcription
                result = self.processor.batch_decode(
                    outputs, 
                    skip_special_tokens=not return_timestamps
                )[0]
                
                if return_timestamps:
                    # Process and format the timestamps
                    segments = []
                    current_segment = {"text": "", "start": 0, "end": 0}
                    words = []
                    
                    for token in result.split():
                        if token.startswith("<|") and token.endswith("|>"):
                            if "time_" in token:
                                timestamp = float(token.replace("<|time_", "").replace("|>", ""))
                                if current_segment["text"]:
                                    current_segment["end"] = timestamp
                                    segments.append(current_segment.copy())
                                    current_segment = {"text": "", "start": timestamp, "end": 0}
                                else:
                                    current_segment["start"] = timestamp
                        else:
                            current_segment["text"] += " " + token if current_segment["text"] else token
                    
                    # Add the last segment if it has text
                    if current_segment["text"] and current_segment["end"] == 0:
                        # Estimate end time if not provided
                        current_segment["end"] = len(audio_array) / sample_rate
                        segments.append(current_segment)
                    
                    batch_results.append({
                        "text": " ".join([s["text"] for s in segments]),
                        "segments": segments
                    })
                else:
                    batch_results.append(result)
            
            return batch_results
        
        results = self.batch_process(
            audio,
            process_batch
        )
        
        return results[0] if single_input else results
    
    def translate(
        self,
        audio: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        source_lang: str = "auto",
        target_lang: str = "en",
        return_timestamps: bool = False
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Transcribe and translate audio to target language
        
        Args:
            audio: Audio file path, numpy array, or list of audio inputs
            source_lang: Source language code or "auto" for auto-detection
            target_lang: Target language code
            return_timestamps: Whether to return timestamps
            
        Returns:
            Translated text or dictionary with text and metadata
        """
        # First transcribe the audio
        if source_lang == "auto":
            # Use Whisper's built-in translation if target is English
            if target_lang == "en":
                return self.transcribe(
                    audio, 
                    language=None,  # Auto-detect
                    task="translate",
                    return_timestamps=return_timestamps
                )
        
        # Otherwise, transcribe first, then translate
        transcription = self.transcribe(
            audio,
            language=source_lang if source_lang != "auto" else None,
            task="transcribe",
            return_timestamps=return_timestamps
        )
        
        # If target is already the source language, return transcription
        if source_lang == target_lang and source_lang != "auto":
            return transcription
        
        # Load translation model if needed
        self._load_translation_model()
        
        # Extract text from transcription
        if isinstance(transcription, list):
            texts = []
            for item in transcription:
                if isinstance(item, dict):
                    texts.append(item["text"])
                else:
                    texts.append(item)
        else:
            if isinstance(transcription, dict):
                texts = [transcription["text"]]
            else:
                texts = [transcription]
            single_text = True
        
        # Map language codes to mBART format
        source_lang_code = self._map_to_mbart_language(source_lang)
        target_lang_code = self._map_to_mbart_language(target_lang)
        
        # Translate texts
        self.translation_tokenizer.src_lang = source_lang_code
        
        translated_texts = []
        for text in texts:
            inputs = self.translation_tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                translated_tokens = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.lang_code_to_id[target_lang_code],
                    max_length=512
                )
                
            translated_text = self.translation_tokenizer.batch_decode(
                translated_tokens, 
                skip_special_tokens=True
            )[0]
            
            translated_texts.append(translated_text)
        
        # Format results to match input format
        if isinstance(transcription, list):
            results = []
            for i, item in enumerate(transcription):
                if isinstance(item, dict):
                    result = item.copy()
                    result["text"] = translated_texts[i]
                    # Update segment texts if available
                    if "segments" in result:
                        # Simplistic approach - replace all segments with the translation
                        if len(result["segments"]) == 1:
                            result["segments"][0]["text"] = translated_texts[i]
                        else:
                            # More complex segmenting would require alignment
                            pass
                    results.append(result)
                else:
                    results.append(translated_texts[i])
        else:
            if isinstance(transcription, dict):
                results = transcription.copy()
                results["text"] = translated_texts[0]
                # Update segment texts if available
                if "segments" in results:
                    # Simplistic approach - replace all segments with the translation
                    if len(results["segments"]) == 1:
                        results["segments"][0]["text"] = translated_texts[0]
            else:
                results = translated_texts[0]
        
        return results
    
    def _map_to_mbart_language(self, lang_code: str) -> str:
        """Map language code to mBART format"""
        if lang_code == "auto":
            return "en_XX"  # Default to English
            
        # Map ISO 639-1 codes to mBART codes
        mapping = {
            "ar": "ar_AR",
            "cs": "cs_CZ",
            "de": "de_DE",
            "en": "en_XX",
            "es": "es_XX",
            "et": "et_EE",
            "fi": "fi_FI",
            "fr": "fr_XX",
            "gu": "gu_IN",
            "hi": "hi_IN",
            "it": "it_IT",
            "ja": "ja_XX",
            "kk": "kk_KZ",
            "ko": "ko_KR",
            "lt": "lt_LT",
            "lv": "lv_LV",
            "my": "my_MM",
            "ne": "ne_NP",
            "nl": "nl_XX",
            "ro": "ro_RO",
            "ru": "ru_RU",
            "si": "si_LK",
            "th": "th_TH",
            "tr": "tr_TR",
            "uk": "uk_UA",
            "vi": "vi_VN",
            "zh": "zh_CN"
        }
        
        return mapping.get(lang_code, "en_XX")

class AudioTranscriber(AudioTextProcessor):
    """Specialized class for audio transcription"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(model_name, None, device, batch_size)
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        language: Optional[str] = None,
        return_timestamps: bool = False
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Transcribe audio to text"""
        return self.transcribe(audio, language, "transcribe", return_timestamps)

class AudioTranslator(AudioTextProcessor):
    """Specialized class for audio translation"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        translation_model: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 1
    ):
        super().__init__(model_name, translation_model, device, batch_size)
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        source_lang: str = "auto",
        target_lang: str = "en",
        return_timestamps: bool = False
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """Translate audio to target language"""
        return self.translate(audio, source_lang, target_lang, return_timestamps)
