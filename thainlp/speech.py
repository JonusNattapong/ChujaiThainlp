"""
ระบบประมวลผลคำพูดภาษาไทย (Thai Speech Processing System)
===========================================================

โมดูลนี้ใช้สำหรับการประมวลผลคำพูดภาษาไทย รวมถึง:
- การรู้จำเสียงพูดภาษาไทยเป็นข้อความ (Speech-to-Text)
- การระบุภาษาที่พูด (Speech Language Identification)
- การถอดเสียงพูดเป็นข้อความ (Transcription)
- การสังเคราะห์เสียงจากข้อความภาษาไทย (Text-to-Speech)
- การระบุตัวผู้พูด (Speaker Identification)
- การแบ่งส่วนเสียงตามผู้พูด (Speaker Diarization)
- การปรับปรุงคุณภาพเสียง (Audio Enhancement)
"""

import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import urllib.request
import zipfile
import torchaudio
import warnings
from typing import List, Dict, Tuple, Union, Optional
from thainlp.tokenize import word_tokenize
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    AutoProcessor, 
    AutoModelForTextToWaveform,
    pipeline
)

# ใช้ SpeechBrain สำหรับการระบุภาษา
try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    print("กำลังติดตั้ง SpeechBrain...")
    import subprocess
    subprocess.check_call(["pip", "install", "speechbrain"])
    from speechbrain.pretrained import EncoderClassifier

class ThaiSpeechProcessor:
    def __init__(self, model_path=None, use_gpu=False, download_models=True):
        """
        คลาสสำหรับการประมวลผลคำพูดภาษาไทย
        
        Parameters:
        -----------
        model_path: str หรือ None
            เส้นทางโฟลเดอร์เก็บโมเดล ASR ถ้าเป็น None จะดาวน์โหลดโมเดลจากอินเทอร์เน็ต
        use_gpu: bool
            ใช้ GPU สำหรับการประมวลผลหรือไม่
        download_models: bool
            ดาวน์โหลดโมเดลที่จำเป็นโดยอัตโนมัติหรือไม่
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.models = {}
        
        print(f"กำลังใช้อุปกรณ์: {self.device}")
        
        # สร้างโฟลเดอร์เก็บโมเดล
        self.model_dir = model_path or os.path.join(os.path.expanduser("~"), ".thainlp_models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        if download_models:
            # โหลดโมเดล ASR สำหรับภาษาไทย
            try:
                self._load_asr_model()
                print("โหลดโมเดล ASR สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล ASR ได้: {str(e)}")
            
            # โหลดโมเดล language identification
            try:
                self._load_lang_id_model()
                print("โหลดโมเดลระบุภาษาสำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดลระบุภาษาได้: {str(e)}")
            
            # โหลดโมเดล TTS สำหรับภาษาไทย
            try:
                self._load_tts_model()
                print("โหลดโมเดล TTS สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล TTS ได้: {str(e)}")
            
            # โหลดโมเดล Speaker Diarization
            try:
                self._load_speaker_diarization_model()
                print("โหลดโมเดลแบ่งส่วนเสียงตามผู้พูดสำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดลแบ่งส่วนเสียงตามผู้พูดได้: {str(e)}")
        
    def _load_asr_model(self):
        """โหลดโมเดล ASR สำหรับภาษาไทย"""
        try:
            # ใช้โมเดล Wav2Vec2 สำหรับภาษาไทยจาก AIResearch
            self.models['asr_processor'] = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
            self.models['asr_model'] = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th").to(self.device)
        except:
            # ใช้โมเดลอื่นๆ ที่รองรับภาษาไทยเป็นทางเลือก
            print("กำลังโหลดโมเดลทางเลือก...")
            self.models['asr_processor'] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            self.models['asr_model'] = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(self.device)
            
    def _load_lang_id_model(self):
        """โหลดโมเดลสำหรับระบุภาษาจากเสียงพูด (VoxLingua107)"""
        # ใช้ SpeechBrain และ VoxLingua107 สำหรับการระบุภาษาจากเสียงพูด
        self.models['lang_id'] = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa", 
            savedir=os.path.join(self.model_dir, "lang_id_voxlingua107")
        )
        
    def _load_tts_model(self):
        """โหลดโมเดล TTS สำหรับภาษาไทย"""
        # ใช้โมเดลที่รองรับภาษาไทย
        try:
            self.models['tts_processor'] = AutoProcessor.from_pretrained("facebook/mms-tts-th")
            self.models['tts_model'] = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-th").to(self.device)
            self.tts_available = True
        except:
            # ใช้โมเดล seamless-m4t-v2 ซึ่งรองรับหลายภาษา รวมถึงภาษาไทย
            self.models['tts'] = pipeline("text-to-speech", model="facebook/seamless-m4t-v2-large")
            self.tts_available = True
        
    def _load_speaker_diarization_model(self):
        """โหลดโมเดลสำหรับแบ่งส่วนเสียงตามผู้พูด"""
        # ใช้ pyannote.audio สำหรับการแบ่งส่วนเสียงตามผู้พูด
        try:
            from pyannote.audio import Pipeline
            self.models['diarization'] = Pipeline.from_pretrained("pyannote/speaker-diarization", 
                                                               use_auth_token=False)
            self.diarization_available = True
        except:
            self.diarization_available = False
            print("ไม่สามารถโหลดโมเดล Speaker Diarization ได้ ฟังก์ชันการแบ่งส่วนเสียงตามผู้พูดจะไม่ทำงาน")
    
    def load_audio(self, file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        โหลดไฟล์เสียงและปรับความถี่การสุ่มตัวอย่างถ้าจำเป็น
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียง
        target_sr: int
            ความถี่การสุ่มตัวอย่างเป้าหมายในหน่วย Hz
            
        Returns:
        --------
        (waveform, sample_rate): Tuple[np.ndarray, int]
            คลื่นเสียงและความถี่การสุ่มตัวอย่าง
        """
        waveform, sample_rate = librosa.load(file_path, sr=target_sr)
        return waveform, sample_rate
    
    def identify_language(self, file_path: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        ระบุภาษาที่พูดในไฟล์เสียง
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียง
        top_k: int
            จำนวนภาษาที่มีความน่าจะเป็นสูงสุดที่จะแสดงผล
            
        Returns:
        --------
        results: List[Dict[str, Union[str, float]]]
            รายการภาษาพร้อมความน่าจะเป็น เรียงลำดับจากมากไปน้อย
        """
        if 'lang_id' not in self.models:
            self._load_lang_id_model()
        
        # ทำนาย
        signal, fs = torchaudio.load(file_path)
        lang_probs = self.models['lang_id'].classify_batch(signal)
        
        # แปลงผลลัพธ์เป็นรูปแบบที่ใช้งานง่าย
        lang_id_to_name = {
            "th": "Thai",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "vi": "Vietnamese",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            # เพิ่มภาษาอื่นๆ ตามต้องการ
        }
        
        predictions = []
        language_names = self.models['lang_id'].hparams.label_encoder.ind2lab
        probs = torch.nn.functional.softmax(lang_probs[0], dim=0)
        
        # สร้างรายการผลลัพธ์
        for i in range(len(language_names)):
            language_code = language_names[i]
            language_name = lang_id_to_name.get(language_code, language_code)
            prob = probs[i].item()
            predictions.append({
                "language_code": language_code,
                "language_name": language_name,
                "probability": prob
            })
        
        # เรียงลำดับตามความน่าจะเป็นและเลือก top_k
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return predictions[:top_k]
        
    def speech_to_text(self, file_path: str, language: str = 'th') -> str:
        """
        แปลงเสียงพูดเป็นข้อความ
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียง
        language: str
            รหัสภาษา (เช่น 'th' สำหรับภาษาไทย, 'en' สำหรับภาษาอังกฤษ)
            
        Returns:
        --------
        text: str
            ข้อความที่แปลงจากเสียงพูด
        """
        if 'asr_model' not in self.models:
            self._load_asr_model()
            
        # โหลดไฟล์เสียง
        waveform, sample_rate = self.load_audio(file_path)
        
        # แปลงเสียงเป็นข้อความ
        inputs = self.models['asr_processor'](
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.models['asr_model'](**inputs).logits
            
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.models['asr_processor'].batch_decode(predicted_ids)[0]
        
        return transcription
        
        def text_to_speech(self, text: str, output_path: str = "output.wav", speaker_id: int = 0) -> str:
            """
            แปลงข้อความภาษาไทยเป็นเสียงพูด
            
            Parameters:
            -----------
            text: str
                ข้อความภาษาไทยที่ต้องการแปลงเป็นเสียง
            output_path: str
                เส้นทางไฟล์สำหรับบันทึกเสียงที่สร้าง
            speaker_id: int
                ID ของเสียงผู้พูด (ถ้ามีหลายเสียงให้เลือก)
                
            Returns:
            --------
            output_path: str
                เส้นทางไฟล์เสียงที่สร้างขึ้น
            """
        if not self.tts_available:
            raise RuntimeError("ฟังก์ชัน TTS ไม่สามารถใช้งานได้ โปรดตรวจสอบว่าโหลดโมเดล TTS สำเร็จแล้ว")
            
        if 'tts_model' in self.models:
            # ใช้โมเดล MMS-TTS
            inputs = self.models['tts_processor'](text=text, return_tensors="pt")
            speech = self.models['tts_model'].generate_speech(
                inputs["input_ids"].to(self.device), 
                speaker_id=torch.tensor([speaker_id], device=self.device), 
                vocoder=None
            )
            
            # แปลงเป็น numpy array และบันทึกเป็นไฟล์ WAV
            speech_numpy = speech.cpu().numpy()
            sf.write(output_path, speech_numpy, self.models['tts_model'].config.sampling_rate)
            
        elif 'tts' in self.models:
            # ใช้ pipeline จาก transformers
            result = self.models['tts'](text, forward_params={"speaker_id": speaker_id})
            sf.write(output_path, result["audio"], result["sampling_rate"])
            
        return output_path

    def enhance_audio(self, file_path: str, output_path: str = "enhanced_audio.wav") -> str:
        """
        ปรับปรุงคุณภาพเสียง โดยลดเสียงรบกวนและเพิ่มความชัดเจนของเสียงพูด
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียงต้นฉบับ
        output_path: str
            เส้นทางสำหรับบันทึกไฟล์เสียงที่ปรับปรุงแล้ว
            
        Returns:
        --------
        output_path: str
            เส้นทางไฟล์เสียงที่ปรับปรุงแล้ว
        """
        try:
            import noisereduce as nr
        except ImportError:
            import subprocess
            print("กำลังติดตั้ง noisereduce...")
            subprocess.check_call(["pip", "install", "noisereduce"])
            import noisereduce as nr
            
        # โหลดไฟล์เสียง
        waveform, sample_rate = self.load_audio(file_path)
        
        # ลดเสียงรบกวน
        enhanced_waveform = nr.reduce_noise(y=waveform, sr=sample_rate)
        
        # ปรับเพิ่มความดังเสียง (normalization)
        if np.abs(enhanced_waveform).max() > 0:
            enhanced_waveform = enhanced_waveform / np.abs(enhanced_waveform).max() * 0.9
        
        # บันทึกเสียงที่ปรับปรุงแล้ว
        sf.write(output_path, enhanced_waveform, sample_rate)
        
        return output_path
        
    def speaker_diarization(self, file_path: str) -> List[Dict]:
        """
        แบ่งส่วนเสียงตามผู้พูด (เพื่อระบุว่าใครพูดเมื่อไร)
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียงบทสนทนา
            
        Returns:
        --------
        segments: List[Dict]
            รายการของช่วงเวลาและผู้พูด
        """
        if not hasattr(self, 'diarization_available') or not self.diarization_available:
            raise RuntimeError("ฟังก์ชันการแบ่งส่วนเสียงตามผู้พูดไม่สามารถใช้งานได้")
            
        # ทำการแบ่งส่วนเสียงตามผู้พูด
        diarization = self.models['diarization'](file_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start
            })
            
        return segments

    def transcribe_with_speaker_diarization(self, file_path: str) -> List[Dict]:
        """
        ถอดเสียงพร้อมระบุผู้พูด
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียงบทสนทนา
            
        Returns:
        --------
        transcript: List[Dict]
            รายการของข้อความพร้อมระบุผู้พูด
        """
        # ระบุส่วนของผู้พูด
        try:
            segments = self.speaker_diarization(file_path)
        except Exception as e:
            print(f"ไม่สามารถแบ่งส่วนเสียงตามผู้พูดได้: {str(e)}")
            return []

        # โหลดไฟล์เสียง
        waveform, sample_rate = librosa.load(file_path, sr=16000)
        
        transcript = []
        for segment in segments:
            # ตัดเสียงตามช่วงเวลาของผู้พูด
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            
            if start_sample >= len(waveform) or start_sample >= end_sample:
                continue
                
            segment_waveform = waveform[start_sample:min(end_sample, len(waveform))]
            
            # บันทึกช่วงเสียงชั่วคราว
            temp_file = f"temp_segment_{segment['speaker']}_{segment['start']:.2f}.wav"
            sf.write(temp_file, segment_waveform, sample_rate)
            
            try:
                # ถอดเสียงเป็นข้อความ
                text = self.speech_to_text(temp_file)
                
                # เพิ่มเข้าในผลลัพธ์
                transcript.append({
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text
                })
            except Exception as e:
                print(f"ไม่สามารถถอดเสียงได้สำหรับ speaker {segment['speaker']}: {str(e)}")
            finally:
                # ลบไฟล์ชั่วคราว
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return transcript
                
    def detect_emotion_from_speech(self, file_path: str) -> Dict[str, float]:
        """
        ตรวจจับอารมณ์จากเสียงพูด
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียงพูด
            
        Returns:
        --------
        emotions: Dict[str, float]
            พจนานุกรมของอารมณ์และคะแนนความน่าจะเป็น
        """
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            raise ImportError("ต้องติดตั้ง SpeechBrain ก่อนใช้งานฟังก์ชันนี้")
            
        # ตรวจสอบว่ามีโมเดล Emotion Recognition หรือไม่
        if 'emotion_recognition' not in self.models:
            try:
                # ใช้โมเดลจาก SpeechBrain สำหรับการรู้จำอารมณ์
                self.models['emotion_recognition'] = EncoderClassifier.from_hparams(
                    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                    savedir=os.path.join(self.model_dir, "emotion_recognition")
                )
            except Exception as e:
                raise RuntimeError(f"ไม่สามารถโหลดโมเดล Emotion Recognition ได้: {str(e)}")
        
        # ทำนายอารมณ์
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            
        out_prob, score, index, text_lab = self.models['emotion_recognition'].classify_batch(waveform)
        
        # แปลงผลลัพธ์เป็นรูปแบบที่ใช้งานง่าย
        emotion_names = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
        emotions = {}
        
        for i, emotion_name in enumerate(emotion_names):
            if i < len(out_prob[0]):
                emotions[emotion_name] = out_prob[0][i].item()
        
        return emotions

    def voice_activity_detection(self, file_path: str, frame_length=1024, hop_length=512, threshold=0.1) -> Dict:
        """
        ตรวจจับช่วงที่มีการพูดในไฟล์เสียง
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียง
        frame_length: int
            ความยาวของเฟรมสำหรับการวิเคราะห์
        hop_length: int
            ระยะกระโดดระหว่างเฟรม
        threshold: float
            ค่าขีดแบ่งของความดังเสียงสำหรับการตรวจจับเสียงพูด
            
        Returns:
        --------
        result: Dict
            ผลลัพธ์ของการตรวจจับช่วงเสียงพูด
        """
        # โหลดไฟล์เสียง
        waveform, sample_rate = self.load_audio(file_path)
        
        # คำนวณ root mean square energy (RMSE)
        rmse = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        
        # ใช้ค่าขีดแบ่งเพื่อระบุช่วงที่มีการพูด
        voiced_frames = rmse > threshold
        
        # แปลงเป็นช่วงเวลา
        frame_times = librosa.frames_to_time(np.arange(len(rmse)), sr=sample_rate, hop_length=hop_length)
        
        # ค้นหาช่วงที่มีการพูด
        segments = []
        in_segment = False
        segment_start = 0
        
        for i in range(len(voiced_frames)):
            if voiced_frames[i] and not in_segment:
                # เริ่มช่วงใหม่
                in_segment = True
                segment_start = frame_times[i]
            elif not voiced_frames[i] and in_segment:
                # จบช่วง
                in_segment = False
                segments.append({
                    "start": segment_start,
                    "end": frame_times[i]
                })
        
        # กรณีที่สิ้นสุดไฟล์แต่ยังอยู่ในช่วงเสียงพูด
        if in_segment:
            segments.append({
                "start": segment_start,
                "end": frame_times[-1]
            })
            
        return {
            "sample_rate": sample_rate,
            "frame_count": len(rmse),
            "segments": segments,
            "total_duration": len(waveform) / sample_rate,
            "speech_duration": sum([seg["end"] - seg["start"] for seg in segments])
        }
        
        def get_speech_statistics(self, file_path: str) -> Dict:
            """
            คำนวณสถิติของเสียงพูด เช่น ระดับเสียง ช่วงเสียง ความเร็วในการพูด
            
            Parameters:
            -----------
            file_path: str
                เส้นทางไปยังไฟล์เสียงพูด
                
            Returns:
            --------
            stats: Dict
                พจนานุกรมของสถิติเสียงพูด
            """
        # โหลดไฟล์เสียง
        waveform, sample_rate = self.load_audio(file_path)
        
        # ตรวจจับช่วงเสียงพูด
        vad_result = self.voice_activity_detection(file_path)
        
        # คำนวณค่าเฉลี่ยของความถี่หลัก (f0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate
        )
        f0_mean = np.nanmean(f0) if sum(~np.isnan(f0)) > 0 else 0
        f0_std = np.nanstd(f0) if sum(~np.isnan(f0)) > 0 else 0
        
        # คำนวณระดับเสียงเฉลี่ย (dB)
        db = librosa.amplitude_to_db(np.abs(waveform))
        db_mean = np.mean(db)
        db_std = np.std(db)
        
        # คำนวณค่า spectrogram
        spec = np.abs(librosa.stft(waveform))
        spec_mean = np.mean(spec)
        spec_std = np.std(spec)
        
        # หาค่าความเร็วในการพูด (จากจำนวนช่วงเสียงพูดต่อวินาที)
        speech_speed = len(vad_result["segments"]) / vad_result["total_duration"] if vad_result["total_duration"] > 0 else 0
        
        # สัดส่วนการพูดเทียบกับความยาวไฟล์ทั้งหมด
        speech_ratio = vad_result["speech_duration"] / vad_result["total_duration"] if vad_result["total_duration"] > 0 else 0
        
        return {
            "f0_mean": float(f0_mean),
            "f0_std": float(f0_std),
            "pitch_hz": float(f0_mean),
            "db_mean": float(db_mean),
            "db_std": float(db_std),
            "volume_db": float(db_mean),
            "spectrogram_mean": float(spec_mean),
            "spectrogram_std": float(spec_std),
            "speech_speed": float(speech_speed),
            "speech_segments_count": len(vad_result["segments"]),
            "speech_duration": float(vad_result["speech_duration"]),
            "total_duration": float(vad_result["total_duration"]),
            "speech_ratio": float(speech_ratio)
        }
    
    def batch_process_speech(self, file_paths: List[str], process_type: str = "transcribe", **kwargs) -> List:
        """
        ประมวลผลไฟล์เสียงจำนวนมากพร้อมกัน
        
        Parameters:
        -----------
        file_paths: List[str]
            รายการเส้นทางไปยังไฟล์เสียง
        process_type: str
            ประเภทการประมวลผล ("transcribe", "lang_id", "statistics", "emotion", etc.)
        **kwargs:
            พารามิเตอร์เพิ่มเติมสำหรับฟังก์ชันประมวลผล
            
        Returns:
        --------
        results: List
            รายการผลลัพธ์ตามประเภทการประมวลผล
        """
        results = []
        
        for i, file_path in enumerate(file_paths):
            try:
                print(f"กำลังประมวลผลไฟล์ {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                
                if process_type == "transcribe":
                    result = self.speech_to_text(file_path, **kwargs)
                elif process_type == "lang_id":
                    result = self.identify_language(file_path, **kwargs)
                elif process_type == "statistics":
                    result = self.get_speech_statistics(file_path)
                elif process_type == "emotion":
                    result = self.detect_emotion_from_speech(file_path)
                elif process_type == "enhance":
                    output_path = kwargs.get("output_dir", "") + f"/enhanced_{os.path.basename(file_path)}"
                    result = self.enhance_audio(file_path, output_path)
                elif process_type == "diarization":
                    result = self.speaker_diarization(file_path)
                elif process_type == "transcribe_diarize":
                    result = self.transcribe_with_speaker_diarization(file_path)
                elif process_type == "vad":
                    result = self.voice_activity_detection(file_path, **kwargs)
                else:
                    raise ValueError(f"ประเภทการประมวลผล '{process_type}' ไม่รองรับ")
                
                results.append({
                    "file_path": file_path,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์ {file_path}: {str(e)}")
                results.append({
                    "file_path": file_path,
                    "error": str(e),
                    "success": False
                })
                
        return results

    def extract_speech_features(self, file_path: str) -> np.ndarray:
        """
        สกัดคุณลักษณะเสียงพูดสำหรับการเรียนรู้ของเครื่อง
        
        Parameters:
        -----------
        file_path: str
            เส้นทางไปยังไฟล์เสียง
            
        Returns:
        --------
        features: np.ndarray
            เวกเตอร์คุณลักษณะเสียง
        """
        # โหลดไฟล์เสียง
        waveform, sample_rate = self.load_audio(file_path)
        
        # สกัดคุณลักษณะ MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # สกัดคุณลักษณะ Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
        spectral_centroids_mean = np.mean(spectral_centroids)
        spectral_centroids_std = np.std(spectral_centroids)
        
        # สกัดคุณลักษณะ Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform)[0]
        zcr_mean = np.mean(zero_crossing_rate)
        zcr_std = np.std(zero_crossing_rate)
        
        # สกัดคุณลักษณะ Chromagram
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # รวมคุณลักษณะทั้งหมด
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            [spectral_centroids_mean, spectral_centroids_std],
            [zcr_mean, zcr_std],
            chroma_mean, chroma_std
        ])
        
        return features
    
    def speech_similarity(self, file_path1: str, file_path2: str) -> float:
        """
        คำนวณความคล้ายคลึงระหว่างเสียงพูดสองไฟล์
        
        Parameters:
        -----------
        file_path1: str
            เส้นทางไปยังไฟล์เสียงที่ 1
        file_path2: str
            เส้นทางไปยังไฟล์เสียงที่ 2
            
        Returns:
        --------
        similarity: float
            คะแนนความคล้ายคลึง (0.0 - 1.0)
        """
        # สกัดคุณลักษณะจากทั้งสองไฟล์
        features1 = self.extract_speech_features(file_path1)
        features2 = self.extract_speech_features(file_path2)
        
        # คำนวณความคล้ายคลึงโดยใช้ cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
        
        return float(similarity)
