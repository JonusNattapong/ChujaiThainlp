"""
Speech Processing Example
"""
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
from thainlp.speech import (
    ThaiTTS,
    ThaiASR,
    VoiceActivityDetector,
    VoiceProcessor,
    AudioUtils,
    SpeechSummarizer,
    synthesize,
    transcribe,
    summarize_speech
)

def main():
    print("=== Text-to-Speech Example ===")
    tts = ThaiTTS()
    text = "สวัสดีครับ ยินดีต้อนรับสู่ห้องสมุดไทยเอ็นแอลพี"
    audio = tts.synthesize(text)
    print(f"Generated audio length: {len(audio)} samples")
    tts.save_to_file(audio, "greeting.wav")
    print("Saved to greeting.wav")
    
    print("\n=== Using synthesize() helper ===")
    audio = synthesize(text)
    print(f"Generated audio length: {len(audio)} samples")

    print("\n=== Speech Recognition Example ===")
    asr = ThaiASR()
    transcription = asr.transcribe("greeting.wav")
    print(f"Transcription: {transcription}")
    
    print("\n=== Using transcribe() helper with Thai validation ===")
    result = transcribe('greeting.wav', validate_thai=True)
    print(f"Transcript: {result['text']}")
    print(f"Valid Thai: {'✓' if result['valid_thai'] else '✗'}")
    if 'corrected_text' in result:
        print(f"Corrected: {result['corrected_text']}")

    print("\n=== Voice Activity Detection ===")
    vad = VoiceActivityDetector()
    segments = vad.detect("greeting.wav")
    print(f"Speech segments (seconds): {segments}")

    # Note: Voice conversion features are under development
    # print("\n=== Voice Conversion Example ===")
    # vp = VoiceProcessor()
    # converted = vp.convert_voice("source.wav", target_voice_id=10)
    # AudioUtils.save_audio(converted, vp.sample_rate, "converted.wav")
    # print("Voice conversion features coming soon...")

    print("\n=== Audio Processing Example ===")
    # Load, normalize and trim silence
    audio, sr = AudioUtils.load_audio("greeting.wav")
    audio = AudioUtils.normalize_audio(audio)
    audio = AudioUtils.trim_silence(audio, sr)
    AudioUtils.save_audio(audio, sr, "processed.wav")
    print("Processed and saved to processed.wav")

    print("\n=== Speech Summarization Example ===")
    summarizer = SpeechSummarizer()
    summary = summarizer.summarize("greeting.wav")
    print(f"Summary: {summary}")
    
    print("\n=== Using summarize_speech() helper ===")
    print(f"Summary: {summarize_speech('greeting.wav')}")

if __name__ == "__main__":
    main()
