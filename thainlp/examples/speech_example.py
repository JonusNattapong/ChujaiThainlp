"""
Speech Processing Example
"""
import numpy as np
from thainlp.speech import (
    ThaiTTS,
    ThaiASR,
    VoiceActivityDetector,
    VoiceProcessor,
    AudioUtils,
    synthesize,
    transcribe
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
    
    print("\n=== Using transcribe() helper ===")
    print(f"Transcript: {transcribe('greeting.wav')}")

    print("\n=== Voice Activity Detection ===")
    vad = VoiceActivityDetector()
    segments = vad.detect("greeting.wav")
    print(f"Speech segments (seconds): {segments}")

    print("\n=== Voice Conversion Example ===")
    vp = VoiceProcessor()
    # Assume we have source.wav and target_voice.wav
    # converted = vp.convert_voice("source.wav", target_voice_id=10)
    # AudioUtils.save_audio(converted, vp.sample_rate, "converted.wav")
    print("Voice conversion demo would run here with actual audio files")

    print("\n=== Audio Processing Example ===")
    # Load, normalize and trim silence
    audio, sr = AudioUtils.load_audio("greeting.wav")
    audio = AudioUtils.normalize_audio(audio)
    audio = AudioUtils.trim_silence(audio, sr)
    AudioUtils.save_audio(audio, sr, "processed.wav")
    print("Processed and saved to processed.wav")

if __name__ == "__main__":
    main()
