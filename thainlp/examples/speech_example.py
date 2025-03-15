#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from thainlp.speech import ThaiSpeechProcessor

# Create an instance of the ThaiSpeechProcessor
processor = ThaiSpeechProcessor()

# --- Speech-to-Text ---
print("\\n--- Speech-to-Text ---")
try:
    # Replace 'path/to/your/thai_audio.wav' with the actual path to a Thai audio file
    thai_audio_file = "thai_audio.wav"  
    text = processor.speech_to_text(thai_audio_file)
    print(f"Transcribed Text: {text}")
except Exception as e:
    print(f"Error in Speech-to-Text: {e}")

# --- Language Identification ---
print("\\n--- Language Identification ---")
try:
    # Replace 'path/to/your/audio_file.wav' with the path to any audio file
    audio_file = "audio_file.wav"
    languages = processor.identify_language(audio_file, top_k=3)
    print(f"Identified Languages: {languages}")
except Exception as e:
    print(f"Error in Language Identification: {e}")

# --- Text-to-Speech ---
print("\\n--- Text-to-Speech ---")
try:
    thai_text = "สวัสดีครับ ยินดีต้อนรับสู่ระบบประมวลผลเสียงภาษาไทย"
    output_audio_file = "output_thai_audio.wav"
    processor.text_to_speech(thai_text, output_audio_file)
    print(f"Generated audio saved to: {output_audio_file}")
except Exception as e:
    print(f"Error in Text-to-Speech: {e}")

# --- Speaker Diarization ---
print("\\n--- Speaker Diarization ---")
try:
    # Replace 'path/to/your/conversation.wav' with a conversation audio file
    conversation_file = "conversation.wav"
    segments = processor.speaker_diarization(conversation_file)
    print("Speaker Diarization Segments:")
    for segment in segments:
        print(f"  Speaker: {segment['speaker']}, Start: {segment['start']:.2f}, End: {segment['end']:.2f}")
except Exception as e:
    print(f"Error in Speaker Diarization: {e}")

# --- Audio Enhancement ---
print("\\n--- Audio Enhancement ---")
try:
    # Replace 'path/to/your/noisy_audio.wav' with a noisy audio file
    noisy_audio_file = "noisy_audio.wav"
    enhanced_audio_file = "enhanced_audio.wav"
    processor.enhance_audio(noisy_audio_file, enhanced_audio_file)
    print(f"Enhanced audio saved to: {enhanced_audio_file}")
except Exception as e:
    print(f"Error in Audio Enhancement: {e}")

# --- Transcribe with Speaker Diarization ---
print("\\n--- Transcribe with Speaker Diarization ---")
try:
    # Replace 'path/to/your/conversation.wav' with the actual path to a conversation audio file.
    conversation_audio_file = "conversation.wav"
    transcript = processor.transcribe_with_speaker_diarization(conversation_audio_file)
    print("Transcription with Speaker Diarization:")
    for entry in transcript:
        print(f"  Speaker: {entry['speaker']}, Start: {entry['start']:.2f}, End: {entry['end']:.2f}, Text: {entry['text']}")
except Exception as e:
    print(f"Error in Transcribe with Speaker Diarization: {e}")

# --- Detect Emotion from Speech ---
print("\\n--- Detect Emotion from Speech ---")
try:
    # Replace 'path/to/your/speech_audio.wav' with the actual path to a speech audio file.
    speech_audio_file = 'speech_audio.wav'
    emotions = processor.detect_emotion_from_speech(speech_audio_file)
    print(f"Detected Emotions: {emotions}")
except Exception as e:
    print(f"Error in Emotion Detection: {e}")

# --- Voice Activity Detection ---
print("\\n--- Voice Activity Detection ---")
try:
    # Replace 'path/to/your/audio_file.wav' with the actual path to an audio file.
    audio_file_vad = 'audio_file.wav'
    vad_result = processor.voice_activity_detection(audio_file_vad)
    print("Voice Activity Detection Result:")
    print(f"  Total Duration: {vad_result['total_duration']:.2f}s")
    print(f"  Speech Duration: {vad_result['speech_duration']:.2f}s")
    print("  Segments:")
    for segment in vad_result['segments']:
        print(f"    Start: {segment['start']:.2f}s, End: {segment['end']:.2f}s")
except Exception as e:
    print(f"Error in Voice Activity Detection: {e}")

# --- Get Speech Statistics ---
print("\\n--- Get Speech Statistics ---")
try:
    # Replace 'path/to/your/speech_audio.wav' with the actual path to a speech audio file.
    speech_audio_file_stats = 'speech_audio.wav'
    statistics = processor.get_speech_statistics(speech_audio_file_stats)
    print("Speech Statistics:")
    for stat, value in statistics.items():
        print(f"  {stat}: {value}")
except Exception as e:
    print(f"Error in Speech Statistics: {e}")

# --- Batch Process Speech (Example: Transcription) ---
print("\\n--- Batch Process Speech (Transcription) ---")
try:
    # Replace with actual paths to multiple audio files
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    results = processor.batch_process_speech(audio_files, process_type="transcribe")
    print("Batch Processing Results:")
    for result in results:
        if result['success']:
            print(f"  File: {result['file_path']}, Transcription: {result['result']}")
        else:
            print(f"  File: {result['file_path']}, Error: {result['error']}")
except Exception as e:
    print(f"Error in Batch Processing: {e}")

# --- Extract Speech Features ---
print("\\n--- Extract Speech Features ---")
try:
    # Replace 'path/to/your/audio_file.wav' with the actual path to an audio file.
    audio_file_features = 'audio_file.wav'
    features = processor.extract_speech_features(audio_file_features)
    print(f"Extracted Speech Features (first 10): {features[:10]}")  # Print only the first 10 features
except Exception as e:
    print(f"Error in Feature Extraction: {e}")

# --- Speech Similarity ---
print("\\n--- Speech Similarity ---")
try:
    # Replace with actual paths to two audio files
    audio_file1 = "audio1.wav"
    audio_file2 = "audio2.wav"
    similarity = processor.speech_similarity(audio_file1, audio_file2)
    print(f"Speech Similarity: {similarity:.4f}")
except Exception as e:
    print(f"Error in Speech Similarity: {e}")

print("\\nExample script finished.")
