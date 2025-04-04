# ChujaiThainlp: Advanced Thai Natural Language Processing

ChujaiThainlp is a comprehensive Thai Natural Language Processing library that provides state-of-the-art models and tools for various NLP and speech processing tasks. The library supports both Thai and English languages with a focus on Thai-specific processing.

## Key Features

- **Speech Processing**
  - Text-to-speech synthesis
  - Automatic speech recognition
  - Voice activity detection  
  - Voice conversion and style transfer
  - Audio processing utilities

[Previous content remains the same until Quick Start section...]

## Quick Start

```python
from thainlp import ThaiNLPPipeline, synthesize, transcribe

# Initialize pipeline
pipeline = ThaiNLPPipeline()

# Speech synthesis
audio = synthesize("สวัสดีครับ ยินดีต้อนรับ") 

# Speech recognition
text = transcribe("audio.wav")

# Voice activity detection  
from thainlp import detect_voice_activity
segments = detect_voice_activity("audio.wav")

[Rest of previous content...]

## Speech Processing Examples

### Text-to-Speech
```python
from thainlp import synthesize

# Generate speech
audio = synthesize("สวัสดีครับ ยินดีต้อนรับ")

# Save to file
from thainlp.speech import AudioUtils
AudioUtils.save_audio(audio, 22050, "greeting.wav")
```

### Speech Recognition
```python 
from thainlp import transcribe

# Transcribe audio file
text = transcribe("audio.wav")
print(f"Transcript: {text}")

# Batch processing
texts = transcribe(["audio1.wav", "audio2.wav"])
```

### Voice Processing
```python
from thainlp.speech import VoiceProcessor

# Initialize voice processor
vp = VoiceProcessor()

# Convert voice
converted = vp.convert_voice("source.wav", target_voice_id=2)

# Style transfer
styled = vp.transfer_style("source.wav", "style_reference.wav")
```

[Rest of previous content...]
