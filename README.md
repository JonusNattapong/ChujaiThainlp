# ChujaiThaiNLP: Advanced Thai Natural Language Processing

<p align="center">
  <img src="thainlp/docs/images/logo.png" alt="ChujaiThaiNLP Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/JonusNattapong/ChujaiThainlp/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/JonusNattapong/ChujaiThainlp"></a>
  <a href="https://pypi.org/project/chujaithai/"><img alt="PyPI" src="https://img.shields.io/pypi/v/chujaithai"></a>
  <a href="https://github.com/JonusNattapong/ChujaiThainlp/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/JonusNattapong/ChujaiThainlp"></a>
</p>

**ChujaiThaiNLP** is a cutting-edge Natural Language Processing library designed specifically for Thai language, with advanced multimodal capabilities for seamless integration of text, speech, vision, and more.

## 🌟 Features

### Core NLP Capabilities

- **Tokenization**: State-of-the-art Thai word segmentation
- **Named Entity Recognition**: Identify entities in Thai text
- **Sentiment Analysis**: Analyze sentiment in Thai content
- **Question Answering**: Extract answers from Thai documents and tables
- **Text Generation**: Generate fluent Thai text
- **Translation**: Translate to and from Thai
- **Summarization**: Generate concise summaries of Thai documents

### Advanced Multimodal Processing

- **Audio-Text**: Transcribe and translate speech in Thai and other languages
- **Image-Text**: Extract text from images (OCR), generate captions, analyze image content
- **Visual Question Answering**: Answer questions about images
- **Document Understanding**: Process and query complex documents with layout understanding
- **Video Processing**: Transcribe and summarize video content
- **Modality Conversion**: Transform between different modalities (text-to-image, image-to-text, etc.)

### Vision Processing

- **Image Classification**: Classify images with standard and zero-shot methods
- **Object Detection**: Detect and locate objects in images
- **Image Segmentation**: Semantic, instance, and panoptic segmentation
- **Visual Features**: Extract and utilize visual features from images
- **Image Generation**: Create images from text descriptions

### Speech Processing

- **Text-to-Speech (TTS)**: Generate natural Thai speech
- **Automatic Speech Recognition (ASR)**: Transcribe Thai speech to text
- **Voice Processing**: Voice activity detection, voice conversion, and more

## 📦 Installation

```bash
pip install chujaithai
```

### Optional Dependencies

```bash
# For speech capabilities
pip install chujaithai[speech]

# For vision capabilities
pip install chujaithai[vision]

# For multimodal capabilities
pip install chujaithai[multimodal]

# For all features
pip install chujaithai[all]
```

## 🚀 Quick Start

### Basic NLP Usage

```python
import thainlp

# Word tokenization
tokens = thainlp.word_tokenize("สวัสดีประเทศไทย")
print(tokens)  # ['สวัสดี', 'ประเทศไทย']

# Named Entity Recognition
entities = thainlp.get_entities("นายกรัฐมนตรีเดินทางไปกรุงเทพมหานคร")
print(entities)  # [{'text': 'นายกรัฐมนตรี', 'label': 'PERSON'}, {'text': 'กรุงเทพมหานคร', 'label': 'LOCATION'}]

# Sentiment Analysis
sentiment = thainlp.get_sentiment("อาหารอร่อยมากๆ บริการดีเยี่ยม")
print(sentiment)  # {'label': 'positive', 'score': 0.95}

# Text Generation
generated = thainlp.generate("ประเทศไทยมีสถานที่ท่องเที่ยวที่สวยงาม")
print(generated)  # "ประเทศไทยมีสถานที่ท่องเที่ยวที่สวยงามมากมาย ไม่ว่าจะเป็นทะเล ภูเขา หรือวัดวาอาราม..."
```

### Multimodal Examples

```python
from thainlp.multimodal import transcribe_audio, caption_image, answer_visual_question, process_multimodal

# Transcribe Thai speech
transcript = transcribe_audio("audio.wav", language="th")
print(transcript)

# Generate image caption
caption = caption_image("image.jpg", prompt="A photo of")
print(caption)

# Visual Question Answering
answer = answer_visual_question("image.jpg", "มีอะไรอยู่ในภาพนี้?")
print(answer)

# Complex multimodal pipeline
result = process_multimodal("document.pdf", [
    {"type": "document_process", "name": "doc"},
    {"type": "document_qa", "name": "answer", "params": {"question": "สรุปเอกสารนี้"}}
])
print(result)
```

### Vision Examples

```python
from thainlp.vision import classify_image, detect_objects, generate_image

# Classify image
classification = classify_image("image.jpg")
print(classification)

# Detect objects
objects = detect_objects("image.jpg")
for obj in objects:
    print(f"{obj['label']}: {obj['score']:.2f} at {obj['box']}")

# Generate image from text
image = generate_image("วิวภูเขาในประเทศไทยที่สวยงาม")
image.save("generated_mountain.jpg")
```

### Speech Examples

```python
from thainlp.speech import synthesize, transcribe

# Text to Speech
audio = synthesize("สวัสดีครับ ยินดีต้อนรับสู่ประเทศไทย", voice_id=0)
AudioUtils.save_audio(audio, "welcome.wav")

# Speech to Text
text = transcribe("speech.wav")
print(text)
```

## 📚 Documentation

For comprehensive documentation, visit our [documentation site](https://chujaithai.github.io/docs/).

## 🧩 Architecture

ChujaiThaiNLP is designed with a modular architecture that enables seamless integration of various modalities:
