# ChujaiThainlp: Advanced Thai Natural Language Processing

ChujaiThainlp is a comprehensive Thai Natural Language Processing library that provides state-of-the-art models and tools for various NLP tasks. The library supports both Thai and English languages with a focus on Thai-specific processing.

## Key Features

- **Text Classification**
  - Token-level classification
  - Named Entity Recognition
  - Part-of-speech tagging
  - Support for fine-tuning on domain data

- **Question Answering**
  - Text-based QA
  - Table QA
  - Multiple answer generation
  - Context-aware processing

- **Translation**
  - Thai ↔ English translation
  - Batch processing
  - Quality scoring
  - Custom vocabulary support

- **Text Generation**
  - Contextual text generation
  - Streaming generation
  - Temperature and sampling control
  - Prompt templates

- **Fill-Mask**
  - Masked token prediction
  - Multiple candidate generation
  - Confidence scoring
  - Support for multiple masks

- **Summarization**
  - Extractive and abstractive summarization
  - Length control
  - ROUGE metrics
  - Batch processing

- **Sentence Similarity**
  - Cross-encoder scoring
  - Bi-encoder embeddings
  - Efficient similarity search
  - Text ranking

## Installation

```bash
pip install chujaithai-nlp
```

## Quick Start

```python
from thainlp.pipelines import ThaiNLPPipeline

# Initialize pipeline
pipeline = ThaiNLPPipeline()

# Basic text analysis
text = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"
result = pipeline.analyze(text, tasks=['tokens', 'entities', 'translation'])

# Question answering
context = "ประเทศไทยมีประชากรประมาณ 70 ล้านคน"
question = "ประเทศไทยมีประชากรเท่าไร"
answer = pipeline.answer_question(question, context)

# Translation
eng_text = pipeline.translate(text, source_lang='th', target_lang='en')

# Text generation
prompt = "เทคโนโลยีปัญญาประดิษฐ์"
generated = pipeline.generate_text(prompt, max_length=100)

# Summarization
long_text = "..." # Your long text here
summary = pipeline.summarize(long_text, ratio=0.3)
```

## Advanced Usage

### Custom Pipeline Components

```python
# Initialize with specific components
pipeline = ThaiNLPPipeline(
    components=['translation', 'qa'],
    device='cuda',
    batch_size=32
)
```

### Batch Processing

```python
# Batch translation
texts = ["สวัสดี", "ขอบคุณ", "ลาก่อน"]
translations = pipeline.translate(texts, source_lang='th', target_lang='en')

# Batch summarization
documents = ["doc1", "doc2", "doc3"]
summaries = pipeline.summarize(documents, ratio=0.3)
```

### Fine-tuning Models

```python
# Get specific component
classifier = pipeline.get_component('classification')

# Fine-tune on custom data
train_data = [
    {"text": "text1", "labels": ["label1"]},
    {"text": "text2", "labels": ["label2"]}
]
classifier.fine_tune(train_data, epochs=3)
```

## Model Hub Integration

```python
from thainlp.model_hub import ModelHub

# Download pre-trained model
hub = ModelHub()
model = hub.load_model('thai-qa-large')

# Use in pipeline
pipeline = ThaiNLPPipeline(qa_model=model)
```

## Components

### Text Classification

- Token-level classification with confidence scores
- Support for custom label sets
- Efficient batch processing
- Model fine-tuning capabilities

### Question Answering

- Support for both text and table inputs
- Multiple answer generation
- Confidence scoring
- Cross-lingual capabilities

### Translation

- Neural machine translation
- Support for multiple language pairs
- Quality metrics
- Custom vocabulary integration

### Text Generation

- Transformer-based generation
- Streaming support
- Advanced sampling controls
- Prompt engineering

### Fill-Mask

- Support for multiple masks
- Confidence scoring
- Batch processing
- Multiple candidates per mask

### Summarization

- Both extractive and abstractive methods
- Length control
- ROUGE metrics
- Batch processing

### Sentence Similarity

- Multiple similarity metrics
- Efficient similarity search
- Cross-lingual support
- Text ranking

## Performance Considerations

- GPU acceleration for improved performance
- Batch processing for efficient handling of multiple inputs
- Caching mechanisms for repeated operations
- Progress monitoring for long-running tasks

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
