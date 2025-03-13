# ThaiNLP Basic

A Python library for Thai Natural Language Processing tasks.

## Installation

```bash
pip install nlpbasiczombitx64
```

## Features

- Thai word tokenization
- Thai character/syllable processing
- Part-of-speech tagging
- Named entity recognition
- Sentiment analysis
- Text summarization
- Spell checking
- Text classification
- Language utilities

## Usage

```python
import thainlp

# Word tokenization (using default PyThaiNLP engine)
tokens = thainlp.word_tokenize("การเดินทางไปท่องเที่ยวที่จังหวัดเชียงใหม่ในฤดูหนาวเป็นประสบการณ์ที่น่าจดจำ")
print(tokens)

# Word tokenization (specifying PyThaiNLP engine)
tokens = thainlp.word_tokenize("นักวิจัยกำลังศึกษาปรากฏการณ์ทางธรรมชาติที่ซับซ้อน", engine="pythainlp:longest")
print(tokens)

# Part-of-speech tagging (using default PyThaiNLP engine and UD tagset)
pos_tags = thainlp.pos_tag("การประชุมวิชาการนานาชาติจะจัดขึ้นในเดือนหน้า")
print(pos_tags)

# Part-of-speech tagging (specifying PyThaiNLP engine and tagset, and converting to ORCHID)
pos_tags = thainlp.pos_tag("เศรษฐกิจของประเทศไทยกำลังฟื้นตัวอย่างช้าๆ", engine="pythainlp", tagset="orchid", return_tagset="ud")
print(pos_tags)

# Text summarization
text = """
การพัฒนาเทคโนโลยีปัญญาประดิษฐ์ (AI) กำลังเปลี่ยนแปลงโลกในหลายด้าน
AI ถูกนำมาใช้ในอุตสาหกรรมต่างๆ เช่น การแพทย์ การเงิน การศึกษา และการขนส่ง
อย่างไรก็ตาม การพัฒนา AI ก็มีความท้าทายหลายประการ เช่น ความเป็นส่วนตัว ความปลอดภัย และจริยธรรม
"""
summary = thainlp.summarize(text, n_sentences=2)
print(summary)

# Spell checking
misspelled = thainlp.spellcheck("ฉันไปเทียวทะเลขาว")
print(misspelled)

# Text classification
label = thainlp.classify("ภาพยนตร์เรื่องนี้สนุกมาก")  # Example: Sentiment analysis
print(label)
```

## Custom Dictionaries and Stopwords

You can use custom dictionaries and stopwords by providing file paths:

```python
from thainlp.resources import combine_dictionaries

words, stopwords = combine_dictionaries(custom_dict_path="path/to/dict.txt",
                                        custom_stopwords_path="path/to/stopwords.txt")

# You can then use these with the tokenization and spell checking functions.
# (Implementation for integrating custom dictionaries into all functions is TODO)
```

## Available PyThaiNLP Engines

For tokenization and POS tagging, you can specify different PyThaiNLP engines.
Refer to the PyThaiNLP documentation for the most up-to-date list. Some common
engines include:

- `newmm` (default): New Maximum Matching
- `longest`: Longest Matching
- `attacut`: Dictionary-based with Attention

## License

MIT