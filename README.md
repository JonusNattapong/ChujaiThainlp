# Thai Natural Language Processing Library

ไลบรารี่สำหรับประมวลผลภาษาธรรมชาติภาษาไทย (Thai NLP) ที่มีประสิทธิภาพและใช้งานง่าย

## คุณสมบัติ

- การแยกคำ (Word Tokenization)
- การวิเคราะห์ความรู้สึก (Sentiment Analysis)
- การระบุชนิดของคำ (POS Tagging)
- การระบุชื่อเฉพาะ (Named Entity Recognition)
- การแปลงเสียงภาษาไทยเป็นอักษรโรมัน (Thai Romanization)

## การติดตั้ง

```bash
pip install thainlp
```

## วิธีการใช้งาน

### 1. การแยกคำ (Word Tokenization)

```python
from thainlp import word_tokenize

text = "สวัสดีครับ ยินดีต้อนรับสู่ไลบรารี่ ThaiNLP"
words = word_tokenize(text)
print(words)
# ผลลัพธ์: ['สวัสดี', 'ครับ', 'ยินดี', 'ต้อนรับ', 'สู่', 'ไลบรารี่', 'ThaiNLP']
```

### 2. การวิเคราะห์ความรู้สึก (Sentiment Analysis)

```python
from thainlp import analyze_sentiment

text = "วันนี้อากาศดีมากๆ ทำให้รู้สึกสดชื่นและมีความสุข"
score, label, words = analyze_sentiment(text)
print(f"คะแนนความรู้สึก: {score}")
print(f"ประเภท: {label}")
print(f"คำที่พบ: {words}")
# ผลลัพธ์:
# คะแนนความรู้สึก: 0.8
# ประเภท: very_positive
# คำที่พบ: {'positive': ['ดี', 'สดชื่น', 'สุข'], 'negative': [], 'neutral': []}
```

### 3. การระบุชนิดของคำ (POS Tagging)

```python
from thainlp import pos_tag

text = "ผมชอบกินข้าวผัดมาก"
pos = pos_tag(text)
print(pos)
# ผลลัพธ์: [('ผม', 'PRON'), ('ชอบ', 'VERB'), ('กิน', 'VERB'), ('ข้าว', 'NOUN'), ('ผัด', 'VERB'), ('มาก', 'ADV')]
```

### 4. การระบุชื่อเฉพาะ (Named Entity Recognition)

```python
from thainlp import find_entities

text = "สมชาย ไปเที่ยวกรุงเทพมหานคร กับบริษัท ABC จำกัด เมื่อวันที่ 1/1/2024"
entities = find_entities(text)
print(entities)
# ผลลัพธ์: [
#   ('สมชาย', 'PERSON', 0, 2, 0.9),
#   ('กรุงเทพมหานคร', 'LOCATION', 8, 15, 1.0),
#   ('บริษัท ABC จำกัด', 'ORGANIZATION', 18, 27, 1.0),
#   ('1/1/2024', 'DATE', 35, 44, 1.0)
# ]
```

### 5. การแปลงเสียงภาษาไทยเป็นอักษรโรมัน (Thai Romanization)

```python
from thainlp import thai_to_roman

text = "สวัสดี"
roman = thai_to_roman(text)
print(roman)
# ผลลัพธ์: 'sawasdee'
```

## การปรับแต่ง

### 1. เพิ่มคำศัพท์ใหม่

```python
from thainlp import add_word

# เพิ่มคำศัพท์ใหม่
add_word("คำใหม่")
```

### 2. เพิ่มคำหยุด (Stopwords)

```python
from thainlp import add_stopword

# เพิ่มคำหยุดใหม่
add_stopword("คำหยุด")
```

### 3. ปรับแต่งพจนานุกรม

```python
from thainlp import load_custom_dictionary

# โหลดพจนานุกรมจากไฟล์
dictionary = load_custom_dictionary("path/to/dictionary.txt")
```

## การมีส่วนร่วม

เรายินดีรับการมีส่วนร่วมจากทุกท่าน! กรุณาตรวจสอบ [CONTRIBUTING.md](CONTRIBUTING.md) สำหรับรายละเอียดเพิ่มเติม

## การทดสอบ

```bash
python -m pytest tests/
```

## การอนุญาต

MIT License - ดูรายละเอียดเพิ่มเติมใน [LICENSE](LICENSE)

## การติดต่อ

- GitHub Issues: [รายงานปัญหา](https://github.com/JonusNattapong)
- Email: zombitx64@gmail.com