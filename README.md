# ThaiNLP

ไลบรารีสำหรับการประมวลผลภาษาธรรมชาติไทย (Thai Natural Language Processing)

## คุณสมบัติ

ThaiNLP มีฟีเจอร์ต่างๆ ดังนี้:

### 1. การตัดคำ (Tokenization)
- ใช้อัลกอริธึม Maximum Matching สำหรับการตัดคำภาษาไทย
- มีดิกชันนารีคำศัพท์ภาษาไทยที่ครอบคลุม

### 2. การแท็กส่วนของคำพูด (Part-of-Speech Tagging)
- ใช้โมเดล Hidden Markov Model (HMM) สำหรับการแท็กส่วนของคำพูด
- รองรับการฝึกโมเดลด้วยข้อมูลที่มีการแท็กไว้แล้ว

### 3. การรู้จำหน่วยงานที่มีชื่อเสียง (Named Entity Recognition)
- ใช้กฎและดิกชันนารีสำหรับการรู้จำหน่วยงานที่มีชื่อเสียง
- รองรับการรู้จำชื่อบุคคล สถานที่ องค์กร วันที่ เวลา เงิน URL และแฮชแท็ก

### 4. การวิเคราะห์อารมณ์ (Sentiment Analysis)
- ใช้ดิกชันนารีคำศัพท์ที่มีอารมณ์บวกและลบ
- รองรับการวิเคราะห์อารมณ์ของข้อความภาษาไทย

### 5. การตรวจสอบการสะกดคำ (Spell Checking)
- ใช้ Edit Distance สำหรับการตรวจสอบและแก้ไขการสะกดคำ
- รองรับการตรวจสอบการสะกดคำภาษาไทย

### 6. การสรุปข้อความ (Text Summarization)
- ใช้อัลกอริธึม TextRank สำหรับการสรุปข้อความ
- รองรับการสรุปข้อความภาษาไทย

### 7. เครื่องมือช่วยเหลือ (Utilities)
- การตรวจสอบอักขระภาษาไทย
- การลบวรรณยุกต์และรูปแบบต่างๆ
- การแปลงข้อความให้เป็นมาตรฐาน
- การแปลงภาษาไทยเป็นอักษรโรมัน
- การตรวจสอบภาษา

## การติดตั้ง

```bash
pip install thainlp
```

## การใช้งาน

### การตัดคำ (Tokenization)

```python
from thainlp import tokenize

text = "ผมชอบกินข้าวที่ร้านอาหารไทย"
tokens = tokenize(text)
print(tokens)  # ['ผม', 'ชอบ', 'กิน', 'ข้าว', 'ที่', 'ร้าน', 'อาหาร', 'ไทย']
```

### การแท็กส่วนของคำพูด (Part-of-Speech Tagging)

```python
from thainlp import pos_tag

text = "ผมกินข้าว"
tokens = tokenize(text)
tagged = pos_tag(tokens)
print(tagged)  # [('ผม', 'PRON'), ('กิน', 'VERB'), ('ข้าว', 'NOUN')]
```

### การรู้จำหน่วยงานที่มีชื่อเสียง (Named Entity Recognition)

```python
from thainlp import extract_entities

text = "คุณสมชายอาศัยอยู่ที่จังหวัดเชียงใหม่และทำงานที่บริษัท ปตท. จำกัด"
entities = extract_entities(text)
print(entities)  # [('PERSON', 'คุณสมชาย', 0, 8), ('LOCATION', 'จังหวัดเชียงใหม่', 17, 31), ('ORGANIZATION', 'บริษัท ปตท. จำกัด', 41, 58)]
```

### การวิเคราะห์อารมณ์ (Sentiment Analysis)

```python
from thainlp import analyze_sentiment

text = "วันนี้อากาศดีมากๆ ฉันมีความสุขมาก"
score, label, words = analyze_sentiment(text)
print(f"Score: {score}, Label: {label}")  # Score: 0.75, Label: very_positive
print(f"Positive words: {words['positive']}")  # Positive words: ['ดี', 'มาก', 'สุข', 'มาก']
```

### การตรวจสอบการสะกดคำ (Spell Checking)

```python
from thainlp import check_spelling

text = "ผมชอบกนิข้าว"  # มีคำผิด "กนิ" (ที่ถูกคือ "กิน")
results = check_spelling(text)
print(results)  # [('กนิ', [('กิน', 0.67), ('กนี', 0.5)])]
```

### การสรุปข้อความ (Text Summarization)

```python
from thainlp import summarize_text

text = """
ประเทศไทยมีประชากรประมาณ 70 ล้านคน ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีกรุงเทพมหานครเป็นเมืองหลวง
ประเทศไทยมีภาษาไทยเป็นภาษาราชการ และมีวัฒนธรรมที่เป็นเอกลักษณ์
อาหารไทยเป็นที่นิยมทั่วโลก เช่น ต้มยำกุ้ง ผัดไทย และแกงเขียวหวาน
ประเทศไทยมีสถานที่ท่องเที่ยวที่สวยงามมากมาย เช่น เกาะพีพี เกาะสมุย และเชียงใหม่
"""
summary = summarize_text(text, num_sentences=2)
print(summary)  # ประเทศไทยมีประชากรประมาณ 70 ล้านคน ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีกรุงเทพมหานครเป็นเมืองหลวง ประเทศไทยมีสถานที่ท่องเที่ยวที่สวยงามมากมาย เช่น เกาะพีพี เกาะสมุย และเชียงใหม่
```

### เครื่องมือช่วยเหลือ (Utilities)

```python
from thainlp import (
    is_thai_char,
    is_thai_word,
    remove_tone_marks,
    remove_diacritics,
    normalize_text,
    count_thai_words,
    extract_thai_text,
    thai_to_roman,
    detect_language
)

# ตรวจสอบอักขระภาษาไทย
print(is_thai_char('ก'))  # True
print(is_thai_char('a'))  # False

# ลบวรรณยุกต์
print(remove_tone_marks('สวัสดี'))  # สวสด

# แปลงภาษาไทยเป็นอักษรโรมัน
print(thai_to_roman('สวัสดี'))  # sawasdee

# ตรวจสอบภาษา
print(detect_language('สวัสดีครับ'))  # thai
print(detect_language('Hello world'))  # english
print(detect_language('สวัสดี Hello'))  # mixed
```

## การพัฒนา

สำหรับนักพัฒนาที่ต้องการมีส่วนร่วมในการพัฒนา ThaiNLP สามารถทำได้ดังนี้:

1. Fork โปรเจคไปยัง GitHub repository ของคุณ
2. Clone โปรเจค:
```bash
git clone https://github.com/yourusername/thainlp.git
cd thainlp
```

3. สร้าง virtual environment และติดตั้ง dependencies:
```bash
python -m venv venv
source venv/bin/activate  # สำหรับ Linux/Mac
venv\Scripts\activate     # สำหรับ Windows
pip install -r requirements.txt
```

4. ทำการแก้ไขและทดสอบ
5. ส่ง Pull Request

## ลิขสิทธิ์

ThaiNLP เผยแพร่ภายใต้ MIT License