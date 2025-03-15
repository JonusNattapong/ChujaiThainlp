# ThaiNLP

ไลบรารีสำหรับการประมวลผลภาษาธรรมชาติไทย (Thai Natural Language Processing)

## คุณสมบัติ

ThaiNLP มีฟีเจอร์ต่างๆ ดังนี้:

### 1. การตัดคำ (Tokenization)
- ใช้อัลกอริธึม Maximum Matching สำหรับการตัดคำภาษาไทย
- รองรับการใช้อัลกอริธึมจาก PyThaiNLP เช่น Longest Matching, Newmm, Attacut
- มีดิกชันนารีคำศัพท์ภาษาไทยที่ครอบคลุม
- รองรับการตัดคำระดับประโยค (Sentence Tokenization)
- รองรับการตัดคำย่อย (Subword Tokenization)

### 2. การแท็กส่วนของคำพูด (Part-of-Speech Tagging)
- ใช้โมเดล Hidden Markov Model (HMM) สำหรับการแท็กส่วนของคำพูด
- รองรับการใช้ Perceptron Tagger จาก PyThaiNLP
- รองรับการใช้ Artagger สำหรับความแม่นยำสูง
- รองรับการฝึกโมเดลด้วยข้อมูลที่มีการแท็กไว้แล้ว

### 3. การรู้จำหน่วยงานที่มีชื่อเสียง (Named Entity Recognition)
- ใช้กฎและดิกชันนารีสำหรับการรู้จำหน่วยงานที่มีชื่อเสียง
- รองรับการรู้จำชื่อบุคคล สถานที่ องค์กร วันที่ เวลา เงิน URL และแฮชแท็ก
- รองรับการใช้โมเดล CRF และ BiLSTM-CRF จาก PyThaiNLP

### 4. การวิเคราะห์อารมณ์ (Sentiment Analysis)
- ใช้ดิกชันนารีคำศัพท์ที่มีอารมณ์บวกและลบ
- รองรับการวิเคราะห์อารมณ์ของข้อความภาษาไทย
- รองรับการใช้โมเดลที่ฝึกด้วย Thai Sentiment Corpus

### 5. การตรวจสอบการสะกดคำ (Spell Checking)
- ใช้ Edit Distance สำหรับการตรวจสอบและแก้ไขการสะกดคำ
- รองรับการตรวจสอบการสะกดคำภาษาไทยด้วย PyThaiNLP
- รองรับการแนะนำคำที่ถูกต้องตามบริบท

### 6. การสรุปข้อความ (Text Summarization)
- ใช้อัลกอริธึม TextRank สำหรับการสรุปข้อความ
- รองรับการสรุปข้อความภาษาไทยแบบสกัดประโยค (Extractive)
- รองรับการสรุปข้อความแบบสร้างใหม่ (Abstractive) ด้วยโมเดลขั้นสูง

### 7. เครื่องมือช่วยเหลือ (Utilities)
- การตรวจสอบอักขระภาษาไทย
- การลบวรรณยุกต์และรูปแบบต่างๆ
- การแปลงข้อความให้เป็นมาตรฐาน
- การแปลงภาษาไทยเป็นอักษรโรมัน (Romanization)
- การจัดรูปแบบวันที่และเวลาในภาษาไทย
- การตรวจสอบภาษา

### 8. การจำแนกข้อความ (Text Classification)
- การจำแนกข้อความตามหมวดหมู่ (เช่น อารมณ์, หัวข้อ, สแปม)
- การจำแนกแบบ Zero-shot โดยไม่ต้องฝึกโมเดลล่วงหน้า
- รองรับการใช้โมเดลที่ฝึกด้วยชุดข้อมูลภาษาไทย

### 9. การจำแนกโทเค็น (Token Classification)
- การแท็กส่วนของคำพูดแบบละเอียด
- การรู้จำหน่วยงานที่มีชื่อเสียงแบบละเอียด
- รองรับการใช้โมเดลที่ฝึกด้วยชุดข้อมูลภาษาไทย

### 10. การตอบคำถาม (Question Answering)
- การตอบคำถามจากบริบทที่กำหนด
- การตอบคำถามจากข้อมูลตาราง
- รองรับการใช้โมเดลที่ฝึกด้วยชุดข้อมูลภาษาไทย

### 11. การแปลภาษา (Translation)
- การแปลระหว่างภาษาไทยและภาษาอังกฤษ
- การตรวจสอบภาษาอัตโนมัติ
- รองรับการใช้โมเดลแปลภาษาจาก PyThaiNLP

### 12. การสกัดคุณลักษณะ (Feature Extraction)
- การสกัดคุณลักษณะพื้นฐานจากข้อความภาษาไทย
- การสกัดคุณลักษณะขั้นสูงสำหรับการวิเคราะห์
- รองรับการใช้ Word Embeddings จาก PyThaiNLP

### 13. การเติมคำในช่องว่าง (Fill-Mask)
- รองรับการเติมคำในช่องว่างด้วยโมเดลภาษาไทย
- รองรับการใช้โมเดล WangchanBERTa และ Gemma
- สามารถเติมหลายช่องว่างพร้อมกันได้
- มีฟังก์ชันวิเคราะห์บริบทของคำ

### 14. การสร้างข้อความ (Text Generation)
- การสร้างข้อความด้วยเทมเพลต
- การสร้างข้อความด้วยรูปแบบ n-gram
- การสร้างข้อความตามรูปแบบไวยากรณ์
- รองรับการใช้โมเดลภาษาขั้นสูงสำหรับภาษาไทย

### 15. การวัดความคล้ายคลึงของประโยค (Sentence Similarity)
- การวัดความคล้ายคลึงด้วย Cosine Similarity
- รองรับการใช้ Sentence Transformers
- สามารถค้นหาประโยคที่คล้ายคลึงกันจากชุดข้อมูลได้
- มีฟังก์ชันจัดกลุ่มประโยคตามความคล้ายคลึง

### 16. การประมวลผลคำพูดภาษาไทย (Thai Speech Processing)
- รองรับการรู้จำเสียงพูด (Speech-to-Text)
- รองรับการสังเคราะห์เสียงพูด (Text-to-Speech)
- มีฟังก์ชันระบุภาษาจากเสียงพูด
- สามารถแบ่งส่วนเสียงตามผู้พูด (Speaker Diarization)
- ปรับปรุงคุณภาพเสียงได้ (Audio Enhancement)
- ตรวจจับอารมณ์จากเสียงพูดได้
- มีฟังก์ชันตรวจจับช่วงที่มีการพูด (Voice Activity Detection)
- คำนวณสถิติของเสียงพูดได้ (ระดับเสียง, ความเร็ว, ฯลฯ)
- รองรับการประมวลผลไฟล์เสียงจำนวนมากพร้อมกัน
- ดูตัวอย่างการใช้งานได้ที่ [examples/speech_example.py](thainlp/examples/speech_example.py)

### 17. การสรุปบทสนทนา (Conversation Summarization)
- สรุปบทสนทนาภาษาไทยแบบอัตโนมัติ
- สกัดประโยคสำคัญและหัวข้อหลัก
- วิเคราะห์สถิติผู้พูด (จำนวนข้อความ, จำนวนคำถาม)
- รองรับการสรุปแยกตามหัวข้อ
- ดูตัวอย่างการใช้งานได้ที่ [examples/conversation_summary_example.py](thainlp/examples/conversation_summary_example.py)

### 18. การเพิ่มข้อมูลภาษาไทย (Thai Data Augmentation)
- มีเทคนิคการเพิ่มข้อมูลที่หลากหลาย เช่น การแทนที่คำพ้องความหมาย, การลบคำ, การสลับคำ, การแทรกคำ, และ back-translation
- รองรับการเพิ่มข้อมูลโดยใช้แม่แบบ (template-based)
- มีฟังก์ชัน EDA (Easy Data Augmentation)
- ดูตัวอย่างการใช้งานได้ที่ [examples/data_augmentation_example.py](thainlp/examples/data_augmentation_example.py)

### 19. การประมวลผลภาษาไทยแบบปกป้องความเป็นส่วนตัว (Privacy-Preserving Thai NLP)
- มีฟังก์ชันสำหรับปกปิดข้อมูลส่วนบุคคล (PII) เช่น ชื่อ, ที่อยู่, เบอร์โทรศัพท์
- ใช้เทคนิค Differential Privacy เพื่อเพิ่ม Noise ในข้อมูล
- รองรับการทำ Hashing สำหรับข้อมูลที่อ่อนไหว
- ดูตัวอย่างการใช้งานได้ที่ [examples/privacy_example.py](thainlp/examples/privacy_example.py)

### 20. ตัวอย่างการใช้งานขั้นสูง
- ตัวอย่างการใช้งานขั้นสูงทั้งหมด: [examples/advanced_usage.py](thainlp/examples/advanced_usage.py)
- ตัวอย่างการประมวลผลเสียง: [examples/speech_example.py](thainlp/examples/speech_example.py)
- ตัวอย่างการเพิ่มข้อมูล: [examples/data_augmentation_example.py](thainlp/examples/data_augmentation_example.py)
- ตัวอย่างการแก้ไขคำผิด: [examples/spell_correction_example.py](thainlp/examples/spell_correction_example.py)
- ตัวอย่างการสรุปบทสนทนา: [examples/conversation_summary_example.py](thainlp/examples/conversation_summary_example.py)

## การติดตั้ง

### การติดตั้งพื้นฐาน
```bash
pip install thainlp
```

### การติดตั้งพร้อมฟีเจอร์ขั้นสูง (รวม PyThaiNLP)
```bash
pip install thainlp[advanced]
```

### การติดตั้งแบบเต็ม (รวมทุกฟีเจอร์)
```bash
pip install thainlp[full]
```

### การติดตั้งสำหรับนักพัฒนา
```bash
pip install thainlp[dev]
```

## การใช้งาน

### การเติมคำในช่องว่าง (Fill-Mask)

```python
from thainlp.generation import ThaiFillMask

# ใช้ WangchanBERTa (default)
fill_mask = ThaiFillMask()
text = "ฉันชอบ<mask>แมว"
predictions = fill_mask.fill_mask(text)
print(predictions)

# ใช้ Gemma
fill_mask_gemma = ThaiFillMask(model_name_or_path="google/gemma-2b-it")
text = "ฉันชอบ<mask>แมว"
predictions = fill_mask_gemma.fill_mask(text)
print(predictions)

# เติมหลายช่องว่าง
text = "ฉัน<mask><mask>แมว"
predictions = fill_mask.fill_multiple_masks(text)
print(predictions)
```

### การวัดความคล้ายคลึงของประโยค (Sentence Similarity)

```python
from thainlp.similarity import ThaiSentenceSimilarity

# ใช้ Sentence Transformers (default)
similarity = ThaiSentenceSimilarity()
text1 = "ฉันชอบกินข้าวผัด"
text2 = "ฉันชอบทานข้าวผัด"
score = similarity.compute_similarity(text1, text2)
print(score)

# ใช้ WangchanBERTa
similarity_wangchan = ThaiSentenceSimilarity(embedding_type="transformer")
text1 = "ฉันชอบกินข้าวผัด"
text2 = "ฉันชอบทานข้าวผัด"
score = similarity_wangchan.compute_similarity(text1, text2)
print(score)

# ค้นหาประโยคที่คล้ายคลึง
corpus = ["ฉันชอบกินข้าว", "ฉันชอบแมว", "อากาศวันนี้ดี"]
query = "ฉันชอบสุนัข"
results = similarity.find_most_similar(query, corpus)
print(results)
```

### การตัดคำ (Tokenization)

#### การตัดคำพื้นฐาน
```python
from thainlp import tokenize

text = "ผมชอบกินข้าวที่ร้านอาหารไทย"
tokens = tokenize(text)
print(tokens)  # ['ผม', 'ชอบ', 'กิน', 'ข้าว', 'ที่', 'ร้าน', 'อาหาร', 'ไทย']
```

#### การตัดคำด้วย PyThaiNLP
```python
from thainlp.tokenization import tokenize_with_pythainlp

text = "ผมชอบกินข้าวที่ร้านอาหารไทย"
tokens = tokenize_with_pythainlp(text, engine="newmm")
print(tokens)  # ['ผม', 'ชอบ', 'กิน', 'ข้าว', 'ที่', 'ร้าน', 'อาหาร', 'ไทย']
```

#### การตัดคำระดับประโยค
```python
from thainlp.tokenization import sentence_tokenize

text = "สวัสดีครับ ผมชื่อสมชาย ผมชอบกินข้าว คุณล่ะชอบกินอะไร"
sentences = sentence_tokenize(text)
print(sentences)  # ['สวัสดีครับ', 'ผมชื่อสมชาย', 'ผมชอบกินข้าว', 'คุณล่ะชอบกินอะไร']
```

### การแท็กส่วนของคำพูด (Part-of-Speech Tagging)

#### การแท็กพื้นฐาน
```python
from thainlp import tokenize, pos_tag

text = "ผมกินข้าว"
tokens = tokenize(text)
tagged = pos_tag(tokens)
print(tagged)  # [('ผม', 'PRON'), ('กิน', 'VERB'), ('ข้าว', 'NOUN')]
```

#### การแท็กด้วย PyThaiNLP
```python
from thainlp.pos_tagging import pos_tag_pythainlp

text = "ผมกินข้าว"
tagged = pos_tag_pythainlp(text, engine="perceptron")
print(tagged)  # [('ผม', 'PPRS'), ('กิน', 'VACT'), ('ข้าว', 'NCMN')]
```

### การรู้จำหน่วยงานที่มีชื่อเสียง (Named Entity Recognition)

#### การรู้จำพื้นฐาน
```python
from thainlp import extract_entities

text = "คุณสมชายอาศัยอยู่ที่จังหวัดเชียงใหม่และทำงานที่บริษัท ปตท. จำกัด"
entities = extract_entities(text)
print(entities)  # [('PERSON', 'คุณสมชาย', 0, 8), ('LOCATION', 'จังหวัดเชียงใหม่', 17, 31), ('ORGANIZATION', 'บริษัท ปตท. จำกัด', 41, 58)]
```

#### การรู้จำด้วย PyThaiNLP
```python
from thainlp.ner import extract_entities_pythainlp

text = "คุณสมชายอาศัยอยู่ที่จังหวัดเชียงใหม่และทำงานที่บริษัท ปตท. จำกัด"
entities = extract_entities_pythainlp(text)
print(entities)  # [('B-PER', 'คุณ'), ('I-PER', 'สมชาย'), ('O', 'อาศัย'), ...]
```

### การวิเคราะห์อารมณ์ (Sentiment Analysis)

#### การวิเคราะห์พื้นฐาน
```python
from thainlp import analyze_sentiment

text = "วันนี้อากาศดีมากๆ ฉันมีความสุขมาก"
score, label, words = analyze_sentiment(text)
print(f"Score: {score}, Label: {label}")  # Score: 0.75, Label: very_positive
print(f"Positive words: {words['positive']}")  # Positive words: ['ดี', 'มาก', 'สุข', 'มาก']
```

#### การวิเคราะห์ด้วย PyThaiNLP
```python
from thainlp.sentiment import analyze_sentiment_pythainlp

text = "วันนี้อากาศดีมากๆ ฉันมีความสุขมาก"
result = analyze_sentiment_pythainlp(text)
print(result)  # {'pos': 0.8, 'neg': 0.1, 'neu': 0.1}
```

### การตรวจสอบการสะกดคำ (Spell Checking)

#### การตรวจสอบพื้นฐาน
```python
from thainlp import check_spelling

text = "ผมชอบกนิข้าว"  # มีคำผิด "กนิ" (ที่ถูกคือ "กิน")
results = check_spelling(text)
print(results)  # [('กนิ', [('กิน', 0.67), ('กนี', 0.5)])]
```

#### การตรวจสอบด้วย PyThaiNLP
```python
from thainlp.spellcheck import check_spelling_pythainlp

text = "ผมชอบกนิข้าว"
results = check_spelling_pythainlp(text)
print(results)  # {'กนิ': 'กิน'}
```

### การสรุปข้อความ (Text Summarization)

#### การสรุปพื้นฐาน
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

#### การสรุปด้วย PyThaiNLP
```python
from thainlp.summarization import summarize_text_pythainlp

text = """
ประเทศไทยมีประชากรประมาณ 70 ล้านคน ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีกรุงเทพมหานครเป็นเมืองหลวง
ประเทศไทยมีภาษาไทยเป็นภาษาราชการ และมีวัฒนธรรมที่เป็นเอกลักษณ์
อาหารไทยเป็นที่นิยมทั่วโลก เช่น ต้มยำกุ้ง ผัดไทย และแกงเขียวหวาน
ประเทศไทยมีสถานที่ท่องเที่ยวที่สวยงามมากมาย เช่น เกาะพีพี เกาะสมุย และเชียงใหม่
"""
summary = summarize_text_pythainlp(text)
print(summary)  # ประเทศไทยมีประชากรประมาณ 70 ล้านคน ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้...
```

### การจำแนกข้อความ (Text Classification)

```python
from thainlp.classification import classify_text, zero_shot_classification

# การจำแนกข้อความตามหมวดหมู่
text = "ร้านนี้อาหารอร่อยมาก บริการดีเยี่ยม"
result = classify_text(text, category="sentiment")
print(result)  # {'label': 'positive', 'score': 0.85}

# การจำแนกแบบ Zero-shot
text = "ผมต้องการสั่งอาหารกลับบ้าน"
labels = ["คำถาม", "คำสั่ง", "ข้อเสนอแนะ"]
result = zero_shot_classification(text, labels)
print(result)  # {'คำสั่ง': 0.75, 'คำถาม': 0.15, 'ข้อเสนอแนะ': 0.1}
```

### การจำแนกโทเค็น (Token Classification)

```python
from thainlp.classification import ThaiTokenClassifier

# การแท็กส่วนของคำพูดแบบละเอียด
text = "ผมกำลังเดินไปที่ตลาด"
classifier = ThaiTokenClassifier()
tokens = text.split()
result = classifier.classify_tokens(tokens, task="pos")
print(result)  # [('ผม', 'PRON'), ('กำลัง', 'AUX'), ('เดิน', 'VERB'), ('ไป', 'VERB'), ('ที่', 'ADP'), ('ตลาด', 'NOUN')]

# การรู้จำหน่วยงานที่มีชื่อเสียง
text = "คุณสมชายทำงานที่บริษัทไทยรุ่งเรือง"
tokens = text.split()
entities = classifier.find_entities(tokens)
print(entities)  # [{'entity': 'PERSON', 'text': 'คุณสมชาย', 'start': 0, 'end': 1, 'score': 0.9}, {'entity': 'ORG', 'text': 'บริษัทไทยรุ่งเรือง', 'start': 3, 'end': 4, 'score': 0.85}]
```

### การตอบคำถาม (Question Answering)

```python
from thainlp.question_answering import answer_question, answer_from_table

# การตอบคำถามจากบริบท
context = "ประเทศไทยมีประชากรประมาณ 70 ล้านคน ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีกรุงเทพมหานครเป็นเมืองหลวง"
question = "ประเทศไทยมีประชากรกี่คน"
answer = answer_question(question, context)
print(answer)  # {'answer': 'ประมาณ 70 ล้านคน', 'start': 16, 'end': 31, 'score': 0.95}

# การตอบคำถามจากตาราง
table = [
    ["ชื่อ", "อายุ", "อาชีพ"],
    ["สมชาย", "30", "วิศวกร"],
    ["สมหญิง", "28", "หมอ"],
    ["สมศักดิ์", "35", "ครู"]
]
question = "สมหญิงอายุเท่าไร"
answer = answer_from_table(question, table)
print(answer)  # {'answer': '28', 'row': 2, 'col': 1, 'score': 0.9}
```

### การแปลภาษา (Translation)

```python
from thainlp.translation import translate_text, detect_language

# การแปลภาษา
text_th = "สวัสดีครับ ผมชื่อสมชาย"
translated = translate_text(text_th, source="th", target="en")
print(translated)  # "Hello, my name is Somchai"

text_en = "Thailand is a beautiful country"
translated = translate_text(text_en, source="en", target="th")
print(translated)  # "ประเทศไทยเป็นประเทศที่สวยงาม"

# การตรวจสอบภาษา
text = "สวัสดีครับ Hello"
detected = detect_language(text)
print(detected)  # {'th': 0.6, 'en': 0.4}
```

### การสกัดคุณลักษณะ (Feature Extraction)

```python
from thainlp.feature_extraction import extract_features, word_embedding

# การสกัดคุณลักษณะพื้นฐาน
text = "สวัสดีครับ ผมชื่อสมชาย ผมอายุ 30 ปี"
features = extract_features(text)
print(features)  # {'word_count': 7, 'char_count': 28, 'thai_char_count': 25, 'digit_count': 2, ...}

# การใช้ Word Embedding
word = "ความสุข"
embedding = word_embedding(word, model="thai2vec")
print(embedding.shape)  # (300,)
```

### การสร้างข้อความ (Text Generation)

```python
from thainlp.generation import generate_text, complete_text

# การสร้างข้อความจากเทมเพลต
template = "สวัสดี{greeting} ฉันชื่อ{name} ฉันเป็น{occupation}"
params = {"greeting": "ครับ", "name": "สมชาย", "occupation": "วิศวกร"}
text = generate_text(template, params)
print(text)  # "สวัสดีครับ ฉันชื่อสมชาย ฉันเป็นวิศวกร"

# การต่อข้อความ
prompt = "ประเทศไทยมี"
completed = complete_text(prompt, max_length=50)
print(completed)  # "ประเทศไทยมีประชากรประมาณ 70 ล้านคน ตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้..."
```

### การวัดความคล้ายคลึงของข้อความ (Text Similarity)

```python
from thainlp.similarity import text_similarity, find_similar_texts

# การวัดความคล้ายคลึง
text1 = "ผมชอบกินข้าวผัด"
text2 = "ฉันชอบทานข้าวผัด"
similarity = text_similarity(text1, text2, method="cosine")
print(similarity)  # 0.85

# การค้นหาข้อความที่คล้ายคลึง
query = "อาหารไทยอร่อย"
texts = ["อาหารไทยรสชาติดี", "ประเทศไทยสวยงาม", "อาหารญี่ปุ่นอร่อย"]
results = find_similar_texts(query, texts)
print(results)  # [{'text': 'อาหารไทยรสชาติดี', 'score': 0.8}, {'text': 'อาหารญี่ปุ่นอร่อย', 'score': 0.6}, {'text': 'ประเทศไทยสวยงาม', 'score': 0.4}]
```

## การใช้งานขั้นสูงกับ PyThaiNLP

ThaiNLP รองรับการใช้งานร่วมกับ PyThaiNLP เพื่อเพิ่มความสามารถในการประมวลผลภาษาไทย

### การติดตั้ง PyThaiNLP

```bash
pip install thainlp[advanced]
```

### ตัวอย่างการใช้งาน PyThaiNLP ผ่าน ThaiNLP

```python
from thainlp.utils import thai_utils

# การแปลงเลขไทยเป็นอารบิก
thai_number = "๑๒๓๔๕"
arabic_number = thai_utils.thai_digit_to_arabic(thai_number)
print(arabic_number)  # "12345"

# การแปลงเลขอารบิกเป็นไทย
arabic_number = "12345"
thai_number = thai_utils.arabic_digit_to_thai(arabic_number)
print(thai_number)  # "๑๒๓๔๕"

# การแปลงวันที่เป็นภาษาไทย
from datetime import datetime
date = datetime.now()
thai_date = thai_utils.thai_strftime(date, "%A %d %B %Y")
print(thai_date)  # "วันอังคาร 15 มิถุนายน 2566"
```

## การพัฒนา

สำหรับนักพัฒนาที่ต้องการมีส่วนร่วมในการพัฒนา ThaiNLP สามารถดูรายละเอียดเพิ่มเติมได้ที่ [CONTRIBUTING.md](CONTRIBUTING.md)

## ลิขสิทธิ์

ThaiNLP เผยแพร่ภายใต้ลิขสิทธิ์ MIT ดูรายละเอียดเพิ่มเติมได้ที่ [LICENSE](LICENSE)
