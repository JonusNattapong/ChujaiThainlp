# Changelog

## [0.4.0] - 2025-04-02

### เพิ่ม

- **Model Hub:** ระบบจัดการโมเดลแบบรวมศูนย์สำหรับโหลดและใช้งานโมเดลต่างๆ
- **Transformer NER:** รองรับการใช้งานโมเดล WangchanBERTa สำหรับการรู้จำหน่วยงานที่มีชื่อเสียง
- **Transformer Sentiment:** รองรับการใช้งานโมเดล WangchanBERTa สำหรับการวิเคราะห์อารมณ์
- เพิ่มตัวอย่างการใช้งาน transformer model (`transformer_ner_example.py`, `transformer_sentiment_example.py`)
- เพิ่มเทสต์สำหรับโมดูล transformer และ Model Hub

### ปรับปรุง

- **Pipeline:** แก้ไขชื่อ class และการใช้งานให้สอดคล้องกัน (`MaximumMatchingTokenizer`, `HMMTagger`, `ThaiNER`)
- **เอกสาร:** อัปเดตวิธีการใช้งานโมเดล transformer และ Model Hub ใน README
- **โค้ด:** ปรับปรุงความชัดเจนของคอมเมนต์และเพิ่ม type hints

## [0.3.0] - 2025-04-02

### เพิ่ม

- อินเตอร์เฟซใหม่ `ThaiTextProcessor` และ `ThaiValidator` ใน `thainlp.utils.thai_text` เพื่อการใช้งานที่ง่ายขึ้นด้วย method chaining และฟังก์ชันสำเร็จรูป
- ฟังก์ชัน `is_valid_thai_word` สำหรับตรวจสอบความถูกต้องของคำไทยตามโครงสร้างพยางค์
- ฟังก์ชัน `get_thai_syllable_pattern` สำหรับสร้าง Regex pattern ของพยางค์ไทย
- ตัวอย่างการใช้งานใหม่ (`thai_text_example.py`) และเอกสารประกอบ (`thai_utils.md`)
- เทสต์เพิ่มเติมสำหรับ `thai_utils` และ `thai_text` เพื่อความครอบคลุม

### ปรับปรุง

- **ประสิทธิภาพ:** แก้ไขการใช้ Lock ใน `thainlp.core.base` ให้เป็น `asyncio.Lock` เพื่อป้องกันการบล็อก event loop และปรับปรุงเทสต์ที่เกี่ยวข้องให้รองรับ async/await
- **`thai_utils.normalize_text`:** เพิ่มการแปลงเลขไทยเป็นอารบิก และการจัดการ invisible characters
- **`thai_utils.remove_diacritics`:** เพิ่มการลบสัญลักษณ์พิเศษอื่นๆ นอกเหนือจากวรรณยุกต์
- **`thai_utils.detect_language`:** เปลี่ยนเป็นการคำนวณสัดส่วนของภาษาต่างๆ แทนการระบุภาษาเดียว
- **`thai_utils.thai_number_to_text`:** เพิ่มการรองรับเลขทศนิยมและปรับปรุงการจัดการเลขหลักต่างๆ
- **`thai_utils.split_sentences`:** เพิ่มคำลงท้ายและคำย่อภาษาไทยเพื่อความแม่นยำในการแบ่งประโยค
- **เอกสาร:** อัปเดต `README.md` ให้สะท้อนการเปลี่ยนแปลงล่าสุดและมีรายละเอียดมากขึ้น

## [0.2.1] - 2023-07-20

### เพิ่ม

- รองรับการใช้งาน PyThaiNLP เพื่อเพิ่มความสามารถขั้นสูงในการประมวลผลภาษาไทย
- ฟังก์ชันขั้นสูงสำหรับการจัดการตัวเลขไทย (thai_number_to_text, thai_text_to_number)
- ฟังก์ชันขั้นสูงสำหรับการจัดการวันที่และเวลาภาษาไทย (format_thai_date, thai_time, thai_day_to_datetime)
- ฟังก์ชันขั้นสูงสำหรับการค้นหาคำที่มีเสียงคล้ายกัน (thai_soundex)
- ฟังก์ชันขั้นสูงสำหรับการแก้ไขคำผิด (spell_correction)
- ฟังก์ชันขั้นสูงสำหรับการจัดการคำหยุดและพยางค์ภาษาไทย (get_thai_stopwords, get_thai_syllables)
- ฟังก์ชันขั้นสูงสำหรับการใช้งาน WordNet ภาษาไทย (get_thai_wordnet_synsets, get_thai_wordnet_synonyms)
- ฟังก์ชันขั้นสูงสำหรับการแปลงเลขไทยและเลขอารบิก (thai_digit_to_arabic_digit, arabic_digit_to_thai_digit)

### ปรับปรุง

- ปรับปรุงการตัดคำภาษาไทยให้รองรับหลายอัลกอริทึมจาก PyThaiNLP (newmm, longest, attacut)
- ปรับปรุงการแท็กส่วนของคำพูดให้รองรับหลายโมเดลจาก PyThaiNLP (perceptron, artagger)
- เพิ่มความสามารถในการแปลงระหว่างรูปแบบแท็กส่วนของคำพูดต่างๆ (UD, ORCHID)
- เพิ่มตัวอย่างการใช้งานฟังก์ชันขั้นสูงใน advanced_usage.py

## [0.2.0] - 2023-07-15

### เพิ่ม

- การจำแนกข้อความ (Text Classification) ด้วยวิธีการใช้คำสำคัญและ Zero-shot
- การจำแนกโทเค็น (Token Classification) สำหรับการแท็กส่วนของคำพูดและการรู้จำหน่วยงานที่มีชื่อเสียงแบบละเอียด
- การตอบคำถาม (Question Answering) สำหรับการตอบคำถามจากบริบทและตาราง
- การแปลภาษา (Translation) ระหว่างภาษาไทยและภาษาอังกฤษ
- การสกัดคุณลักษณะ (Feature Extraction) สำหรับการวิเคราะห์ข้อความภาษาไทย
- การสร้างข้อความ (Text Generation) ด้วยเทมเพลต, n-gram และรูปแบบไวยากรณ์
- การวัดความคล้ายคลึงของข้อความ (Text Similarity) ด้วยวิธีการต่างๆ
- ตัวอย่างการใช้งานขั้นสูง (Advanced Usage Examples)

### ปรับปรุง

- เพิ่มประสิทธิภาพการวิเคราะห์อารมณ์ด้วยดิกชันนารีที่ครอบคลุมมากขึ้น
- ปรับปรุงการแท็กส่วนของคำพูดให้มีความแม่นยำมากขึ้น
- เพิ่มทรัพยากรภาษาไทย (Thai Resources) สำหรับการประมวลผลภาษาธรรมชาติ

## [0.1.0] - 2023-07-01

### เพิ่ม

- การตัดคำ (Tokenization) ด้วยอัลกอริธึม Maximum Matching
- การแท็กส่วนของคำพูด (Part-of-Speech Tagging) ด้วย Hidden Markov Model
- การรู้จำหน่วยงานที่มีชื่อเสียง (Named Entity Recognition) ด้วยกฎและดิกชันนารี
- การวิเคราะห์อารมณ์ (Sentiment Analysis) ด้วยดิกชันนารีคำศัพท์
- การตรวจสอบการสะกดคำ (Spell Checking) ด้วย Edit Distance
- การสรุปข้อความ (Text Summarization) ด้วยอัลกอริธึม TextRank
- เครื่องมือช่วยเหลือ (Utilities) สำหรับการประมวลผลภาษาไทย
