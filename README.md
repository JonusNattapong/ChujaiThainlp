# ChujaiThainlp: ไลบรารีประมวลผลภาษาไทย

**ChujaiThainlp** เป็นไลบรารีภาษา Python ที่ออกแบบมาเพื่อการประมวลผลภาษาไทยโดยเฉพาะ โดยมุ่งเน้นความง่ายในการใช้งาน ประสิทธิภาพ และความสามารถที่ครอบคลุม ตั้งแต่การจัดการข้อความพื้นฐานไปจนถึงการวิเคราะห์ภาษาขั้นสูง

## 🚀 คุณสมบัติหลัก

- ✨ **อินเตอร์เฟซที่ใช้งานง่าย:** ออกแบบด้วย method chaining และฟังก์ชันสำเร็จรูป เพื่อความสะดวกในการพัฒนา
- 🔄 **ความสามารถหลากหลาย:** รองรับการจัดการข้อความภาษาไทยในรูปแบบต่างๆ เช่น การปรับแต่ง การแปลง และการแยกข้อมูล
- 📊 **การวิเคราะห์ข้อความ:** มีเครื่องมือสำหรับวิเคราะห์สัดส่วนภาษา นับประเภทอักขระ และแบ่งประโยค
- ⚡ **ประสิทธิภาพ:** พัฒนาโดยคำนึงถึงประสิทธิภาพในการประมวลผลข้อความขนาดใหญ่
- 🧪 **ความน่าเชื่อถือ:** ผ่านการทดสอบอย่างครอบคลุมเพื่อให้มั่นใจในความถูกต้องของการทำงาน

## ⚙️ การติดตั้ง

สามารถติดตั้งไลบรารี ChujaiThainlp ผ่าน pip:

```bash
pip install thainlp
```

หากต้องการติดตั้งเวอร์ชันล่าสุดจากแหล่งข้อมูลโดยตรง:

```bash
pip install git+https://github.com/username/thainlp.git
```

## 📖 คู่มือการใช้งาน

### 1. ฟังก์ชันสำเร็จรูป (Quick Functions)

สำหรับงานประมวลผลทั่วไป สามารถใช้ฟังก์ชันสำเร็จรูปได้โดยตรง:

```python
from thainlp.utils.thai_text import normalize_text, romanize, extract_thai, split_sentences

# การปรับแต่งข้อความ (Normalization)
# ลบช่องว่างซ้ำซ้อน, แปลงเลขไทย, จัดการอักขระพิเศษ
text = normalize_text("  สวัสดี   ครับ  ๑๒๓ ")
# ผลลัพธ์: "สวัสดี ครับ 123"

# การแปลงเป็นอักษรโรมัน (Romanization)
roman = romanize("สวัสดี")
# ผลลัพธ์: "sawatdi"

# การแยกเฉพาะข้อความภาษาไทย (Thai Text Extraction)
thai = extract_thai("Hello สวัสดี 123")
# ผลลัพธ์: "สวัสดี"

# การแบ่งประโยค (Sentence Splitting)
sentences = split_sentences("สวัสดีครับ วันนี้อากาศดีนะครับ ไปไหนมาครับ")
# ผลลัพธ์: ["สวัสดีครับ", "วันนี้อากาศดีนะครับ", "ไปไหนมาครับ"]
```

### 2. การประมวลผลแบบต่อเนื่อง (Method Chaining)

ใช้คลาส `ThaiTextProcessor` เพื่อประมวลผลข้อความหลายขั้นตอนอย่างต่อเนื่อง:

```python
from thainlp.utils.thai_text import process_text

# ตัวอย่างการประมวลผลหลายขั้นตอน
text_input = "สวัสดี ๑๒๓ Hello! ภาษาไทย"
result = (process_text(text_input)
         .normalize()          # 1. ปรับรูปแบบข้อความ
         .to_arabic_digits()   # 2. แปลงเลขไทยเป็นอารบิก
         .extract_thai()       # 3. แยกเฉพาะข้อความภาษาไทย
         .to_roman()          # 4. แปลงเป็นอักษรโรมัน
         .get_text())         # 5. ดึงผลลัพธ์สุดท้าย
# ผลลัพธ์: "sawatdi phasathai"

# ตัวอย่างการวิเคราะห์ข้อความ
text_analysis = "Hello สวัสดีครับ 123"
processor = process_text(text_analysis)

# วิเคราะห์สัดส่วนภาษา
ratios = processor.get_script_ratios()
print(f"Thai Ratio: {ratios['thai']:.2%}")
print(f"Latin Ratio: {ratios['latin']:.2%}")

# นับประเภทอักขระ
counts = processor.get_character_counts()
print(f"Consonants: {counts.get('consonants', 0)}")
print(f"Vowels: {counts.get('vowels', 0)}")
```

### 3. การตรวจสอบความถูกต้อง (Validation)

ใช้คลาส `ThaiValidator` สำหรับตรวจสอบลักษณะของข้อความภาษาไทย:

```python
from thainlp.utils.thai_text import ThaiValidator

validator = ThaiValidator()

# ตรวจสอบว่าเป็นอักขระไทยหรือไม่
is_thai_char = validator.is_thai_char("ก")  # True
is_thai_char = validator.is_thai_char("A")  # False

# ตรวจสอบว่าเป็นคำไทยที่ถูกต้องตามโครงสร้างพยางค์หรือไม่
is_valid_word = validator.is_valid_word("สวัสดี")  # True
is_valid_word = validator.is_valid_word("ก า")    # False (มีช่องว่าง)
is_valid_word = validator.is_valid_word("hello")  # False

# ดูรูปแบบ Regex สำหรับพยางค์ไทย
syllable_pattern = validator.get_syllable_pattern()
```

## 🎯 ภาพรวมความสามารถ

| หมวดหมู่             | ความสามารถ                                                                 |
| :------------------- | :------------------------------------------------------------------------- |
| **การจัดการข้อความ** | `normalize_text`, `remove_tone_marks`, `remove_diacritics`, `extract_thai` |
| **การแปลงรูปแบบ**   | `thai_to_roman`, `thai_digit_to_arabic_digit`, `arabic_digit_to_thai_digit` |
| **การวิเคราะห์**     | `detect_language`, `count_thai_characters`, `split_sentences`              |
| **การตรวจสอบ**     | `is_thai_char`, `is_thai_word`, `is_valid_thai_word`                       |
| **การแปลงตัวเลข**   | `thai_number_to_text`                                                      |

## 📚 เอกสารและตัวอย่าง

- **ตัวอย่างการใช้งาน:** ดูโค้ดตัวอย่างเพิ่มเติมได้ในไดเรกทอรี `thainlp/examples/`
  - [`thai_text_example.py`](thainlp/examples/thai_text_example.py): ตัวอย่างการใช้อินเตอร์เฟซหลัก
  - [`thai_utils_example.py`](thainlp/examples/thai_utils_example.py): ตัวอย่างการใช้ฟังก์ชันพื้นฐาน
- **เอกสารอ้างอิง:** อ่านคำอธิบายฟังก์ชันโดยละเอียดได้ที่ [`thainlp/docs/thai_utils.md`](thainlp/docs/thai_utils.md)

## 🛠️ สำหรับนักพัฒนา

### การตั้งค่าสภาพแวดล้อม

```bash
# Clone repository
git clone https://github.com/username/thainlp.git
cd thainlp

# สร้างและเปิดใช้งาน virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate  # บน Linux/macOS
# venv\Scripts\activate  # บน Windows

# ติดตั้งไลบรารีและ dependencies สำหรับการพัฒนา
pip install -e ".[dev]"
```

### การรันชุดทดสอบ

```bash
pytest tests/
```

## 🤝 การมีส่วนร่วม

เรายินดีต้อนรับการมีส่วนร่วมในทุกรูปแบบ! หากท่านสนใจ กรุณาศึกษา [แนวทางการมีส่วนร่วม](CONTRIBUTING.md) ของเรา และสามารถเปิด Issue หรือส่ง Pull Request ได้

## 📜 สัญญาอนุญาต

ไลบรารีนี้เผยแพร่ภายใต้สัญญาอนุญาต MIT ดูรายละเอียดเพิ่มเติมได้ที่ไฟล์ [LICENSE](LICENSE)

## 👥 ทีมผู้พัฒนา

ThaiNLP พัฒนาโดยทีมงานและผู้มีส่วนร่วม สามารถดูรายชื่อได้ที่ [CONTRIBUTORS.md](CONTRIBUTORS.md)

## 📫 ช่องทางการติดต่อ

- **รายงานปัญหา/ข้อเสนอแนะ:** [GitHub Issues](https://github.com/username/thainlp/issues)
- **สอบถามทั่วไป:** [zombitx64@gmail.com](mailto:zombitx64@gmail.com)
