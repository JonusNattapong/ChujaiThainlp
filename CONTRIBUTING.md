# วิธีการมีส่วนร่วม (Contributing)

ขอบคุณที่สนใจมีส่วนร่วมในการพัฒนา ThaiNLP! เอกสารนี้จะแนะนำวิธีการมีส่วนร่วมในการพัฒนาไลบรารี่

## การเตรียมสภาพแวดล้อมสำหรับการพัฒนา

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
pip install -r requirements-dev.txt
```

## การพัฒนา

### 1. สร้าง Branch ใหม่

```bash
git checkout -b feature/your-feature-name
# หรือ
git checkout -b fix/your-fix-name
```

### 2. การเขียนโค้ด

- ใช้ PEP 8 style guide
- เพิ่ม docstring สำหรับฟังก์ชันและคลาส
- เขียน type hints
- เพิ่ม comments ที่จำเป็น

ตัวอย่าง:

```python
def analyze_sentiment(text: str) -> Tuple[float, str, Dict[str, List[str]]]:
    """
    วิเคราะห์ความรู้สึกของข้อความภาษาไทย
    
    Args:
        text (str): ข้อความภาษาไทยที่ต้องการวิเคราะห์
        
    Returns:
        Tuple[float, str, Dict[str, List[str]]]: (คะแนนความรู้สึก, ประเภท, คำที่พบ)
    """
    # โค้ดของคุณที่นี่
```

### 3. การเขียนเทสต์

- เขียน unit tests สำหรับฟังก์ชันใหม่
- ตรวจสอบให้แน่ใจว่าเทสต์ทั้งหมดผ่าน
- เพิ่ม test cases ที่ครอบคลุมกรณีต่างๆ

ตัวอย่าง:

```python
def test_analyze_sentiment():
    text = "วันนี้อากาศดีมากๆ"
    score, label, words = analyze_sentiment(text)
    assert score > 0
    assert label == "very_positive"
    assert "ดี" in words['positive']
```

### 4. การอัปเดตเอกสาร

- อัปเดต README.md ถ้ามีการเปลี่ยนแปลงที่สำคัญ
- อัปเดต CHANGELOG.md ตามรูปแบบที่กำหนด
- เพิ่มหรือแก้ไข docstrings ตามความเหมาะสม

### 5. การตรวจสอบโค้ด

```bash
# ตรวจสอบ style
flake8 thainlp tests

# ตรวจสอบ type hints
mypy thainlp

# รันเทสต์
pytest tests/
```

## การส่ง Pull Request

1. Commit การเปลี่ยนแปลงของคุณ:

```bash
git add .
git commit -m "feat: เพิ่มฟีเจอร์ใหม่"  # หรือ "fix: แก้ไขบั๊ก"
```

2. Push ไปยัง repository ของคุณ:

```bash
git push origin feature/your-feature-name
```

3. สร้าง Pull Request บน GitHub:
   - เลือก branch ที่ต้องการ merge
   - กรอกรายละเอียดการเปลี่ยนแปลง
   - อ้างอิง issue ที่เกี่ยวข้อง (ถ้ามี)

## แนวทางการ Commit

ใช้ conventional commits:

- `feat:` สำหรับฟีเจอร์ใหม่
- `fix:` สำหรับการแก้ไขบั๊ก
- `docs:` สำหรับการแก้ไขเอกสาร
- `style:` สำหรับการแก้ไขรูปแบบโค้ด
- `refactor:` สำหรับการปรับปรุงโค้ด
- `test:` สำหรับการเพิ่มหรือแก้ไขเทสต์
- `chore:` สำหรับการเปลี่ยนแปลงอื่นๆ

## การรายงานปัญหา (Issues)

เมื่อรายงานปัญหา กรุณาระบุ:

1. เวอร์ชันของไลบรารี่ที่ใช้
2. ขั้นตอนการทำซ้ำปัญหา
3. ข้อความ error ที่พบ (ถ้ามี)
4. ตัวอย่างโค้ดที่ทำให้เกิดปัญหา

## การติดต่อ

- สร้าง issue สำหรับการอภิปราย
- ติดต่อผ่าน email: <zombitx64@gmail.com>

## การอนุญาต

การมีส่วนร่วมของคุณจะถูกอนุญาตภายใต้ MIT License เช่นเดียวกับโปรเจค

ขอบคุณที่สนใจมีส่วนร่วมในการพัฒนา ThaiNLP!
