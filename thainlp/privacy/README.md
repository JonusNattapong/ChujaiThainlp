# Privacy Module Development Guide

## การพัฒนาโมดูล Privacy

โมดูล Privacy ถูกออกแบบให้สามารถพัฒนาและปรับปรุงได้อย่างต่อเนื่อง ผ่านการเรียนรู้จากตัวอย่างใหม่ๆ

### โครงสร้างโมดูล

```
privacy/
├── __init__.py
├── model_handler.py      # จัดการโมดลและการฝึกฝน
├── examples/            # ตัวอย่างการใช้งาน
│   └── privacy_training_example.py
└── models/             # โฟลเดอร์เก็บโมดล
    └── privacy/
        ├── examples/   # ตัวอย่างที่ใช้ฝึกฝน
        └── model_info.json
```

### การฝึกฝนโมดล

1. สร้าง instance ของ PrivacyPreservingNLP:
```python
privacy = PrivacyPreservingNLP(
    model_dir="./models/privacy_development"
)
```

2. ฝึกฝนด้วยตัวอย่างใหม่:
```python
privacy.train_on_example(
    text="นายสมชาย ใจดี โทร 081-234-5678",
    spans=[
        ("PERSON", 0, 13),
        ("PHONE", 18, 30)
    ]
)
```

3. ตรวจสอบประสิทธิภาพ:
```python
metrics = privacy.get_performance_metrics()
print(f"ความแม่นยำ: {metrics['accuracy']:.2%}")
```

### การปรับแต่งพารามิเตอร์

- `epsilon`: ควบคุมระดับของ differential privacy (ค่าน้อย = ความเป็นส่วนตัวสูง)
- `noise_scale`: ปรับระดับ noise ใน vector embedding
- `validation_threshold`: กำหนดค่าความเชื่อมั่นขั้นต่ำในการตรวจจับ
- `fuzzy_threshold`: ค่าความคล้ายคลึงขั้นต่ำสำหรับ fuzzy matching

### การเพิ่มรูปแบบการตรวจจับใหม่

1. เพิ่ม pattern ใน `pii_patterns`:
```python
self.pii_patterns.update({
    'รูปแบบใหม่': r'regex_pattern'
})
```

2. เพิ่มการตรวจสอบใน `_validate_pii`:
```python
elif label == 'รูปแบบใหม่':
    return self._custom_validation(text)
```

### Best Practices

1. **การเก็บตัวอย่าง**:
   - เก็บตัวอย่างที่หลากหลาย
   - ครอบคลุมภาษาท้องถิ่นต่างๆ
   - มีทั้งกรณีปกติและข้อยกเว้น

2. **การปรับแต่ง Threshold**:
   - เริ่มด้วยค่าสูง (0.95) แล้วค่อยๆ ปรับลด
   - แยก threshold ตามประเภทข้อมูล
   - ใช้ข้อมูลจริงในการปรับแต่ง

3. **การทดสอบ**:
   - ทดสอบกับข้อมูลหลากหลายรูปแบบ
   - ตรวจสอบ false positives/negatives
   - วัดประสิทธิภาพอย่างต่อเนื่อง

4. **การอัพเดทโมดล**:
   - บันทึกการเปลี่ยนแปลง
   - ทดสอบก่อนใช้งานจริง
   - สำรองข้อมูลเดิมไว้เสมอ

### ตัวอย่างการใช้งาน

ดูตัวอย่างการใช้งานได้ที่ `examples/privacy_training_example.py`

### การติดตามประสิทธิภาพ

โมดูลจะบันทึกประสิทธิภาพไว้ใน `model_info.json` ซึ่งประกอบด้วย:
- ความแม่นยำ (Accuracy)
- จำนวนตัวอย่างที่ผ่านการฝึกฝน
- วันที่อัพเดทล่าสุด
- ประวัติการปรับปรุง

### การอัพเดทในอนาคต

- [ ] เพิ่มการรองรับภาษาท้องถิ่นเพิ่มเติม
- [ ] ปรับปรุงการตรวจจับแบบ fuzzy
- [ ] เพิ่มความสามารถในการเรียนรู้อัตโนมัติ
- [ ] พัฒนาการป้องกันการรั่วไหลของข้อมูล