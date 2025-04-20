"""
ตัวอย่างการใช้งานโมดูล Privacy สำหรับการพัฒนาและฝึกฝน
"""

import os
from thainlp.privacy import PrivacyPreservingNLP

def main():
    # สร้าง instance ของ PrivacyPreservingNLP พร้อมระบุ directory สำหรับเก็บโมเดล
    privacy = PrivacyPreservingNLP(
        epsilon=0.1,
        noise_scale=0.1,
        handle_dialects=True,
        model_dir="./models/privacy_development"
    )

    # ตัวอย่างข้อความสำหรับฝึกฝน
    training_examples = [
        {
            "text": "นายสมชาย ใจดี อาศัยอยู่บ้านเลขที่ 123 หมู่ 4 ต.ในเมือง อ.เมือง จ.เชียงใหม่ 50000 โทร 081-234-5678",
            "spans": [
                ("PERSON", 0, 13),  # นายสมชาย ใจดี
                ("ADDRESS", 22, 71),  # บ้านเลขที่ 123 หมู่ 4...
                ("PHONE", 76, 88)  # 081-234-5678
            ]
        },
        {
            "text": "อาจ๋านสมศรี สอนที่โรงเรียนบ้านดอยสวย ติดต่อ LINE: teacher_somsri",
            "spans": [
                ("PERSON", 0, 10),  # อาจ๋านสมศรี
                ("ORGANIZATION", 19, 37),  # โรงเรียนบ้านดอยสวย
                ("LINE ID", 46, 64)  # LINE: teacher_somsri
            ]
        },
        {
            "text": "คุณวิชัย ขายของที่ตลาดวโรรส เบอร์โทรศัพท์ 053-999888 อีเมล wichai@email.com",
            "spans": [
                ("PERSON", 0, 8),  # คุณวิชัย
                ("LOCATION", 16, 26),  # ตลาดวโรรส
                ("PHONE", 40, 51),  # 053-999888
                ("EMAIL", 58, 74)  # wichai@email.com
            ]
        }
    ]

    # ฝึกฝนโมเดลด้วยตัวอย่าง
    for example in training_examples:
        privacy.train_on_example(example["text"], example["spans"])
        
        # ทดสอบการตรวจจับหลังฝึกฝน
        anonymized = privacy.anonymize_text(example["text"])
        print("\nต้นฉบับ:", example["text"])
        print("หลังปกปิด:", anonymized)
        
    # แสดงประสิทธิภาพของโมเดล
    performance = privacy.get_performance_metrics()
    print("\nประสิทธิภาพของโมเดล:")
    print(f"ความแม่นยำ: {performance['accuracy']:.2%}")
    print(f"จำนวนตัวอย่างที่ฝึกฝน: {performance['samples']}")

if __name__ == "__main__":
    main()