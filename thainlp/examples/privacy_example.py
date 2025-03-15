from thainlp.privacy import PrivacyPreservingNLP

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    privacy_nlp = PrivacyPreservingNLP(epsilon=0.2)
    
    # ตัวอย่างข้อความที่มีข้อมูลส่วนบุคคล
    text = """
    นายสมชาย ใจดี เกิดวันที่ 15 มกราคม 2535 
    อยู่บ้านเลขที่ 123/45 ถ.พหลโยธิน แขวงจตุจักร กรุงเทพฯ 10900
    เบอร์โทรศัพท์ 086-123-4567 อีเมล somchai@example.com
    เลขบัตรประชาชน 1234567890123
    ทำงานที่บริษัท ไทยรุ่งเรือง จำกัด
    """
    
    # แสดงผลลัพธ์
    result = privacy_nlp.process_sensitive_document(text)
    
    print("ข้อความที่ปกปิดข้อมูลส่วนบุคคลแล้ว:")
    print(result['anonymized_text'])
    
    print("\nคำที่อ่อนไหวถูกแปลงเป็น hash:")
    print(result['hashed_tokens'][:10])  # แสดงเฉพาะ 10 โทเค็นแรก
