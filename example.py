"""
Simple example of Thai text error correction
"""
def main():
    try:
        from thainlp.agent.error_correction_agent import ThaiErrorCorrectionAgent
        
        print("\nทดสอบระบบแก้ไขข้อความภาษาไทย")
        print("============================")
        
        # สร้าง error correction agent
        agent = ThaiErrorCorrectionAgent()
        
        # ทดสอบการแก้ไขข้อความ
        test_cases = [
            ("พิมพ์ผิดแป้นพิมพ์", "l;ylfu"),                    # สวัสดี
            ("ตัวซ้ำ", "เธอออออ คิดถึงงงง มากกกกก"),          # เธอ คิดถึง มาก
            ("สะกดผิด", "เเปนเรือง ทีเกิดขึน"),                # เป็นเรื่อง ที่เกิดขึ้น
            ("ผสม", "k;k เเละะะ งัยยย"),                      # ค่ะ และ ไง
            ("สไตล์โซเชียล", "55555 จ้าาาา")                   # ฮฮฮ จ้า
        ]
        
        for test_type, text in test_cases:
            print(f"\nทดสอบ: {test_type}")
            print(f"Input: {text}")
            
            # แก้ไขข้อความ
            result = agent.correct_text(text, aggressive=True)
            print(f"Output: {result['corrected']}")
            print(f"ความมั่นใจ: {result['confidence']:.2f}")
            
            # แสดงรายละเอียดการแก้ไข
            if result['corrections']:
                print("\nรายละเอียดการแก้ไข:")
                for c in result['corrections']:
                    print(f"- {c['type']}: {c['original']} -> {c['corrected']}")
                    
        # ทดสอบการวิเคราะห์ข้อผิดพลาด
        print("\nการวิเคราะห์ข้อผิดพลาด:")
        text = "helloooo สบายดีีีี มากกก"
        analysis = agent.analyze_errors(text)
        print(f"Text: {text}")
        print(f"Analysis: {analysis}")

    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()