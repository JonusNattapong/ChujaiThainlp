"""
Simple example of Thai text error correction without external dependencies
"""
import os
import sys

def main():
    try:
        # Add parent directory to Python path for imports
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        sys.path.insert(0, project_root)
        
        from thainlp.agent.error_correction_agent import ThaiErrorCorrectionAgent
        
        print("Thai Text Error Correction Examples")
        print("==================================")
        print(f"Project root: {project_root}\n")
        
        agent = ThaiErrorCorrectionAgent()
        
        examples = [
            ("พิมพ์ผิดแป้นพิมพ์", "l;ylfu", "สวัสดี"),  # สวัสดี
            ("ตัวอักษรซ้ำ", "เธอออออ คิดถึงงงง มากกกกก", "เธอ คิดถึง มาก"),
            ("พิมพ์ผิด", "เเปนเรือง ทีเกิดขึน", "เป็นเรื่อง ที่เกิดขึ้น"),
            ("ผสม", "k;k เเละะะ งัยยย", "ค่ะ และ ไง"),
            ("สไตล์โซเชียล", "55555 จ้าาาา", "ฮฮฮ จ้า"),
        ]
        
        for test_type, input_text, expected in examples:
            print(f"\nทดสอบ: {test_type}")
            print(f"Input: {input_text}")
            
            result = agent.correct_text(input_text, aggressive=True)
            print(f"Output: {result['corrected']}")
            print(f"Expected: {expected}")
            print(f"Corrections: {len(result['corrections'])}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            # Show detailed corrections
            if result['corrections']:
                print("\nรายละเอียดการแก้ไข:")
                for c in result['corrections']:
                    print(f"- {c['type']}: {c['original']} -> {c['corrected']}")
                    
        print("\nการวิเคราะห์ข้อผิดพลาด:")
        text = "helloooo สบายดีีีี มากกก"
        analysis = agent.analyze_errors(text)
        print(f"Text: {text}")
        print(f"Analysis: {analysis}")
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()