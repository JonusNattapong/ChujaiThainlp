#!/usr/bin/env python3
from thainlp.generation.fill_mask import ThaiFillMask

def main():
    """ตัวอย่างการใช้งานระบบเติมคำในช่องว่างภาษาไทย"""
    print("=== ตัวอย่างการใช้งานระบบเติมคำในช่องว่างภาษาไทย ===")
    
    # 1. เติมคำในช่องว่างพื้นฐานด้วย WangchanBERTa
    print("\n1. เติมคำในช่องว่างพื้นฐานด้วย WangchanBERTa:")
    fill_mask = ThaiFillMask(model_name_or_path="google-bert/bert-base-multilingual-cased")

    text = "ฉัน<mask>แมว"
    print(f"ข้อความ: {text}")
    predictions = fill_mask.fill_mask(text, top_k=5)
    print("คำทำนาย:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. '{pred['token']}' (คะแนน: {pred['score']:.3f})")

if __name__ == "__main__":
    main()
