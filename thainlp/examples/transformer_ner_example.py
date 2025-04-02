"""
Example usage of the Transformer-based NER model via Model Hub.
"""
# Ensure dependencies are installed: pip install thainlp[hf]
# (Installs transformers, torch/tensorflow, sentencepiece)

import logging
from thainlp.ner import tag_transformer, DEFAULT_TRANSFORMER_MODEL

# Configure logging to see model loading messages (optional)
logging.basicConfig(level=logging.INFO)

def main():
    print("=== Transformer NER Example ===")
    print(f"Using default model: {DEFAULT_TRANSFORMER_MODEL}")

    text1 = "คุณสมชายเดินทางไปจังหวัดเชียงใหม่เมื่อวานนี้"
    text2 = "ฉันทำงานที่บริษัท ไทยเอ็นแอลพี จำกัด แถวสยาม"
    text3 = "ดูคอนเสิร์ต BLACKPINK ที่สนามราชมังคลากีฬาสถาน สนุกมาก"

    print(f"\nInput: {text1}")
    entities1 = tag_transformer(text1)
    print("Entities:", entities1)

    print(f"\nInput: {text2}")
    entities2 = tag_transformer(text2)
    print("Entities:", entities2)

    print(f"\nInput: {text3}")
    entities3 = tag_transformer(text3)
    print("Entities:", entities3)

    # Example of using a specific model (if another was registered)
    # print("\nUsing specific model (hypothetical):")
    # try:
    #     entities_other = tag_transformer(text1, model_name="another_ner_model")
    #     print("Entities:", entities_other)
    # except ValueError as e:
    #     print(f"Error: {e}") # Expected if model not registered or invalid

if __name__ == "__main__":
    main()
