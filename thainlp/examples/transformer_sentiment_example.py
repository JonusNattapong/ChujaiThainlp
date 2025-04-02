"""
Example usage of the Transformer-based Sentiment Analysis model via Model Hub.
"""
# Ensure dependencies are installed: pip install thainlp[hf]
# (Installs transformers, torch/tensorflow, sentencepiece)

import logging
from thainlp.sentiment import analyze_transformer, DEFAULT_TRANSFORMER_MODEL

# Configure logging to see model loading messages (optional)
logging.basicConfig(level=logging.INFO)

def main():
    print("=== Transformer Sentiment Analysis Example ===")
    print(f"Using default model: {DEFAULT_TRANSFORMER_MODEL}")

    texts = [
        "หนังสนุกมาก ชอบสุดๆ",
        "อาหารรสชาติแย่มาก ไม่แนะนำเลย",
        "วันนี้อากาศดี ไม่มีอะไรพิเศษ",
        "บริการดีเยี่ยม ประทับใจมากค่ะ",
        "รอนานมาก หงุดหงิด",
    ]

    for text in texts:
        print(f"\nInput: {text}")
        sentiment = analyze_transformer(text)
        if sentiment:
            print(f"Sentiment: Label='{sentiment['label']}', Score={sentiment['score']:.4f}")
        else:
            print("Sentiment: Could not analyze.")

    # Example of using a specific model (if another was registered)
    # print("\nUsing specific model (hypothetical):")
    # try:
    #     sentiment_other = analyze_transformer(texts[0], model_name="another_sentiment_model")
    #     if sentiment_other:
    #           print(f"Sentiment: Label='{sentiment_other['label']}', Score={sentiment_other['score']:.4f}")
    #     else:
    #           print("Sentiment: Could not analyze.")
    # except ValueError as e:
    #     print(f"Error: {e}") # Expected if model not registered or invalid

if __name__ == "__main__":
    main()
