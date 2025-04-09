"""
Thai Dialect Processing Example

This example demonstrates the advanced Thai dialect processing capabilities
in ChujaiThainlp, including detection, translation, and speech synthesis.
"""

import sys
import os
import time
from typing import Dict, List, Any

# Add parent directory to path to run this file directly
sys.path.append(os.path.abspath(".."))

from thainlp.dialects.dialect_processor import ThaiDialectProcessor
from thainlp.dialects.dialect_tokenizer import DialectTokenizer
from thainlp.optimization.dialect_optimizer import DialectOptimizer, batch_detect_dialect
from thainlp.speech.dialect_adapter import ThaiDialectSpeechAdapter, adapt_tts_for_dialect


def dialect_detection_example():
    """Example of dialect detection"""
    print("\n=== Thai Dialect Detection Example ===")
    
    # Initialize dialect processor
    processor = ThaiDialectProcessor(use_ml=True)
    
    # Example texts in different dialects
    examples = {
        "northern": "เปิ้นบ่อยู่เฮือนเจ้า ไปซื้อของก่อนเน้อ",
        "northeastern": "อีหลีเฮ็ดหยังอยู่ ข้อยสิไปตลาดเด้อ กินเข้าแล้วบ่",
        "southern": "ฉานไปตลาดหนิ อาหารหรอยนักวันนี้แอ",
        "central": "ผมกำลังจะไปตลาดครับ อาหารอร่อยมากวันนี้นะ",
        "pattani_malay": "อาเกาะ นะ เปอกี ปาซะ มากัน นี่ ซาดะ บาญะ"
    }
    
    # Detect dialect for each example
    for dialect_name, text in examples.items():
        print(f"\nText ({dialect_name}): {text}")
        
        # Detect dialect
        results = processor.detect_dialect(text)
        
        # Print results
        print("Detected dialects:")
        for detected_dialect, confidence in sorted(results.items(), key=lambda x: -x[1]):
            if confidence > 0.05:  # Only show dialects with significant confidence
                print(f"  - {detected_dialect}: {confidence:.2f}")
        
        # If a regional dialect was detected, analyze further
        primary_dialect = max(results.items(), key=lambda x: x[1])[0]
        if primary_dialect in ["northern", "northeastern", "southern"] and results[primary_dialect] > 0.5:
            regional_results = processor.detect_regional_dialect(text, primary_dialect)
            print("Regional dialect detection:")
            for region, confidence in sorted(regional_results.items(), key=lambda x: -x[1]):
                if confidence > 0.1:
                    print(f"  - {region}: {confidence:.2f}")


def dialect_translation_example():
    """Example of dialect translation"""
    print("\n=== Thai Dialect Translation Example ===")
    
    processor = ThaiDialectProcessor()
    
    # Example texts in different dialects
    examples = {
        "northern": "เจ้ากิ๋นข้าวแล้วกา อั๋นจะไป๋ตลาดเจ้า",
        "northeastern": "สบายดีบ่ เฮ็ดหยังอยู่ กินเข้าแล้วบ่",
        "southern": "ฉานจะไปตลาดจัง อาหารหรอยนักวันนี้หนิ"
    }
    
    # Translate to standard Thai
    print("\nTranslating to Standard Thai:")
    for dialect_name, text in examples.items():
        print(f"\nOriginal ({dialect_name}): {text}")
        translated = processor.translate_to_standard(text, dialect_name)
        print(f"Translated (standard): {translated}")
    
    # Translate from standard Thai to dialects
    standard_text = "สวัสดีครับ คุณสบายดีไหม ผมกำลังจะไปตลาด อาหารอร่อยมากวันนี้"
    print("\nTranslating from Standard Thai:")
    print(f"\nOriginal (standard): {standard_text}")
    
    for dialect in ["northern", "northeastern", "southern"]:
        translated = processor.translate_from_standard(standard_text, dialect)
        print(f"Translated ({dialect}): {translated}")


def dialect_tokenization_example():
    """Example of dialect-aware tokenization"""
    print("\n=== Thai Dialect-Aware Tokenization Example ===")
    
    # Initialize dialect tokenizer
    tokenizer = DialectTokenizer(auto_detect=True)
    
    # Example texts in different dialects
    examples = {
        "northern": "เปิ้นกำลังมาละเจ้า จะไปก๋าดเจ้า",
        "northeastern": "อีหลีสิไปไส อยู่ตรงนี่นำกัน กะสิเอาบ่",
        "southern": "ตั๋วกินข้าวกับนิ ไปวั่นมาวั่น ใจ้ชั่วเหอะ"
    }
    
    # Tokenize each example
    for dialect_name, text in examples.items():
        print(f"\nText ({dialect_name}): {text}")
        
        # Standard tokenization
        tokens = tokenizer.tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Tokenize with dialect preservation
        tokens_with_dialect = tokenizer.tokenize_and_preserve_dialectal(text)
        print("Tokens with dialect markers:")
        for token, dialect in tokens_with_dialect:
            dialect_str = dialect if dialect else "standard"
            print(f"  '{token}' ({dialect_str})")
        
        # Calculate dialectal diversity
        diversity = tokenizer.get_dialectal_diversity(text)
        print(f"Dialectal diversity score: {diversity:.2f}")


def batch_processing_example():
    """Example of batch processing for dialects"""
    print("\n=== Thai Dialect Batch Processing Example ===")
    
    # Create sample texts
    texts = [
        "สวัสดีครับ ผมชื่อสมชาย",
        "เปิ้นบ่อยู่เฮือนเจ้า ไปซื้อของก่อน",
        "อีหลีเฮ็ดหยังอยู่ ข้อยสิไปตลาด",
        "ฉานไปตลาดหนิ อาหารหรอยนัก",
        "สบายดีครับ คุณมาจากไหน",
        "อากาศฮ้อนเหลือเปิ้น อู้กำเมืองได้ก่อ",
        "เว้าภาษาอีสานเป็นบ่ สิไปไสกะไปเดอ",
        "หวัดดีหนิ ไปท่าไหนมา",
        "อาเกาะ นะ เปอกี ปาซะ"
    ]
    
    print(f"Processing {len(texts)} texts in batch mode...")
    
    # Initialize optimizer
    optimizer = DialectOptimizer(max_workers=4)
    
    # Time batch processing
    start_time = time.time()
    results = optimizer.batch_detect_dialect(texts)
    elapsed = time.time() - start_time
    
    print(f"Batch processing completed in {elapsed:.3f} seconds")
    
    # Show results
    for i, (text, result) in enumerate(zip(texts, results)):
        primary_dialect = max(result.items(), key=lambda x: x[1])[0]
        confidence = result[primary_dialect]
        print(f"{i+1}. \"{text[:30]}{'...' if len(text) > 30 else ''}\" -> {primary_dialect} ({confidence:.2f})")
    
    # Analyze dialect distribution
    print("\nDialect distribution analysis:")
    distribution = optimizer.analyze_dialect_distribution(texts)
    
    # Print distribution percentages
    for dialect, percentage in distribution["percentages"].items():
        if percentage > 0:
            print(f"  - {dialect}: {percentage:.1f}% ({distribution['counts'][dialect]} texts)")
            

def speech_synthesis_example():
    """Example of dialect-aware speech synthesis"""
    print("\n=== Thai Dialect Speech Synthesis Example ===")
    
    # Initialize speech adapter
    adapter = ThaiDialectSpeechAdapter()
    
    # Example text
    text = "สวัสดีครับ วันนี้อากาศดีมาก"
    print(f"\nText: {text}")
    
    # Example of base TTS parameters
    tts_params = {
        "pitch_factor": 1.0,
        "speed_factor": 1.0,
        "volume": 1.0
    }
    
    # Apply dialect adaptations
    for dialect in ["northern", "northeastern", "southern", "central"]:
        adapted_params = adapter.adapt_speech_parameters(
            text=text,
            tts_params=tts_params,
            dialect=dialect
        )
        
        # Print adapted parameters (without the accent_params detail)
        display_params = {k: v for k, v in adapted_params.items() if k != "accent_params"}
        print(f"\n{dialect.capitalize()} dialect parameters:")
        for param, value in display_params.items():
            print(f"  - {param}: {value}")
    
    # Create a custom voice profile
    profile_name = adapter.create_dialect_voice_profile(
        dialect="northeastern",
        region="อีสานเหนือ",
        custom_params={"pitch_shift": 1.15, "speech_rate": 0.95}
    )
    
    print(f"\nCreated custom voice profile: {profile_name}")
    profile = adapter.get_voice_profile(profile_name)
    if profile:
        print(f"Profile details: {profile['dialect']} ({profile['region']})")


if __name__ == "__main__":
    print("Thai Dialect Processing Examples")
    print("=" * 50)
    
    # Run all examples
    dialect_detection_example()
    dialect_translation_example()
    dialect_tokenization_example()
    batch_processing_example()
    speech_synthesis_example()
    
    print("\nAll examples completed.")