"""
Example of using Thai dialect functionality
"""
import sys
import os
from typing import Dict, List, Any, Union

# Add parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thainlp.dialects import (
    ThaiDialectProcessor, 
    DialectTokenizer,
    detect_dialect,
    translate_to_standard,
    translate_from_standard,
    get_dialect_features,
    get_dialect_info
)

def demonstrate_dialect_detection():
    """Demonstrate dialect detection"""
    print("=== Dialect Detection ===")
    
    # Northern Thai examples
    northern_text = "อั๋นจะไป๋ตลาดเน้อ กิ๋นข้าวแล้วกา"
    print(f"\nNorthern Thai text: {northern_text}")
    northern_result = detect_dialect(northern_text)
    print("Detected dialects:")
    for dialect, score in northern_result.items():
        if score > 0.01:  # Only show non-zero scores
            print(f"- {dialect}: {score:.2f}")
    
    # Northeastern Thai examples
    northeastern_text = "ข้อยสิไปตลาด กินเข้าแล้วบ่ อาหารแซบหลาย"
    print(f"\nNortheastern Thai text: {northeastern_text}")
    northeastern_result = detect_dialect(northeastern_text)
    print("Detected dialects:")
    for dialect, score in northeastern_result.items():
        if score > 0.01:
            print(f"- {dialect}: {score:.2f}")
    
    # Southern Thai examples
    southern_text = "ฉานจะไปตลาด กินข้าวแล้วหรือหนิ อาหารหรอยนัก"
    print(f"\nSouthern Thai text: {southern_text}")
    southern_result = detect_dialect(southern_text)
    print("Detected dialects:")
    for dialect, score in southern_result.items():
        if score > 0.01:
            print(f"- {dialect}: {score:.2f}")
    
    # Central/Standard Thai examples
    central_text = "ผมจะไปตลาด กินข้าวหรือยัง อาหารอร่อยมาก"
    print(f"\nCentral Thai text: {central_text}")
    central_result = detect_dialect(central_text)
    print("Detected dialects:")
    for dialect, score in central_result.items():
        if score > 0.01:
            print(f"- {dialect}: {score:.2f}")
    
    # Pattani Malay examples
    pattani_text = "อาเกาะ นะ เปอกี ปาซะ มากัน แล็ฮ กือ บือลูม"
    print(f"\nPattani Malay text: {pattani_text}")
    pattani_result = detect_dialect(pattani_text)
    print("Detected dialects:")
    for dialect, score in pattani_result.items():
        if score > 0.01:
            print(f"- {dialect}: {score:.2f}")
            
    # Mixed dialects
    mixed_text = "ผมกิ๋นข้าวที่ร้านอาหารเด้อ แซบหลาย"
    print(f"\nMixed Thai dialects: {mixed_text}")
    mixed_result = detect_dialect(mixed_text)
    print("Detected dialects:")
    for dialect, score in mixed_result.items():
        if score > 0.01:
            print(f"- {dialect}: {score:.2f}")

def demonstrate_regional_dialect_detection():
    """Demonstrate regional dialect variation detection"""
    print("\n=== Regional Dialect Detection ===")
    
    processor = ThaiDialectProcessor()
    
    # Northern Thai regional variations
    northern_cm_text = "เปิ้นกำลังมาละเจ้า จะไปก๋าดเจ้า"
    print(f"\nNorthern Thai (Chiang Mai style): {northern_cm_text}")
    northern_result = processor.detect_dialect(northern_cm_text)
    primary_dialect = max(northern_result, key=lambda k: northern_result[k])
    print(f"Primary dialect: {primary_dialect} (confidence: {northern_result[primary_dialect]:.2f})")
    
    # Detect regional variation
    regional_result = processor.detect_regional_dialect(northern_cm_text, primary_dialect)
    print("Regional variations:")
    for region, score in regional_result.items():
        if score > 0.01:
            print(f"- {region}: {score:.2f}")
    
    # Northeastern Thai regional variations
    northeastern_north_text = "เจ้าสิไปไสมาสิบ่บอก บักหล่า สิไปเด้อ"
    print(f"\nNortheastern Thai (Northern Isan style): {northeastern_north_text}")
    northeastern_result = processor.detect_dialect(northeastern_north_text)
    primary_dialect = max(northeastern_result, key=lambda k: northeastern_result[k])
    print(f"Primary dialect: {primary_dialect} (confidence: {northeastern_result[primary_dialect]:.2f})")
    
    # Detect regional variation
    regional_result = processor.detect_regional_dialect(northeastern_north_text, primary_dialect)
    print("Regional variations:")
    for region, score in regional_result.items():
        if score > 0.01:
            print(f"- {region}: {score:.2f}")
    
    # Southern Thai regional variations
    southern_phuket_text = "กินข้าวมั่งนุ้ย ใต้ๆ หรอยมาก"
    print(f"\nSouthern Thai (Phuket style): {southern_phuket_text}")
    southern_result = processor.detect_dialect(southern_phuket_text)
    primary_dialect = max(southern_result, key=lambda k: southern_result[k])
    print(f"Primary dialect: {primary_dialect} (confidence: {southern_result[primary_dialect]:.2f})")
    
    # Detect regional variation
    regional_result = processor.detect_regional_dialect(southern_phuket_text, primary_dialect)
    print("Regional variations:")
    for region, score in regional_result.items():
        if score > 0.01:
            print(f"- {region}: {score:.2f}")

def demonstrate_dialect_translation():
    """Demonstrate translation between dialects and standard Thai"""
    print("\n=== Dialect Translation ===")
    
    processor = ThaiDialectProcessor()
    
    # 1. Northern Thai to Standard Thai
    northern_text = "กิ๋นข้าวแล้วกา อั๋นจะไป๋ตลาด"
    std_text = processor.translate_to_standard(northern_text, "northern")
    print(f"\nNorthern Thai: {northern_text}")
    print(f"Standard Thai: {std_text}")
    
    # 2. Northeastern Thai to Standard Thai
    northeastern_text = "กินเข้าแล้วบ่ ข้อยสิไปตลาด"
    std_text = processor.translate_to_standard(northeastern_text, "northeastern")
    print(f"\nNortheastern Thai: {northeastern_text}")
    print(f"Standard Thai: {std_text}")
    
    # 3. Southern Thai to Standard Thai
    southern_text = "กินข้าวแล้วหรือหนิ ฉานจะไปตลาด"
    std_text = processor.translate_to_standard(southern_text, "southern")
    print(f"\nSouthern Thai: {southern_text}")
    print(f"Standard Thai: {std_text}")
    
    # 4. Pattani Malay to Standard Thai
    pattani_text = "มากัน แล็ฮ กือ บือลูม อาเกาะ นะ เปอกี ปาซะ"
    std_text = processor.translate_to_standard(pattani_text, "pattani_malay")
    print(f"\nPattani Malay: {pattani_text}")
    print(f"Standard Thai: {std_text}")
    
    # 5. Standard Thai to Northern Thai
    standard_text = "กินข้าวหรือยัง ผมจะไปตลาด"
    northern_text = processor.translate_from_standard(standard_text, "northern")
    print(f"\nStandard Thai: {standard_text}")
    print(f"Northern Thai: {northern_text}")
    
    # 6. Standard Thai to Northeastern Thai
    northeastern_text = processor.translate_from_standard(standard_text, "northeastern")
    print(f"Northeastern Thai: {northeastern_text}")
    
    # 7. Standard Thai to Southern Thai
    southern_text = processor.translate_from_standard(standard_text, "southern")
    print(f"Southern Thai: {southern_text}")
    
    # 8. Standard Thai to Pattani Malay
    pattani_text = processor.translate_from_standard(standard_text, "pattani_malay")
    print(f"Pattani Malay: {pattani_text}")

def demonstrate_dialect_tokenization():
    """Demonstrate dialect-aware tokenization"""
    print("\n=== Dialect-Aware Tokenization ===")
    
    # Create dialect tokenizer
    tokenizer = DialectTokenizer()
    
    # Test texts
    texts = {
        "northern": "อั๋นจะไป๋ตลาดเน้อ กิ๋นข้าวแล้วกา",
        "northeastern": "ข้อยสิไปตลาด กินเข้าแล้วบ่",
        "southern": "ฉานจะไปตลาด กินข้าวแล้วหรือหนิ",
        "central": "ผมจะไปตลาด กินข้าวหรือยัง",
        "pattani_malay": "อาเกาะ นะ เปอกี ปาซะ มากัน แล็ฮ กือ บือลูม"
    }
    
    for dialect, text in texts.items():
        print(f"\n{dialect.capitalize()} text: {text}")
        
        # Tokenize with explicit dialect
        tokens = tokenizer.tokenize(text, dialect=dialect)
        print(f"Tokens ({dialect}): {tokens}")
        
        # Auto-detect dialect and tokenize
        result = tokenizer.detect_and_tokenize(text)
        print(f"Auto-detected dialect: {result['dialect']} (confidence: {result['dialect_confidence']:.2f})")
        print(f"Tokens (auto): {result['tokens']}")

def demonstrate_dialect_features():
    """Demonstrate dialect feature analysis"""
    print("\n=== Dialect Features ===")
    
    processor = ThaiDialectProcessor()
    
    for dialect in ["northern", "northeastern", "southern", "central", "pattani_malay"]:
        info = processor.get_dialect_info(dialect)
        features = processor.get_dialect_features(dialect)
        examples = processor.get_example_phrases(dialect)
        
        print(f"\n{info['name']} ({info['thai_name']})")
        print(f"ISO code: {info['code']}")
        
        if "regions" in info:
            print(f"Main regions: {', '.join(info['regions'][:3])}...")
        
        # Show some distinctive features
        print("\nDistinctive features:")
        if "particles" in features:
            print(f"- Particles: {', '.join(features['particles'][:5])}...")
        if "pronouns" in features:
            print(f"- Pronouns: {', '.join(features['pronouns'][:5])}...")
        if "tones" in features:
            print(f"- Tones: {', '.join(features['tones'])}")
        if "script" in features:
            print(f"- Script: {', '.join(features['script'])}")
        if "influences" in features:
            print(f"- Influences: {', '.join(features['influences'])}")
        
        # Show example vocabulary
        if "vocabulary" in features:
            print("\nSample vocabulary (Standard → Dialect):")
            sample_vocab = list(features["vocabulary"].items())[:5]
            max_len = max(len(std) for std, _ in sample_vocab)
            for std_word, dialect_word in sample_vocab:
                print(f"- {std_word.ljust(max_len)} → {dialect_word}")
        
        # Show example phrases
        if examples:
            print("\nExample phrases (Dialect → Standard):")
            for dialect_phrase, std_phrase in examples[:3]:
                print(f"- {dialect_phrase} → {std_phrase}")
        
        print("-" * 50)

def demonstrate_regional_dialect_examples():
    """Demonstrate regional dialect variations with examples"""
    print("\n=== Regional Dialect Variations ===")
    
    processor = ThaiDialectProcessor()
    
    regions = {
        "northern": ["เชียงใหม่-ลำพูน", "เชียงราย-พะเยา-ลำปาง", "น่าน-แพร่"],
        "northeastern": ["อีสานเหนือ", "อีสานกลาง", "อีสานใต้"],
        "southern": ["upper_south", "middle_south", "lower_south", "phuket_trang"]
    }
    
    for dialect, region_list in regions.items():
        print(f"\n{processor.get_dialect_info(dialect)['name']} regional variations:")
        
        for region in region_list:
            variation_details = processor.dialect_variations[dialect][region]
            examples = processor.get_regional_dialect_examples(dialect, region)
            
            print(f"\n  {region}:")
            print(f"  Description: {variation_details['description']}")
            print(f"  Distinctive words: {', '.join(variation_details['distinctive_words'][:3])}...")
            
            if examples:
                print("  Examples:")
                for regional_phrase, std_phrase in examples:
                    print(f"  - {regional_phrase} → {std_phrase}")
            
            print()

def speech_synthesis_with_dialect():
    """Demonstrate speech synthesis with dialect support"""
    try:
        from thainlp.speech import ThaiTTS
        
        print("\n=== Text-to-Speech with Dialect Support ===")
        print("Note: This feature requires additional speech models")
        
        try:
            tts = ThaiTTS()
            text = "สวัสดีครับ ยินดีต้อนรับสู่ระบบไทยเอ็นแอลพี"
            
            # Standard Thai
            print(f"\nGenerating speech for: {text}")
            print("- Standard Thai: Use default TTS parameters")
            
            # Northern dialect adaptation
            print("- Northern Thai: Adjust TTS parameters for northern dialect")
            # In a real implementation, you'd adjust speech parameters here
            
            # Northeastern dialect adaptation
            print("- Northeastern Thai: Adjust TTS parameters for northeastern dialect")
            # In a real implementation, you'd adjust speech parameters here
            
            # Southern dialect adaptation
            print("- Southern Thai: Adjust TTS parameters for southern dialect")
            # In a real implementation, you'd adjust speech parameters here
            
            # Pattani Malay adaptation
            print("- Pattani Malay: Adjust TTS parameters for Pattani Malay accent")
            # In a real implementation, you'd adjust speech parameters here
            
        except ValueError as e:
            print(f"\nError loading TTS model: {e}")
            print("To use this feature, make sure the required TTS models are installed.")
            print("Please check the documentation for proper model installation.")
        
    except ImportError:
        print("\n=== Text-to-Speech with Dialect Support ===")
        print("Speech module is not available. Install with: pip install chujaithai[speech]")

def main():
    """Run all demonstrations"""
    print("Thai Dialect Processing Examples")
    print("===============================")
    
    demonstrate_dialect_detection()
    demonstrate_regional_dialect_detection()
    demonstrate_dialect_translation()
    demonstrate_dialect_tokenization()
    demonstrate_dialect_features()
    demonstrate_regional_dialect_examples()
    speech_synthesis_with_dialect()

if __name__ == "__main__":
    main()