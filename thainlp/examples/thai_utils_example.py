"""
Examples of using Thai language utilities
"""
from thainlp.utils import thai_utils

def demonstrate_character_utils():
    """Show usage of character-related utilities"""
    print("=== Character Utilities ===")
    
    # Check Thai characters
    text = "สวัสดี Hello ๑๒๓"
    print(f"\nAnalyzing text: {text}")
    
    # Get character counts
    counts = thai_utils.count_thai_characters(text)
    print("\nCharacter counts:")
    for char_type, count in counts.items():
        print(f"- {char_type}: {count}")
        
    # Get character types
    types = thai_utils.get_thai_character_types()
    print("\nCharacter type ranges:")
    for type_name, chars in types.items():
        print(f"- {type_name}: {chars}")

def demonstrate_word_analysis():
    """Show usage of word analysis utilities"""
    print("\n=== Word Analysis ===")
    
    # Check valid Thai words
    words = ["สวัสดี", "ประเทศ", "ก า", "Thai", "ไทย123"]
    print("\nChecking Thai words:")
    for word in words:
        valid = thai_utils.is_valid_thai_word(word)
        print(f"- '{word}': {'Valid' if valid else 'Invalid'} Thai word")
        
    # Get syllable pattern
    pattern = thai_utils.get_thai_syllable_pattern()
    print(f"\nThai syllable pattern: {pattern}")

def demonstrate_text_processing():
    """Show usage of text processing utilities"""
    print("\n=== Text Processing ===")
    
    # Text normalization
    text = "  สวัสดี   ครับ  ๑๒๓  "
    normalized = thai_utils.normalize_text(text)
    print(f"\nNormalizing: '{text}'")
    print(f"Result: '{normalized}'")
    
    # Language detection
    texts = [
        "สวัสดีครับ นี่เป็นข้อความภาษาไทย",
        "This is English text",
        "สวัสดี This is mixed ๑๒๓"
    ]
    print("\nLanguage detection:")
    for t in texts:
        ratios = thai_utils.detect_language(t)
        print(f"\nText: {t}")
        for script, ratio in ratios.items():
            if ratio > 0:
                print(f"- {script}: {ratio:.2%}")

def demonstrate_sentence_splitting():
    """Show usage of sentence splitting"""
    print("\n=== Sentence Splitting ===")
    
    texts = [
        "สวัสดีครับ วันนี้อากาศดีนะครับ ไปไหนมาครับ",
        "ดร. สมศักดิ์ทำงานที่ กทม. เมื่อวานนี้ และวันนี้ก็มาอีก",
        "เธอพูดว่า 'ฉันจะไป' แล้วก็ไปจริงๆ"
    ]
    
    print("\nSplitting sentences:")
    for text in texts:
        print(f"\nInput: {text}")
        sentences = thai_utils.split_thai_sentences(text)
        print("Results:")
        for i, sent in enumerate(sentences, 1):
            print(f"{i}. {sent}")

def demonstrate_number_conversion():
    """Show usage of number conversion utilities"""
    print("\n=== Number Conversion ===")
    
    # Convert numbers to Thai text
    numbers = [0, 12, 21, 100, 101, 1234, 1.5, -42]
    print("\nConverting numbers to Thai text:")
    for num in numbers:
        text = thai_utils.thai_number_to_text(num)
        print(f"- {num} -> {text}")
        
    # Convert between Thai and Arabic digits
    thai_nums = "๑๒๓๔๕"
    arabic = thai_utils.thai_digit_to_arabic_digit(thai_nums)
    back_to_thai = thai_utils.arabic_digit_to_thai_digit(arabic)
    print(f"\nThai digits: {thai_nums}")
    print(f"To Arabic: {arabic}")
    print(f"Back to Thai: {back_to_thai}")

def main():
    """Run all demonstrations"""
    print("Thai Language Utilities Examples")
    print("===============================")
    
    demonstrate_character_utils()
    demonstrate_word_analysis()
    demonstrate_text_processing()
    demonstrate_sentence_splitting()
    demonstrate_number_conversion()

if __name__ == "__main__":
    main()
