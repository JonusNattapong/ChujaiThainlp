"""
Examples of using the simplified Thai text processing interface
"""
from thainlp.utils.thai_text import (
    process_text,
    ThaiTextProcessor,
    ThaiValidator,
    normalize_text,
    romanize,
    extract_thai,
    split_sentences
)

def demonstrate_quick_functions():
    """Show usage of pre-built quick functions"""
    print("=== Quick Functions ===")
    
    text = "  สวัสดี   Hello  ๑๒๓  "
    print(f"\nOriginal text: '{text}'")
    
    # Using pre-built functions
    print(f"Normalized: '{normalize_text(text)}'")
    print(f"Romanized: '{romanize(text)}'")
    print(f"Thai only: '{extract_thai(text)}'")
    
    # Split sentences
    text = "สวัสดีครับ วันนี้อากาศดีนะครับ ไปไหนมาครับ"
    sentences = split_sentences(text)
    print("\nSplit sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"{i}. {sent}")

def demonstrate_method_chaining():
    """Show usage of method chaining with ThaiTextProcessor"""
    print("\n=== Method Chaining ===")
    
    # Process text with multiple operations
    text = "สวัสดี ๑๒๓ Hello! ภาษาไทย"
    result = (process_text(text)
             .normalize()
             .extract_thai()
             .to_roman()
             .get_text())
    print(f"\nMulti-step processing: '{result}'")
    
    # Get script ratios
    text = "Hello สวัสดี 123 ๑๒๓"
    ratios = (process_text(text)
             .normalize()
             .get_script_ratios())
    print("\nScript ratios:")
    for script, ratio in ratios.items():
        if ratio > 0:
            print(f"- {script}: {ratio:.2%}")
            
    # Process numbers
    number = 1234.5
    text = ThaiTextProcessor.number_to_thai(number)
    print(f"\nNumber {number} in Thai: {text}")

def demonstrate_text_analysis():
    """Show usage of text analysis features"""
    print("\n=== Text Analysis ===")
    
    text = "สวัสดีครับ ๑๒๓"
    processor = ThaiTextProcessor()
    processor.load(text)
    
    # Get character counts
    counts = processor.get_character_counts()
    print("\nCharacter counts:")
    for char_type, count in counts.items():
        print(f"- {char_type}: {count}")
        
    # Check for Thai text
    print(f"\nContains Thai: {processor.has_thai()}")

def demonstrate_validation():
    """Show usage of Thai text validation"""
    print("\n=== Text Validation ===")
    
    validator = ThaiValidator()
    
    # Check characters
    chars = ["ก", "a", "๑", " "]
    print("\nChecking characters:")
    for char in chars:
        result = "Thai" if validator.is_thai_char(char) else "Not Thai"
        print(f"- '{char}': {result}")
        
    # Validate words
    words = ["สวัสดี", "ก า", "hello", "ไทย123"]
    print("\nValidating words:")
    for word in words:
        result = "Valid" if validator.is_valid_word(word) else "Invalid"
        print(f"- '{word}': {result}")
        
    # Get character type info
    types = validator.get_character_types()
    print("\nThai character types:")
    for type_name, chars in types.items():
        print(f"- {type_name}: {chars}")

def main():
    """Run all demonstrations"""
    print("Thai Text Processing Examples")
    print("============================")
    
    demonstrate_quick_functions()
    demonstrate_method_chaining()
    demonstrate_text_analysis()
    demonstrate_validation()

if __name__ == "__main__":
    main()
