# Thai Language Utilities

Module `thainlp.utils.thai_utils` provides comprehensive utilities for Thai language processing.

## Character & Word Functions

### Character Detection

```python
from thainlp.utils import thai_utils

# Check single Thai character
is_thai = thai_utils.is_thai_char("ก")  # True

# Check if word contains Thai characters
has_thai = thai_utils.is_thai_word("Hello ไทย")  # True
```

### Character Types

```python
# Get character type ranges
types = thai_utils.get_thai_character_types()
# Returns:
# {
#   'consonants': 'ก-ฮ',
#   'vowels': 'ะัาำิีึืุูเแโใไๅ',
#   'tonemarks': '่้๊๋',
#   'digits': '๐-๙',
#   'all': '-'
# }

# Count character types
counts = thai_utils.count_thai_characters("สวัสดี ๑๒๓")
# Returns:
# {
#   'consonants': 4,
#   'vowels': 1,
#   'tonemarks': 1,
#   'digits': 3
# }
```

### Word Validation

```python
# Check if word follows valid Thai syllable patterns
valid = thai_utils.is_valid_thai_word("สวัสดี")  # True
valid = thai_utils.is_valid_thai_word("ก า")  # False (space in middle)

# Get Thai syllable pattern
pattern = thai_utils.get_thai_syllable_pattern()
```

## Text Processing

### Text Normalization

```python
# Normalize text (spaces, line endings, invisible chars)
text = thai_utils.normalize_text("  สวัสดี   ครับ  \r\n")  # "สวัสดี ครับ"

# Remove tone marks
text = thai_utils.remove_tone_marks("สวัสดี")  # "สวสด"

# Remove all diacritics
text = thai_utils.remove_diacritics("กรุ๊ปกฤษณ์")  # "กรปกฤษณ"
```

### Language Detection

```python
# Detect script usage ratios
ratios = thai_utils.detect_language("สวัสดี Hello ๑๒๓")
# Returns:
# {
#   'thai': 0.4,      # Thai script ratio
#   'latin': 0.3333,  # Latin script ratio
#   'chinese': 0,     # Chinese characters ratio
#   'japanese': 0,    # Japanese characters ratio
#   'korean': 0,      # Korean characters ratio
#   'other': 0.2667   # Other scripts ratio
# }
```

### Sentence Splitting

```python
# Split Thai text into sentences with smart rules
sentences = thai_utils.split_thai_sentences("สวัสดีครับ วันนี้อากาศดีนะครับ")
# Returns: ["สวัสดีครับ", "วันนี้อากาศดีนะครับ"]

# Handles abbreviations:
text = "ดร. สมศักดิ์ทำงานที่ กทม."
sentences = thai_utils.split_thai_sentences(text)  # Returns single sentence

# Handles conjunctions:
text = "เขาไปตลาด และซื้อผลไม้มา"
sentences = thai_utils.split_thai_sentences(text)  # Returns single sentence
```

## Number Processing

### Number to Thai Text

```python
# Basic numbers
thai_utils.thai_number_to_text(5)      # "ห้า"
thai_utils.thai_number_to_text(21)     # "ยี่สิบเอ็ด"
thai_utils.thai_number_to_text(100)    # "หนึ่งร้อย"

# Special cases
thai_utils.thai_number_to_text(0)      # "ศูนย์"
thai_utils.thai_number_to_text(-42)    # "ลบสี่สิบสอง"
thai_utils.thai_number_to_text(1.5)    # "หนึ่งจุดห้า"
```

### Digit Conversion

```python
# Thai to Arabic digits
thai_utils.thai_digit_to_arabic_digit("๑๒๓")  # "123"

# Arabic to Thai digits
thai_utils.arabic_digit_to_thai_digit("123")   # "๑๒๓"
```

## Romanization

```python
# Basic romanization of Thai text
thai_utils.thai_to_roman("สวัสดี")  # "sawatdi"
thai_utils.thai_to_roman("กรุงเทพ")  # "krungthep"
```

## Complete Example

See `thainlp/examples/thai_utils_example.py` for a comprehensive demonstration of all features:

```python
from thainlp.utils import thai_utils

# Character analysis
text = "สวัสดี Hello ๑๒๓"
counts = thai_utils.count_thai_characters(text)
print(f"Character counts: {counts}")

# Word validation
words = ["สวัสดี", "ก า", "Thai"]
for word in words:
    valid = thai_utils.is_valid_thai_word(word)
    print(f"'{word}': {'Valid' if valid else 'Invalid'} Thai word")

# Text processing
text = "  สวัสดี   ครับ  ๑๒๓  "
normalized = thai_utils.normalize_text(text)
print(f"Normalized: '{normalized}'")

# Language detection
text = "สวัสดี This is mixed ๑๒๓"
ratios = thai_utils.detect_language(text)
for script, ratio in ratios.items():
    if ratio > 0:
        print(f"{script}: {ratio:.2%}")

# Number conversion
num = 1234.5
thai_text = thai_utils.thai_number_to_text(num)
print(f"{num} -> {thai_text}")
```

## Best Practices

1. Always normalize text before processing:

   ```python
   text = thai_utils.normalize_text(raw_text)
   ```

2. Use appropriate character type functions:

   ```python
   # For single characters
   if thai_utils.is_thai_char(char):
       # Process Thai character

   # For words/text
   if thai_utils.is_thai_word(text):
       # Process Thai text
   ```

3. Handle script mixing appropriately:

   ```python
   ratios = thai_utils.detect_language(text)
   if ratios['thai'] > 0.8:  # Predominantly Thai
       # Process as Thai text
   ```

4. Consider sentence boundaries:

   ```python
   sentences = thai_utils.split_thai_sentences(text)
   for sentence in sentences:
       # Process each sentence
   ```

## Performance Tips

1. Cache syllable patterns for repeated word validation:

   ```python
   pattern = thai_utils.get_thai_syllable_pattern()
   # Reuse pattern for multiple validations
   ```

2. Pre-compile regular expressions for performance:

   ```python
   import re
   pattern = re.compile(thai_utils.get_thai_syllable_pattern())
   ```

3. Use batch processing when possible:

   ```python
   words = text.split()
   valid_words = [w for w in words if thai_utils.is_valid_thai_word(w)]
