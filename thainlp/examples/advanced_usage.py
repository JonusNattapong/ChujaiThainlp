"""
Advanced Usage Examples for ThaiNLP Library
"""

import thainlp
import datetime

def classification_examples():
    """Examples of text and token classification"""
    print("\n=== Text Classification Examples ===")
    
    # Text classification
    text = "ร้านอาหารนี้อร่อยมาก บรรยากาศดี แนะนำให้มาลอง"
    result = thainlp.classify_text(text, category="sentiment")
    print(f"Sentiment classification: {result}")
    
    # Zero-shot classification
    text = "วันนี้ท้องฟ้าสวยมาก มีแดดแต่ไม่ร้อนเกินไป อากาศดีเหมาะแก่การออกไปเดินเล่น"
    labels = ["สภาพอากาศ", "อาหาร", "การท่องเที่ยว", "กีฬา"]
    result = thainlp.zero_shot_classification(text, labels)
    print(f"Zero-shot classification: {result}")
    
    # Token classification
    tokens = ["ผม", "ชอบ", "กิน", "ข้าวผัด", "ที่", "ร้าน", "อาหาร", "ไทย"]
    result = thainlp.classify_tokens(tokens, task="pos")
    print(f"POS tagging: {result}")
    
    # Named entity recognition
    tokens = ["นาย", "สมชาย", "เดินทาง", "ไป", "กรุงเทพ", "เมื่อวาน"]
    result = thainlp.find_entities(tokens)
    print(f"Named entities: {result}")

def question_answering_examples():
    """Examples of question answering"""
    print("\n=== Question Answering Examples ===")
    
    # Simple question answering
    context = """
    กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย ก่อตั้งเมื่อปี พ.ศ. 2325 โดยพระบาทสมเด็จพระพุทธยอดฟ้าจุฬาโลกมหาราช
    มีประชากรประมาณ 8 ล้านคน และเป็นศูนย์กลางเศรษฐกิจ การศึกษา และวัฒนธรรมของประเทศ
    """
    question = "กรุงเทพมหานครก่อตั้งเมื่อปีใด"
    result = thainlp.answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {result}")
    
    # Table question answering
    table = [
        {"ชื่อ": "สมชาย", "อายุ": "25", "อาชีพ": "วิศวกร"},
        {"ชื่อ": "สมหญิง", "อายุ": "30", "อาชีพ": "แพทย์"},
        {"ชื่อ": "สมศักดิ์", "อายุ": "45", "อาชีพ": "ครู"}
    ]
    question = "ใครอายุมากที่สุด"
    result = thainlp.answer_from_table(question, table)
    print(f"Question: {question}")
    print(f"Answer: {result}")

def translation_examples():
    """Examples of translation"""
    print("\n=== Translation Examples ===")
    
    # Thai to English translation
    thai_text = "สวัสดีครับ ผมชื่อสมชาย ผมชอบกินอาหารไทย"
    result = thainlp.translate_text(thai_text, source_lang="th", target_lang="en")
    print(f"Thai to English: {result}")
    
    # English to Thai translation
    english_text = "Hello, my name is John. I like Thai food."
    result = thainlp.translate_text(english_text, source_lang="en", target_lang="th")
    print(f"English to Thai: {result}")
    
    # Language detection
    mixed_text = "Hello สวัสดีครับ How are you?"
    result = thainlp.detect_language_translation(mixed_text)
    print(f"Detected language: {result}")

def feature_extraction_examples():
    """Examples of feature extraction"""
    print("\n=== Feature Extraction Examples ===")
    
    # Basic feature extraction
    text = "สวัสดีครับ นี่เป็นตัวอย่างข้อความภาษาไทยสำหรับการสกัดคุณลักษณะ"
    tokens = ["สวัสดีครับ", "นี่", "เป็น", "ตัวอย่าง", "ข้อความ", "ภาษาไทย", "สำหรับ", "การ", "สกัด", "คุณลักษณะ"]
    features = thainlp.extract_features(text, tokens)
    print(f"Basic features: {list(features.keys())}")
    
    # Advanced feature extraction
    pos_tags = [("สวัสดีครับ", "INTJ"), ("นี่", "PRON"), ("เป็น", "VERB"), ("ตัวอย่าง", "NOUN"), 
                ("ข้อความ", "NOUN"), ("ภาษาไทย", "NOUN"), ("สำหรับ", "ADP"), ("การ", "NOUN"), 
                ("สกัด", "VERB"), ("คุณลักษณะ", "NOUN")]
    advanced_features = thainlp.extract_advanced_features(text, tokens, pos_tags)
    print(f"Advanced features: {list(advanced_features.keys())}")
    
    # Document vector creation
    vector = thainlp.create_document_vector(features, vector_size=10)
    print(f"Document vector (size 10): {vector}")

def text_generation_examples():
    """Examples of text generation"""
    print("\n=== Text Generation Examples ===")
    
    # Template-based generation
    template_text = thainlp.generate_text(method="template", template_type="greeting")
    print(f"Template-based text: {template_text}")
    
    # Pattern-based generation
    pattern = ["PRON", "VERB", "NOUN"]
    pattern_text = thainlp.generate_text(method="pattern", pattern=pattern)
    print(f"Pattern-based text: {pattern_text}")
    
    # Paragraph generation
    paragraph = thainlp.generate_paragraph(num_sentences=3)
    print(f"Generated paragraph: {paragraph}")
    
    # Text completion
    prefix = "สวัสดีครับ วันนี้"
    completion = thainlp.complete_text(prefix, length=5)
    print(f"Text completion: {completion}")

def text_similarity_examples():
    """Examples of text similarity"""
    print("\n=== Text Similarity Examples ===")
    
    # Calculate similarity
    text1 = "ผมชอบกินอาหารไทย โดยเฉพาะต้มยำกุ้ง"
    text2 = "ฉันชอบอาหารไทยมาก ต้มยำกุ้งเป็นอาหารโปรด"
    similarity = thainlp.calculate_similarity(text1, text2, method="combined")
    print(f"Similarity between texts: {similarity}")
    
    # Find most similar
    query = "อาหารไทยรสชาติเผ็ด"
    texts = [
        "ต้มยำกุ้งเป็นอาหารไทยที่มีรสเผ็ด",
        "ข้าวผัดเป็นอาหารที่ทำง่าย",
        "แกงเขียวหวานมีรสชาติเผ็ดและหวาน",
        "ส้มตำไทยเป็นอาหารที่มีรสเปรี้ยวและเผ็ด"
    ]
    most_similar = thainlp.find_most_similar(query, texts)
    print(f"Most similar texts: {most_similar}")
    
    # Check for duplicates
    text1 = "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย"
    text2 = "กรุงเทพฯ เป็นเมืองหลวงของประเทศไทย"
    is_dup = thainlp.is_duplicate(text1, text2, threshold=0.7)
    print(f"Are texts duplicates? {is_dup}")

def pythainlp_advanced_examples():
    """Examples of advanced features using PyThaiNLP"""
    print("\n=== PyThaiNLP Advanced Features Examples ===")
    
    # Thai number to text
    number = 12345
    text = thainlp.thai_number_to_text(number)
    print(f"Number {number} in Thai text: {text}")
    
    # Thai text to number
    text = "หนึ่งหมื่นสองพันสามร้อยสี่สิบห้า"
    number = thainlp.thai_text_to_number(text)
    print(f"Thai text '{text}' to number: {number}")
    
    # Format Thai date
    today = datetime.datetime.now()
    thai_date = thainlp.format_thai_date(today, "%A %d %B %Y")
    print(f"Today in Thai format: {thai_date}")
    
    # Thai soundex
    word = "สวัสดี"
    soundex = thainlp.thai_soundex(word)
    print(f"Soundex for '{word}': {soundex}")
    
    # Spell correction
    misspelled = "สวัสดร"
    corrections = thainlp.spell_correction(misspelled)
    print(f"Spell correction for '{misspelled}': {corrections}")
    
    # Thai stopwords
    stopwords = thainlp.get_thai_stopwords()
    print(f"Thai stopwords (first 10): {list(stopwords)[:10]}")
    
    # Thai syllables
    syllables = thainlp.get_thai_syllables()
    print(f"Thai syllables (first 10): {list(syllables)[:10]}")
    
    # Thai WordNet
    word = "รัก"
    synsets = thainlp.get_thai_wordnet_synsets(word)
    print(f"WordNet synsets for '{word}': {synsets}")
    
    synonyms = thainlp.get_thai_wordnet_synonyms(word)
    print(f"Synonyms for '{word}': {synonyms}")
    
    # Thai digit conversion
    text_with_thai_digits = "๑๒๓๔๕"
    arabic_digits = thainlp.thai_digit_to_arabic_digit(text_with_thai_digits)
    print(f"Thai digits '{text_with_thai_digits}' to Arabic: {arabic_digits}")
    
    text_with_arabic_digits = "12345"
    thai_digits = thainlp.arabic_digit_to_thai_digit(text_with_arabic_digits)
    print(f"Arabic digits '{text_with_arabic_digits}' to Thai: {thai_digits}")
    
    # Thai time
    time_str = "14:30"
    thai_time_text = thainlp.thai_time(time_str)
    print(f"Time '{time_str}' in Thai: {thai_time_text}")
    
    # Thai day to datetime
    day_text = "พรุ่งนี้"
    day_datetime = thainlp.thai_day_to_datetime(day_text)
    print(f"Thai day '{day_text}' to datetime: {day_datetime}")

def tokenization_advanced_examples():
    """Examples of advanced tokenization using PyThaiNLP"""
    print("\n=== Advanced Tokenization Examples ===")
    
    from thainlp.tokenization.maximum_matching import tokenize, sentence_tokenize, subword_tokenize, create_custom_tokenizer
    
    # Different tokenization engines
    text = "ผมชอบกินข้าวที่ร้านอาหารไทยเพราะอร่อยมาก"
    
    tokens_newmm = tokenize(text, engine="pythainlp:newmm")
    print(f"Tokenization with newmm: {tokens_newmm}")
    
    tokens_longest = tokenize(text, engine="pythainlp:longest")
    print(f"Tokenization with longest matching: {tokens_longest}")
    
    tokens_attacut = tokenize(text, engine="pythainlp:attacut")
    print(f"Tokenization with attacut: {tokens_attacut}")
    
    # Sentence tokenization
    text_multi = "สวัสดีครับ วันนี้อากาศดีมาก ผมกำลังเรียนภาษาไทย คุณล่ะครับ เรียนอะไรอยู่"
    sentences = sentence_tokenize(text_multi)
    print(f"Sentence tokenization: {sentences}")
    
    # Subword tokenization
    text_word = "ความสุข"
    subwords_tcc = subword_tokenize(text_word, engine="tcc")
    print(f"TCC subword tokenization of '{text_word}': {subwords_tcc}")
    
    subwords_syllable = subword_tokenize(text_word, engine="syllable")
    print(f"Syllable tokenization of '{text_word}': {subwords_syllable}")
    
    # Custom tokenizer
    custom_dict = {"ผม", "ชอบ", "กิน", "ข้าว", "ที่", "ร้าน", "อาหาร", "ไทย", "อร่อย", "มาก"}
    custom_tokenizer = create_custom_tokenizer(custom_dict)
    custom_tokens = custom_tokenizer(text)
    print(f"Custom tokenization: {custom_tokens}")

def pos_tagging_advanced_examples():
    """Examples of advanced POS tagging using PyThaiNLP"""
    print("\n=== Advanced POS Tagging Examples ===")
    
    from thainlp.pos_tagging.hmm_tagger import pos_tag, get_pos_tag_list, convert_tag_schema
    
    # Different POS tagging engines
    text = "ผมชอบกินข้าวที่ร้านอาหารไทย"
    
    pos_perceptron = pos_tag(text, engine="pythainlp:perceptron")
    print(f"POS tagging with perceptron: {pos_perceptron}")
    
    pos_artagger = pos_tag(text, engine="pythainlp:artagger")
    print(f"POS tagging with artagger: {pos_artagger}")
    
    # Get available POS tags
    pos_tags = get_pos_tag_list()
    print(f"Available POS tags: {list(pos_tags.keys())}")
    print(f"Examples for NOUN: {pos_tags.get('NOUN', [])}")
    
    # Convert between tag schemas
    ud_tags = [("ผม", "PRON"), ("ชอบ", "VERB"), ("กิน", "VERB"), ("ข้าว", "NOUN")]
    orchid_tags = convert_tag_schema(ud_tags, source="ud", target="orchid")
    print(f"UD tags: {ud_tags}")
    print(f"Converted to ORCHID: {orchid_tags}")

def main():
    """Run all examples"""
    print("ThaiNLP Advanced Usage Examples")
    print("===============================")
    
    classification_examples()
    question_answering_examples()
    translation_examples()
    feature_extraction_examples()
    text_generation_examples()
    text_similarity_examples()
    
    # Advanced examples using PyThaiNLP
    pythainlp_advanced_examples()
    tokenization_advanced_examples()
    pos_tagging_advanced_examples()

if __name__ == "__main__":
    main() 