"""
Advanced usage examples for ThaiNLP
"""
from typing import List
from ..tokenization import word_tokenize
from ..tag import pos_tag
from ..thai_spell_correction import ThaiSpellChecker
from ..summarization.summarizer import ThaiSummarizer
from ..generation.text_generator import TextGenerator
from ..similarity.sentence_similarity import SentenceSimilarity
from ..qa.question_answering import QuestionAnswering
from ..translation.translator import Translator

def tokenization_examples():
    """Examples of advanced tokenization"""
    text = "สวัสดีครับ นี่คือตัวอย่างการตัดคำภาษาไทย"
    
    # Basic tokenization
    tokens = word_tokenize(text)
    print(f"Basic tokenization: {tokens}")
    
    # With POS tagging
    pos_tags = pos_tag(text)
    print(f"POS tagging: {pos_tags}")

def spell_check_examples():
    """Examples of spell checking"""
    checker = ThaiSpellChecker()
    
    # Check individual word
    word = "สวัสดร"  # Misspelled
    suggestions = checker.suggest(word)
    print(f"Spell check suggestions for '{word}': {suggestions}")

def summarization_examples():
    """Examples of text summarization"""
    summarizer = ThaiSummarizer()
    
    text = """
    ประเทศไทยตั้งอยู่ในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีพื้นที่ประมาณ 513,000 ตารางกิโลเมตร 
    มีประชากรประมาณ 70 ล้านคน มีกรุงเทพมหานครเป็นเมืองหลวง ประเทศไทยมีภาษาไทยเป็นภาษาราชการ 
    และมีพุทธศาสนาเป็นศาสนาประจำชาติ มีการปกครองระบอบประชาธิปไตยอันมีพระมหากษัตริย์ทรงเป็นประมุข
    """
    
    summary = summarizer.summarize(text, ratio=0.3)
    print(f"Text summary: {summary}")

def generation_examples():
    """Examples of text generation"""
    generator = TextGenerator()
    
    prompt = "วันนี้อากาศ"
    generations = generator.generate(prompt, max_length=50, num_return_sequences=3)
    print(f"Generated continuations for '{prompt}':")
    for gen in generations:
        print(f"- {gen}")

def similarity_examples():
    """Examples of text similarity"""
    sim = SentenceSimilarity()
    
    text1 = "วันนี้อากาศดีมาก"
    text2 = "อากาศวันนี้ดีจัง"
    
    score = sim.get_similarity(text1, text2)
    print(f"Similarity between '{text1}' and '{text2}': {score:.3f}")

def qa_examples():
    """Examples of question answering"""
    qa = QuestionAnswering()
    
    context = """
    ประเทศไทยมีประชากรประมาณ 70 ล้านคน มีกรุงเทพมหานครเป็นเมืองหลวง 
    ภาษาไทยเป็นภาษาราชการ และมีพุทธศาสนาเป็นศาสนาประจำชาติ
    """
    question = "เมืองหลวงของประเทศไทยคือที่ไหน?"
    
    answer = qa.answer_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

def translation_examples():
    """Examples of translation"""
    translator = Translator()
    
    text = "สวัสดีครับ สบายดีไหม"
    translation = translator.translate(text, source_lang='th', target_lang='en')
    print(f"Translation of '{text}': {translation}")

def run_all_examples():
    """Run all advanced examples"""
    print("\n=== Tokenization Examples ===")
    tokenization_examples()
    
    print("\n=== Spell Check Examples ===")
    spell_check_examples()
    
    print("\n=== Summarization Examples ===")
    summarization_examples()
    
    print("\n=== Generation Examples ===")
    generation_examples()
    
    print("\n=== Similarity Examples ===")
    similarity_examples()
    
    print("\n=== Question Answering Examples ===")
    qa_examples()
    
    print("\n=== Translation Examples ===")
    translation_examples()

if __name__ == "__main__":
    run_all_examples()
