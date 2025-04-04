"""
Example usage of the Thai NLP pipeline
"""
from typing import Dict, Any
from ..pipelines.thai_nlp_pipeline import ThaiNLPPipeline

def main():
    # Initialize pipeline with all components
    pipeline = ThaiNLPPipeline(
        components=[
            'classification',
            'qa',
            'translation',
            'generation',
            'fill_mask',
            'summarization',
            'similarity'
        ],
        device='cuda',
        batch_size=32
    )
    
    # Example 1: Basic text analysis
    text = """
    กรุงเทพมหานครเป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย 
    เป็นศูนย์กลางการปกครอง การศึกษา การคมนาคมขนส่ง การเงินการธนาคาร การพาณิชย์ 
    การสื่อสาร และความเจริญของประเทศ
    """
    
    print("Example 1: Text Analysis")
    print("-" * 50)
    
    results = pipeline.analyze(
        text,
        tasks=['tokens', 'entities', 'translation', 'summary']
    )
    
    print("Entities:", [t['token'] for t in results['entities']])
    print("\nEnglish Translation:", results['translation']['en'])
    print("\nSummary:", results['summary'])
    print("\n")
    
    # Example 2: Question Answering
    context = """
    ประเทศไทยมีประชากรประมาณ 70 ล้านคน มีกรุงเทพมหานครเป็นเมืองหลวง 
    ภาษาราชการคือภาษาไทย สกุลเงินที่ใช้คือบาท 
    ประเทศไทยมีการปกครองระบอบประชาธิปไตยอันมีพระมหากษัตริย์ทรงเป็นประมุข
    """
    
    questions = [
        "ประเทศไทยมีประชากรเท่าไร",
        "เมืองหลวงของประเทศไทยคือที่ไหน",
        "ประเทศไทยใช้สกุลเงินอะไร"
    ]
    
    print("Example 2: Question Answering")
    print("-" * 50)
    
    for question in questions:
        answer = pipeline.answer_question(
            question=question,
            context=context,
            return_scores=True
        )
        print(f"Q: {question}")
        print(f"A: {answer['answer']} (confidence: {answer['score']:.2f})\n")
        
    # Example 3: Text Generation
    print("Example 3: Text Generation")
    print("-" * 50)
    
    prompt = "เทคโนโลยีปัญญาประดิษฐ์ในปัจจุบัน"
    generated = pipeline.generate_text(
        prompt=prompt,
        max_length=100,
        num_return_sequences=2,
        temperature=0.7
    )
    
    print("Prompt:", prompt)
    print("\nGenerated Text 1:", generated[0])
    print("Generated Text 2:", generated[1])
    print("\n")
    
    # Example 4: Fill-Mask
    print("Example 4: Fill-Mask")
    print("-" * 50)
    
    masked_text = "ประเทศไทยมี[MASK]ที่หลากหลาย"
    predictions = pipeline.fill_mask(
        text=masked_text,
        top_k=3
    )
    
    print("Masked Text:", masked_text)
    print("\nTop Predictions:")
    for pred in predictions:
        print(f"- {pred['token']} (score: {pred['score']:.2f})")
    print("\n")
    
    # Example 5: Text Similarity
    print("Example 5: Text Similarity")
    print("-" * 50)
    
    text1 = "ร้านอาหารนี้อร่อยมาก"
    text2 = "อาหารที่ร้านนี้รสชาติดีมาก"
    text3 = "วันนี้อากาศร้อนมาก"
    
    similarity = pipeline.calculate_similarity(text1, text2)
    print(f"Similarity between:\n'{text1}' and\n'{text2}'")
    print(f"Score: {similarity:.2f}\n")
    
    similarity = pipeline.calculate_similarity(text1, text3)
    print(f"Similarity between:\n'{text1}' and\n'{text3}'")
    print(f"Score: {similarity:.2f}\n")
    
    # Example 6: Document Search
    print("Example 6: Document Search")
    print("-" * 50)
    
    documents = [
        "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
        "เชียงใหม่เป็นจังหวัดทางภาคเหนือของประเทศไทย",
        "ภูเก็ตเป็นจังหวัดท่องเที่ยวทางภาคใต้",
        "อยุธยาเคยเป็นราชธานีเก่าของไทย",
    ]
    
    query = "เมืองหลวงของประเทศไทย"
    similar_docs = pipeline.find_similar_texts(
        query=query,
        candidates=documents,
        top_k=2
    )
    
    print("Query:", query)
    print("\nMost Similar Documents:")
    for doc, score in similar_docs:
        print(f"- {doc} (score: {score:.2f})")
    print("\n")
    
    # Example 7: Cross-Lingual Capabilities
    print("Example 7: Cross-Lingual Capabilities")
    print("-" * 50)
    
    english_text = """
    Thailand is a Southeast Asian country known for its beautiful beaches,
    ornate temples, and rich culture. The capital city is Bangkok.
    """
    
    # Translate to Thai
    thai_translation = pipeline.translate(
        text=english_text,
        source_lang='en',
        target_lang='th'
    )
    print("English Text:", english_text)
    print("\nThai Translation:", thai_translation)
    
    # Generate Thai summary
    thai_summary = pipeline.summarize(
        text=thai_translation,
        ratio=0.5
    )
    print("\nThai Summary:", thai_summary)
    
    # Translate summary back to English
    english_summary = pipeline.translate(
        text=thai_summary,
        source_lang='th',
        target_lang='en'
    )
    print("\nEnglish Summary:", english_summary)

if __name__ == "__main__":
    main()
