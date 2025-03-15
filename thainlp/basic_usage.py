"""
Basic usage examples for ThaiNLP library
"""

def tokenization_example():
    """Tokenization example"""
    from thainlp import tokenize
    
    text = "ผมชอบกินข้าวที่ร้านอาหารไทย"
    tokens = tokenize(text)
    print(tokens)  # ['ผม', 'ชอบ', 'กิน', 'ข้าว', 'ที่', 'ร้าน', 'อาหาร', 'ไทย']

def sentiment_example():
    """Sentiment analysis example"""
    from thainlp import analyze_sentiment
    
    text = "วันนี้อากาศดีมากๆ ฉันมีความสุขมาก"
    score, label = analyze_sentiment(text)
    print(f"Sentiment: {label} (score: {score})")  # Sentiment: positive (score: 0.75)

def ner_example():
    """Named Entity Recognition example"""
    from thainlp import extract_entities
    
    text = "คุณสมชายอาศัยอยู่ที่กรุงเทพมหานคร"
    entities = extract_entities(text)
    print(entities)  # [('PERSON', 'คุณสมชาย'), ('LOCATION', 'กรุงเทพมหานคร')]

def fill_mask_example():
    """Fill-Mask example"""
    from thainlp.models.fill_mask import ThaiFillMask
    
    # Initialize model
    fill_mask = ThaiFillMask()
    
    # Simple mask filling
    text = "ผมชอบกิน[MASK]ที่ร้านอาหารไทย"
    results = fill_mask.fill_mask(text, top_k=3)
    print("Fill-Mask results:")
    for preds in results:
        for pred in preds:
            print(f"Text: {pred['text']}, Score: {pred.get('score', 'N/A')}")
    
    # Generate masked text
    text = "ประเทศไทยมีประชากรประมาณ 70 ล้านคน"
    masked = fill_mask.generate_masks(text, mask_ratio=0.2)
    print(f"\nGenerated masked text: {masked}")
    
    # Batch processing
    texts = [
        "วันนี้[MASK]ดีมาก",
        "ผม[MASK]ไปโรงเรียน"
    ]
    results = fill_mask.fill_mask_batch(texts, top_k=2)
    print("\nBatch Fill-Mask results:")
    for text, preds in zip(texts, results):
        print(f"\nInput: {text}")
        for mask_preds in preds:
            for pred in mask_preds:
                print(f"Text: {pred['text']}, Score: {pred.get('score', 'N/A')}")

def sentence_similarity_example():
    """Sentence Similarity example"""
    from thainlp.similarity.sentence_similarity import ThaiSentenceSimilarity
    
    # Initialize similarity model
    sim_model = ThaiSentenceSimilarity()
    
    # Compare two sentences
    text1 = "ผมชอบกินข้าวผัด"
    text2 = "ฉันชอบทานข้าวผัด"
    similarity = sim_model.compute_similarity(text1, text2)
    print(f"\nSimilarity score: {similarity:.2f}")
    
    # Find similar sentences
    query = "อาหารไทยอร่อย"
    candidates = [
        "อาหารไทยรสชาติดี",
        "ประเทศไทยสวยงาม",
        "อาหารญี่ปุ่นอร่อย"
    ]
    results = sim_model.find_most_similar(query, candidates, top_k=2)
    print("\nMost similar sentences:")
    for result in results:
        print(f"Text: {result['text']}, Score: {result['similarity']:.2f}")
    
    # Create similarity matrix
    texts = [
        "ผมชอบกินข้าวผัด",
        "ฉันชอบทานข้าวผัด",
        "อาหารไทยอร่อยมาก"
    ]
    matrix = sim_model.compute_similarity_matrix(texts)
    print("\nSimilarity matrix:")
    print(matrix)
    
    # Cluster similar texts
    texts = [
        "ผมชอบกินข้าวผัด",
        "ฉันชอบทานข้าวผัด",
        "อาหารไทยอร่อยมาก",
        "ประเทศไทยสวยงาม",
        "เมืองไทยมีสถานที่ท่องเที่ยวสวยๆ"
    ]
    clusters = sim_model.cluster_texts(texts, n_clusters=2)
    print("\nText clusters:")
    for label, cluster_texts in clusters['clusters'].items():
        print(f"\nCluster {label}:")
        for text in cluster_texts:
            print(f"- {text}")

def main():
    """Run all examples"""
    print("\n=== Tokenization Example ===")
    tokenization_example()
    
    print("\n=== Sentiment Analysis Example ===")
    sentiment_example()
    
    print("\n=== Named Entity Recognition Example ===")
    ner_example()
    
    print("\n=== Fill-Mask Example ===")
    fill_mask_example()
    
    print("\n=== Sentence Similarity Example ===")
    sentence_similarity_example()

if __name__ == "__main__":
    main()