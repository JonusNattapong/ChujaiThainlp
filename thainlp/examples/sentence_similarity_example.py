#!/usr/bin/env python3
from thainlp.similarity import ThaiSentenceSimilarity

def main():
    """ตัวอย่างการใช้งานระบบวัดความคล้ายคลึงของประโยคภาษาไทย"""
    print("=== ตัวอย่างการใช้งานระบบวัดความคล้ายคลึงของประโยคภาษาไทย ===")
    
    similarity = ThaiSentenceSimilarity()
    
    # 1. คำนวณความคล้ายคลึงระหว่างสองประโยค
    print("\n1. คำนวณความคล้ายคลึงระหว่างสองประโยค:")
    text1 = "ฉันชอบกินข้าวผัด"
    text2 = "ผมชอบทานข้าวผัด"
    score = similarity.compute_similarity(text1, text2)
    print(f"ประโยค 1: {text1}")
    print(f"ประโยค 2: {text2}")
    print(f"คะแนนความคล้ายคลึง: {score:.3f}")
    
    # 2. ใช้ WangchanBERTa
    print("\n2. ใช้ WangchanBERTa:")
    similarity_wangchan = ThaiSentenceSimilarity(embedding_type="transformer")
    score = similarity_wangchan.compute_similarity(text1, text2)
    print(f"คะแนนความคล้ายคลึง (WangchanBERTa): {score:.3f}")
    
    # 3. ค้นหาประโยคที่คล้ายคลึง
    print("\n3. ค้นหาประโยคที่คล้ายคลึง:")
    corpus = [
        "ฉันชอบกินข้าว",
        "ฉันชอบแมว",
        "อากาศวันนี้ดี",
        "ผมชอบกินก๋วยเตี๋ยว",
        "วันนี้ฝนตก"
    ]
    query = "ฉันชอบทานอาหาร"
    results = similarity.find_most_similar(query, corpus, top_k=3)
    print(f"ประโยคค้นหา: {query}")
    print("\nผลลัพธ์ที่คล้ายคลึง:")
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result['text']}' (คะแนน: {result['score']:.3f})")
    
    # 4. จัดกลุ่มประโยคตามความคล้ายคลึง
    print("\n4. จัดกลุ่มประโยคตามความคล้ายคลึง:")
    sentences = [
        "ฉันชอบกินข้าวผัด",
        "ผมชอบทานข้าวผัด",
        "อากาศวันนี้ดีมาก",
        "วันนี้ท้องฟ้าแจ่มใส",
        "ผมชอบดูหนัง",
        "ฉันชอบดูภาพยนตร์"
    ]
    clusters = similarity.cluster_sentences(sentences, num_clusters=3)
    print("\nกลุ่มประโยค:")
    for i, cluster in enumerate(clusters, 1):
        print(f"\nกลุ่มที่ {i}:")
        for sentence in cluster:
            print(f"- {sentence}")
    
    # 5. สร้าง Embedding
    print("\n5. สร้าง Embedding:")
    text = "ประเทศไทยมีวัฒนธรรมที่เป็นเอกลักษณ์"
    embedding = similarity.get_sentence_embedding(text)
    print(f"ประโยค: {text}")
    print(f"ขนาด Embedding: {embedding.shape}")
    
    # 6. วิเคราะห์ความคล้ายคลึงแบบละเอียด
    print("\n6. วิเคราะห์ความคล้ายคลึงแบบละเอียด:")
    text1 = "ร้านนี้อาหารอร่อยมาก บริการดีเยี่ยม"
    text2 = "ร้านนี้อาหารดี พนักงานบริการดี"
    analysis = similarity.analyze_similarity(text1, text2)
    print(f"ประโยค 1: {text1}")
    print(f"ประโยค 2: {text2}")
    print("\nผลการวิเคราะห์:")
    print(f"คะแนนรวม: {analysis['overall_score']:.3f}")
    print(f"คำที่เหมือนกัน: {', '.join(analysis['common_words'])}")
    print(f"ความคล้ายคลึงระดับความหมาย: {analysis['semantic_similarity']:.3f}")
    print(f"ความคล้ายคลึงระดับโครงสร้าง: {analysis['structural_similarity']:.3f}")

if __name__ == "__main__":
    main()