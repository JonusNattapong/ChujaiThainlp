from thainlp.thai_data_augmentation import ThaiDataAugmenter

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สร้างตัวเพิ่มข้อมูล
    data_augmenter = ThaiDataAugmenter(use_built_in=True)
    
    # ตัวอย่างข้อความที่ต้องการเพิ่มข้อมูล
    example_text = "ผมชอบกินอาหารไทยที่มีรสชาติเผ็ดและหวาน"
    print(f"ข้อความต้นฉบับ: {example_text}\n")
    
    # 1. การแทนที่คำด้วยคำพ้องความหมาย
    synonym_text = data_augmenter._synonym_replacement(example_text)
    print(f"1. แทนที่คำด้วยคำพ้องความหมาย: {synonym_text}")
    
    # 2. การลบคำแบบสุ่ม
    deletion_text = data_augmenter._random_deletion(example_text)
    print(f"2. ลบคำแบบสุ่ม: {deletion_text}")
    
    # 3. การสลับตำแหน่งคำแบบสุ่ม
    swap_text = data_augmenter._random_swap(example_text)
    print(f"3. สลับตำแหน่งคำแบบสุ่ม: {swap_text}")
    
    # 4. การแทรกคำแบบสุ่ม
    insertion_text = data_augmenter._random_insertion(example_text)
    print(f"4. แทรกคำแบบสุ่ม: {insertion_text}")
    
    # 5. การเพิ่มข้อมูลโดยใช้หลายเทคนิค
    augmented_texts = data_augmenter.augment_text(example_text, n_aug=3)
    print("\n5. การเพิ่มข้อมูลโดยใช้หลายเทคนิค:")
    for i, text in enumerate(augmented_texts):
        print(f"   {i+1}. {text}")
    
    # 6. การเพิ่มข้อมูลด้วย EDA
    eda_texts = data_augmenter.eda(example_text, n_aug=3)
    print("\n6. การเพิ่มข้อมูลด้วย EDA:")
    for i, text in enumerate(eda_texts):
        print(f"   {i+1}. {text}")
    
    # 7. การเพิ่มข้อมูลด้วยแม่แบบ
    templates = [
        "ฉัน{กริยา}{กรรม}ที่{สถานที่}",
        "วันนี้ฉันจะไป{กริยา}{กรรม}ที่{สถานที่}",
        "{กรรม}ที่{สถานที่}มี{คุณภาพ}มาก"
    ]
    
    slots = {
        "กริยา": ["กิน", "ทาน", "รับประทาน", "ชอบ", "ชิม"],
        "กรรม": ["อาหารไทย", "อาหารจีน", "อาหารญี่ปุ่น", "ส้มตำ", "ต้มยำกุ้ง", "แกงเขียวหวาน"],
        "สถานที่": ["ร้านอาหาร", "บ้าน", "ตลาด", "ห้างสรรพสินค้า", "โรงอาหาร"],
        "คุณภาพ": ["อร่อย", "รสชาติดี", "รสจัดจ้าน", "รสกลมกล่อม", "รสเผ็ด"]
    }
    
    template_texts = data_augmenter.augment_with_templates(templates, slots, n_aug=3)
    print("\n7. การเพิ่มข้อมูลด้วยแม่แบบ:")
    for i, text in enumerate(template_texts):
        print(f"   {i+1}. {text}")
    
    # 8. การเพิ่มข้อมูลชุดข้อมูล
    dataset = [
        "วันนี้อากาศดีมาก",
        "อาหารไทยรสชาติเผ็ด",
        "เขาชอบดูหนังที่โรงภาพยนตร์"
    ]
    labels = ["บรรยากาศ", "อาหาร", "กิจกรรม"]
    
    augmented_dataset, augmented_labels = data_augmenter.augment_dataset(
        dataset, labels, n_aug_per_text=2
    )
    
    print("\n8. การเพิ่มข้อมูลชุดข้อมูล:")
    print("   ชุดข้อมูลต้นฉบับ:")
    for text, label in zip(dataset, labels):
        print(f"   - [{label}] {text}")
        
    print("\n   ชุดข้อมูลที่เพิ่มแล้ว:")
    for text, label in zip(augmented_dataset, augmented_labels):
        print(f"   - [{label}] {text}")
