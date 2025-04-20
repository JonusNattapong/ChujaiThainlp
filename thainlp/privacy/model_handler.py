"""
โมดูลสำหรับจัดการโมเดลและการพัฒนาปรับปรุง
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

class PrivacyModelHandler:
    def __init__(self, base_path: str = "./models/privacy"):
        """
        จัดการโมเดลสำหรับการปกป้องความเป็นส่วนตัว
        
        Args:
            base_path: ที่อยู่โฟลเดอร์สำหรับเก็บโมเดล
        """
        self.base_path = base_path
        self.model_info = {
            'version': '0.1.0',
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
            'samples_processed': 0,
            'performance': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        }
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(base_path, exist_ok=True)
        
    def save_training_example(self, 
                            text: str, 
                            entities: List[Tuple[str, int, int, float]],
                            dialect: str = None):
        """บันทึกตัวอย่างสำหรับการฝึกฝน
        
        Args:
            text: ข้อความ
            entities: รายการ entity (ประเภท, ตำแหน่งเริ่ม, ตำแหน่งจบ, ความน่าจะเป็น)
            dialect: ภาษาถิ่น (ถ้ามี)
        """
        example = {
            'text': text,
            'entities': [
                {
                    'type': t,
                    'start': s,
                    'end': e,
                    'prob': p
                } for t, s, e, p in entities
            ],
            'dialect': dialect,
            'timestamp': datetime.now().isoformat()
        }
        
        # บันทึกลงไฟล์
        filename = f"example_{self.model_info['samples_processed']}.json"
        path = os.path.join(self.base_path, 'examples', filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(example, f, ensure_ascii=False, indent=2)
            
        self.model_info['samples_processed'] += 1
        
    def update_performance(self, metrics: Dict[str, float]):
        """อัพเดทค่าประสิทธิภาพของโมเดล
        
        Args:
            metrics: ค่าวัดประสิทธิภาพต่างๆ
        """
        self.model_info['performance'].update(metrics)
        self.model_info['updated'] = datetime.now().isoformat()
        
        # บันทึกข้อมูลโมเดล
        with open(os.path.join(self.base_path, 'model_info.json'), 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, ensure_ascii=False, indent=2)
            
    def load_training_examples(self) -> List[Dict]:
        """โหลดตัวอย่างทั้งหมดสำหรับการฝึกฝน"""
        examples = []
        example_dir = os.path.join(self.base_path, 'examples')
        
        if not os.path.exists(example_dir):
            return examples
            
        for filename in os.listdir(example_dir):
            if filename.endswith('.json'):
                with open(os.path.join(example_dir, filename), 'r', encoding='utf-8') as f:
                    examples.append(json.load(f))
                    
        return examples
        
    def calculate_thresholds(self, examples: List[Dict]) -> Dict[str, float]:
        """คำนวณค่า threshold ที่เหมาะสมจากตัวอย่าง
        
        Args:
            examples: รายการตัวอย่างที่ใช้ในการคำนวณ
            
        Returns:
            Dict ของค่า threshold แยกตามประเภท
        """
        entity_probs = defaultdict(list)
        
        # รวบรวมค่าความน่าจะเป็นแยกตามประเภท
        for example in examples:
            for entity in example['entities']:
                entity_probs[entity['type']].append(entity['prob'])
                
        # คำนวณ threshold จากเปอร์เซ็นไทล์ที่ 5
        thresholds = {}
        for entity_type, probs in entity_probs.items():
            if probs:
                thresholds[entity_type] = np.percentile(probs, 5)
            else:
                thresholds[entity_type] = 0.5
                
        return thresholds
        
    def get_model_info(self) -> Dict:
        """ดึงข้อมูลของโมเดลปัจจุบัน"""
        return self.model_info.copy()
        
    def get_latest_performance(self) -> Dict[str, float]:
        """ดึงค่าประสิทธิภาพล่าสุดของโมเดล"""
        return self.model_info['performance'].copy()