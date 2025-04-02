"""
Thai language resources
"""
from typing import Set, List, Dict
import json
import os

# Basic Thai stopwords
THAI_STOPWORDS = {
    'ไว้', 'ไม่', 'ไป', 'ได้', 'ให้', 'ใน', 'โดย', 'แห่ง',
    'แล้ว', 'และ', 'แต่', 'เอง', 'เห็น', 'เลย', 'เริ่ม',
    'เรา', 'เมื่อ', 'เพื่อ', 'เพราะ', 'เป็นการ', 'เป็น',
    'เปิด', 'เนื่องจาก', 'เดียว', 'เด๋ียว', 'เซ็น', 'เช่น',
    'เฉพาะ', 'เคย', 'เข้า', 'เขา', 'อีก', 'อาจ', 'อะไร',
    'ออก', 'อย่าง', 'อยู่', 'อยาก', 'หาก', 'หลาย', 'หรือ',
    'หนึ่ง', 'ส่วน', 'ส่ง', 'สุด', 'สําหรับ', 'ว่า', 'วัน',
    'ลง', 'ร่วม', 'ราย', 'รับ', 'ระหว่าง', 'รวม', 'ยัง',
    'มี', 'มาก', 'มา', 'พร้อม', 'พบ', 'ผ่าน', 'ผล', 'บาง',
    'น่า', 'นี้', 'นํา', 'นั้น', 'นัก', 'นอกจาก', 'ทุก',
    'ที่สุด', 'ที่', 'ทําให้', 'ทํา', 'ทาง', 'ทั้งนี้', 'ถ้า',
    'ถูก', 'ถึง', 'ต้อง', 'ต่างๆ', 'ต่าง', 'ตาม', 'ตั้ง',
    'ดัง', 'ด้าน', 'ด้วย', 'ตั้งแต่', 'เดิม', 'เกิน', 'เกิด',
    'เก่า', 'เก็บ', 'เขต', 'เขา', 'เงิน', 'เจ้า', 'เฉย'
}

# Basic Thai words
THAI_WORDS = {
    'กิน', 'เดิน', 'วิ่ง', 'นอน', 'ดู', 'ฟัง', 'พูด', 'อ่าน',
    'เขียน', 'ทำ', 'บ้าน', 'รถ', 'หมา', 'แมว', 'คน', 'ต้นไม้',
    'ดอกไม้', 'น้ำ', 'อาหาร', 'ข้าว', 'หนังสือ', 'โต๊ะ', 'เก้าอี้',
    'ประตู', 'หน้าต่าง', 'ทีวี', 'โทรศัพท์', 'คอมพิวเตอร์', 'ดี',
    'สวย', 'ใหญ่', 'เล็ก', 'สูง', 'ต่ำ', 'เร็ว', 'ช้า', 'ร้อน',
    'เย็น', 'หนัก', 'เบา', 'ง่าย', 'ยาก', 'แดง', 'เขียว', 'น้ำเงิน',
    'เหลือง', 'ขาว', 'ดำ', 'ฉัน', 'เธอ', 'เขา', 'มัน', 'พวกเรา',
    'พวกเขา', 'ที่นี่', 'ที่นั่น', 'วันนี้', 'พรุ่งนี้', 'เมื่อวาน'
}

def get_stopwords() -> Set[str]:
    """Get Thai stopwords"""
    return THAI_STOPWORDS.copy()

def get_words() -> Set[str]:
    """Get basic Thai words"""
    return THAI_WORDS.copy()

def add_words(words: Set[str]):
    """Add custom words to the dictionary"""
    THAI_WORDS.update(words)

def save_words(filepath: str):
    """Save word dictionary to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(list(THAI_WORDS), f, ensure_ascii=False, indent=2)

def load_words(filepath: str):
    """Load word dictionary from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        words = json.load(f)
        THAI_WORDS.update(words)

# Dictionary paths
DICT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'dict')

def get_dictionary_path(name: str) -> str:
    """Get path to dictionary file"""
    return os.path.join(DICT_DIR, f'{name}.txt')

# Initialize with custom dictionaries if available
try:
    custom_dict_path = get_dictionary_path('custom')
    if os.path.exists(custom_dict_path):
        load_words(custom_dict_path)
except:
    pass # Ignore errors loading custom dictionary
