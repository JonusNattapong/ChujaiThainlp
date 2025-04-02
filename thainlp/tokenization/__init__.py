"""
Thai tokenization utilities
"""
from typing import List
import re
from .maximum_matching import MaximumMatchingTokenizer

# Common Thai words for tokenization
THAI_WORDS = {
    "การ", "กับ", "ก็", "ก่อน", "ขณะ", "ขึ้น", "ของ", "ครับ", "ค่ะ", "ครั้ง", 
    "ความ", "คือ", "จะ", "จัด", "จาก", "จึง", "ช่วง", "ซึ่ง", "ดัง", "ด้วย",
    "ด้าน", "ต้อง", "ถึง", "ต่างๆ", "ที่", "ทุก", "ทาง", "ทั้ง", "ทำ", "ที่สุด",
    "นี้", "นั้น", "นัก", "นั้น", "นี้", "นั้น", "ใน", "ให้", "หรือ", "และ",
    "แล้ว", "ว่า", "วัน", "ไว้", "ว่า", "เพื่อ", "เมื่อ", "เรา", "เริ่ม", "เลย",
    "เวลา", "ส่วน", "ส่ง", "ส่วน", "สามารถ", "สิ่ง", "หาก", "ออก", "อะไร", "อาจ",
    "อีก", "เขา", "เพียง", "เพราะ", "เปิด", "เป็น", "แบบ", "แต่", "เอง", "เอง",
    "เคย", "เคย", "เข้า", "เช่น", "เฉพาะ", "เคย", "เคย", "เคย", "เคย", "เคย"
}

def word_tokenize(text: str) -> List[str]:
    """Tokenize Thai text using maximum matching algorithm"""
    tokenizer = MaximumMatchingTokenizer()
    return tokenizer.tokenize(text)

class Tokenizer:
    """Base tokenizer class"""
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError