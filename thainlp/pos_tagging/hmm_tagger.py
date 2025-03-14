"""
Part-of-Speech Tagging for Thai text using Hidden Markov Model
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import re
import warnings
import numpy as np
from collections import defaultdict

try:
    import pythainlp
    from pythainlp.tag import pos_tag as pythainlp_pos_tag
    from pythainlp.tag.perceptron import PerceptronTagger
    from pythainlp.corpus import thai_words
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    warnings.warn("PyThaiNLP not found. Using simplified POS tagging.")

# Simplified POS dictionary for when PyThaiNLP is not available
_THAI_POS_DICT = {
    "สวัสดี": "INTJ",
    "ประเทศ": "NOUN",
    "ไทย": "PROPN",
    "คน": "NOUN",
    "กิน": "VERB",
    "ข้าว": "NOUN",
    "น้ำ": "NOUN",
    "รถ": "NOUN",
    "บ้าน": "NOUN",
    "เมือง": "NOUN",
    "จังหวัด": "NOUN",
    "เชียงใหม่": "PROPN",
    "กรุงเทพ": "PROPN",
    "ภูเก็ต": "PROPN",
    "ท่องเที่ยว": "VERB",
    "เดินทาง": "VERB",
    "นักวิจัย": "NOUN",
    "ศึกษา": "VERB",
    "ปรากฏการณ์": "NOUN",
    "ธรรมชาติ": "NOUN",
    "ซับซ้อน": "ADJ",
    "การประชุม": "NOUN",
    "วิชาการ": "ADJ",
    "นานาชาติ": "ADJ",
    "เศรษฐกิจ": "NOUN",
    "ฟื้นตัว": "VERB",
    "อย่าง": "ADV",
    "ช้า": "ADV",
    "รวดเร็ว": "ADJ",
    "ปัญญาประดิษฐ์": "NOUN",
    "เทคโนโลยี": "NOUN",
    "เปลี่ยนแปลง": "VERB",
    "อุตสาหกรรม": "NOUN",
    "การแพทย์": "NOUN",
    "การเงิน": "NOUN",
    "การศึกษา": "NOUN",
    "การขนส่ง": "NOUN",
    "การท่องเที่ยว": "NOUN",
    "การเกษตร": "NOUN",
    "การสื่อสาร": "NOUN",
    "การพัฒนา": "NOUN",
    "การวิจัย": "NOUN",
    "การค้า": "NOUN",
    "การลงทุน": "NOUN",
    "การผลิต": "NOUN",
    "การบริโภค": "NOUN",
    "การส่งออก": "NOUN",
    "การนำเข้า": "NOUN",
    "การแข่งขัน": "NOUN",
    "การเติบโต": "NOUN",
    "การพัฒนา": "NOUN",
    "การปรับปรุง": "NOUN",
    "การเปลี่ยนแปลง": "NOUN",
    "การเรียนรู้": "NOUN",
    "การสอน": "NOUN",
    "การฝึกอบรม": "NOUN",
    "การทดสอบ": "NOUN",
    "การทดลอง": "NOUN",
    "การวิเคราะห์": "NOUN",
    "การสังเคราะห์": "NOUN",
    "การประเมิน": "NOUN",
    "การตรวจสอบ": "NOUN",
    "การติดตาม": "NOUN",
    "การควบคุม": "NOUN",
    "การจัดการ": "NOUN",
    "การบริหาร": "NOUN",
    "การวางแผน": "NOUN",
    "การดำเนินการ": "NOUN",
    "การปฏิบัติ": "NOUN",
    "การทำงาน": "NOUN",
    "การใช้งาน": "NOUN",
    "การพัฒนา": "NOUN",
    "การออกแบบ": "NOUN",
    "การสร้าง": "NOUN",
    "การผลิต": "NOUN",
    "การประกอบ": "NOUN",
    "การติดตั้ง": "NOUN",
    "การบำรุงรักษา": "NOUN",
    "การซ่อมแซม": "NOUN",
    "การทดสอบ": "NOUN",
    "การตรวจสอบ": "NOUN",
    "การรับรอง": "NOUN",
    "การรับประกัน": "NOUN",
    "การขาย": "NOUN",
    "การตลาด": "NOUN",
    "การโฆษณา": "NOUN",
    "การประชาสัมพันธ์": "NOUN",
    "การบริการ": "NOUN",
    "การสนับสนุน": "NOUN",
    "การช่วยเหลือ": "NOUN",
    "การแก้ไข": "NOUN",
    "การปรับปรุง": "NOUN",
    "การพัฒนา": "NOUN",
    "การเพิ่ม": "NOUN",
    "การลด": "NOUN",
    "การขยาย": "NOUN",
    "การหด": "NOUN",
    "การเติบโต": "NOUN",
    "การถดถอย": "NOUN",
    "การฟื้นตัว": "NOUN",
    "การล่ม": "NOUN",
    "การล้ม": "NOUN",
    "การเกิด": "NOUN",
    "การตาย": "NOUN",
    "การเริ่ม": "NOUN",
    "การจบ": "NOUN",
    "การเปิด": "NOUN",
    "การปิด": "NOUN",
    "การเข้า": "NOUN",
    "การออก": "NOUN",
    "การขึ้น": "NOUN",
    "การลง": "NOUN",
    "การไป": "NOUN",
    "การมา": "NOUN",
    "การถึง": "NOUN",
    "การกลับ": "NOUN",
    "การหยุด": "NOUN",
    "การพัก": "NOUN",
    "การนอน": "NOUN",
    "การตื่น": "NOUN",
    "การกิน": "NOUN",
    "การดื่ม": "NOUN",
    "การเล่น": "NOUN",
    "การทำงาน": "NOUN",
    "การเรียน": "NOUN",
    "การสอน": "NOUN",
    "การอ่าน": "NOUN",
    "การเขียน": "NOUN",
    "การพูด": "NOUN",
    "การฟัง": "NOUN",
    "การดู": "NOUN",
    "การเห็น": "NOUN",
    "การคิด": "NOUN",
    "การรู้สึก": "NOUN",
    "การรับรู้": "NOUN",
    "การเข้าใจ": "NOUN",
    "การจำ": "NOUN",
    "การลืม": "NOUN",
    "การรัก": "NOUN",
    "การเกลียด": "NOUN",
    "การชอบ": "NOUN",
    "การไม่ชอบ": "NOUN",
    "การสุข": "NOUN",
    "การทุกข์": "NOUN",
    "การสบาย": "NOUN",
    "การเจ็บ": "NOUN",
    "การป่วย": "NOUN",
    "การหาย": "NOUN",
    "การเป็น": "NOUN",
    "การตาย": "NOUN",
    "ผม": "PRON",
    "ฉัน": "PRON",
    "เขา": "PRON",
    "เธอ": "PRON",
    "มัน": "PRON",
    "เรา": "PRON",
    "พวกเขา": "PRON",
    "พวกเรา": "PRON",
    "ที่": "PRON",
    "อัน": "PRON",
    "ตัว": "PRON",
    "คน": "PRON",
    "ใคร": "PRON",
    "อะไร": "PRON",
    "ไหน": "PRON",
    "เมื่อไร": "PRON",
    "อย่างไร": "PRON",
    "ทำไม": "PRON",
    "และ": "CONJ",
    "หรือ": "CONJ",
    "แต่": "CONJ",
    "เพราะ": "CONJ",
    "เนื่องจาก": "CONJ",
    "ดังนั้น": "CONJ",
    "เพราะฉะนั้น": "CONJ",
    "ถ้า": "CONJ",
    "ถึงแม้": "CONJ",
    "แม้ว่า": "CONJ",
    "ก็ตาม": "CONJ",
    "จนกระทั่ง": "CONJ",
    "ตั้งแต่": "CONJ",
    "เมื่อ": "CONJ",
    "ใน": "PREP",
    "นอก": "PREP",
    "บน": "PREP",
    "ล่าง": "PREP",
    "ข้าง": "PREP",
    "หน้า": "PREP",
    "หลัง": "PREP",
    "ระหว่าง": "PREP",
    "ท่ามกลาง": "PREP",
    "ด้วย": "PREP",
    "โดย": "PREP",
    "แก่": "PREP",
    "สำหรับ": "PREP",
    "กับ": "PREP",
    "แด่": "PREP",
    "ต่อ": "PREP",
    "จาก": "PREP",
    "ถึง": "PREP",
}

class HMMTagger:
    """Hidden Markov Model for POS tagging"""
    
    def __init__(self):
        """Initialize HMM tagger"""
        self.tag_counts = defaultdict(int)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tags = set()
        self.vocab = set()
        
    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """
        Train HMM tagger on tagged sentences
        
        Args:
            tagged_sentences (List[List[Tuple[str, str]]]): List of tagged sentences
        """
        # Count occurrences
        for sentence in tagged_sentences:
            prev_tag = "<START>"
            for word, tag in sentence:
                self.tag_counts[tag] += 1
                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][word] += 1
                self.tags.add(tag)
                self.vocab.add(word)
                prev_tag = tag
            
            # Add transition to end
            self.transition_counts[prev_tag]["<END>"] += 1
            
        # Add start and end tags
        self.tags.add("<START>")
        self.tags.add("<END>")
        
        # Calculate probabilities
        self.transition_prob = {}
        for prev_tag, next_tags in self.transition_counts.items():
            self.transition_prob[prev_tag] = {}
            total = sum(next_tags.values())
            for next_tag, count in next_tags.items():
                self.transition_prob[prev_tag][next_tag] = count / total
                
        self.emission_prob = {}
        for tag, words in self.emission_counts.items():
            self.emission_prob[tag] = {}
            total = sum(words.values())
            for word, count in words.items():
                self.emission_prob[tag][word] = count / total
                
    def viterbi(self, sentence: List[str]) -> List[str]:
        """
        Viterbi algorithm for finding most likely tag sequence
        
        Args:
            sentence (List[str]): List of words
            
        Returns:
            List[str]: List of predicted tags
        """
        # Initialize
        viterbi = [{}]
        backpointer = [{}]
        
        # Base case
        for tag in self.tags:
            if tag not in ["<START>", "<END>"]:
                word = sentence[0]
                # Handle unknown words
                if word not in self.vocab:
                    # Use a small probability for unknown words
                    emission_p = 0.001
                else:
                    emission_p = self.emission_prob.get(tag, {}).get(word, 0.001)
                    
                transition_p = self.transition_prob.get("<START>", {}).get(tag, 0.001)
                viterbi[0][tag] = transition_p * emission_p
                backpointer[0][tag] = "<START>"
                
        # Recursion
        for t in range(1, len(sentence)):
            viterbi.append({})
            backpointer.append({})
            word = sentence[t]
            
            for tag in self.tags:
                if tag not in ["<START>", "<END>"]:
                    # Find the best previous tag
                    best_prev_tag = None
                    best_prob = 0
                    
                    for prev_tag in self.tags:
                        if prev_tag not in ["<START>", "<END>"]:
                            # Skip if previous tag has zero probability
                            if prev_tag not in viterbi[t-1]:
                                continue
                                
                            transition_p = self.transition_prob.get(prev_tag, {}).get(tag, 0.001)
                            prob = viterbi[t-1][prev_tag] * transition_p
                            
                            if prob > best_prob:
                                best_prob = prob
                                best_prev_tag = prev_tag
                                
                    # Handle unknown words
                    if word not in self.vocab:
                        emission_p = 0.001
                    else:
                        emission_p = self.emission_prob.get(tag, {}).get(word, 0.001)
                        
                    viterbi[t][tag] = best_prob * emission_p
                    backpointer[t][tag] = best_prev_tag
                    
        # Termination
        best_last_tag = None
        best_prob = 0
        
        for tag, prob in viterbi[-1].items():
            transition_p = self.transition_prob.get(tag, {}).get("<END>", 0.001)
            prob = prob * transition_p
            
            if prob > best_prob:
                best_prob = prob
                best_last_tag = tag
                
        # Backtrack
        best_path = [best_last_tag]
        for t in range(len(sentence) - 1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])
            
        return best_path
        
    def tag(self, sentence: List[str]) -> List[Tuple[str, str]]:
        """
        Tag a sentence
        
        Args:
            sentence (List[str]): List of words
            
        Returns:
            List[Tuple[str, str]]: List of (word, tag) pairs
        """
        tags = self.viterbi(sentence)
        return list(zip(sentence, tags))

def _simple_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Simple dictionary-based POS tagging
    
    Args:
        tokens (List[str]): List of tokens
        
    Returns:
        List[Tuple[str, str]]: List of (token, tag) pairs
    """
    result = []
    for token in tokens:
        # Check if token is in dictionary
        if token in _THAI_POS_DICT:
            tag = _THAI_POS_DICT[token]
        else:
            # Apply simple rules for unknown words
            if token.isdigit():
                tag = "NUM"
            elif re.match(r'^[ก-๛]+$', token):
                # Default for unknown Thai words
                tag = "NOUN"
            else:
                tag = "X"  # Unknown
                
        result.append((token, tag))
        
    return result

def train_and_tag(tokens: List[str], model: str = "pythainlp", corpus: Optional[List[List[Tuple[str, str]]]] = None) -> List[Tuple[str, str]]:
    """
    Train HMM model and tag tokens
    
    Args:
        tokens (List[str]): List of tokens to tag
        model (str): Model to use
               - 'pythainlp': Use PyThaiNLP POS tagger (recommended)
               - 'pythainlp:perceptron': Use PyThaiNLP Perceptron tagger
               - 'pythainlp:artagger': Use PyThaiNLP Artagger
               - 'hmm': Use HMM tagger
               - 'simple': Use simple dictionary-based tagger
        corpus (Optional[List[List[Tuple[str, str]]]]): Training corpus for HMM model
        
    Returns:
        List[Tuple[str, str]]: List of (token, tag) pairs
    """
    if model.startswith("pythainlp"):
        if not PYTHAINLP_AVAILABLE:
            warnings.warn("PyThaiNLP not available. Falling back to simple POS tagging.")
            return _simple_pos_tag(tokens)
            
        if ":" in model:
            _, engine = model.split(":", 1)
            return pythainlp_pos_tag(tokens, engine=engine)
        else:
            # Default to perceptron
            return pythainlp_pos_tag(tokens, engine="perceptron")
    elif model == "hmm":
        if corpus is None:
            warnings.warn("No training corpus provided for HMM. Falling back to simple POS tagging.")
            return _simple_pos_tag(tokens)
            
        tagger = HMMTagger()
        tagger.train(corpus)
        return tagger.tag(tokens)
    elif model == "simple":
        return _simple_pos_tag(tokens)
    else:
        raise ValueError(f"POS tagging model '{model}' is not supported.")

def pos_tag(text: Union[str, List[str]], engine: str = "pythainlp") -> List[Tuple[str, str]]:
    """
    Tag parts of speech in Thai text
    
    Args:
        text (Union[str, List[str]]): Thai text or list of tokens
        engine (str): Engine for POS tagging
               - 'pythainlp': Use PyThaiNLP tagger (recommended)
               - 'pythainlp:perceptron': Use PyThaiNLP Perceptron tagger
               - 'pythainlp:artagger': Use PyThaiNLP Artagger
               - 'hmm': Use HMM tagger
               - 'simple': Use simple dictionary-based tagger
               
    Returns:
        List[Tuple[str, str]]: List of (word, pos_tag) tuples
    """
    # Tokenize if input is a string
    if isinstance(text, str):
        if PYTHAINLP_AVAILABLE:
            from pythainlp.tokenize import word_tokenize
            tokens = word_tokenize(text)
        else:
            from thainlp.tokenization.maximum_matching import tokenize
            tokens = tokenize(text)
    else:
        tokens = text
        
    return train_and_tag(tokens, model=engine)

def get_pos_tag_list() -> Dict[str, List[str]]:
    """
    Get list of available POS tags with examples
    
    Returns:
        Dict[str, List[str]]: Dictionary of POS tags and example words
    """
    if PYTHAINLP_AVAILABLE:
        # Get POS tags from PyThaiNLP
        pos_tags = {}
        for word, tag in _THAI_POS_DICT.items():
            if tag not in pos_tags:
                pos_tags[tag] = []
            if len(pos_tags[tag]) < 5:  # Limit to 5 examples per tag
                pos_tags[tag].append(word)
        return pos_tags
    else:
        # Create from our dictionary
        pos_tags = {}
        for word, tag in _THAI_POS_DICT.items():
            if tag not in pos_tags:
                pos_tags[tag] = []
            if len(pos_tags[tag]) < 5:  # Limit to 5 examples per tag
                pos_tags[tag].append(word)
        return pos_tags

def convert_tag_schema(tagged_tokens: List[Tuple[str, str]], source: str = "ud", target: str = "orchid") -> List[Tuple[str, str]]:
    """
    Convert between different POS tag schemas
    
    Args:
        tagged_tokens (List[Tuple[str, str]]): List of (token, tag) pairs
        source (str): Source tag schema (ud, orchid)
        target (str): Target tag schema (ud, orchid)
        
    Returns:
        List[Tuple[str, str]]: List of (token, tag) pairs with converted tags
    """
    if not PYTHAINLP_AVAILABLE:
        warnings.warn("PyThaiNLP not available. Cannot convert tag schema.")
        return tagged_tokens
        
    if source == target:
        return tagged_tokens
        
    # Conversion mappings
    ud_to_orchid = {
        "NOUN": "NCMN",
        "PROPN": "NPRP",
        "VERB": "VACT",
        "ADJ": "ADJT",
        "ADV": "ADVN",
        "PRON": "PRON",
        "DET": "DDAN",
        "NUM": "NCNM",
        "CONJ": "JCRG",
        "PREP": "RPRE",
        "PART": "PART",
        "INTJ": "INTJ",
        "PUNCT": "PUNC",
        "SYM": "PUNC",
        "X": "FIXN"
    }
    
    orchid_to_ud = {v: k for k, v in ud_to_orchid.items()}
    
    if source == "ud" and target == "orchid":
        mapping = ud_to_orchid
    elif source == "orchid" and target == "ud":
        mapping = orchid_to_ud
    else:
        raise ValueError(f"Conversion from {source} to {target} is not supported")
        
    result = []
    for token, tag in tagged_tokens:
        new_tag = mapping.get(tag, tag)  # Use original if not in mapping
        result.append((token, new_tag))
        
    return result

def train_perceptron_tagger(corpus: List[List[Tuple[str, str]]]) -> Any:
    """
    Train a Perceptron POS tagger on a custom corpus
    
    Args:
        corpus (List[List[Tuple[str, str]]]): Training corpus
        
    Returns:
        Any: Trained tagger object
    """
    if not PYTHAINLP_AVAILABLE:
        raise ImportError("PyThaiNLP is required for Perceptron tagger")
        
    tagger = PerceptronTagger()
    
    # Format corpus for training
    formatted_corpus = []
    for sentence in corpus:
        words = [word for word, _ in sentence]
        tags = [tag for _, tag in sentence]
        formatted_corpus.append((words, tags))
        
    tagger.train(formatted_corpus)
    return tagger 