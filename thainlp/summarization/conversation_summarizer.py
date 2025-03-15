"""
ระบบสรุปบทสนทนาอัตโนมัติสำหรับภาษาไทย
บทคัดย่อ: โมดูลนี้ใช้สำหรับสรุปบทสนทนาภาษาไทยโดยอัตโนมัติ ด้วยเทคนิคการสกัดเนื้อหาสำคัญ
และการสร้างข้อความสรุป สามารถทำงานกับทั้งบทสนทนาแบบ 1 ต่อ 1 และแบบกลุ่ม
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Union
from collections import Counter
from thainlp.tokenize import word_tokenize, sentence_tokenize
from thainlp.utils.thai_utils import normalize_text, get_thai_stopwords
from thainlp.question_answering.qa_system import answer_question
import thainlp
import pandas as pd
import networkx as nx

class ConversationSummarizer:
    def __init__(self, summarization_ratio=0.3, min_summary_sentences=3):
        """
        คลาสสำหรับสรุปบทสนทนาภาษาไทยอัตโนมัติ
        
        Parameters:
        -----------
        summarization_ratio: float
            สัดส่วนของข้อความที่จะนำมาสรุป (0.0 - 1.0)
        min_summary_sentences: int
            จำนวนประโยคขั้นต่ำสำหรับการสรุป
        """
        self.summarization_ratio = summarization_ratio
        self.min_summary_sentences = min_summary_sentences
        self.stopwords = get_thai_stopwords()
        
    def parse_conversation(self, conversation: str) -> List[Dict]:
        """แยกบทสนทนาเป็นข้อความของแต่ละคน"""
        # รูปแบบสำหรับบทสนทนาแบบ "ชื่อ: ข้อความ"
        pattern = r'([^:]+):(.*?)(?=\n[^:]+:|$)'
        
        messages = []
        matches = re.findall(pattern, conversation, re.DOTALL)
        
        for speaker, text in matches:
            messages.append({
                'speaker': speaker.strip(),
                'message': text.strip()
            })
            
        return messages
        
    def extract_important_sentences(self, conversation: List[Dict], top_n=None) -> List[str]:
        """สกัดประโยคสำคัญจากบทสนทนา"""
        # รวมข้อความทั้งหมด
        full_text = " ".join([msg['message'] for msg in conversation])
        
        # แบ่งเป็นประโยค
        sentences = sentence_tokenize(full_text)
        
        if not sentences:
            return []
            
        # กำหนดจำนวนประโยคที่จะสกัด
        if top_n is None:
            top_n = max(self.min_summary_sentences, int(len(sentences) * self.summarization_ratio))
        
        # คำนวณความถี่ของคำในบทสนทนา
        word_freq = Counter()
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [w for w in words if w not in self.stopwords and len(w) > 1]
            word_freq.update(words)
            
        # คำนวณคะแนนของแต่ละประโยค
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [w for w in words if w not in self.stopwords and len(w) > 1]
            
            # คำนวณคะแนนจากความถี่ของคำ
            score = sum(word_freq[word] for word in words) / max(len(words), 1)
            
            sentence_scores.append((sentence, score))
            
        # เรียงลำดับประโยคตามคะแนน
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        # เลือกประโยคที่มีคะแนนสูงสุด top_n ประโยค
        top_sentences = [sentence for sentence, _ in sorted_sentences[:top_n]]
        
        # เรียงประโยคตามลำดับที่ปรากฏในบทสนทนา
        ordered_sentences = [s for s in sentences if s in top_sentences]
        
        return ordered_sentences
    
    def extract_key_topics(self, conversation: List[Dict], num_topics=5) -> List[str]:
        """สกัดหัวข้อสำคัญในบทสนทนา"""
        # รวมข้อความทั้งหมด
        full_text = " ".join([msg['message'] for msg in conversation])
        
        # ตัดคำและกรองคำหยุด
        words = word_tokenize(full_text)
        filtered_words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        # นับความถี่ของคำ
        word_freq = Counter(filtered_words)
        
        # ดึงคำที่มีความถี่สูงสุด
        top_words = word_freq.most_common(num_topics * 2)
        
        # กรองเฉพาะคำที่น่าจะเป็นหัวข้อ (ไม่ใช่คำทั่วไป)
        topics = []
        for word, freq in top_words:
            if word not in self.stopwords and len(word) > 1:
                topics.append(word)
                if len(topics) >= num_topics:
                    break

        return topics

    def extract_speaker_stats(self, conversation: List[Dict]) -> Dict:
        """คำนวณสถิติผู้พูดในบทสนทนา"""
        stats = {}
        
        # นับจำนวนข้อความและความยาวข้อความของแต่ละคน
        for msg in conversation:
            speaker = msg['speaker']
            message = msg['message']
            
            if speaker not in stats:
                stats[speaker] = {
                    'message_count': 0,
                    'total_length': 0,
                    'questions_asked': 0
                }
            
            stats[speaker]['message_count'] += 1
            stats[speaker]['total_length'] += len(message)
            
            # นับจำนวนคำถาม (ข้อความที่มีเครื่องหมาย ?)
            if '?' in message:
                stats[speaker]['questions_asked'] += 1
                
        return stats
    
    def generate_conversation_summary(self, conversation: str) -> Dict:
        """สร้างสรุปบทสนทนาอัตโนมัติ"""
        # แยกบทสนทนาเป็นข้อความของแต่ละคน
        messages = self.parse_conversation(conversation)
        
        if not messages:
            return {
                'summary': '',
                'important_sentences': [],
                'key_topics': [],
                'speaker_stats': {},
                'qa_summary': {}
            }
        
        # สกัดประโยคสำคัญ
        important_sentences = self.extract_important_sentences(messages)
        
        # สกัดหัวข้อสำคัญ
        key_topics = self.extract_key_topics(messages)
        
        # สถิติผู้พูด
        speaker_stats = self.extract_speaker_stats(messages)
        
        # สรุปข้อความ
        full_text = " ".join([msg['message'] for msg in messages])
        
        # สร้างคำถามสำคัญเพื่อช่วยในการสรุป
        qa_summary = {
            "หัวข้อหลักของการสนทนา": answer_question("บทสนทนานี้เกี่ยวกับอะไร", full_text)['answer'],
            "ประเด็นสำคัญ": answer_question("ประเด็นสำคัญในบทสนทนานี้คืออะไร", full_text)['answer'],
            "ข้อตกลง": answer_question("มีข้อตกลงหรือข้อสรุปอะไรในบทสนทนานี้", full_text)['answer']
        }
        
        # สร้างสรุปแบบบรรยาย
        summary_text = f"บทสนทนาระหว่าง {', '.join(speaker_stats.keys())} "
        summary_text += f"เกี่ยวกับ{qa_summary['หัวข้อหลักของการสนทนา']} "
        summary_text += f"ประเด็นสำคัญคือ {qa_summary['ประเด็นสำคัญ']} "
        
        if qa_summary['ข้อตกลง'] and qa_summary['ข้อตกลง'] != "ไม่มีข้อตกลงชัดเจน":
            summary_text += f"โดยมีข้อสรุปคือ {qa_summary['ข้อตกลง']}"
        
        return {
            'summary': summary_text,
            'important_sentences': important_sentences,
            'key_topics': key_topics,
            'speaker_stats': speaker_stats,
            'qa_summary': qa_summary
        }
    
    def summarize_conversation_by_topics(self, conversation: str) -> Dict:
        """สรุปบทสนทนาแยกตามหัวข้อ"""
        # แยกบทสนทนาเป็นข้อความของแต่ละคน
        messages = self.parse_conversation(conversation)
        
        if not messages:
            return {}
        
        # สร้าง DataFrame จากข้อความ
        df = pd.DataFrame(messages)
        
        # สร้าง embeddings สำหรับแต่ละข้อความ
        embeddings = []
        for msg in df['message']:
            try:
                emb = thainlp.feature_extraction.create_document_vector(msg)
                embeddings.append(emb)
            except:
                # ถ้ามีปัญหาในการสร้าง embedding ให้ใช้ vector ศูนย์แทน
                embeddings.append(np.zeros(300))
        
        # ทำ clustering เพื่อจับกลุ่มข้อความตามหัวข้อ
        from sklearn.cluster import AgglomerativeClustering
        
        # ประมาณจำนวน clusters ตามความยาวของบทสนทนา
        n_clusters = max(2, min(10, len(messages) // 5))
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        df['cluster'] = clustering.fit_predict(embeddings)
        
        # สรุปแต่ละ cluster
        topic_summaries = {}
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_messages = df[df['cluster'] == cluster_id]
            
            # รวมข้อความใน cluster
            cluster_text = " ".join(cluster_messages['message'])
            
            # สกัดหัวข้อ
            words = word_tokenize(cluster_text)
            filtered_words = [w for w in words if w not in self.stopwords and len(w) > 1]
            top_words = Counter(filtered_words).most_common(3)
            
            # สร้างชื่อ topic
            topic_name = " ".join([word for word, _ in top_words])
            
            # สรุปประโยคสำคัญ
            cluster_msgs = [{'speaker': row['speaker'], 'message': row['message']} 
                           for _, row in cluster_messages.iterrows()]
            important_sentences = self.extract_important_sentences(cluster_msgs, top_n=2)
            
            topic_summaries[topic_name] = {
                'messages_count': len(cluster_messages),
                'speakers': list(cluster_messages['speaker'].unique()),
                'key_sentences': important_sentences
            }
            
        return topic_summaries
