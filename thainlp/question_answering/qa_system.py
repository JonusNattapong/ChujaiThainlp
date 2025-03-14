"""
Question Answering System for Thai Text
"""

from typing import List, Dict, Tuple, Union, Optional
import re
from collections import Counter

class ThaiQuestionAnswering:
    def __init__(self):
        """Initialize ThaiQuestionAnswering"""
        # Question types and their patterns
        self.question_types = {
            'PERSON': [r'ใคร', r'บุคคลใด'],
            'LOCATION': [r'ที่ไหน', r'สถานที่ใด', r'จังหวัดอะไร'],
            'TIME': [r'เมื่อไร', r'เวลาใด', r'กี่โมง', r'วันที่เท่าไร'],
            'REASON': [r'ทำไม', r'เพราะอะไร', r'สาเหตุใด'],
            'METHOD': [r'อย่างไร', r'วิธีใด', r'ด้วยวิธีไหน'],
            'QUANTITY': [r'กี่', r'เท่าไร', r'จำนวนเท่าใด'],
            'DEFINITION': [r'คืออะไร', r'หมายถึงอะไร', r'นิยามว่าอย่างไร'],
            'BOOLEAN': [r'ใช่ไหม', r'หรือไม่', r'จริงหรือ', r'หรือเปล่า']
        }
        
        # Entity types for answer extraction
        self.entity_types = {
            'PERSON': ['PERSON'],
            'LOCATION': ['LOCATION'],
            'TIME': ['DATE', 'TIME'],
            'QUANTITY': ['NUMBER', 'PERCENT', 'MONEY'],
            'DEFINITION': ['DEFINITION']
        }
        
        # Keywords for answer extraction
        self.keywords = {
            'REASON': ['เพราะ', 'เนื่องจาก', 'สาเหตุ', 'ดังนั้น', 'เพื่อ'],
            'METHOD': ['โดย', 'ด้วย', 'ผ่าน', 'ใช้', 'วิธี']
        }
        
    def _identify_question_type(self, question: str) -> str:
        """
        Identify the type of question
        
        Args:
            question (str): Input question
            
        Returns:
            str: Question type
        """
        for q_type, patterns in self.question_types.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return q_type
        return 'UNKNOWN'
        
    def _extract_question_focus(self, question: str) -> List[str]:
        """
        Extract the focus of the question (important keywords)
        
        Args:
            question (str): Input question
            
        Returns:
            List[str]: List of focus keywords
        """
        # Remove question words
        for q_type, patterns in self.question_types.items():
            for pattern in patterns:
                question = re.sub(pattern, '', question)
                
        # Split into words and remove common words
        words = question.split()
        stopwords = ['ที่', 'และ', 'ใน', 'ของ', 'ให้', 'ได้', 'ไป', 'มา', 'เป็น', 'กับ']
        focus_words = [word for word in words if word not in stopwords]
        
        return focus_words
        
    def _find_candidate_answers(self, context: str, question_type: str, focus_words: List[str]) -> List[str]:
        """
        Find candidate answers in the context
        
        Args:
            context (str): Input context
            question_type (str): Type of question
            focus_words (List[str]): Focus keywords from question
            
        Returns:
            List[str]: List of candidate answers
        """
        sentences = re.split(r'[.!?]\s+', context)
        candidates = []
        
        # Score each sentence based on focus words
        for sentence in sentences:
            score = sum(1 for word in focus_words if word in sentence)
            if score > 0:
                candidates.append((sentence, score))
                
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Extract specific answer based on question type
        answers = []
        
        for sentence, _ in candidates[:3]:  # Consider top 3 sentences
            if question_type in self.entity_types:
                # Extract entities of specific type
                entity_types = self.entity_types[question_type]
                # This is a simplified version - in a real system, use NER
                for entity_type in entity_types:
                    # Simple pattern matching for demonstration
                    if entity_type == 'PERSON':
                        matches = re.findall(r'(นาย|นาง|นางสาว|คุณ)[ก-๛]+', sentence)
                        answers.extend(matches)
                    elif entity_type == 'LOCATION':
                        matches = re.findall(r'(จังหวัด|อำเภอ|ตำบล|ประเทศ)[ก-๛]+', sentence)
                        answers.extend(matches)
                    elif entity_type == 'DATE':
                        matches = re.findall(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', sentence)
                        answers.extend(matches)
                    elif entity_type == 'TIME':
                        matches = re.findall(r'\d{1,2}:\d{2}', sentence)
                        answers.extend(matches)
                    elif entity_type == 'NUMBER':
                        matches = re.findall(r'\d+', sentence)
                        answers.extend(matches)
            elif question_type in self.keywords:
                # Extract phrases with specific keywords
                keywords = self.keywords[question_type]
                for keyword in keywords:
                    pattern = f'{keyword}(.*?)[.!?]'
                    matches = re.findall(pattern, sentence + '.')
                    answers.extend([match.strip() for match in matches])
            elif question_type == 'BOOLEAN':
                # For yes/no questions, return the sentence
                answers.append(sentence)
            else:
                # For other types, return the sentence
                answers.append(sentence)
                
        return answers
        
    def _rank_answers(self, answers: List[str], focus_words: List[str]) -> List[str]:
        """
        Rank answers based on relevance to focus words
        
        Args:
            answers (List[str]): List of candidate answers
            focus_words (List[str]): Focus keywords from question
            
        Returns:
            List[str]: Ranked list of answers
        """
        if not answers:
            return []
            
        # Score each answer based on focus words
        scored_answers = []
        for answer in answers:
            score = sum(1 for word in focus_words if word in answer)
            scored_answers.append((answer, score))
            
        # Sort by score
        scored_answers.sort(key=lambda x: x[1], reverse=True)
        
        # Return ranked answers
        return [answer for answer, _ in scored_answers]
        
    def answer_question(self, question: str, context: str) -> Dict[str, Union[str, float]]:
        """
        Answer a question based on the context
        
        Args:
            question (str): Input question
            context (str): Input context
            
        Returns:
            Dict[str, Union[str, float]]: Answer with confidence score
        """
        # Identify question type
        question_type = self._identify_question_type(question)
        
        # Extract focus words
        focus_words = self._extract_question_focus(question)
        
        # Find candidate answers
        candidates = self._find_candidate_answers(context, question_type, focus_words)
        
        # Rank answers
        ranked_answers = self._rank_answers(candidates, focus_words)
        
        if not ranked_answers:
            return {
                'answer': 'ไม่พบคำตอบในบริบทที่ให้มา',
                'score': 0.0
            }
            
        # Calculate confidence score
        score = min(1.0, 0.3 + 0.7 * (len(focus_words) / max(len(question.split()), 1)))
        
        return {
            'answer': ranked_answers[0],
            'score': score
        }
        
    def answer_from_table(self, question: str, table: List[Dict[str, str]]) -> Dict[str, Union[str, float]]:
        """
        Answer a question based on tabular data
        
        Args:
            question (str): Input question
            table (List[Dict[str, str]]): Table data as list of dictionaries
            
        Returns:
            Dict[str, Union[str, float]]: Answer with confidence score
        """
        # Extract focus words
        focus_words = self._extract_question_focus(question)
        
        # Find relevant rows
        relevant_rows = []
        for row in table:
            score = 0
            for word in focus_words:
                for value in row.values():
                    if word in value:
                        score += 1
            if score > 0:
                relevant_rows.append((row, score))
                
        # Sort by relevance
        relevant_rows.sort(key=lambda x: x[1], reverse=True)
        
        if not relevant_rows:
            return {
                'answer': 'ไม่พบคำตอบในตารางที่ให้มา',
                'score': 0.0
            }
            
        # Get most relevant row
        best_row, _ = relevant_rows[0]
        
        # Identify question type
        question_type = self._identify_question_type(question)
        
        # Extract answer based on question type
        answer = ''
        if question_type == 'PERSON':
            for key in ['ชื่อ', 'บุคคล', 'คน', 'ผู้']:
                if any(key in k for k in best_row.keys()):
                    for k, v in best_row.items():
                        if key in k:
                            answer = v
                            break
        elif question_type == 'LOCATION':
            for key in ['สถานที่', 'ที่อยู่', 'จังหวัด', 'ประเทศ']:
                if any(key in k for k in best_row.keys()):
                    for k, v in best_row.items():
                        if key in k:
                            answer = v
                            break
        elif question_type == 'TIME':
            for key in ['เวลา', 'วันที่', 'วัน', 'เดือน', 'ปี']:
                if any(key in k for k in best_row.keys()):
                    for k, v in best_row.items():
                        if key in k:
                            answer = v
                            break
        elif question_type == 'QUANTITY':
            for key in ['จำนวน', 'ราคา', 'ค่า', 'ปริมาณ']:
                if any(key in k for k in best_row.keys()):
                    for k, v in best_row.items():
                        if key in k:
                            answer = v
                            break
        
        # If no specific answer found, return the whole row
        if not answer:
            answer = ', '.join([f'{k}: {v}' for k, v in best_row.items()])
            
        # Calculate confidence score
        score = min(1.0, 0.3 + 0.7 * (len(focus_words) / max(len(question.split()), 1)))
        
        return {
            'answer': answer,
            'score': score
        }

def answer_question(question: str, context: str) -> Dict[str, Union[str, float]]:
    """
    Answer a question based on the context
    
    Args:
        question (str): Input question
        context (str): Input context
        
    Returns:
        Dict[str, Union[str, float]]: Answer with confidence score
    """
    qa = ThaiQuestionAnswering()
    return qa.answer_question(question, context)

def answer_from_table(question: str, table: List[Dict[str, str]]) -> Dict[str, Union[str, float]]:
    """
    Answer a question based on tabular data
    
    Args:
        question (str): Input question
        table (List[Dict[str, str]]): Table data as list of dictionaries
        
    Returns:
        Dict[str, Union[str, float]]: Answer with confidence score
    """
    qa = ThaiQuestionAnswering()
    return qa.answer_from_table(question, table) 