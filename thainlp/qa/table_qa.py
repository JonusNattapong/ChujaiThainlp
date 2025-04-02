"""
Question answering for tabular data in Thai
"""
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from ..tokenization import word_tokenize
from ..similarity.sentence_similarity import SentenceSimilarity
from ..core.transformers import TransformerBase

class TableQuestionAnswering(TransformerBase):
    def __init__(self, model_name: str = ""):
        super().__init__(model_name)
        self.similarity = SentenceSimilarity()
        
    def answer(
        self,
        question: str,
        table: pd.DataFrame,
        max_matches: int = 3
    ) -> Dict[str, Any]:
        """Answer question based on tabular data
        
        Args:
            question: Question text
            table: Pandas DataFrame containing table data
            max_matches: Maximum number of matching cells to consider
            
        Returns:
            Dict containing:
            - answer: Answer text
            - score: Confidence score
            - matches: List of matching cell locations
        """
        # Tokenize question
        q_tokens = word_tokenize(question)
        
        # Find matching cells
        matches = self._find_matching_cells(q_tokens, table, max_matches)
        if not matches:
            return {
                'answer': '',
                'score': 0.0,
                'matches': []
            }
            
        # Construct answer from matches
        answer, score = self._construct_answer(matches, table)
        
        return {
            'answer': answer,
            'score': score,
            'matches': [m[0] for m in matches]
        }
        
    def _find_matching_cells(
        self,
        q_tokens: List[str],
        table: pd.DataFrame,
        max_matches: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Find table cells matching question tokens"""
        matches = []
        
        # Search each cell
        for i, row in table.iterrows():
            for j, cell in enumerate(row):
                cell_str = str(cell)
                cell_tokens = word_tokenize(cell_str)
                
                # Calculate match score
                overlap = set(q_tokens) & set(cell_tokens)
                if not overlap:
                    continue
                    
                score = len(overlap) / len(q_tokens)
                matches.append(((i, j), score))
                
        # Sort by score and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_matches]
        
    def _construct_answer(
        self,
        matches: List[Tuple[Tuple[int, int], float]],
        table: pd.DataFrame
    ) -> Tuple[str, float]:
        """Construct answer from matching cells"""
        if len(matches) == 1:
            # Single match - return cell value
            (i, j), score = matches[0]
            return str(table.iloc[i, j]), score
            
        # Multiple matches - combine related cells
        values = []
        for (i, j), score in matches:
            # Get cell value
            value = str(table.iloc[i, j])
            
            # Get column header
            header = table.columns[j]
            
            # Format as key-value pair
            values.append(f"{header}: {value}")
            
        answer = ", ".join(values)
        avg_score = sum(s for _, s in matches) / len(matches)
        
        return answer, avg_score
        
    def answer_multiple(
        self,
        questions: List[str],
        table: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions about same table"""
        return [
            self.answer(q, table)
            for q in questions
        ]