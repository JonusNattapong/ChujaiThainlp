"""
Table Question Answering for Thai Text using Transformer models
"""

from typing import List, Dict, Union, Optional, Any
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForTableQuestionAnswering,
    AutoTokenizer,
    TapasConfig,
    TapasTokenizer
)
from pythainlp.tokenize import word_tokenize
from pythainlp.translate import translate
from ..core.transformers import TransformerBase

class ThaiTableQA(TransformerBase):
    """Advanced table question answering for Thai text"""
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        use_sql_queries: bool = True,
        max_rows: int = 100,
        max_cols: int = 50,
        **kwargs
    ):
        """Initialize table QA model
        
        Args:
            model_name_or_path: Name or path of the model
            use_sql_queries: Whether to support SQL-like queries
            max_rows: Maximum number of table rows to process
            max_cols: Maximum number of table columns to process
            **kwargs: Additional arguments for model initialization
        """
        if model_name_or_path is None:
            model_name_or_path = "google/tapas-base-finetuned-wtq"
            
        self.use_sql_queries = use_sql_queries
        self.max_rows = max_rows
        self.max_cols = max_cols
        
        # Initialize TAPAS configuration
        config = TapasConfig.from_pretrained(
            model_name_or_path,
            num_aggregation_labels=4,  # none, sum, average, count
            use_answer_as_supervision=True
        )
        
        super().__init__(
            model_name_or_path=model_name_or_path,
            task_type="table-question-answering",
            config=config,
            **kwargs
        )
        
        # Initialize tokenizer specifically for TAPAS
        self.tokenizer = TapasTokenizer.from_pretrained(model_name_or_path)
        
        # SQL query templates
        self.sql_templates = {
            'select': "SELECT {cols} FROM table WHERE {conds}",
            'count': "SELECT COUNT({col}) FROM table WHERE {conds}",
            'sum': "SELECT SUM({col}) FROM table WHERE {conds}",
            'average': "SELECT AVG({col}) FROM table WHERE {conds}",
            'max': "SELECT MAX({col}) FROM table WHERE {conds}",
            'min': "SELECT MIN({col}) FROM table WHERE {conds}"
        }
        
    def _preprocess_table(self, table: Union[pd.DataFrame, str, Dict]) -> pd.DataFrame:
        """Preprocess input table
        
        Args:
            table: Input table as DataFrame, path, or dictionary
            
        Returns:
            Preprocessed pandas DataFrame
        """
        if isinstance(table, str):
            # Load table from file
            if table.endswith('.csv'):
                df = pd.read_csv(table)
            elif table.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(table)
            elif table.endswith('.json'):
                df = pd.read_json(table)
            else:
                raise ValueError(f"Unsupported file format: {table}")
        elif isinstance(table, dict):
            df = pd.DataFrame.from_dict(table)
        elif isinstance(table, pd.DataFrame):
            df = table.copy()
        else:
            raise ValueError("Table must be a DataFrame, file path, or dictionary")
            
        # Limit table size
        df = df.iloc[:self.max_rows, :self.max_cols]
        
        # Convert all column names to string
        df.columns = df.columns.astype(str)
        
        # Convert all values to string
        df = df.astype(str)
        
        return df
        
    def _translate_question(self, question: str, to_lang: str = 'en') -> str:
        """Translate question between Thai and English
        
        Args:
            question: Input question
            to_lang: Target language code
            
        Returns:
            Translated question
        """
        return translate(question, 'th', to_lang)
        
    def _generate_sql_query(
        self,
        question: str,
        table: pd.DataFrame
    ) -> str:
        """Generate SQL-like query from natural language question
        
        Args:
            question: Question in Thai
            table: Input table
            
        Returns:
            SQL-like query
        """
        # Translate question to English for better pattern matching
        eng_question = self._translate_question(question)
        
        # Identify query type
        query_type = 'select'  # default
        if any(w in eng_question.lower() for w in ['how many', 'count']):
            query_type = 'count'
        elif any(w in eng_question.lower() for w in ['sum', 'total']):
            query_type = 'sum'
        elif any(w in eng_question.lower() for w in ['average', 'mean']):
            query_type = 'average'
        elif any(w in eng_question.lower() for w in ['maximum', 'highest']):
            query_type = 'max'
        elif any(w in eng_question.lower() for w in ['minimum', 'lowest']):
            query_type = 'min'
            
        # Extract column names mentioned in question
        cols = []
        for col in table.columns:
            if col.lower() in eng_question.lower():
                cols.append(col)
                
        # Generate conditions based on values in table
        conds = []
        for col in cols:
            for val in table[col].unique():
                if str(val).lower() in eng_question.lower():
                    conds.append(f"{col} = '{val}'")
                    
        # Build query
        if query_type == 'select':
            cols_str = ', '.join(cols) if cols else '*'
            conds_str = ' AND '.join(conds) if conds else '1=1'
            query = self.sql_templates['select'].format(cols=cols_str, conds=conds_str)
        else:
            col = cols[0] if cols else table.columns[0]
            conds_str = ' AND '.join(conds) if conds else '1=1'
            query = self.sql_templates[query_type].format(col=col, conds=conds_str)
            
        return query
        
    def answer(
        self,
        question: str,
        table: Union[pd.DataFrame, str, Dict],
        return_sql: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Answer question about table
        
        Args:
            question: Question in Thai
            table: Input table
            return_sql: Whether to return generated SQL query
            **kwargs: Additional arguments for model
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Preprocess table
        df = self._preprocess_table(table)
        
        # Generate SQL query if requested
        sql_query = None
        if self.use_sql_queries and return_sql:
            sql_query = self._generate_sql_query(question, df)
        
        # Prepare inputs for model
        inputs = self.tokenizer(
            table=df,
            queries=[question],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model predictions
        outputs = self.model(**inputs)
        
        # Process outputs
        answer_cells = []
        aggregation_type = None
        
        # Get selected cells
        logits = outputs.logits.detach().cpu().numpy()
        predictions = np.where(logits > 0.5)
        for row, col in zip(predictions[1], predictions[2]):
            if row < len(df) and col < len(df.columns):
                answer_cells.append({
                    'value': df.iloc[row, col],
                    'row': row,
                    'column': df.columns[col]
                })
        
        # Get aggregation type
        if hasattr(outputs, 'aggregation_logits'):
            agg_logits = outputs.aggregation_logits.detach().cpu().numpy()
            agg_pred = np.argmax(agg_logits)
            aggregation_type = ['NONE', 'SUM', 'AVERAGE', 'COUNT'][agg_pred]
        
        # Prepare result
        result = {
            'answer_cells': answer_cells,
            'aggregation_type': aggregation_type,
            'confidence': float(outputs.logits.max())
        }
        
        if sql_query:
            result['sql_query'] = sql_query
            
        return result
        
    def batch_answer(
        self,
        questions: List[str],
        table: Union[pd.DataFrame, str, Dict],
        batch_size: int = 8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions about table
        
        Args:
            questions: List of questions in Thai
            table: Input table
            batch_size: Batch size for processing
            **kwargs: Additional arguments for model
            
        Returns:
            List of answer dictionaries
        """
        results = []
        
        # Process questions in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_results = [
                self.answer(q, table, **kwargs)
                for q in batch_questions
            ]
            results.extend(batch_results)
            
        return results 