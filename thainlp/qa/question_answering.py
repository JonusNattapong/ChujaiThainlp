"""
Advanced Thai question answering system with specialized preprocessing and multilingual support
"""
from typing import List, Dict, Tuple, Optional, Union
import re
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer, util
from ..core.transformers import TransformerBase
from ..tokenization import word_tokenize
from ..thai_preprocessor import ThaiTextPreprocessor
from ..similarity.sentence_similarity import SentenceSimilarity

class ThaiQuestionAnswering(TransformerBase):
    """Thai-specific question answering model with enhanced preprocessing"""
    
    def __init__(self, 
                 model_name: str = "monsoon-nlp/bert-base-thai-squad",
                 retriever_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8,
                 max_seq_length: int = 384,
                 doc_stride: int = 128):
        """Initialize Thai QA model
        
        Args:
            model_name: Name of QA model optimized for Thai
            retriever_model: Model for passage retrieval
            device: Device to run model on
            batch_size: Batch size for processing
            max_seq_length: Maximum sequence length
            doc_stride: Stride for document processing
        """
        super().__init__(model_name)
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.preprocessor = ThaiTextPreprocessor()
        
        # Load models
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load retriever model
        self.retriever = SentenceTransformer(retriever_model).to(device)
        
        # Thai-specific patterns
        self.thai_question_words = {
            'ใคร', 'อะไร', 'ที่ไหน', 'เมื่อไหร่', 'เมื่อไร',
            'ทำไม', 'อย่างไร', 'เท่าไหร่', 'เท่าไร', 'กี่'
        }
        
    def preprocess_thai_text(self, text: str) -> str:
        """Enhanced preprocessing for Thai text
        
        Args:
            text: Input Thai text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = self.preprocessor.preprocess(text)
        
        # Handle Thai-specific patterns
        text = self._normalize_thai_numbers(text)
        text = self._expand_thai_abbreviations(text)
        text = self._handle_thai_particles(text)
        
        return text
        
    def _normalize_thai_numbers(self, text: str) -> str:
        """Convert Thai numerals to Arabic numerals"""
        thai_digits = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
        return text.translate(thai_digits)
        
    def _expand_thai_abbreviations(self, text: str) -> str:
        """Expand common Thai abbreviations"""
        abbrev_map = {
            'กทม': 'กรุงเทพมหานคร',
            'จว.': 'จังหวัด',
            'อ.': 'อำเภอ',
            'ต.': 'ตำบล',
            'รร.': 'โรงเรียน',
            'มรภ.': 'มหาวิทยาลัยราชภัฏ',
            'รพ.': 'โรงพยาบาล'
        }
        
        for abbrev, full in abbrev_map.items():
            text = re.sub(rf'\b{abbrev}\b', full, text)
            
        return text
        
    def _handle_thai_particles(self, text: str) -> str:
        """Handle Thai particles and question markers"""
        # Remove redundant question particles
        text = re.sub(r'(ไหม|หรือไม่|มั้ย|รึเปล่า)\s*\?', '?', text)
        
        # Normalize question markers
        text = re.sub(r'[\?？]+', '?', text)
        
        return text
        
    def _split_into_thai_passages(self, text: str, max_len: int = 500) -> List[str]:
        """Split Thai text into passages respecting sentence boundaries
        
        Args:
            text: Thai text to split
            max_len: Maximum passage length
            
        Returns:
            List of passages
        """
        # Thai sentence boundary markers
        boundaries = {'|', '.', '?', '!', '\n', '।', '။', '၏', '。', '！', '？'}
        
        passages = []
        current = []
        current_len = 0
        
        # Split into sentences using Thai-aware rules
        tokens = word_tokenize(text)
        sentence = []
        
        for token in tokens:
            sentence.append(token)
            if any(b in token for b in boundaries):
                if sentence:
                    joined = ''.join(sentence)
                    if current_len + len(joined) > max_len and current:
                        passages.append(''.join(current))
                        current = []
                        current_len = 0
                        
                    current.append(joined)
                    current_len += len(joined)
                    sentence = []
                    
        if sentence:
            joined = ''.join(sentence)
            if current_len + len(joined) <= max_len:
                current.append(joined)
            else:
                if current:
                    passages.append(''.join(current))
                current = [joined]
                
        if current:
            passages.append(''.join(current))
            
        return passages
        
    def answer_question(self,
                       question: Union[str, List[str]],
                       context: Union[str, pd.DataFrame],
                       max_answer_len: int = 100,
                       return_scores: bool = True,
                       num_answers: int = 1,
                       translate_answer: bool = False) -> Union[Dict, List[Dict]]:
        """Answer Thai questions with enhanced features
        
        Args:
            question: Question or list of questions in Thai
            context: Text passage or pandas DataFrame
            max_answer_len: Maximum answer length
            return_scores: Whether to return confidence scores
            num_answers: Number of answers to return per question
            translate_answer: Whether to translate answer to English
            
        Returns:
            Dict or list of dicts containing:
            - answer: Extracted answer text
            - translated_answer: English translation (if requested)
            - score: Confidence score (if return_scores=True)
            - context: Source passage
            - start: Start position in context (text QA only)
            - end: End position in context (text QA only)
        """
        # Preprocess inputs
        if isinstance(question, str):
            questions = [question]
            single_query = True
        else:
            questions = question
            single_query = False
            
        questions = [self.preprocess_thai_text(q) for q in questions]
        
        if isinstance(context, str):
            context = self.preprocess_thai_text(context)
            
        # Get answers using appropriate method
        if isinstance(context, pd.DataFrame):
            results = self._table_qa(questions, context, num_answers)
        else:
            results = self._text_qa(
                questions,
                context,
                max_answer_len,
                return_scores,
                num_answers
            )
            
        # Translate answers if requested
        if translate_answer:
            translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-th-en",
                device=self.device
            )
            
            for result in results:
                if isinstance(result, dict):
                    result['translated_answer'] = translator(result['answer'])[0]['translation_text']
                elif isinstance(result, list):
                    for r in result:
                        r['translated_answer'] = translator(r['answer'])[0]['translation_text']
                        
        return results[0] if single_query else results
        
    def _text_qa(self,
                 questions: List[str],
                 context: str,
                 max_answer_len: int,
                 return_scores: bool,
                 num_answers: int) -> List[Dict]:
        """Process Thai text-based questions with improved context handling"""
        # Split context into passages using Thai-aware splitting
        passages = self._split_into_thai_passages(context)
        
        # Get passage embeddings
        passage_embeddings = self.retriever.encode(
            passages,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        results = []
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            
            # Get question embeddings
            question_embeddings = self.retriever.encode(
                batch_questions,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Find relevant passages using improved similarity
            scores = util.cos_sim(question_embeddings, passage_embeddings)
            rerank_scores = self._rerank_passages(batch_questions, passages, scores)
            
            top_k = min(3, len(passages))
            top_passages = {}
            
            for q_idx, q_scores in enumerate(rerank_scores):
                top_indices = q_scores.topk(top_k).indices
                q_passages = [passages[idx] for idx in top_indices]
                top_passages[q_idx] = q_passages
                
            # Get answers from relevant passages
            batch_results = []
            for q_idx, q in enumerate(batch_questions):
                q_results = []
                
                for passage in top_passages[q_idx]:
                    inputs = self.tokenizer(
                        q,
                        passage,
                        max_length=self.max_seq_length,
                        truncation=True,
                        stride=self.doc_stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        
                    start_logits = outputs.start_logits
                    end_logits = outputs.end_logits
                    
                    start_end_pairs = self._get_best_spans(
                        start_logits[0],
                        end_logits[0],
                        max_answer_len,
                        num_answers
                    )
                    
                    for start_idx, end_idx, score in start_end_pairs:
                        answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
                        answer = self.tokenizer.decode(answer_tokens)
                        
                        result = {
                            'answer': answer,
                            'context': passage,
                            'start': start_idx,
                            'end': end_idx
                        }
                        
                        if return_scores:
                            result['score'] = score.item()
                            
                        q_results.append(result)
                
                # Sort and filter results
                if return_scores:
                    q_results = sorted(
                        q_results,
                        key=lambda x: x['score'],
                        reverse=True
                    )[:num_answers]
                else:
                    q_results = q_results[:num_answers]
                    
                if num_answers == 1:
                    batch_results.append(q_results[0])
                else:
                    batch_results.append(q_results)
                    
            results.extend(batch_results)
            
        return results
        
    def _rerank_passages(self,
                        questions: List[str],
                        passages: List[str],
                        initial_scores: torch.Tensor) -> torch.Tensor:
        """Rerank passages using additional Thai-specific features"""
        final_scores = initial_scores.clone()
        
        for i, question in enumerate(questions):
            # Check for question word matches
            q_words = set(word_tokenize(question))
            question_types = q_words.intersection(self.thai_question_words)
            
            for j, passage in enumerate(passages):
                p_words = set(word_tokenize(passage))
                
                # Boost score for passages containing question words
                if question_types & p_words:
                    final_scores[i, j] *= 1.2
                    
                # Boost score for numeric matches in quantity questions
                if {'เท่าไหร่', 'เท่าไร', 'กี่'} & question_types:
                    if bool(re.search(r'[0-9๐-๙]', passage)):
                        final_scores[i, j] *= 1.1
                        
                # Boost score for temporal expressions in time questions
                if {'เมื่อไหร่', 'เมื่อไร'} & question_types:
                    if bool(re.search(r'วันที่|เดือน|ปี|เวลา|ช่วง', passage)):
                        final_scores[i, j] *= 1.1
                        
        return final_scores
        
    def fine_tune(self,
                 train_data: List[Dict],
                 val_data: Optional[List[Dict]] = None,
                 epochs: int = 3,
                 learning_rate: float = 3e-5,
                 warmup_steps: int = 500):
        """Fine-tune the QA model on Thai data
        
        Args:
            train_data: List of dicts with 'question', 'context', 'answer' keys
            val_data: Optional validation data
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        from transformers import get_linear_schedule_with_warmup
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Create scheduler
        num_training_steps = epochs * len(train_data) // self.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(train_data), self.batch_size):
                batch_data = train_data[i:i + self.batch_size]
                
                # Preprocess batch data
                questions = [self.preprocess_thai_text(d['question']) for d in batch_data]
                contexts = [self.preprocess_thai_text(d['context']) for d in batch_data]
                
                # Prepare inputs
                inputs = self.tokenizer(
                    questions,
                    contexts,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Prepare answer positions
                start_positions = []
                end_positions = []
                
                for data in batch_data:
                    answer_start = data['context'].find(data['answer'])
                    answer_end = answer_start + len(data['answer'])
                    
                    char_to_token = inputs.char_to_token(
                        len(start_positions),
                        answer_start
                    )
                    start_positions.append(char_to_token)
                    
                    char_to_token = inputs.char_to_token(
                        len(end_positions),
                        answer_end
                    )
                    end_positions.append(char_to_token)
                    
                start_positions = torch.tensor(start_positions).to(self.device)
                end_positions = torch.tensor(end_positions).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    **inputs,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Validation
            if val_data:
                self.model.eval()
                val_loss = 0
                correct_answers = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_data), self.batch_size):
                        batch_data = val_data[i:i + self.batch_size]
                        
                        questions = [self.preprocess_thai_text(d['question']) for d in batch_data]
                        contexts = [self.preprocess_thai_text(d['context']) for d in batch_data]
                        
                        inputs = self.tokenizer(
                            questions,
                            contexts,
                            max_length=self.max_seq_length,
                            truncation=True,
                            padding=True,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        for j, data in enumerate(batch_data):
                            predicted_answer = self.answer_question(
                                data['question'],
                                data['context']
                            )['answer']
                            
                            if predicted_answer.lower() in data['answer'].lower():
                                correct_answers += 1
                                
                accuracy = correct_answers / len(val_data)
                print(f"Validation Accuracy: {accuracy:.4f}")
                
                self.model.train()

def answer_question(question: str,
                   context: str,
                   model: str = "monsoon-nlp/bert-base-thai-squad",
                   **kwargs) -> Dict:
    """
    Simplified interface for Thai question answering
    
    Args:
        question: Question in Thai
        context: Context text in Thai
        model: Model name to use
        **kwargs: Additional arguments for ThaiQuestionAnswering
        
    Returns:
        Dict containing answer and metadata
    """
    qa = ThaiQuestionAnswering(model_name=model)
    return qa.answer_question(question, context, **kwargs)