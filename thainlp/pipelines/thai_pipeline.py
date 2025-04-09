"""
User-friendly Thai NLP pipeline combining multiple functionalities
"""
from typing import Optional, Union, List, Dict, Any
import pandas as pd
from ..tokenization import word_tokenize
from ..ner import ThaiNamedEntityRecognition
from ..sentiment import ThaiSentimentAnalyzer
from ..qa import ThaiQuestionAnswering, TableQuestionAnswering
from ..generation import ThaiTextGenerator
from ..similarity import ThaiTextAnalyzer
from ..dialects import ThaiDialectProcessor, DialectTokenizer
from ..utils.thai_utils import normalize_text, clean_thai_text

class ThaiNLPPipeline:
    """
    Unified Thai NLP pipeline providing easy access to all functionality
    
    Example usage:
        nlp = ThaiNLPPipeline()
        
        # Basic text processing
        tokens = nlp.tokenize("สวัสดีครับ")
        
        # Named entities
        entities = nlp.get_entities("นายสมชาย ไปกรุงเทพ")
        
        # Sentiment analysis
        sentiment = nlp.analyze_sentiment("อาหารอร่อยมาก")
        
        # Question answering
        answer = nlp.answer_question("ใครไปกรุงเทพ?", "นายสมชาย ไปกรุงเทพ")
        
        # Text generation
        text = nlp.generate("วันนี้อากาศ")
        
        # Dialect processing
        dialect = nlp.detect_dialect("ข้อยสิไปตลาด")
    """
    
    def __init__(self, 
                 load_all: bool = False,
                 device: str = "cuda"):
        """Initialize pipeline
        
        Args:
            load_all: Whether to load all components at initialization
            device: Device to use for models
        """
        self.device = device
        self._tokenizer = None
        self._ner = None
        self._sentiment = None
        self._qa = None
        self._table_qa = None
        self._generator = None
        self._analyzer = None
        self._dialect_processor = None
        self._dialect_tokenizer = None
        
        # Load components if requested
        if load_all:
            self.get_tokenizer()
            self.get_ner()
            self.get_sentiment()
            self.get_qa()
            self.get_table_qa()
            self.get_generator()
            self.get_analyzer()
            self.get_dialect_processor()
            self.get_dialect_tokenizer()

    def get_tokenizer(self):
        """Get or initialize tokenizer"""
        if self._tokenizer is None:
            from ..tokenization import ThaiTokenizer
            self._tokenizer = ThaiTokenizer()
        return self._tokenizer

    def get_ner(self):
        """Get or initialize NER"""
        if self._ner is None:
            self._ner = ThaiNamedEntityRecognition()
        return self._ner
        
    def get_sentiment(self):
        """Get or initialize sentiment analyzer"""
        if self._sentiment is None:
            self._sentiment = ThaiSentimentAnalyzer()
        return self._sentiment
        
    def get_qa(self):
        """Get or initialize QA"""
        if self._qa is None:
            self._qa = ThaiQuestionAnswering()
        return self._qa
        
    def get_table_qa(self):
        """Get or initialize Table QA"""
        if self._table_qa is None:
            self._table_qa = TableQuestionAnswering()
        return self._table_qa
        
    def get_generator(self):
        """Get or initialize text generator"""
        if self._generator is None:
            self._generator = ThaiTextGenerator()
        return self._generator
        
    def get_analyzer(self):
        """Get or initialize text analyzer"""
        if self._analyzer is None:
            self._analyzer = ThaiTextAnalyzer()
        return self._analyzer

    def get_dialect_processor(self):
        """Get or initialize dialect processor"""
        if self._dialect_processor is None:
            self._dialect_processor = ThaiDialectProcessor()
        return self._dialect_processor

    def get_dialect_tokenizer(self):
        """Get or initialize dialect-aware tokenizer"""
        if self._dialect_tokenizer is None:
            self._dialect_tokenizer = DialectTokenizer()
        return self._dialect_tokenizer

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Thai text into words"""
        return word_tokenize(text)

    def get_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        return self.get_ner().extract_entities(text)

    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of text"""
        return self.get_sentiment().analyze_sentiment(text)
        
    def answer_question(self, 
                       question: str,
                       context: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Answer question based on context
        
        Args:
            question: Question text
            context: Text passage or pandas DataFrame for table QA
            
        Returns:
            Dictionary containing answer and confidence score
        """
        if isinstance(context, pd.DataFrame):
            return self.get_table_qa().answer(question, context)
        return self.get_qa().answer_question(question, context)
        
    def generate(self, 
                prompt: str,
                max_length: int = 50,
                num_sequences: int = 1,
                **kwargs) -> Union[str, List[str]]:
        """Generate text from prompt"""
        return self.get_generator().generate_text(
            prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            **kwargs
        )
        
    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        return self.get_analyzer().semantic_similarity(text1, text2)
        
    def detect_dialect(self, text: str) -> Dict[str, float]:
        """Detect Thai dialect in text
        
        Args:
            text: Thai text to analyze
            
        Returns:
            Dictionary mapping dialect codes to confidence scores
        """
        return self.get_dialect_processor().detect_dialect(text)
        
    def tokenize_with_dialect(self, 
                             text: str, 
                             dialect: Optional[str] = None,
                             auto_detect: bool = False) -> Union[List[str], Dict[str, Any]]:
        """Tokenize text with dialect awareness
        
        Args:
            text: Text to tokenize
            dialect: Specific dialect to use (northern, northeastern, southern, central)
            auto_detect: Whether to automatically detect the dialect
            
        Returns:
            If auto_detect is True, returns a dictionary with dialect information and tokens.
            Otherwise, returns a list of tokens.
        """
        tokenizer = self.get_dialect_tokenizer()
        
        if auto_detect:
            return tokenizer.detect_and_tokenize(text)
        else:
            return tokenizer.tokenize(text, dialect=dialect)
            
    def translate_dialect(self, 
                         text: str, 
                         source_dialect: Optional[str] = None,
                         target_dialect: str = "central",
                         auto_detect: bool = False) -> str:
        """Translate between Thai dialects
        
        Args:
            text: Text to translate
            source_dialect: Source dialect code (if None and auto_detect is True, will be detected)
            target_dialect: Target dialect code
            auto_detect: Whether to automatically detect the source dialect
            
        Returns:
            Translated text
        """
        processor = self.get_dialect_processor()
        
        # Auto-detect source dialect if needed
        if source_dialect is None and auto_detect:
            dialect_scores = processor.detect_dialect(text)
            source_dialect = max(dialect_scores, key=lambda k: dialect_scores[k])
        
        # If translating to standard Thai
        if target_dialect == "central":
            return processor.translate_to_standard(text, source_dialect)
        
        # If translating from standard Thai to dialect
        if source_dialect == "central":
            return processor.translate_from_standard(text, target_dialect)
            
        # If translating between dialects, go through standard Thai
        std_text = processor.translate_to_standard(text, source_dialect)
        return processor.translate_from_standard(std_text, target_dialect)
    
    def preprocess(self, text: str, normalize: bool = True) -> str:
        """Preprocess Thai text
        
        Args:
            text: Input text
            normalize: Whether to normalize Thai characters
            
        Returns:
            Preprocessed text
        """
        if normalize:
            text = normalize_text(text)
        return clean_thai_text(text)

    def process(self, 
                text: str,
                tasks: List[str] = None) -> Dict[str, Any]:
        """Process text with multiple tasks
        
        Args:
            text: Input text
            tasks: List of tasks to perform, options:
                  ["tokenize", "ner", "sentiment", "preprocess", "dialect"]
                  If None, performs all tasks
                  
        Returns:
            Dictionary containing results of all tasks
        """
        if tasks is None:
            tasks = ["tokenize", "ner", "sentiment", "preprocess"]
            
        results = {}
        
        if "preprocess" in tasks:
            results["preprocessed"] = self.preprocess(text)
            
        if "tokenize" in tasks:
            results["tokens"] = self.tokenize(text)
            
        if "ner" in tasks:
            results["entities"] = self.get_entities(text)
            
        if "sentiment" in tasks:
            results["sentiment"] = self.analyze_sentiment(text)
            
        if "dialect" in tasks:
            # Include dialect detection
            results["dialect"] = self.detect_dialect(text)
            
            # Include dialect-aware tokenization
            dialect_tokens = self.tokenize_with_dialect(text, auto_detect=True)
            results["dialect_tokenization"] = dialect_tokens
            
        return results