"""
Unified Thai NLP pipeline combining multiple components
"""
from typing import List, Dict, Optional, Union, Any
from ..classification.token_classifier import TokenClassifier
from ..qa.question_answering import QuestionAnswering
from ..translation.translator import Translator
from ..generation.text_generator import TextGenerator
from ..generation.fill_mask import FillMask
from ..summarization.summarizer import Summarizer
from ..similarity.sentence_similarity import SentenceSimilarity
from ..extensions.monitoring import ProgressTracker

class ThaiNLPPipeline:
    """Pipeline combining multiple Thai NLP capabilities"""
    
    def __init__(self,
                 components: Optional[List[str]] = None,
                 device: str = "cuda",
                 batch_size: int = 32):
        """Initialize pipeline with specified components
        
        Args:
            components: List of components to initialize
                       ('classification', 'qa', 'translation', 'generation',
                        'fill_mask', 'summarization', 'similarity')
            device: Device to run models on
            batch_size: Default batch size for processing
        """
        self.device = device
        self.batch_size = batch_size
        self.progress = ProgressTracker()
        
        # Initialize requested components
        if not components:
            components = [
                'classification', 'qa', 'translation', 'generation',
                'fill_mask', 'summarization', 'similarity'
            ]
            
        self.components = {}
        
        if 'classification' in components:
            self.components['classification'] = TokenClassifier(
                model_name="xlm-roberta-base",
                device=device,
                batch_size=batch_size
            )
            
        if 'qa' in components:
            self.components['qa'] = QuestionAnswering(
                model_name="xlm-roberta-large-squad2",
                device=device,
                batch_size=batch_size
            )
            
        if 'translation' in components:
            self.components['translation'] = Translator(
                model_name="facebook/m2m100_418M",
                device=device,
                batch_size=batch_size
            )
            
        if 'generation' in components:
            self.components['generation'] = TextGenerator(
                model_name="facebook/xglm-7.5B",
                device=device,
                batch_size=batch_size
            )
            
        if 'fill_mask' in components:
            self.components['fill_mask'] = FillMask(
                model_name="xlm-roberta-large",
                device=device,
                batch_size=batch_size
            )
            
        if 'summarization' in components:
            self.components['summarization'] = Summarizer(
                model_name="facebook/bart-large-cnn",
                device=device,
                batch_size=batch_size
            )
            
        if 'similarity' in components:
            self.components['similarity'] = SentenceSimilarity(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                device=device,
                batch_size=batch_size
            )
    
    def analyze(self,
                text: str,
                tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run multiple analyses on text
        
        Args:
            text: Input text to analyze
            tasks: List of tasks to perform
                  ('tokens', 'entities', 'translation', 'summary', etc)
                  
        Returns:
            Dict containing results for each requested task
        """
        results = {}
        
        if not tasks:
            tasks = ['tokens', 'entities', 'translation', 'summary']
            
        # Track progress
        self.progress.start_task(len(tasks))
        
        for task in tasks:
            if task == 'tokens':
                if 'classification' in self.components:
                    results['tokens'] = self.components['classification'].classify_tokens(
                        text,
                        return_confidence=True
                    )
                    
            elif task == 'entities':
                if 'classification' in self.components:
                    token_info = self.components['classification'].classify_tokens(text)
                    results['entities'] = [
                        t for t in token_info 
                        if t.get('named_entity')
                    ]
                    
            elif task == 'translation':
                if 'translation' in self.components:
                    results['translation'] = {
                        'en': self.components['translation'].translate(
                            text,
                            source_lang='th',
                            target_lang='en'
                        ),
                        'th': self.components['translation'].translate(
                            text,
                            source_lang='en',
                            target_lang='th'
                        ) if not self._is_thai(text) else None
                    }
                    
            elif task == 'summary':
                if 'summarization' in self.components:
                    results['summary'] = self.components['summarization'].summarize(
                        text,
                        ratio=0.3,
                        min_length=30,
                        max_length=130
                    )
                    
            elif task == 'generated':
                if 'generation' in self.components:
                    results['generated'] = self.components['generation'].generate(
                        text,
                        max_length=100,
                        num_return_sequences=3
                    )
                    
            elif task == 'filled_masks':
                if 'fill_mask' in self.components:
                    results['filled_masks'] = self.components['fill_mask'].fill_mask(
                        text,
                        top_k=5
                    )
                    
            self.progress.update(1)
            
        self.progress.end_task()
        return results
    
    def answer_question(self,
                       question: str,
                       context: Union[str, Dict],
                       **kwargs) -> Dict[str, Any]:
        """Answer question using provided context
        
        Args:
            question: Question to answer
            context: Text passage or structured data
            **kwargs: Additional parameters for QA model
            
        Returns:
            Dict containing answer and metadata
        """
        if 'qa' not in self.components:
            raise ValueError("QA component not initialized")
            
        return self.components['qa'].answer_question(
            question,
            context,
            **kwargs
        )
    
    def translate(self,
                 text: Union[str, List[str]],
                 source_lang: str = "th",
                 target_lang: str = "en",
                 **kwargs) -> Union[str, List[str]]:
        """Translate text between languages
        
        Args:
            text: Text(s) to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional translation parameters
            
        Returns:
            Translated text(s)
        """
        if 'translation' not in self.components:
            raise ValueError("Translation component not initialized")
            
        return self.components['translation'].translate(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            **kwargs
        )
    
    def generate_text(self,
                     prompt: Union[str, List[str]],
                     **kwargs) -> Union[str, List[str]]:
        """Generate text from prompt(s)
        
        Args:
            prompt: Input prompt(s)
            **kwargs: Generation parameters
            
        Returns:
            Generated text(s)
        """
        if 'generation' not in self.components:
            raise ValueError("Generation component not initialized")
            
        return self.components['generation'].generate(
            prompt,
            **kwargs
        )
    
    def summarize(self,
                 text: Union[str, List[str]],
                 **kwargs) -> Union[str, List[str]]:
        """Generate summary of text(s)
        
        Args:
            text: Text(s) to summarize
            **kwargs: Summarization parameters
            
        Returns:
            Generated summary/summaries
        """
        if 'summarization' not in self.components:
            raise ValueError("Summarization component not initialized")
            
        return self.components['summarization'].summarize(
            text,
            **kwargs
        )
    
    def calculate_similarity(self,
                           texts1: Union[str, List[str]],
                           texts2: Union[str, List[str]],
                           **kwargs) -> Union[float, List[float]]:
        """Calculate similarity between text pairs
        
        Args:
            texts1: First text(s)
            texts2: Second text(s)
            **kwargs: Similarity parameters
            
        Returns:
            Similarity score(s)
        """
        if 'similarity' not in self.components:
            raise ValueError("Similarity component not initialized")
            
        return self.components['similarity'].get_similarity(
            texts1,
            texts2,
            **kwargs
        )
    
    def find_similar_texts(self,
                          query: Union[str, List[str]],
                          candidates: List[str],
                          **kwargs) -> Union[List[tuple], List[List[tuple]]]:
        """Find most similar texts from candidates
        
        Args:
            query: Query text(s)
            candidates: Candidate texts to search
            **kwargs: Search parameters
            
        Returns:
            Most similar texts with scores
        """
        if 'similarity' not in self.components:
            raise ValueError("Similarity component not initialized")
            
        return self.components['similarity'].find_most_similar(
            query,
            candidates,
            **kwargs
        )
    
    def fill_mask(self,
                 text: Union[str, List[str]],
                 **kwargs) -> Union[List[Dict], List[List[Dict]]]:
        """Fill masked tokens in text(s)
        
        Args:
            text: Text(s) with mask tokens
            **kwargs: Mask filling parameters
            
        Returns:
            Predictions for masked tokens
        """
        if 'fill_mask' not in self.components:
            raise ValueError("Fill-mask component not initialized")
            
        return self.components['fill_mask'].fill_mask(
            text,
            **kwargs
        )
    
    def _is_thai(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        for char in text:
            if '\u0E00' <= char <= '\u0E7F':
                return True
        return False

    def get_component(self, name: str) -> Any:
        """Get specific component by name"""
        if name not in self.components:
            raise ValueError(f"Component {name} not initialized")
        return self.components[name]