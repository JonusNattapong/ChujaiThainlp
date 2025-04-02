"""
Standard Thai NLP Pipeline (Tokenize -> POS -> NER)
"""
from typing import List, Tuple, Dict, Any, Callable
from .base import Pipeline
from thainlp.tokenization import maximum_matching
from thainlp.pos_tagging import hmm_tagger
from thainlp.ner import rule_based # Using rule_based NER as default for now
from thainlp.utils.thai_text import process_text

# Define a default tokenizer instance
default_tokenizer = maximum_matching.MaximumMatchingTokenizer()
# Define a default POS tagger instance
default_pos_tagger = hmm_tagger.HMMTagger()
# Define a default NER tagger instance
default_ner_tagger = rule_based.ThaiNER() # Using ThaiNER from rule_based module

class StandardThaiPipeline(Pipeline):
    """
    A standard pipeline for common Thai NLP tasks:
    1. Sentence Splitting
    2. Tokenization
    3. Part-of-Speech (POS) Tagging
    4. Named Entity Recognition (NER) Tagging

    Returns a list of sentences, where each sentence is a list of tuples:
    [(token, pos_tag, ner_tag), ...]
    """
    
    def _get_default_processors(self) -> List[Callable]:
        """Returns the default sequence of processors."""
        return [
            self._split_sentences,
            self._tokenize_sentences,
            self._tag_pos,
            self._tag_ner
        ]

    def _split_sentences(self, text: str) -> List[str]:
        """Processor for splitting text into sentences."""
        return process_text(text).normalize().get_sentences()

    def _tokenize_sentences(self, sentences: List[str]) -> List[List[str]]:
        """Processor for tokenizing each sentence."""
        return [default_tokenizer.tokenize(sentence) for sentence in sentences]

    def _tag_pos(self, tokenized_sentences: List[List[str]]) -> List[List[Tuple[str, str]]]:
        """Processor for POS tagging tokenized sentences."""
        return [default_pos_tagger.tag(tokens) for tokens in tokenized_sentences]

    def _tag_ner(self, pos_tagged_sentences: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str, str]]]:
        """Processor for NER tagging POS-tagged sentences."""
        ner_tagged_sentences = []
        for sentence in pos_tagged_sentences:
            # NER tagger expects list of tokens, not (token, pos) tuples
            tokens = [token for token, pos in sentence]
            # Use the default NER tagger's extract_entities method
            ner_results = default_ner_tagger.extract_entities(" ".join(tokens))

            # Map NER results back to tokens
            # NER results are (entity_text, type, start, end) tuples
            token_ner_tags = ['O'] * len(tokens) # Default to Outside
            current_pos = 0
            token_spans = []
            for token in tokens:
                start = current_pos
                end = start + len(token)
                token_spans.append((start, end))
                # Add 1 for the space between tokens when joined
                current_pos = end + 1

            for entity_text, entity_type, ent_start, ent_end in ner_results:
                # Find tokens that fall within the entity span
                for i, (tok_start, tok_end) in enumerate(token_spans):
                    # Check for overlap
                    if max(tok_start, ent_start) < min(tok_end, ent_end):
                        # Apply BIO tagging scheme
                        if token_ner_tags[i] == 'O': # Start of entity
                            token_ner_tags[i] = f"B-{entity_type}"
                        else: # Inside entity
                            token_ner_tags[i] = f"I-{entity_type}"

            # Combine token, pos, and ner tags
            combined_sentence = []
            for i, (token, pos) in enumerate(sentence):
                ner_tag = token_ner_tags[i]
                combined_sentence.append((token, pos, ner_tag))
            ner_tagged_sentences.append(combined_sentence)
        return ner_tagged_sentences

    def __call__(self, text: str, **kwargs) -> List[List[Tuple[str, str, str]]]:
        """
        Process the input text through the standard pipeline.

        Args:
            text (str): The input Thai text.
            **kwargs: Additional arguments (currently unused but kept for compatibility).

        Returns:
            List[List[Tuple[str, str, str]]]:
                A list of sentences. Each sentence is a list of tuples,
                where each tuple contains (token, pos_tag, ner_tag).
        """
        processed_data = text
        # Ensure processors are initialized
        if not hasattr(self, 'processors'):
             self.processors = self._get_default_processors()

        for processor in self.processors:
            processed_data = processor(processed_data)

        # Validate output format
        if not isinstance(processed_data, list):
             print(f"Warning: Pipeline output type mismatch. Expected List, got {type(processed_data)}")
             return []
        if processed_data and not isinstance(processed_data[0], list):
             print(f"Warning: Pipeline output type mismatch. Expected List[List], got List[{type(processed_data[0])}]")
             return []

        return processed_data

# Example usage
if __name__ == "__main__":
    pipeline = StandardThaiPipeline()
    text = "นายสมชายเดินทางไปกรุงเทพมหานครเมื่อวานนี้ เวลา 10.00 น. เพื่อพบกับ ดร.สมศรี ที่บริษัท ABC จำกัด"
    
    print(f"Processing text: {text}")
    results = pipeline(text)
    
    print("\nPipeline Results:")
    for i, sentence in enumerate(results):
        print(f"\nSentence {i+1}:")
        for token, pos, ner in sentence:
            print(f"- Token: {token:<15} POS: {pos:<10} NER: {ner}")
