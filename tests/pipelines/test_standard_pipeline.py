"""
Tests for the Standard Thai NLP Pipeline
"""
import unittest
from thainlp.pipelines import StandardThaiPipeline

class TestStandardThaiPipeline(unittest.TestCase):
    """Test the StandardThaiPipeline class"""
    
    def setUp(self):
        """Set up the test case"""
        self.pipeline = StandardThaiPipeline()
        self.text1 = "แมวกินปลา"
        self.text2 = "นายสมชายเดินทางไปกรุงเทพฯ เมื่อวานนี้"
        self.text3 = "" # Empty text
        self.text4 = "English text only" # Non-Thai text
        
    def test_pipeline_initialization(self):
        """Test pipeline initializes with default processors"""
        self.assertIsInstance(self.pipeline, StandardThaiPipeline)
        processors = self.pipeline.get_processor_names()
        self.assertIn('_split_sentences', processors)
        self.assertIn('_tokenize_sentences', processors)
        self.assertIn('_tag_pos', processors)
        self.assertIn('_tag_ner', processors)
        
    def test_pipeline_call_simple(self):
        """Test pipeline processing for a simple sentence"""
        results = self.pipeline(self.text1)
        
        # Check overall structure (list of sentences)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1) # Should be one sentence
        
        # Check sentence structure (list of tuples)
        sentence = results[0]
        self.assertIsInstance(sentence, list)
        self.assertGreater(len(sentence), 0) # Should have tokens
        
        # Check token structure (tuple of 3 strings)
        token_data = sentence[0]
        self.assertIsInstance(token_data, tuple)
        self.assertEqual(len(token_data), 3)
        self.assertTrue(all(isinstance(item, str) for item in token_data))
        
        # Check specific token data (example)
        tokens = [t[0] for t in sentence]
        self.assertIn("แมว", tokens)
        self.assertIn("กิน", tokens)
        self.assertIn("ปลา", tokens)
        
    def test_pipeline_call_complex(self):
        """Test pipeline processing for a more complex sentence"""
        results = self.pipeline(self.text2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1) 
        
        sentence = results[0]
        self.assertIsInstance(sentence, list)
        
        # Check if NER tags are present and reasonable (example)
        found_person = False
        found_location = False
        for token, pos, ner in sentence:
            if ner == 'B-PERSON' or ner == 'I-PERSON':
                found_person = True
            if ner == 'B-LOCATION' or ner == 'I-LOCATION':
                found_location = True
                
        # Note: RuleBasedNER might not be perfect, adjust assertion if needed
        self.assertTrue(found_person, "Expected PERSON tag not found")
        self.assertTrue(found_location, "Expected LOCATION tag not found")
        
    def test_pipeline_empty_text(self):
        """Test pipeline with empty input text"""
        results = self.pipeline(self.text3)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0) # Expect empty list for empty input
        
    def test_pipeline_non_thai_text(self):
        """Test pipeline with non-Thai input text"""
        results = self.pipeline(self.text4)
        self.assertIsInstance(results, list)
        # Depending on implementation, might return empty or process non-Thai words
        # Let's assume it processes them but tags might be default/unknown
        self.assertEqual(len(results), 1) 
        sentence = results[0]
        self.assertGreater(len(sentence), 0)
        # Check if tags are default/unknown (e.g., 'X' or 'O')
        token, pos, ner = sentence[0]
        # self.assertEqual(pos, 'X') # Example assertion, adjust based on actual tagger behavior
        self.assertEqual(ner, 'O') # NER usually defaults to 'O'

if __name__ == '__main__':
    unittest.main()
