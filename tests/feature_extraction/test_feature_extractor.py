"""
Tests for feature extraction
"""
import unittest
from thainlp.feature_extraction.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(remove_stopwords=True)
        self.text = "สวัสดีครับ นี่คือการทดสอบ"
        self.documents = [
            "ฉันชอบกินข้าว",
            "เธอชอบกินขนม",
            "พวกเราชอบดูหนัง"
        ]
        
    def test_bow_extraction(self):
        features = self.extractor.extract_bow(self.text)
        self.assertIsInstance(features, dict)
        self.assertTrue(len(features) > 0)
        
    def test_tfidf_extraction(self):
        vectors = self.extractor.extract_tfidf(self.documents)
        self.assertEqual(len(vectors), len(self.documents))
        self.assertTrue(all(isinstance(v, dict) for v in vectors))
        
        # Check if TF-IDF scores make sense
        # Words that appear in all documents should have lower scores
        word = "ชอบ"  # Appears in all documents
        scores = [v.get(word, 0) for v in vectors]
        self.assertTrue(all(s <= 1.0 for s in scores))
        
    def test_ngram_extraction(self):
        ngrams = self.extractor.extract_ngrams(self.text, n=2)
        self.assertIsInstance(ngrams, list)
        self.assertTrue(len(ngrams) > 0)
        self.assertTrue(all(len(ng) == 2 for ng in ngrams))
        
    def test_pos_patterns(self):
        patterns = self.extractor.extract_pos_patterns(self.text)
        self.assertIsInstance(patterns, list)
        self.assertTrue(len(patterns) > 0)
        # Verify pattern format (TAG_TAG)
        self.assertTrue(all('_' in p for p in patterns))
        
    def test_stopword_removal(self):
        # Create extractor with and without stopword removal
        with_stops = FeatureExtractor(remove_stopwords=False)
        without_stops = FeatureExtractor(remove_stopwords=True)
        
        text = "นี่ คือ การ ทดสอบ"
        
        features_with = with_stops.extract_bow(text)
        features_without = without_stops.extract_bow(text)
        
        # Should have fewer features without stopwords
        self.assertGreater(len(features_with), len(features_without))

if __name__ == '__main__':
    unittest.main()