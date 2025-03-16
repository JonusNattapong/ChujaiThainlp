"""Unit tests for thainlp/core/base.py"""
import unittest
import time
import threading
from thainlp.core.base import (
    SecurityManager,
    RateLimiter,
    Cache,
    MetricsCollector,
    ABTesting,
    ThaiNLPBase
)

class TestSecurityManager(unittest.TestCase):
    def setUp(self):
        self.manager = SecurityManager()
        
    def test_rate_limiting(self):
        client_id = "test_client"
        # Test within rate limit
        for _ in range(100):
            self.assertTrue(self.manager.rate_limit_check(client_id))
        # Test exceeding rate limit
        self.assertFalse(self.manager.rate_limit_check(client_id))
        
    def test_input_validation(self):
        # Test valid input
        self.assertTrue(self.manager.input_validation("valid text"))
        # Test empty input
        self.assertFalse(self.manager.input_validation(""))
        # Test oversized input
        self.assertFalse(self.manager.input_validation("a" * 10001))
        
    def test_text_hashing(self):
        text = "test text"
        hash1 = self.manager.hash_text(text)
        hash2 = self.manager.hash_text(text)
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, self.manager.hash_text("different text"))

class TestRateLimiter(unittest.TestCase):
    def setUp(self):
        self.limiter = RateLimiter(max_requests=2, window_seconds=1)
        
    def test_rate_limiting(self):
        client_id = "test_client"
        # First two requests should pass
        self.assertTrue(self.limiter.is_allowed(client_id))
        self.assertTrue(self.limiter.is_allowed(client_id))
        # Third request should fail
        self.assertFalse(self.limiter.is_allowed(client_id))
        # After window expires, should allow again
        time.sleep(1.1)
        self.assertTrue(self.limiter.is_allowed(client_id))

class TestCache(unittest.TestCase):
    def setUp(self):
        self.cache = Cache(max_size=2)
        
    def test_cache_operations(self):
        # Test basic set/get
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        # Test cache eviction
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        # Allow time for eviction processing
        time.sleep(0.1)
        self.assertIsNone(self.cache.get("key1"))
        
    def test_cache_clear(self):
        self.cache.set("key1", "value1")
        self.cache.clear()
        self.assertIsNone(self.cache.get("key1"))

class TestMetricsCollector(unittest.TestCase):
    def setUp(self):
        self.collector = MetricsCollector(max_window_seconds=1)
        
    def test_metrics_recording(self):
        self.collector.record("test_metric", 1)
        self.collector.record("test_metric", 2)
        stats = self.collector.get_stats("test_metric")
        self.assertIn('count', stats)
        self.assertEqual(stats.get('count', 0), 2)
        self.assertEqual(stats.get('avg', 0), 1.5)
        # Test window expiration
        time.sleep(1.1)
        stats = self.collector.get_stats("test_metric")
        self.assertIn('count', stats)
        self.assertEqual(stats.get('count', 0), 0)

class TestABTesting(unittest.TestCase):
    def setUp(self):
        self.ab = ABTesting()
        self.ab.register_experiment("test_exp", [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2}
        ])
        
    def test_variant_assignment(self):
        user1 = "user1"
        user2 = "user2"
        variant1 = self.ab.get_variant("test_exp", user1)
        variant2 = self.ab.get_variant("test_exp", user2)
        self.assertIn(variant1['name'], ["A", "B"])
        self.assertIn(variant2['name'], ["A", "B"])
        
    def test_result_recording(self):
        variant = self.ab.get_variant("test_exp", "user1")
        self.ab.record_result("test_exp", variant, True)
        stats = self.ab.get_experiment_stats("test_exp")
        self.assertGreater(stats[str(variant)]['total'], 0)

class TestThaiNLPBase(unittest.TestCase):
    def setUp(self):
        class TestImplementation(ThaiNLPBase):
            def _process(self, text, task):
                return f"processed:{text}:{task}"
                
        self.nlp = TestImplementation()
        
    def test_processing(self):
        result = self.nlp.process("test", "task", "client1")
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['result'], 'processed:test:task')
        
    def test_error_handling(self):
        class FailingImplementation(ThaiNLPBase):
            def _process(self, text, task):
                raise ValueError("test error")
                
        failing_nlp = FailingImplementation()
        result = failing_nlp.process("test", "task", "client1")
        self.assertEqual(result['status'], 'error')
        self.assertIn("test error", result['error'])

if __name__ == '__main__':
    unittest.main()
