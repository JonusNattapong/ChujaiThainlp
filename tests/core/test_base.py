"""Unit tests for thainlp/core/base.py"""
import unittest
import asyncio
import time
from typing import Any
from thainlp.core.base import (
    SecurityManager,
    RateLimiter,
    Cache,
    MetricsCollector,
    ABTesting,
    ThaiNLPBase,
    MetricType
)

class AsyncTestCase(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def tearDown(self):
        self.loop.close()
        
    def async_test(self, coro):
        return self.loop.run_until_complete(coro)

class TestSecurityManager(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.manager = SecurityManager()
        
    async def test_rate_limiting(self):
        client_id = "test_client"
        # Test within rate limit
        for _ in range(100):
            self.assertTrue(await self.manager.rate_limit_check(client_id))
        # Test exceeding rate limit
        self.assertFalse(await self.manager.rate_limit_check(client_id))
        
    def test_input_validation(self):
        # Test valid input
        self.assertTrue(self.manager.input_validation("valid text"))
        # Test empty input
        self.assertFalse(self.manager.input_validation(""))
        # Test non-string input
        self.assertFalse(self.manager.input_validation(None))
        # Test oversized input
        self.assertFalse(self.manager.input_validation("a" * 10001))
        # Test invalid Thai text
        self.assertFalse(self.manager.input_validation("ภาษาไทยที่ผิดปกติ"))
        # Test suspicious patterns
        self.assertFalse(self.manager.input_validation("<script>alert(1)</script>"))
        self.assertFalse(self.manager.input_validation("SELECT * FROM users"))
        # Test valid Thai text
        self.assertTrue(self.manager.input_validation("สวัสดีครับ"))
        
    def test_text_hashing(self):
        text = "test text"
        hash1 = self.manager.hash_text(text)
        hash2 = self.manager.hash_text(text)
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, self.manager.hash_text("different text"))

class TestRateLimiter(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.limiter = RateLimiter(max_requests=2, window_seconds=1)
        
    async def test_rate_limiting(self):
        client_id = "test_client"
        # First two requests should pass
        self.assertTrue(await self.limiter.is_allowed(client_id))
        self.assertTrue(await self.limiter.is_allowed(client_id))
        # Third request should fail
        self.assertFalse(await self.limiter.is_allowed(client_id))
        # After window expires, should allow again
        time.sleep(1.1)
        self.assertTrue(await self.limiter.is_allowed(client_id))

class TestCache(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.cache: Cache[str] = Cache(max_size=2)
        
    async def test_cache_operations(self):
        # Test basic set/get with LRU ordering
        await self.cache.set("key1", "value1")
        self.assertEqual(await self.cache.get("key1"), "value1")
        
        # Test cache eviction
        await self.cache.set("key2", "value2")
        await self.cache.set("key3", "value3")
        self.assertIsNone(await self.cache.get("key1"))  # Should be evicted
        self.assertEqual(await self.cache.get("key2"), "value2")  # Should still exist
        
        # Test LRU update on access
        await self.cache.get("key2")  # Access key2, making key3 the LRU
        await self.cache.set("key4", "value4")
        self.assertIsNone(await self.cache.get("key3"))  # Should be evicted
        self.assertEqual(await self.cache.get("key2"), "value2")  # Should still exist
        
    async def test_cache_stats(self):
        await self.cache.set("key1", "value1")
        await self.cache.get("key1")  # Hit
        await self.cache.get("nonexistent")  # Miss
        
        stats = await self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)
        
    async def test_cache_clear(self):
        await self.cache.set("key1", "value1")
        await self.cache.clear()
        stats = await self.cache.get_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["hits"], 0)

class TestMetricsCollector(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.collector = MetricsCollector(max_window_seconds=1)
        
    async def test_metrics_recording(self):
        # Test different metric types
        await self.collector.record("requests", 1, MetricType.COUNTER)
        await self.collector.record("latency", 0.5, MetricType.HISTOGRAM)
        await self.collector.record("cpu_usage", 75.5, MetricType.GAUGE)
        
        # Test counter stats
        stats = self.collector.get_stats("requests")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["latest"], 1)
        
        # Test histogram stats
        stats = self.collector.get_stats("latency")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["avg"], 0.5)
        
        # Test gauge stats
        stats = self.collector.get_stats("cpu_usage")
        self.assertEqual(stats["latest"], 75.5)
        
        # Test metrics by type
        counters = self.collector.get_metrics_by_type(MetricType.COUNTER)
        self.assertIn("requests", counters)
        
    async def test_metrics_window(self):
        await self.collector.record("test", 1, MetricType.COUNTER)
        time.sleep(1.1)  # Wait for window expiration
        stats = self.collector.get_stats("test")
        self.assertEqual(stats.get("count", 0), 0)

class TestABTesting(AsyncTestCase):
    def setUp(self):
        super().setUp()
        self.ab = ABTesting()
        self.ab.register_experiment("test_exp", [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2}
        ])
        
    def test_variant_assignment(self):
        user1, user2 = "user1", "user2"
        variant1 = self.ab.get_variant("test_exp", user1)
        variant2 = self.ab.get_variant("test_exp", user2)
        self.assertIn(variant1["name"], ["A", "B"])
        self.assertIn(variant2["name"], ["A", "B"])
        
    async def test_result_recording(self):
        variant = self.ab.get_variant("test_exp", "user1")
        await self.ab.record_result("test_exp", variant, True)
        stats = await self.ab.get_experiment_stats("test_exp")
        self.assertGreater(stats[str(variant)]["total"], 0)

class TestThaiNLPBase(AsyncTestCase):
    def setUp(self):
        super().setUp()
        
        class TestImplementation(ThaiNLPBase[str]):
            async def _process(self, text: str, task: str, correlation_id: str) -> str:
                return f"processed:{text}:{task}"
                
        self.nlp = TestImplementation()
        
    async def test_processing(self):
        result = await self.nlp.process("test", "task", "client1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "processed:test:task")
        self.assertIn("correlation_id", result)
        
    async def test_error_handling(self):
        class FailingImplementation(ThaiNLPBase[Any]):
            async def _process(self, text: str, task: str, correlation_id: str) -> Any:
                raise ValueError("test error")
                
        failing_nlp = FailingImplementation()
        result = await failing_nlp.process("test", "task", "client1")
        self.assertEqual(result["status"], "error")
        self.assertIn("test error", result["error"])
        self.assertIn("correlation_id", result)

if __name__ == "__main__":
    unittest.main()
