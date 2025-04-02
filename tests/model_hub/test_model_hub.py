"""
Tests for the ThaiNLP Model Hub (Registry and Manager)
"""
import unittest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock huggingface_hub before importing manager to avoid actual downloads in tests
# We need to mock hf_hub_download
mock_hf_download = MagicMock()

# Now import the modules
from thainlp.model_hub import ModelManager, list_models, registry

# Define a dummy cache directory for tests
TEST_CACHE_DIR = Path("./.test_cache/thainlp")

class TestModelRegistry(unittest.TestCase):
    """Test model registry functions"""

    def test_list_models_all(self):
        """Test listing all models"""
        models = list_models()
        self.assertIsInstance(models, list)
        # Check if default built-in models are listed
        model_names = [m['name'] for m in models]
        self.assertIn("hmm_pos", model_names)
        self.assertIn("rule_based_ner", model_names)

    def test_list_models_filtered(self):
        """Test filtering models"""
        # Filter by task
        pos_models = list_models(task="pos_tag")
        self.assertTrue(all(m['task'] == 'pos_tag' for m in pos_models))
        self.assertGreaterEqual(len(pos_models), 1) # At least hmm_pos

        # Filter by source
        thainlp_models = list_models(source="thainlp")
        self.assertTrue(all(m['source'] == 'thainlp' for m in thainlp_models))
        self.assertGreaterEqual(len(thainlp_models), 2) # hmm_pos and rule_based_ner

        # Filter by non-existent task
        non_existent = list_models(task="non_existent_task")
        self.assertEqual(len(non_existent), 0)

    def test_get_model_info_exists(self):
        """Test getting info for an existing model"""
        info = registry.get_model_info("hmm_pos")
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], "hmm_pos")
        self.assertEqual(info['task'], "pos_tag")

    def test_get_model_info_not_exists(self):
        """Test getting info for a non-existent model"""
        info = registry.get_model_info("non_existent_model")
        self.assertIsNone(info)

class TestModelManager(unittest.TestCase):
    """Test ModelManager class"""

    def setUp(self):
        """Set up test environment"""
        # Ensure test cache directory exists and is clean (optional)
        TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.manager = ModelManager(cache_dir=TEST_CACHE_DIR)
        # Reset mocks for each test
        mock_hf_download.reset_mock()

    # Patch hf_hub_download for tests involving HF models
    @patch('thainlp.model_hub.manager.hf_hub_download', mock_hf_download)
    def test_get_model_path_hf(self):
        """Test getting path for a mock Hugging Face model"""
        # Add a temporary mock HF model to the registry for this test
        mock_hf_model_name = "mock_hf_ner"
        mock_hf_id = "org/mock-hf-ner-model"
        # Use a copy of the original registry to avoid modifying it globally across tests
        original_registry = registry._MODEL_REGISTRY.copy()
        registry._MODEL_REGISTRY[mock_hf_model_name] = {
            "name": mock_hf_model_name, "task": "ner", "description": "Mock HF",
            "source": "huggingface", "hf_id": mock_hf_id,
            "framework": "pytorch", "tags": ["mock", "hf"]
        }

        try:
            # Configure the mock download function
            mock_config_path = TEST_CACHE_DIR / "huggingface" / "--".join(["models", *mock_hf_id.split('/')]) / "snapshots" / "dummy_hash" / "config.json"
            mock_config_path.parent.mkdir(parents=True, exist_ok=True)
            mock_config_path.touch()
            mock_hf_download.return_value = str(mock_config_path)

            path = self.manager.get_model_path(mock_hf_model_name)
            self.assertIsNotNone(path)
            self.assertEqual(path, mock_config_path.parent)
            mock_hf_download.assert_called_once_with(
                repo_id=mock_hf_id,
                filename="config.json",
                cache_dir=TEST_CACHE_DIR / "huggingface",
                force_download=False
            )
        finally:
            # Restore original registry
            registry._MODEL_REGISTRY = original_registry
            # Clean up dummy file/dirs (optional but good practice)
            # import shutil
            # if mock_config_path.parent.parent.parent.exists():
            #      shutil.rmtree(mock_config_path.parent.parent.parent)


    def test_get_model_path_builtin(self):
        """Test getting path for a built-in model (should be None)"""
        path = self.manager.get_model_path("hmm_pos")
        self.assertIsNone(path) # Built-in models don't have managed paths

    def test_get_model_path_not_found(self):
        """Test getting path for a non-existent model"""
        path = self.manager.get_model_path("non_existent_model")
        self.assertIsNone(path)

    # Patch the HMMPOSTagger where it's imported and used
    @patch('thainlp.pos_tagging.hmm_tagger.HMMPOSTagger')
    def test_load_model_builtin_pos(self, mock_hmm_tagger_class):
        """Test loading a built-in POS tagger"""
        mock_instance = MagicMock()
        mock_hmm_tagger_class.return_value = mock_instance

        model = self.manager.load_model("hmm_pos")
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_instance)
        mock_hmm_tagger_class.assert_called_once() # Check if class was instantiated

    # Patch the RuleBasedNER where it's imported and used
    @patch('thainlp.ner.rule_based.RuleBasedNER')
    def test_load_model_builtin_ner(self, mock_rb_ner_class):
        """Test loading a built-in NER tagger"""
        mock_instance = MagicMock()
        mock_rb_ner_class.return_value = mock_instance

        model = self.manager.load_model("rule_based_ner")
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_instance)
        mock_rb_ner_class.assert_called_once() # Check if class was instantiated

    def test_load_model_not_found(self):
        """Test loading a non-existent model"""
        model = self.manager.load_model("non_existent_model")
        self.assertIsNone(model)

    # Patch the transformers classes where they're imported and used
    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForTokenClassification')
    def test_load_model_hf_ner(self, mock_auto_model_class, mock_auto_tokenizer_class):
        """Test loading a mock Hugging Face NER model"""
        # Mock the from_pretrained methods
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
        mock_auto_model_class.from_pretrained.return_value = mock_model_instance

        # Add mock HF model
        mock_hf_model_name = "mock_hf_ner"
        mock_hf_id = "org/mock-hf-ner-model"
        # Use a copy of the original registry to avoid modifying it globally across tests
        original_registry = registry._MODEL_REGISTRY.copy()
        registry._MODEL_REGISTRY[mock_hf_model_name] = {
            "name": mock_hf_model_name, "task": "ner", "description": "Mock HF",
            "source": "huggingface", "hf_id": mock_hf_id,
            "framework": "pytorch", "tags": ["mock", "hf"]
        }

        try:
            loaded_data = self.manager.load_model(mock_hf_model_name)
            self.assertIsNotNone(loaded_data)
            model, tokenizer = loaded_data
            self.assertEqual(model, mock_model_instance)
            self.assertEqual(tokenizer, mock_tokenizer_instance)

            cache_path_expected = self.manager.cache_dir / "huggingface"
            # Assert that the mocked from_pretrained methods were called
            mock_auto_tokenizer_class.from_pretrained.assert_called_once_with(mock_hf_id, cache_dir=cache_path_expected)
            mock_auto_model_class.from_pretrained.assert_called_once_with(mock_hf_id, cache_dir=cache_path_expected)
        finally:
            # Clean up: Restore original registry
            registry._MODEL_REGISTRY = original_registry


    def tearDown(self):
        """Clean up test cache directory"""
        # import shutil
        # if TEST_CACHE_DIR.exists():
        #     shutil.rmtree(TEST_CACHE_DIR)
        pass # Avoid deleting cache for now, can be enabled if needed

if __name__ == '__main__':
    unittest.main()
