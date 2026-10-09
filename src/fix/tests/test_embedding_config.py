"""Configuration regression checks; no model requests or downloads."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'src/chatdku/config.py'
spec = importlib.util.spec_from_file_location('embedding_config_under_test', PATH)
config = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = config
spec.loader.exec_module(config)


class EmbeddingConfigTests(unittest.TestCase):
    def settings(self, **changes):
        env = {
            'CHATDKU_EMBEDDING_PROVIDER': 'openai',
            'CHATDKU_EMBEDDING_BASE_URL': 'https://school.example/v1',
            'CHATDKU_OPENAI_EMBEDDING_MODEL': 'Qwen3-Embedding-8B',
            'CHATDKU_EMBEDDING_API_KEY': 'test-only-not-a-real-key',
        }
        env.update(changes)
        with patch.dict(os.environ, env, clear=True), patch.object(config, 'load_dotenv'):
            return config.Settings.from_environment()

    def test_explicit_openai_overrides_stale_local_model(self):
        s = self.settings(CHATDKU_LOCAL_EMBEDDING_MODEL='stale-local-model')
        self.assertEqual(s.resolved_embedding_provider, 'openai')

    def test_auto_selects_openai(self):
        s = self.settings(CHATDKU_EMBEDDING_PROVIDER='auto')
        self.assertEqual(s.resolved_embedding_provider, 'openai')

    def test_explicit_local_remains_authoritative(self):
        s = self.settings(CHATDKU_EMBEDDING_PROVIDER='local', CHATDKU_LOCAL_EMBEDDING_MODEL='local-model')
        self.assertEqual(s.resolved_embedding_provider, 'local')

    def test_explicit_tei_remains_authoritative(self):
        s = self.settings(CHATDKU_EMBEDDING_PROVIDER='tei', CHATDKU_TEI_EMBEDDING_MODEL='tei-model')
        self.assertEqual(s.resolved_embedding_provider, 'tei')

    def test_missing_key_fails(self):
        with self.assertRaisesRegex(ValueError, 'CHATDKU_EMBEDDING_API_KEY'):
            self.settings(CHATDKU_EMBEDDING_API_KEY='')

    def test_missing_url_fails(self):
        with self.assertRaisesRegex(ValueError, 'CHATDKU_EMBEDDING_BASE_URL'):
            self.settings(CHATDKU_EMBEDDING_BASE_URL='')

    def test_invalid_batch_fails(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.settings(CHATDKU_EMBEDDING_BATCH_SIZE='0')

    def test_public_values_do_not_contain_key(self):
        self.assertNotIn('test-only-not-a-real-key', repr(self.settings().public_values()))


if __name__ == '__main__':
    unittest.main()
