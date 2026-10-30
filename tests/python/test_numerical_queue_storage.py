"""The active numerical queue must be unambiguous and fully readable."""
from copy import deepcopy
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'bench'))
from scripts.bench import deferred_numerical_experiments as queue
from scripts.bench.adapter_common import load


class NumericalQueueStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {
            'summary': {'inventory_tasks': 184, 'tasks_with_native_recipes': 4},
            'rules': [], 'supplements': [{'obligations': []}],
            'input_sha256': {'source.py': 'a' * 64},
            'boundary': 'Numerical conformance remains deferred; 数值验证待办',
        }
        self.path = self.root / (queue.OUTPUT + '.json.gz')
        self.plain = self.root / (queue.OUTPUT + '.json')

    def generate(self):
        queue.write_outputs(self.root, self.data, check=False)

    def test_roundtrip_deterministic_and_one_active_path(self):
        self.generate()
        first = self.path.read_bytes()
        self.assertEqual(load(self.path), self.data)
        self.assertFalse(self.plain.exists())
        self.generate()
        self.assertEqual(first, self.path.read_bytes())
        queue.write_outputs(self.root, self.data, check=True)

    def test_changed_declaration_evidence_is_stale(self):
        self.generate()
        changed = deepcopy(self.data)
        changed['input_sha256']['source.py'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Stale deferred experiment queue'):
            queue.write_outputs(self.root, changed, check=True)

    def test_truncated_and_crc_corrupted_gzip_rejected(self):
        self.generate()
        original = self.path.read_bytes()
        bad_crc = bytearray(original)
        bad_crc[-8] ^= 1
        for invalid in (original[:-4], bytes(bad_crc)):
            with self.subTest(size=len(invalid)):
                self.path.write_bytes(invalid)
                with self.assertRaises((OSError, EOFError, ValueError)):
                    queue.write_outputs(self.root, self.data, check=True)

    def test_plain_queue_never_silently_shadows_compressed(self):
        self.generate()
        self.plain.write_bytes(gzip.decompress(self.path.read_bytes()))
        for check in (True, False):
            with self.subTest(check=check), self.assertRaisesRegex(ValueError, 'Competing plain'):
                queue.write_outputs(self.root, self.data, check=check)

    def test_old_path_only_is_not_a_successful_check(self):
        self.generate()
        self.plain.write_bytes(gzip.decompress(self.path.read_bytes()))
        self.path.unlink()
        with self.assertRaises(ValueError):
            queue.write_outputs(self.root, self.data, check=True)

    def test_missing_queue_or_stale_document_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing compressed'):
            queue.write_outputs(self.root, self.data, check=True)
        self.generate()
        (self.root / (queue.OUTPUT + '.md')).write_text('stale\n')
        with self.assertRaisesRegex(ValueError, 'Stale deferred experiment handoff'):
            queue.write_outputs(self.root, self.data, check=True)

    def test_cli_default_uses_compressed_queue_for_write_and_check(self):
        with patch.object(queue, 'collect', return_value=self.data):
            for arguments in ([], ['--check']):
                with patch('sys.argv', ['deferred_numerical_experiments.py',
                                       '--root', str(self.root), *arguments]):
                    queue.main()
        self.assertTrue(self.path.is_file())
        self.assertFalse(self.plain.exists())


if __name__ == '__main__':
    unittest.main()
