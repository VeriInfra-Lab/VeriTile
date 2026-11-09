"""Regression checks for the homepage proof fingerprint."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'record_home_demo', Path(__file__).with_name('record-home-demo.py'))
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.demo = self.write('bench/examples/VectorAdd/RealCorrect.lean',
                               'import Mathlib\nimport VeriTile.Direct\n')
        self.write('VeriTile/Direct.lean', '/-\nimport VeriTile.Unused\n-/\nimport VeriTile.Transitive\n')
        self.write('VeriTile/Transitive.lean', 'def checked : Nat := 1\n')
        self.write('VeriTile/Unused.lean', 'def unrelated : Nat := 1\n')
        for name in ('lean-toolchain', 'lakefile.toml', 'lake-manifest.json',
                     'site/scripts/record-home-demo.py'):
            self.write(name, 'fixture\n')
        overrides = patch.multiple(recorder, ROOT=self.root, PROOF_SOURCES=(self.demo,))
        overrides.start()
        self.addCleanup(overrides.stop)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_unrelated_modules_do_not_change_fingerprint(self):
        initial = recorder.fingerprint()
        self.write('VeriTile/Unused.lean', 'def unrelated : Nat := 2\n')
        self.write('VeriTile/New.lean', 'def newModule : Nat := 1\n')
        self.assertEqual(recorder.fingerprint(), initial)

    def test_proof_sources_and_pinned_inputs_change_fingerprint(self):
        initial = recorder.fingerprint()
        for name in ('bench/examples/VectorAdd/RealCorrect.lean', 'VeriTile/Direct.lean',
                     'VeriTile/Transitive.lean', 'lean-toolchain', 'lake-manifest.json'):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_text()
                self.write(name, original + '\n-- Change to a tracked proof input.\n')
                self.assertNotEqual(recorder.fingerprint(), initial)
                self.write(name, original)
