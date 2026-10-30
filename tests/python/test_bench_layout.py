"""Reject reintroduced version trees and benchmark infrastructure clutter."""
from pathlib import Path
import tempfile
import unittest

from scripts.bench.final_inventory import BENCH_SECTIONS, validate_layout


class BenchLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for section in BENCH_SECTIONS:
            (self.root / 'bench' / section).mkdir(parents=True)
        self.task = self.root / 'bench/tritonbench_g/add'
        self.task.mkdir()
        base = 'bench/tritonbench_g/add/'
        self.ledger = {'tasks': [{'task': 'add', 'kernel_revision': {
            'source': base + 'add.py', 'kernel_source': base + 'Kernels.lean',
            'public_entries': [base + name + '.lean' for name in ('Contract', 'Correct', 'FPEquiv')],
        }}]}

    def test_infrastructure_and_extra_tasks_are_rejected(self):
        validate_layout(self.root, self.ledger)
        for path in (self.root / 'bench/repairs', self.root / 'bench/README.md',
                     self.task.with_name('add_old')):
            with self.subTest(path=path):
                path.touch()
                with self.assertRaises(ValueError):
                    validate_layout(self.root, self.ledger)
                path.unlink()

    def test_version_directory_and_noncanonical_source_are_rejected(self):
        version = self.task / 'revision001'
        version.mkdir()
        with self.assertRaisesRegex(ValueError, 'version directory'):
            validate_layout(self.root, self.ledger)
        version.rmdir()
        self.ledger['tasks'][0]['kernel_revision']['source'] = 'old/add.py'
        with self.assertRaisesRegex(ValueError, 'canonical task entry'):
            validate_layout(self.root, self.ledger)


if __name__ == '__main__':
    unittest.main()
