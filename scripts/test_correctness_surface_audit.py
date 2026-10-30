"""Regress source-order and prose false positives in the named-surface gate."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from bench.audit_source import correctness_surface_evidence

ROOT = Path(__file__).resolve().parents[1]


class CorrectnessSurfaceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.port = self.root / 'fixture'
        (self.port / 'Proofs').mkdir(parents=True)
        self.source = self.port / 'Kernels.lean'
        self.source.write_text('import bench.tritonbench_g.fixture.Proofs.Source\n')
        (self.port / 'Correct.lean').write_text('')
        (self.port / 'FPEquiv.lean').write_text('')
        (self.port / 'Proofs/Source.lean').write_text('def source := triton { x = 1 }\n')
        self.doc = self.root / 'proof_blockers.md'
        self.doc.write_text('## Correctness-Surface Blockers\n')



    def test_legacy_declaration_before_its_body(self):
        for path in self.port.rglob('*.lean'):
            path.unlink()
        self.source = self.port / 'Fixture.lean'
        self.source.write_text('def surface := ComputeCorrect.Realizes_without_Rounding\n'
                               'def source := triton { x = 1 }\n')
        self.assertEqual(correctness_surface_evidence(self.source), ([self.source], []))







if __name__ == '__main__':
    unittest.main()
