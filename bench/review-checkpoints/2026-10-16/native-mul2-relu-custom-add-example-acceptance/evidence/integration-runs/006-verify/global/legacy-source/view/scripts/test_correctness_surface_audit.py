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

    def gate(self):
        # Execute the real production shell gate's Python, including registry
        # validation. Other corpus gates need unrelated source fixtures.
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        match = re.search(
            r'if python3 - "\$\{PORTS_ROOT\}" "\$\{PORTS_ROOT\}/proof_blockers.md"'
            r" <<'PY'\n(.*?)\nPY\n", script, re.S)
        self.assertIsNotNone(match)
        return subprocess.run([sys.executable, '-', str(self.root), str(self.doc)],
                              input=match[1], text=True, capture_output=True, cwd=ROOT)

    def test_split_declaration_precedes_imported_body(self):
        path = self.port / 'Correct.lean'
        path.write_text('specification correctness : ComputeCorrect.RealizesFrame_without_Rounding k s w e := proof\n')
        self.assertEqual(correctness_surface_evidence(self.source), ([path], []))
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('1 ports on a named correctness surface', result.stdout)

    def test_legacy_declaration_before_its_body(self):
        for path in self.port.rglob('*.lean'):
            path.unlink()
        self.source = self.port / 'Fixture.lean'
        self.source.write_text('def surface := ComputeCorrect.Realizes_without_Rounding\n'
                               'def source := triton { x = 1 }\n')
        self.assertEqual(correctness_surface_evidence(self.source), ([self.source], []))

    def test_surfaces_only_in_comments_and_strings_fail(self):
        samples = [
            '/- ComputeCorrect.Realizes_without_Rounding -/',
            '/- nested /- ⊨ -/ ComputeRefine.Realizes -/',
            '-- ComputeCorrect.General',
            'def note := "ComputeCorrect.RealizesFrame_without_Rounding"',
            'def note := r##"correct_target and ⊨"##',
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                (self.port / 'Correct.lean').write_text(sample + '\n')
                self.assertEqual(correctness_surface_evidence(self.source)[0], [])
                result = self.gate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('no named correctness surface', result.stdout)

    def test_registered_blocker_preamble_with_triton_in_prose(self):
        self.source.write_text('/- triton { is mentioned in prose.\n'
                               'Correctness-surface blocker: full proof pending\n-/\n'
                               'def source := triton { x = 1 }\n')
        self.doc.write_text('## Correctness-Surface Blockers\n| `fixture` | pending |\n')
        files, markers = correctness_surface_evidence(self.source)
        self.assertEqual(files, [])
        self.assertEqual(markers, [' full proof pending'])
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stale_marker_and_registry_still_fail(self):
        self.source.write_text('-- Correctness-surface blocker: pending\n'
                               'def source := triton { x = 1 }\n')
        (self.port / 'Correct.lean').write_text('def surface := ComputeCorrect.Realizes_without_Rounding\n')
        self.doc.write_text('## Correctness-Surface Blockers\n| `fixture` | pending |\n')
        result = self.gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale Correctness-surface blocker marker', result.stdout)
        self.assertIn('stale proof_blockers.md correctness-surface row', result.stdout)

    def test_unregistered_and_empty_markers_still_fail(self):
        for description in ['', ' pending']:
            with self.subTest(description=description):
                self.source.write_text('-- Correctness-surface blocker:' + description + '\n'
                                       'def source := triton { x = 1 }\n')
                result = self.gate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('lacks a', result.stdout)

    def test_imported_file_marker_is_not_hidden_by_source_order(self):
        (self.port / 'Proofs/Legacy.lean').write_text(
            '/- Correctness-surface status\n'
            'Correctness-surface blocker: legacy frame pending\n-/\n'
            'def legacy := triton { x = 1 }\n')
        self.doc.write_text('## Correctness-Surface Blockers\n| `fixture` | pending |\n')
        self.assertEqual(correctness_surface_evidence(self.source)[1],
                         [' legacy frame pending'])
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (self.port / 'Correct.lean').write_text(
            'def surface := ComputeCorrect.RealizesFrame_without_Rounding\n')
        result = self.gate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('stale Correctness-surface blocker marker', result.stdout)


if __name__ == '__main__':
    unittest.main()
