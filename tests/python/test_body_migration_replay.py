"""A body match or successful declaration build cannot certify migration completion."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.bench import replay_body_migrations as runner

ROOT = Path(__file__).resolve().parents[2]


class MigrationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = self.root / 'bench/tritonbench_g/dequantize_rowwise'
        base.mkdir(parents=True)
        original = ROOT / 'bench/tritonbench_g/dequantize_rowwise'
        for name in ('dequantize_rowwise.py', 'Kernels.lean', 'jit-correspondence.json'):
            (base / name).write_bytes((original / name).read_bytes())
        # Body matching alone cannot certify even these displayed specifications.
        for name in ('Correct.lean', 'FPEquiv.lean'):
            (base / name).write_text('namespace Fixture'+name[:-5]+'\nspecification proved : True := by trivial\nend Fixture'+name[:-5]+'\n')
        (self.root / 'validation').mkdir()
        (self.root / runner.DEFAULT).write_text(json.dumps({
            'schema': 1, 'scope': 'explicit_body_migration_batch',
            'cases': [{'path': 'bench/tritonbench_g/dequantize_rowwise',
                       'python': 'bench/tritonbench_g/dequantize_rowwise/dequantize_rowwise.py',
                       'public_theorems': {
                           'bench/tritonbench_g/dequantize_rowwise/Correct.lean': ['FixtureCorrect.proved'],
                           'bench/tritonbench_g/dequantize_rowwise/FPEquiv.lean': ['FixtureFPEquiv.proved']}}],
        }))
        for name in ('lean-toolchain', 'lakefile.toml', 'lake-manifest.json'):
            (self.root / name).write_text('fixture\n')
        dependency = self.root / 'VeriTile/Triton/DSL/CallSignature.lean'
        dependency.parent.mkdir(parents=True)
        dependency.write_text('-- fixture dependency; no transitive imports\n')

    def invoke(self, *args):
        with patch.object(runner, 'ROOT', self.root), patch('sys.argv', ['runner', *args]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = runner.main()
        return result, output.getvalue()

    def test_source_match_explicitly_does_not_certify_proof_completion(self):
        result, output = self.invoke('--check')
        self.assertEqual(result, 0)
        self.assertIn('proof completion has NOT been certified', output)
        self.assertFalse(list(self.root.glob('*replay*.json')))

    def test_build_cannot_write_a_completion_receipt(self):
        receipt = self.root / 'receipt.json'
        with patch.object(runner.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 0)
            with self.assertRaisesRegex(ValueError, 'Only official proof replay'):
                self.invoke('--build', '--report', str(receipt))
            run.assert_not_called()
        self.assertFalse(receipt.exists())

    def test_unsolved_prop_is_not_a_public_proof(self):
        entry = self.root / 'bench/tritonbench_g/dequantize_rowwise/Correct.lean'
        entry.write_text('def unsolvedProblem : Prop := False\n')
        with self.assertRaisesRegex(ValueError, 'not a proved specification'):
            self.invoke('--check')

    def test_comparator_empty_module_dummy_cannot_replace_required_proofs(self):
        receipt = self.root / 'receipt.json'
        dummy = {'accepted': True, 'theorem_count': 0, 'theorems': []}
        with patch.object(runner, 'require_tools'), \
                patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(runner.check_comparator, 'check', return_value=dummy):
            result, _ = self.invoke('--replay', '--workspace', str(self.root / 'judge'),
                                    '--report', str(receipt))
        self.assertEqual(result, 1)
        report = json.loads(receipt.read_text())
        self.assertFalse(report['accepted'])
        self.assertFalse(report['cases'][0]['accepted'])
        self.assertTrue(all('not replayed' in r['error'] for r in report['replay_results']))

    def test_reused_judge_snapshot_is_rejected(self):
        judge = self.root / 'judge'
        judge.mkdir()
        (judge / 'ready.json').write_text('{}')
        with patch.object(runner, 'require_tools') as preflight:
            with self.assertRaisesRegex(ValueError, 'fresh empty snapshot'):
                self.invoke('--replay', '--workspace', str(judge),
                            '--report', str(self.root / 'receipt.json'))
            preflight.assert_not_called()

    def test_dependency_edit_during_replay_prevents_acceptance(self):
        receipt = self.root / 'receipt.json'
        judge = self.root / 'judge'
        dependency = self.root / 'VeriTile/Changed.lean'
        dependency.parent.mkdir(exist_ok=True)
        dependency.write_text('def answer := 1\n')
        def replay(*_args, **_kwargs):
            dependency.write_text('def answer := 2\n')
            return {'accepted': True, 'theorems': ['FixtureCorrect.proved','FixtureFPEquiv.proved'], 'theorem_count': 2}
        with patch.object(runner, 'require_tools'), \
                patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                patch.object(runner.check_comparator, 'check', side_effect=replay) as check:
            result, _ = self.invoke('--replay', '--workspace', str(judge), '--report', str(receipt))
        self.assertEqual(check.call_count, 2)
        self.assertEqual(result, 1)
        report = json.loads(receipt.read_text())
        self.assertFalse(report['source_unchanged'])
        self.assertFalse(report['accepted'])
        self.assertFalse(report['cases'][0]['accepted'])


if __name__ == '__main__':
    unittest.main()
