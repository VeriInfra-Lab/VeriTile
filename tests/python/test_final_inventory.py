"""Select current public entries and preserve complete corpus coverage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'bench'))
from scripts.bench.final_inventory import BENCH_SECTIONS, ci_targets, coverage, local, modules, read, selected_tasks
from scripts.bench.audit_trust_prep import prepare_source


class FinalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for section in BENCH_SECTIONS:
            (self.root / 'bench' / section).mkdir(parents=True)
        self.directory = self.root / 'bench/tritonbench_g/fixture'
        self.directory.mkdir()
        files = {'Kernels.lean': 'def source := 1\n',
                 'RealCorrect.lean': 'specification result : source = 1 := by rfl\n',
                 'FPEquiv.lean': 'specification pair : True := by trivial\n',
                 'Contract.lean': 'def condition := True\n',
                 'fixture.py': 'def kernel(x): return x\n'}
        for name, body in files.items():
            (self.directory / name).write_text(body)
        base = self.directory.relative_to(self.root).as_posix()
        self.selection = {'kernel_source': base + '/Kernels.lean', 'source': base + '/fixture.py',
                          'public_entries': [base + '/' + name for name in
                                             ('Contract.lean', 'RealCorrect.lean', 'FPEquiv.lean')],
                          'proof_scope': 'one complete invocation and frame'}
        self.ledger = {'tasks': [{'task': 'fixture', 'kernel_revision': self.selection}]}
        (self.root / 'validation').mkdir()
        self.ledger_path = self.root / 'validation/task_progress.json'
        self.ledger_path.write_text(json.dumps(self.ledger))

    def test_coverage_reads_current_statements_without_old_review(self):
        rows = coverage(self.root)['rows']
        self.assertEqual([row['declaration'] for row in rows], ['result', 'pair'])
        self.assertEqual(rows[0]['scope'], self.selection['proof_scope'])
        reader = self.directory / 'RealCorrect.lean'
        reader.write_text('specification result : source = source := by rfl\n')
        self.assertIn('source = source', coverage(self.root)['rows'][0]['statement'])

    def test_canonical_reader_must_publish_a_specification(self):
        (self.directory / 'RealCorrect.lean').write_text('def theoremResult := True\n')
        with self.assertRaisesRegex(ValueError, 'no public specification'):
            coverage(self.root)

    def test_duplicate_or_escaping_entry_is_rejected(self):
        entries = self.selection['public_entries']
        entries.append(entries[0])
        with self.assertRaisesRegex(ValueError, 'invalid selected public entries'):
            selected_tasks(self.root, self.ledger)
        entries[-1] = 'bench/examples/Other/RealCorrect.lean'
        with self.assertRaisesRegex(ValueError, 'invalid selected public entries'):
            selected_tasks(self.root, self.ledger)

    def test_missing_and_escaping_sources(self):
        for name in ('absent.lean', '../escape.lean'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'missing or escaping'):
                local(self.root, name)

    def test_duplicate_json_keys_rejected(self):
        (self.root / 'duplicate.json').write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            read(self.root, 'duplicate.json')

    def test_imports_ignore_strings_comments_and_keep_continuations(self):
        source = ('module\n/- import bench.fake -/\nimport all Init.Data.Nat.Power2.Basic\n'
                  'import VeriTile.Triton\n  VeriTile.Spec bench.fixture.IO\n'
                  'def label := "import bench.fake"\nimport bench.fixture.Proofs\n')
        self.assertEqual(modules(source), {'Init.Data.Nat.Power2.Basic', 'VeriTile.Triton',
                                          'VeriTile.Spec', 'bench.fixture.IO', 'bench.fixture.Proofs'})

    def test_repaired_reader_audits_transitive_kernel(self):
        helper = self.root / 'bench/tritonbench_g/fixture/Proofs/Real.lean'
        helper.parent.mkdir(parents=True)
        helper.write_text('import bench.tritonbench_g.fixture.IO\n')
        io = helper.parent.parent / 'IO.lean'
        io.write_text('import bench.tritonbench_g.fixture.Kernels\n')
        source = 'import bench.tritonbench_g.fixture.Proofs.Real\ndef expectedSpec := 42\n'
        audited = prepare_source(source, 'bench/tritonbench_g/fixture/Correct.lean',
                                 self.root / 'scripts/absent.tsv')
        self.assertIn('#auditModuleAxioms', audited)
        self.assertIn('#auditModuleSpecs from [bench.tritonbench_g.fixture.IO, '
                      'bench.tritonbench_g.fixture.Kernels, bench.tritonbench_g.fixture.Proofs.Real]', audited)

    def test_audit_import_preserves_multiline_source(self):
        source = ('/- import bench.fake -/\nimport VeriTile.Spec\n'
                  '  bench.tritonbench_g.fixture.Kernels\ndef expectedSpec := 42\n')
        audited = prepare_source(source, 'bench/tritonbench_g/fixture/Correct.lean',
                                 self.root / 'scripts/absent.tsv')
        self.assertIn('import VeriTile.Meta.StatementAudit\nimport VeriTile.Spec\n'
                      '  bench.tritonbench_g.fixture.Kernels\n', audited)
        self.assertIn('#auditModuleSpecs from [bench.tritonbench_g.fixture.Kernels]', audited)
        self.assertNotIn('from [bench.fake', audited)

    def test_moved_regression_imports_keep_their_transitive_source_audit(self):
        directory = self.root / 'tests/lean'
        directory.mkdir(parents=True)
        (directory / 'Kernel.lean').write_text('import tests.lean.Helper\ndef kernel := 1\n')
        (directory / 'Helper.lean').write_text('def helper := 1\n')
        audited = prepare_source('import tests.lean.Kernel\ndef expectedSpec := 1\n',
                                 'tests/lean/Reader.lean', self.root / 'scripts/absent.tsv')
        self.assertIn('#auditModuleSpecs from [tests.lean.Helper, tests.lean.Kernel]', audited)


class FinalCITargetTests(unittest.TestCase):
    def setUp(self):
        self.prefix = 'bench/tritonbench_g/task/'
        self.entries = [self.prefix + p + '.lean' for p in ('Contract', 'Correct', 'FPEquiv')]
        self.kernel = self.prefix + 'Kernels.lean'
        self.helper = self.prefix + 'Proofs/Helper.lean'
        self.tests = ['tests/lean/Regression.lean', 'bench/examples/Example/Correct.lean']
        self.rows = {'task': {'public_entries': self.entries, 'kernel_source': self.kernel}}
        self.paths = self.entries + [self.kernel, self.helper] + self.tests + ['VeriTile/Shared.lean']
        self.imports = {self.entries[1]: ['VeriTile.Shared'],
                        'VeriTile/Shared.lean': ['bench.tritonbench_g.task.Proofs.Helper']}

    def test_public_entries_sources_and_tests_cover_transitive_build(self):
        targets = ci_targets(self.paths, self.imports, self.rows)
        self.assertEqual(targets, sorted(self.entries + [self.kernel] + self.tests))
        self.assertNotIn(self.helper, targets)

    def test_retained_helper_outside_closure_is_selected_directly(self):
        self.imports['VeriTile/Shared.lean'] = []
        self.assertIn(self.helper, ci_targets(self.paths, self.imports, self.rows))

    def test_missing_canonical_source_is_rejected(self):
        self.paths.remove(self.kernel)
        with self.assertRaisesRegex(ValueError, 'entry outside retained corpus'):
            ci_targets(self.paths, self.imports, self.rows)


if __name__ == '__main__':
    unittest.main()
