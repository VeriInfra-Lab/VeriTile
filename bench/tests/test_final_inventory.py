"""Final source binding must reject omission, stale contracts and scope inflation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_inventory import digest, local, modules, read, validate_review
from audit_trust_prep import prepare_source


class FinalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, body in [('Kernel.lean', 'def source := 1\n'),
                           ('Correct.lean', 'specification result : source = 1 := by rfl\n'),
                           ('source.py', 'def kernel(x): return x\n')]:
            (self.root / name).write_text(body)
        selection = {'kernel_source': 'Kernel.lean', 'source': 'source.py',
                     'public_entries': ['Correct.lean']}
        self.ledger = {'tasks': [{'task': 'fixture', 'kernel_revision': selection}]}
        self.row = {'task': 'fixture', 'kernel_source': 'Kernel.lean', 'python_source': 'source.py',
            'public_entries': ['Correct.lean'], 'mathematical_target': 'identity',
            'required_conditions': ['valid IO'], 'scope': 'one complete invocation and frame',
            'canonical_connection': 'source execution theorem', 'fp_program_pair': ['source', 'candidate'],
            'numerical_dependencies': {'kind': 'strict_structure', 'rules': []},
            'source_sha256': {p.name: digest(p) for p in self.root.iterdir()},
            'readers': [{'path': 'Correct.lean', 'sha256': digest(self.root / 'Correct.lean'),
                         'specifications': [{'name': 'result'}]}],
            'review_status': 'reviewed_current_contract',
            'native_execution': 'not_established_by_this_review'}
        self.review = {'schema': 1, 'kind': 'final_selected_contract_review', 'tasks': [self.row]}

    def check(self):
        return validate_review(self.root, self.review, self.ledger, expected_tasks=1)

    def test_complete_selected_review(self):
        self.assertEqual(set(self.check()), {'fixture'})

    def test_missing_duplicate_or_unknown_task(self):
        for rows in ([], [self.row, self.row], [{**self.row, 'task': 'other'}]):
            with self.subTest(rows=len(rows)):
                self.review['tasks'] = rows
                with self.assertRaisesRegex(ValueError, 'every selected task'):
                    self.check()

    def test_source_or_public_statement_change(self):
        for name in ('Kernel.lean', 'Correct.lean', 'source.py'):
            path = self.root / name
            original = path.read_text()
            path.write_text(original + '# changed\n')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'source changed'):
                self.check()
            path.write_text(original)

    def test_observed_rehash_does_not_refresh_reader_review(self):
        path = self.root / 'Correct.lean'
        path.write_text('specification result : True := by trivial\n')
        self.row['source_sha256'][path.name] = digest(path)
        with self.assertRaisesRegex(ValueError, 'reader changed'):
            self.check()

    def test_source_selection_cannot_change_with_old_review(self):
        self.ledger['tasks'][0]['kernel_revision']['kernel_source'] = 'Correct.lean'
        with self.assertRaisesRegex(ValueError, 'selected revision changed'):
            self.check()

    def test_public_specification_omission_is_rejected(self):
        self.row['readers'][0]['specifications'] = []
        with self.assertRaisesRegex(ValueError, 'specification inventory changed'):
            self.check()

    def test_build_or_metadata_cannot_claim_native_execution(self):
        self.row['native_execution'] = 'passed'
        with self.assertRaisesRegex(ValueError, 'native-execution claim'):
            self.check()

    def test_conditions_pair_and_numerical_boundary_are_required(self):
        for key in ('required_conditions', 'fp_program_pair', 'numerical_dependencies', 'readers'):
            before = deepcopy(self.row[key])
            self.row[key] = []
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'explicit review field'):
                self.check()
            self.row[key] = before

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
        helper = self.root / 'bench/repairs/fixture/Proofs/Real.lean'
        helper.parent.mkdir(parents=True)
        helper.write_text('import bench.repairs.fixture.IO\n')
        io = helper.parent.parent / 'IO.lean'
        io.write_text('import bench.repairs.fixture.Kernels\n')
        source = 'import bench.repairs.fixture.Proofs.Real\ndef expectedSpec := 42\n'
        audited = prepare_source(source, 'bench/repairs/fixture/Correct.lean',
                                 self.root / 'scripts/absent.tsv')
        self.assertIn('#auditModuleAxioms', audited)
        self.assertIn('#auditModuleSpecs from [bench.repairs.fixture.IO, '
                      'bench.repairs.fixture.Kernels, bench.repairs.fixture.Proofs.Real]', audited)

    def test_audit_import_preserves_multiline_source(self):
        source = ('/- import bench.fake -/\nimport VeriTile.Spec\n'
                  '  bench.repairs.fixture.Kernels\ndef expectedSpec := 42\n')
        audited = prepare_source(source, 'bench/repairs/fixture/Correct.lean',
                                 self.root / 'scripts/absent.tsv')
        self.assertIn('import VeriTile.Meta.StatementAudit\nimport VeriTile.Spec\n'
                      '  bench.repairs.fixture.Kernels\n', audited)
        self.assertIn('#auditModuleSpecs from [bench.repairs.fixture.Kernels]', audited)
        self.assertNotIn('from [bench.fake', audited)


if __name__ == '__main__':
    unittest.main()
