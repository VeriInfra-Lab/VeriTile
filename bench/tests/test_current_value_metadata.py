"""Supplement metadata rejection tests; fixtures make no compiled-proof claims."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter_common import canonical, hash_file, sha, write_json
from current_value_metadata import check_supplement_scope, value_scope, check


class CurrentValueMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = {'task': 't', 'required_value_stages': ['main'], 'declarations': [{
            'file': 'bench/tritonbench_g/t/NewCorrect.lean', 'declaration': 'value',
            'statement_sha256': 'a' * 64, 'model_kind': 'backend_value_contract',
            'role': 'parametric_value_contract', 'scope_relation': 'full_declared_stage',
            'source_variant': 'original_source_translation', 'covers_stages': ['main'],
            'dtype_profile': 'all original FloatDType parameters',
            'dtype_scope_relation': 'complete_original_declared_domain',
            'unclosed_numeric_callbacks': [], 'extra_numeric_law_ids': [],
            'native_binding_status': 'not_established'}]}
        self.supplement = {'schema': 1, 'tasks': [self.task]}
        self.bind_review()

    def bind_review(self):
        write_json(self.root / 'review.json', {'reviewed_value_scope_sha256': {
            't': sha(canonical(value_scope(self.task)))}})
        self.task['review'] = {'path': 'review.json', 'sha256': hash_file(self.root / 'review.json')}

    def check(self):
        return check_supplement_scope(self.supplement, self.root)

    def test_explicit_reviewed_dtype_stage_summary_has_no_whole_program_claim(self):
        result = self.check()[0]
        self.assertEqual(['main'], result['original_full_declared_value_stages'])
        self.assertEqual([], result['uncovered_declared_stages'])
        self.assertEqual('requires_separate_183_scope_ledger', result['whole_original_program_coverage'])

    def test_missing_stage_inventory_rejected(self):
        del self.task['required_value_stages']
        with self.assertRaisesRegex(ValueError, 'explicit required stage'):
            self.check()

    def test_dtype_relabel_needs_new_scope_review(self):
        self.task['declarations'][0]['dtype_profile'] = 'fp32 only'
        with self.assertRaisesRegex(ValueError, 'lack exact current review'):
            self.check()

    def test_original_scope_metadata_changes_need_new_value_scope_review(self):
        self.task['declarations'][0]['original_scope'] = {'stages': ['stage:original.py:1:run']}
        with self.assertRaisesRegex(ValueError, 'lack exact current review'):
            self.check()

    def test_partial_supplement_does_not_require_original_scope_metadata(self):
        self.task['declarations'][0]['dtype_scope_relation'] = 'selected_explicit_profiles_only'
        self.bind_review()
        self.assertEqual('requires_separate_183_scope_ledger', self.check()[0]['whole_original_program_coverage'])

    def test_missing_dtype_profile_not_inferred_from_theorem_name(self):
        self.task['declarations'][0]['declaration'] = 'all_fp32'
        del self.task['declarations'][0]['dtype_profile']; self.bind_review()
        with self.assertRaisesRegex(ValueError, 'explicit typed dtype profile'):
            self.check()

    def test_undeclared_stage_rejected(self):
        self.task['declarations'][0]['covers_stages'] = ['another']; self.bind_review()
        with self.assertRaisesRegex(ValueError, 'undeclared/duplicate'):
            self.check()

    def test_nonvalue_cannot_fill_stage(self):
        self.task['declarations'][0].update(model_kind='protocol_contract', role='protocol_contract')
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'nonvalue supplement promoted'):
            self.check()

    def test_strict_specialization_does_not_fill_full_stage(self):
        self.task['declarations'][0]['scope_relation'] = 'strict_specialization'; self.bind_review()
        result = self.check()[0]
        self.assertEqual(['main'], result['uncovered_declared_stages'])
        self.assertEqual([], result['original_full_declared_value_stages'])

    def test_candidate_proof_does_not_fill_original_stage(self):
        self.task['declarations'][0]['source_variant'] = 'compile_repaired_candidate'; self.bind_review()
        self.assertEqual(['main'], self.check()[0]['uncovered_declared_stages'])

    def test_metadata_cannot_claim_native_binding(self):
        self.task['declarations'][0]['native_binding_status'] = 'passed'; self.bind_review()
        with self.assertRaisesRegex(ValueError, 'native-binding claim'):
            self.check()

    def test_frozen_complete_replay_failure_is_not_skipped(self):
        frozen = {'root': self.root, 'manifest': {}}
        with patch('current_value_metadata.baseline', return_value=frozen), \
                patch('current_value_metadata.subprocess.run', return_value=type('Exit', (), {'returncode': 9})()):
            with self.assertRaisesRegex(ValueError, 'frozen complete value metadata replay failed: 9'):
                check(self.root, self.root, {}, {}, {}, self.root, {}, None)


if __name__ == '__main__':
    unittest.main()
