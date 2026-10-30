"""Local rejection must not bypass source authority for otherwise valid rows."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'bench'))
from scripts.bench import check_original_program_scope as gate
from scripts.bench import exact_scope_gate
from scripts.bench import deferred_numerical_experiments
from scripts.bench.adapter_common import canonical, sha
from scripts.bench.exact_numerical_bindings import extend_queue


class NumericalScopePrecheckTests(unittest.TestCase):
    def setUp(self):
        self.callback = 'Fixture.Numerics.add'
        self.reference = {'file': 'bench/tritonbench_g/fixture/Correct.lean',
                          'declaration': 'source_outcomes', 'statement_sha256': 'a' * 64}
        case = {'dtype_profile': {'inputs': {'x': 'fp32'}, 'outputs': {'out': 'fp32'},
                                 'compute': {'value': 'fp32'}},
                'configuration': {'N': 3, 'outcome': 'returned_active'}}
        bindings = [{'id': name, 'task': 'fixture', 'declarations': [self.reference],
                     'callbacks': [self.callback], **deepcopy(case)} for name in ('first', 'second')]
        rule = {'id': 'fixture_add', 'status': 'pending_numeric_validation',
                'queue_status': 'pending_numeric_validation',
                'affected_tasks': ['fixture'], 'scope_bindings': bindings}
        references = [{'callback': self.callback, 'assumption_id': rule['id'],
                       'scope_binding_id': b['id']} for b in bindings]
        table = {'task': 'fixture', 'declaration': self.reference,
                 'callback_union': [self.callback],
                 'cases': [{**case, 'required_callbacks': [self.callback],
                            'required_binding_refs': references}]}
        self.authority = {'rules': [rule], 'provenance': {}, 'requirements': [table],
                          'input_sha256': {'fixture': 'b' * 64},
                          'unmigrated_descriptive_followups': {}, 'summary': {}}
        self.legacy = {'schema': 1, 'rules': [], 'summary': {},
                       'tasks': [{'task': 'fixture', 'rule_ids': []}]}
        self.queue = extend_queue(self.legacy, self.authority)
        self.row = {'deferred_numerical_assumptions': [
            {'assumption_id': rule['id'], 'scope_binding_id': b['id'],
             'kind': 'numerical_rule_conformance', 'rule_sha256': sha(canonical(rule)),
             **deepcopy(case), 'required_validation': 'Validate both scalar ports.'}
            for b in bindings],
            'callback_assumptions': {self.callback: [
                {k: r[k] for k in ('assumption_id', 'scope_binding_id')} for r in references]}}
        key = (self.reference['file'], self.reference['declaration'])
        self.public = {key: {**self.reference, 'unclosed_numeric_callbacks': [self.callback]}}
        self.cases = {key: {canonical(case)}}
        self.derive = self.enterContext(patch.object(exact_scope_gate, 'derive_authority',
                                                    return_value=self.authority))
        self.collect = self.enterContext(patch.object(deferred_numerical_experiments,
                                                     'collect_legacy', return_value=self.legacy))
        # Queue equality and exact port validation execute their real implementations.
        self.queue_check = self.enterContext(patch.object(exact_scope_gate, 'validate_exact_queue',
                                                         wraps=exact_scope_gate.validate_exact_queue))

    def validate(self):
        return gate.validate_used_numerics('fixture', self.row, self.public, self.cases,
                                          self.queue, root=Path('/fixture'))

    def test_omitted_referenced_deferral_rejects_before_authority(self):
        self.row['deferred_numerical_assumptions'].pop()
        with self.assertRaisesRegex(ValueError, 'incompatible with registered numerical rule'):
            self.validate()
        self.derive.assert_not_called()
        self.queue_check.assert_not_called()

    def test_removing_deferral_and_link_still_fails_exact_port_check(self):
        self.row['deferred_numerical_assumptions'].pop()
        self.row['callback_assumptions'][self.callback].pop()
        with self.assertRaisesRegex(ValueError, 'exact scalar port omitted'):
            self.validate()
        self.derive.assert_called_once()
        self.queue_check.assert_called_once()

    def test_valid_calls_rederive_and_later_source_failure_is_observed(self):
        expected = (self.row['deferred_numerical_assumptions'], self.row['callback_assumptions'])
        self.assertEqual(self.validate(), expected)
        self.assertEqual(self.validate(), expected)
        self.assertEqual(self.derive.call_count, 2)
        self.assertEqual(self.collect.call_count, 2)
        self.assertEqual(self.queue_check.call_count, 2)
        self.derive.side_effect = ValueError('source authority changed')
        with self.assertRaisesRegex(ValueError, 'source authority changed'):
            self.validate()
        self.assertEqual(self.derive.call_count, 3)
        self.assertEqual(self.queue_check.call_count, 2)

    def test_valid_references_do_not_authorize_stale_queue(self):
        self.queue['exact_numerical_inputs']['fixture'] = 'c' * 64
        with self.assertRaises(ValueError):
            self.validate()
        self.derive.assert_called_once()
        self.queue_check.assert_called_once()

    def test_missing_callback_names_still_require_authority(self):
        self.row['callback_assumptions'].clear()
        with self.assertRaisesRegex(ValueError, 'unclosed numerical callbacks'):
            self.validate()
        self.derive.assert_called_once()
        self.queue_check.assert_called_once()

    def test_local_reference_error_precedes_simultaneous_authority_failure(self):
        self.row['deferred_numerical_assumptions'].pop()
        self.derive.side_effect = ValueError('source authority changed')
        with self.assertRaisesRegex(ValueError, 'incompatible with registered numerical rule'):
            self.validate()
        self.derive.assert_not_called()


if __name__ == '__main__':
    unittest.main()
