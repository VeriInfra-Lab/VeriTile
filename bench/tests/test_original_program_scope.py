"""Scope-gate fixtures exercise rejection paths; none is a Lean certificate."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_current_source_adapters as fixtures
from adapter_common import canonical, hash_file, sha, write_json
from check_original_program_scope import (DIMENSIONS, source_inventory, task_bindings,
                                         validate_task, validate_deferrals, original_python, check)


class OriginalProgramScopeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdapterTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.catalog = self.fixture.root, self.fixture.catalog
        self.python = 'bench/tritonbench_g/t/original.py'
        self.fixture.put(self.python, 'def run(x):\n    if x.dtype == tl.float16:\n        return x\n    return x\n')
        digest = hash_file(self.root / self.python)
        self.catalog['python_input_sha256'] = {self.python: digest}
        self.catalog['python_sources'] = {self.python: {'sha256': digest, 'variant': 'original'}}
        self.task = self.catalog['tasks'][0]
        self.task['declarations'][0].update(source_variant='original_source_translation',
            model_kind='backend_value_contract', role='parametric_value_contract',
            scope_relation='full_declared_stage', dtype_scope_relation='complete_original_declared_domain')
        inventory = source_inventory(self.root, {self.python: digest})
        proof = {k: self.task['declarations'][0][k] for k in ('file', 'declaration', 'statement_sha256')}
        self.case = {'dtype_profile': {'inputs': {'x': 'fp32'}, 'outputs': {'result': 'fp32'},
                                       'compute': {'value': 'fp32'}},
                     'configuration': {'branch': 'all'}}
        self.task['declarations'][0]['covers_stages'] = [item['id'] for item in inventory['stages']]
        self.task['declarations'][0]['original_scope'] = {
            'source_inventory_sha256': sha(canonical(inventory)),
            **{dimension: [item['id'] for item in inventory[dimension]] for dimension in DIMENSIONS},
            'cases': {item['id']: [deepcopy(self.case)]
                      for dimension in DIMENSIONS for item in inventory[dimension]}}
        self.row = {'task': 't', 'scope_schema': 2, 'status': 'verified', 'structural_gaps': [],
            'input_sha256': task_bindings(self.root, self.catalog, self.task),
            'source_inventory': inventory,
            'coverage': {dimension: [{'id': item['id'], 'status': 'covered',
                'explanation': 'Synthetic review fixture; not a semantic proof.',
                'cases': [deepcopy(self.case)], 'proofs': [deepcopy(proof)]}
                for item in inventory[dimension]] for dimension in DIMENSIONS},
            'deferred_numerical_assumptions': [], 'callback_assumptions': {}}
        self.rule = {'id': 'fixture_rounding', 'status': 'pending_exact_binding',
                     'affected_tasks': ['t'], 'profile': {'dtype': 'fp32'},
                     'scope_bindings': [{'id': 'fp32_case', 'task': 't',
                         'declarations': [deepcopy(proof)], 'callbacks': [], **deepcopy(self.case)}]}
        self.queue = {'rules': [self.rule]}
        self.bind_review()

    def bind_review(self):
        review = {'schema': 1, 'scope_schema': 2, 'kind': 'original_program_scope_review', 'task': 't',
            'reviewer': 'root', 'status': 'reviewed', 'catalog_sha256': sha(canonical(self.catalog)),
            'input_sha256': self.row['input_sha256'],
            'source_inventory_sha256': sha(canonical(self.row['source_inventory'])),
            'coverage_sha256': sha(canonical(self.row['coverage'])),
            'deferred_numerical_sha256': sha(canonical(self.row['deferred_numerical_assumptions'])),
            'callback_assumptions_sha256': sha(canonical(self.row['callback_assumptions'])),
            'structural_gaps': [], 'semantic_review': 'Synthetic test artifact; no Lean claim.'}
        write_json(self.root / 'scope-review.json', review)
        self.row['review'] = {'path': 'scope-review.json', 'sha256': hash_file(self.root / 'scope-review.json')}

    def validate(self):
        return validate_task(self.root, self.catalog, self.task, self.row, self.queue)

    def defer(self):
        self.task['declarations'][0]['extra_numeric_law_ids'] = [self.rule['id']]
        self.row['status'] = 'verified_with_deferred_numerics'
        self.row['deferred_numerical_assumptions'] = [{
            'assumption_id': self.rule['id'], 'kind': 'numerical_rule_conformance',
            'scope_binding_id': 'fp32_case', 'rule_sha256': sha(canonical(self.rule)),
            **deepcopy(self.case),
            'required_validation': 'Bind exact fp32 rule to target compiler/device.'}]
        self.bind_review()

    def test_reviewed_complete_fixture_is_accepted_only_at_scope_layer(self):
        self.assertEqual('verified', self.validate())

    def test_old_nonempty_prose_schema_cannot_be_accepted(self):
        self.row.pop('scope_schema')
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'schema 2'):
            self.validate()

    def test_rehashed_selected_dtype_profile_cannot_cover_whole_domain(self):
        self.task['declarations'][0]['dtype_scope_relation'] = 'selected_explicit_profiles_only'
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'no full original execution'):
            self.validate()

    def test_rehashed_kernel_stage_label_cannot_cover_host(self):
        self.task['declarations'][0]['covers_stages'] = ['kernel_only']
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'stage labels'):
            self.validate()

    def test_full_stage_evidence_without_original_scope_is_rejected(self):
        self.task['declarations'][0].pop('original_scope')
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'explicit original scope'):
            self.validate()

    def test_new_review_cannot_make_wrong_dtype_case_compatible(self):
        self.row['coverage']['dtypes'][0]['cases'][0]['dtype_profile']['inputs']['x'] = 'fp64'
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'does not match required dtype/configuration'):
            self.validate()

    def test_new_review_cannot_make_wrong_constexpr_case_compatible(self):
        self.row['coverage']['branches'][0]['cases'][0]['configuration'] = {'branch': 'true_only'}
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'does not match required dtype/configuration'):
            self.validate()

    def test_declared_execution_must_cover_exact_branch(self):
        scope = self.task['declarations'][0]['original_scope']
        ident = scope['branches'].pop()
        scope['cases'].pop(ident)
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'no full original execution'):
            self.validate()

    def test_dtype_profiles_are_structured_not_arbitrary_prose(self):
        for profile in ('all dtypes', {'inputs': {'x': 'invented'}, 'outputs': {'x': 'fp32'}, 'compute': {}}):
            with self.subTest(profile=profile):
                self.row['coverage']['dtypes'][0]['cases'][0]['dtype_profile'] = profile
                self.bind_review()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_pending_cannot_pass_even_with_hashes_and_proof_names(self):
        self.row['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'scope remains'):
            self.validate()

    def test_each_dimension_is_mandatory(self):
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                saved = self.row['coverage'].pop(dimension)
                with self.assertRaisesRegex(ValueError, 'missing scope dimension'):
                    self.validate()
                self.row['coverage'][dimension] = saved

    def test_omitted_branch_cannot_be_a_dtype_or_numerical_deferral(self):
        self.row['coverage']['branches'] = []
        with self.assertRaisesRegex(ValueError, 'missing/extra branches'):
            self.validate()

    def test_gap_cannot_be_relabelled_as_numerical(self):
        self.row['structural_gaps'] = [{'kind': 'branch', 'can_defer_as_numeric_assumption': True}]
        with self.assertRaisesRegex(ValueError, 'structural/scope gaps'):
            self.validate()

    def test_new_source_branch_invalidates_inventory(self):
        self.fixture.put(self.python, 'def run(x):\n    return x if x else 0\n')
        digest = hash_file(self.root / self.python)
        self.catalog['python_sources'][self.python]['sha256'] = digest
        self.catalog['python_input_sha256'][self.python] = digest
        self.row['input_sha256'] = task_bindings(self.root, self.catalog, self.task)
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            self.validate()

    def test_helper_input_omission_fails(self):
        self.row['input_sha256']['lean'].pop(self.fixture.helper)
        with self.assertRaisesRegex(ValueError, 'bindings changed or incomplete'):
            self.validate()

    def test_covered_label_requires_actual_public_proof_reference(self):
        self.row['coverage']['stages'][0]['proofs'] = []
        with self.assertRaisesRegex(ValueError, 'lacks public proof'):
            self.validate()

    def test_candidate_and_standalone_models_cannot_cover_original(self):
        for variant in ('compile_repaired_candidate', 'standalone_sc_model', 'observed_compiled_graph'):
            with self.subTest(variant=variant):
                self.task['declarations'][0]['source_variant'] = variant
                with self.assertRaisesRegex(ValueError, 'no full original execution'):
                    self.validate()

    def test_identity_frame_protocol_and_fixed_witness_cannot_cover_scope(self):
        for model, role, relation in [
                ('recurrence_identity', 'recurrence_identity', 'not_an_execution_contract'),
                ('memory_frame_contract', 'memory_frame_contract', 'not_an_execution_contract'),
                ('protocol_contract', 'protocol_contract', 'not_an_execution_contract'),
                ('bit_interpreter_contract', 'parametric_value_contract', 'strict_specialization')]:
            with self.subTest(model=model):
                self.task['declarations'][0].update(model_kind=model, role=role, scope_relation=relation)
                with self.assertRaisesRegex(ValueError, 'no full original execution'):
                    self.validate()

    def test_explicit_external_original_is_inventoried_and_bound(self):
        external = 'bench/review-checkpoints/fixture/native-original/t.py'
        self.fixture.put(external, 'def external_host(x):\n    return x\n')
        digest = hash_file(self.root / external)
        self.task['source_views'][0]['original_python'] = external
        self.catalog['python_sources'][external] = {'sha256': digest, 'variant': 'original'}
        originals = original_python(self.catalog, 't')
        self.assertEqual({self.python, external}, set(originals))
        self.assertIn(external, task_bindings(self.root, self.catalog, self.task)['python'])
        self.assertIn('external_host', {r['function'] for r in source_inventory(self.root, originals)['stages']})

    def test_explicit_external_original_requires_original_variant_binding(self):
        external = 'bench/review-checkpoints/fixture/candidate/t.py'
        self.task['source_views'][0]['original_python'] = external
        self.catalog['python_sources'][external] = {'sha256': 'a' * 64, 'variant': 'candidate'}
        with self.assertRaisesRegex(ValueError, 'lacks original catalog binding'):
            original_python(self.catalog, 't')

    def test_other_task_original_cannot_be_used_as_this_task_source(self):
        other = 'bench/tritonbench_g/other/original.py'
        self.task['source_views'][0]['original_python'] = other
        self.catalog['python_sources'][other] = {'sha256': 'a' * 64, 'variant': 'original'}
        with self.assertRaisesRegex(ValueError, 'belongs to another task'):
            original_python(self.catalog, 't')

    def test_shared_external_original_cannot_silently_move_between_tasks(self):
        external = 'bench/review-checkpoints/fixture/native-original/t.py'
        self.task['source_views'][0]['original_python'] = external
        self.catalog['python_sources'][external] = {'sha256': 'a' * 64, 'variant': 'original'}
        self.catalog['tasks'].append({'task': 'other', 'source_views': [{'original_python': external}]})
        with self.assertRaisesRegex(ValueError, 'assigned to multiple tasks'):
            original_python(self.catalog, 't')

    def test_production_function_cannot_be_test_harness(self):
        self.row['coverage']['stages'][0]['status'] = 'test_harness'
        with self.assertRaisesRegex(ValueError, 'production scope cannot be excluded'):
            self.validate()

    def test_test_named_jit_and_production_called_test_are_not_harnesses(self):
        self.fixture.put(self.python, '@triton.jit\ndef test_kernel():\n    return 1\n'
                         'def run():\n    return test_helper()\n'
                         'def test_helper():\n    return 2\n'
                         'def test_only():\n    return 3\n')
        inventory = source_inventory(self.root, {self.python: hash_file(self.root / self.python)})
        states = {row['function']: row['may_be_test_harness'] for row in inventory['stages']}
        self.assertEqual({'test_kernel': False, 'run': False, 'test_helper': False, 'test_only': True}, states)

    def test_edited_obligation_requires_new_exact_review(self):
        self.row['coverage']['dtypes'][0]['explanation'] = 'Updated semantic review.'
        with self.assertRaisesRegex(ValueError, 'does not bind exact'):
            self.validate()

    def test_registered_rule_is_explicitly_conditional(self):
        self.defer()
        self.assertEqual('verified_with_deferred_numerics', self.validate())

    def test_registered_callback_requires_matching_rule_declaration_and_case(self):
        self.rule['scope_bindings'][0]['callbacks'] = ['Backend.exp2']
        self.task['declarations'][0]['unclosed_numeric_callbacks'] = ['Backend.exp2']
        self.defer()
        self.row['callback_assumptions'] = {'Backend.exp2': [
            {'assumption_id': self.rule['id'], 'scope_binding_id': 'fp32_case'}]}
        self.bind_review()
        self.assertEqual('verified_with_deferred_numerics', self.validate())

    def test_callback_cannot_use_unrelated_registered_rounding_rule(self):
        self.task['declarations'][0]['unclosed_numeric_callbacks'] = ['Backend.exp2']
        self.defer()
        self.row['callback_assumptions'] = {'Backend.exp2': [
            {'assumption_id': self.rule['id'], 'scope_binding_id': 'fp32_case'}]}
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'callback is incompatible'):
            self.validate()

    def test_callback_rule_name_string_is_not_a_scope_binding(self):
        self.task['declarations'][0]['unclosed_numeric_callbacks'] = ['Backend.exp2']
        self.defer()
        self.row['callback_assumptions'] = {'Backend.exp2': self.rule['id']}
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'explicit numerical scope binding'):
            self.validate()

    def test_fp32_rule_cannot_be_deferred_at_fp64_configuration(self):
        self.defer()
        item = self.row['deferred_numerical_assumptions'][0]
        item['dtype_profile']['inputs']['x'] = 'fp64'
        item['configuration'] = {'branch': 'other'}
        self.bind_review()
        with self.assertRaisesRegex(ValueError, 'does not match registered binding'):
            self.validate()

    def test_queue_rule_without_explicit_scope_binding_is_not_enough(self):
        self.rule.pop('scope_bindings')
        self.defer()
        with self.assertRaisesRegex(ValueError, 'registered task scope binding'):
            self.validate()

    def test_same_task_rule_for_another_theorem_does_not_discharge_assumption(self):
        self.rule['scope_bindings'][0]['declarations'][0]['declaration'] = 'another_contract'
        self.defer()
        with self.assertRaisesRegex(ValueError, 'wrong declaration/dtype/configuration'):
            self.validate()

    def test_rule_binding_for_stale_statement_does_not_discharge_assumption(self):
        self.rule['scope_bindings'][0]['declarations'][0]['statement_sha256'] = 'b' * 64
        self.defer()
        with self.assertRaisesRegex(ValueError, 'wrong declaration/dtype/configuration'):
            self.validate()

    def test_adding_a_second_case_requires_its_own_numeric_registration(self):
        other_case = deepcopy(self.case)
        other_case['configuration'] = {'branch': 'second'}
        for rows in self.row['coverage'].values():
            for row in rows:
                row['cases'].append(deepcopy(other_case))
        for cases in self.task['declarations'][0]['original_scope']['cases'].values():
            cases.append(deepcopy(other_case))
        self.defer()
        with self.assertRaisesRegex(ValueError, 'wrong declaration/dtype/configuration'):
            self.validate()

    def test_one_rule_may_have_two_distinct_exact_registered_cases(self):
        other_case = deepcopy(self.case)
        other_case['configuration'] = {'branch': 'second'}
        for rows in self.row['coverage'].values():
            for row in rows:
                row['cases'].append(deepcopy(other_case))
        for cases in self.task['declarations'][0]['original_scope']['cases'].values():
            cases.append(deepcopy(other_case))
        second_binding = deepcopy(self.rule['scope_bindings'][0])
        second_binding.update(id='second_case', **other_case)
        self.rule['scope_bindings'].append(second_binding)
        self.defer()
        second = deepcopy(self.row['deferred_numerical_assumptions'][0])
        second.update(scope_binding_id='second_case', **other_case)
        self.row['deferred_numerical_assumptions'].append(second)
        self.bind_review()
        self.assertEqual('verified_with_deferred_numerics', self.validate())

    def test_used_numerical_rule_cannot_disappear(self):
        self.task['declarations'][0]['extra_numeric_law_ids'] = [self.rule['id']]
        with self.assertRaisesRegex(ValueError, 'assumptions omitted'):
            self.validate()

    def test_backend_callback_requires_explicit_registered_rule(self):
        self.task['declarations'][0]['unclosed_numeric_callbacks'] = ['Backend.exp2']
        with self.assertRaisesRegex(ValueError, 'callbacks lack exact'):
            self.validate()

    def test_unknown_rule_cannot_be_deferred(self):
        self.defer()
        self.row['deferred_numerical_assumptions'][0]['assumption_id'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'unregistered numerical'):
            self.validate()

    def test_refuted_rule_cannot_be_deferred(self):
        self.defer()
        self.rule['status'] = 'not_an_experiment'
        with self.assertRaisesRegex(ValueError, 'refuted/nondeferrable'):
            self.validate()

    def test_other_task_rule_cannot_be_deferred(self):
        self.defer()
        self.rule['affected_tasks'] = ['another']
        with self.assertRaisesRegex(ValueError, 'another task'):
            self.validate()

    def test_deferral_requires_dtype_configuration_and_followup(self):
        self.defer()
        for field, empty in [('dtype_profile', ''), ('configuration', {}), ('required_validation', '')]:
            with self.subTest(field=field):
                item = self.row['deferred_numerical_assumptions'][0]
                saved = item[field]; item[field] = empty
                with self.assertRaises(ValueError):
                    self.validate()
                item[field] = saved

    def test_a_partial_population_cannot_be_used_as_183_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'exactly 183'):
            check(self.root, self.catalog, {'schema': 1, 'kind': 'original_program_scope_ledger'})


if __name__ == '__main__':
    unittest.main()
