"""Keep scope, arithmetic-model adequacy and numerical evidence separate.

The tiny Nat/True source below is a parser/binding fixture only. Assigning model
reviews to it exercises inventory policy; it is not evidence about actual Lean
floating-point theorems or native execution.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
import numerical_obligations as numerical


class NumericalObligationsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = 'bench/tritonbench_g/example/Correct.lean'
        source = self.root / self.file
        source.parent.mkdir(parents=True)
        source.write_text(
            'specification first (n : Nat) (h : 0 < n) : 0 < n := h\n'
            'specification second : True := by trivial\n')
        self.evidence = 'experiments/example/report.json'
        evidence = self.root / self.evidence
        evidence.parent.mkdir(parents=True)
        evidence.write_text('{"decision":"ACCEPT","instance":"exact-profile"}\n')
        self.rows = [
            {'file': self.file, 'declaration': name, 'numeric_model': 'mathematical',
             'coverage_level': 'specialization'}
            for name in ('first', 'second')
        ]
        statements = numerical.statement_map(self.rows, self.root)
        self.review = {
            'schema': 1,
            'reviewed_on': '2026-10-14',
            'policy': {
                'normal_input_conditions': 'accepted',
                'configuration_and_stage_scope': 'accepted',
            },
            'input_sha256': {
                name: numerical.digest((self.root / name).read_bytes())
                for name in (self.file, self.evidence)
            },
            'obligations': [],
            'declarations': [
                {**row,
                 'statement_sha256': numerical.digest(
                     statements[row['file'], row['declaration']].encode()),
                 'review_status': 'reviewed',
                 'review_basis': 'Explicit review of the actual statement and its definitions.',
                 'fp_model_review': self.model_review('mathematical_contract'),
                 'extra_numeric_ids': [], 'shared_model_ids': []}
                for row in self.rows
            ],
            'native_boundaries': ['No native arithmetic result is claimed by this fixture.'],
        }

    def model_review(self, kind):
        return {
            'kind': kind,
            'review_basis': 'Synthetic classifier fixture; no Lean semantic claim is made.',
            'evidence': [{'path': self.file}],
        }

    def test_embedded_source_hash_cannot_disagree_with_bound_input(self):
        evidence = self.review['declarations'][0]['fp_model_review']['evidence'][0]
        evidence['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Stale embedded evidence fingerprint'):
            numerical.validate_review(self.review, self.rows, self.root)

    def set_model(self, kind, *, row=0, label=None):
        self.review['declarations'][row]['fp_model_review'] = self.model_review(kind)
        if label is not None:
            self.rows[row]['numeric_model'] = label
            self.review['declarations'][row]['numeric_model'] = label

    def add_obligation(self, name='add', *, kind='reported_atom',
                       status='reported_evidence_available', row=0):
        obligation = {
            'id': name, 'kind': kind, 'evidence_status': status,
            'formula': 'add_fp32(a, b) = add_fp32(b, a)',
            'domain': 'Finite actual operands at the rewrite site.',
            'profiles': {'input': 'fp32', 'compute': 'fp32', 'output': 'fp32'},
            'experiment': 'Replay the bound instance and review the exact Lean binding.',
            'evidence': [{'path': self.evidence, 'description': 'Exact-profile result'}],
        }
        self.review['obligations'].append(obligation)
        field = 'shared_model_ids' if kind == 'shared_model_requirement' else 'extra_numeric_ids'
        self.review['declarations'][row][field].append(name)
        return obligation

    def validate(self):
        return numerical.validate_review(self.review, self.rows, self.root)

    def first_status(self):
        reviewed, _ = self.validate()
        return reviewed[self.file, 'first']['numerical_status']

    def task(self, reviewed, *, source=True, declarations=None):
        return numerical.task_status({
            'task': 'example', 'source_entry': self.file if source else None,
            'public_declarations': self.rows if declarations is None else declarations,
        }, reviewed)

    def test_configuration_and_stage_labels_do_not_create_numeric_obligations(self):
        for level in ('full_value_candidate', 'specialization',
                      'precomputed_input_slice', 'pre_rounding_slice', 'projection_only'):
            with self.subTest(level=level):
                for row in self.rows:
                    row['coverage_level'] = level
                reviewed, _ = self.validate()
                task = self.task(reviewed)
                self.assertEqual(task['numerical_status'], 'no_extra_numeric_law')
                self.assertIn('scope_accepted', task['scope_policy'])
                self.assertFalse(task['hardware_or_independent_replay_accepted'])
                self.assertEqual(task['extra_numeric_ids'], [])

    def test_empty_fp_review_is_not_a_missing_rule(self):
        for row in range(len(self.rows)):
            self.set_model('backend_equivalence', row=row, label='admitted_fp')
        reviewed, obligations = self.validate()
        self.assertEqual(obligations, {})
        self.assertTrue(all(r['numerical_status'] == 'no_extra_numeric_law'
                            for r in reviewed.values()))

    def test_shared_model_is_visible_without_becoming_an_extra_law(self):
        self.add_obligation('idempotence', kind='shared_model_requirement',
                            status='shared_model_contract')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'no_extra_numeric_law')
        self.assertEqual(task['shared_model_ids'], ['idempotence'])
        self.assertEqual(task['extra_numeric_ids'], [])

    def test_shared_model_cannot_be_placed_on_extra_axis(self):
        self.add_obligation('idempotence', kind='shared_model_requirement',
                            status='shared_model_contract')
        row = self.review['declarations'][0]
        row['extra_numeric_ids'], row['shared_model_ids'] = row['shared_model_ids'], []
        with self.assertRaisesRegex(ValueError, 'wrong axis'):
            self.validate()

    def test_extra_law_cannot_be_hidden_on_shared_axis(self):
        self.add_obligation()
        row = self.review['declarations'][0]
        row['shared_model_ids'], row['extra_numeric_ids'] = row['extra_numeric_ids'], []
        with self.assertRaisesRegex(ValueError, 'wrong axis'):
            self.validate()

    def test_extra_law_cannot_use_shared_contract_evidence_status(self):
        self.add_obligation(status='shared_model_contract')
        with self.assertRaisesRegex(ValueError, 'cannot be hidden'):
            self.validate()

    def test_shared_model_cannot_claim_experiment_acceptance(self):
        self.add_obligation(kind='shared_model_requirement')
        with self.assertRaisesRegex(ValueError, 'separate axis'):
            self.validate()

    def test_pending_law_stays_pending_when_scope_is_accepted(self):
        for status in ('binding_validation_pending', 'experiment_pending'):
            with self.subTest(status=status):
                self.review['obligations'] = []
                self.review['declarations'][0]['extra_numeric_ids'] = []
                self.add_obligation(status=status)
                reviewed, _ = self.validate()
                self.assertEqual(self.task(reviewed)['numerical_status'],
                                 'awaiting_numeric_validation')

    def test_existing_report_is_evidence_without_hardware_acceptance(self):
        self.add_obligation()
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'existing_numeric_evidence')
        self.assertFalse(task['hardware_or_independent_replay_accepted'])

    def test_identity_counterexample_takes_priority_over_existing_and_pending(self):
        self.add_obligation('reported')
        self.add_obligation('pending', status='experiment_pending')
        self.add_obligation('round_identity', kind='explicit_numeric_law',
                            status='counterexample_to_unrestricted_form')
        self.assertEqual(self.first_status(), 'requires_numeric_model_revision')

    def test_task_combination_takes_strongest_status_and_deduplicates_ids(self):
        self.add_obligation('reported')
        self.review['declarations'][1]['extra_numeric_ids'].append('reported')
        self.add_obligation('pending', status='binding_validation_pending', row=1)
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'awaiting_numeric_validation')
        self.assertEqual(task['extra_numeric_ids'], ['pending', 'reported'])
        self.add_obligation('round_identity', kind='explicit_numeric_law',
                            status='counterexample_to_unrestricted_form', row=1)
        reviewed, _ = self.validate()
        self.assertEqual(self.task(reviewed)['numerical_status'],
                         'requires_numeric_model_revision')

    def test_source_translation_blocker_is_separate_from_numeric_axis(self):
        self.rows[0]['coverage_level'] = 'blocked_summary'
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertTrue(task['source_translation_blocked'])
        self.assertEqual(task['numerical_status'], 'no_extra_numeric_law')
        self.assertIn('omitted producers', task['scope_note'])

    def test_missing_source_cannot_pass_by_having_no_numeric_entries(self):
        task = self.task({}, source=False, declarations=[])
        self.assertEqual(task['numerical_status'], 'missing_source')
        self.assertFalse(task['hardware_or_independent_replay_accepted'])

    def test_imported_task_without_public_reviews_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no numerical-review entries'):
            self.task({}, declarations=[])

    def test_aggregation_handles_empty_generator_and_invalid_status(self):
        self.assertEqual(numerical.aggregate_status(iter(())), 'no_extra_numeric_law')
        statuses = iter(('existing_numeric_evidence', 'awaiting_numeric_validation'))
        self.assertEqual(numerical.aggregate_status(statuses), 'awaiting_numeric_validation')
        with self.assertRaises(ValueError):
            numerical.aggregate_status(['silently_accepted'])

    def test_unknown_obligation_is_rejected(self):
        self.review['declarations'][0]['extra_numeric_ids'].append('unreviewed')
        with self.assertRaisesRegex(ValueError, 'Unreviewed numerical assumption'):
            self.validate()

    def test_missing_public_declaration_review_is_rejected(self):
        self.review['declarations'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing numerical reviews'):
            self.validate()

    def test_unexpected_and_duplicate_public_reviews_are_rejected(self):
        for name in ('first', 'unknown'):
            with self.subTest(name=name):
                duplicate = copy.deepcopy(self.review['declarations'][0])
                duplicate['declaration'] = name
                self.review['declarations'].append(duplicate)
                with self.assertRaisesRegex(ValueError, 'Unexpected or duplicate'):
                    self.validate()
                self.review['declarations'].pop()

    def test_duplicate_input_public_records_are_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate public record'):
            self.validate()

    def test_actual_missing_declaration_is_not_trusted_from_review(self):
        self.rows[0]['declaration'] = 'absent_from_source'
        with self.assertRaisesRegex(ValueError, 'Unresolved numerical-review declaration'):
            self.validate()

    def test_public_source_hash_detects_proof_only_changes(self):
        path = self.root / self.file
        path.write_text(path.read_text().replace(':= h', ':= by exact h'))
        with self.assertRaisesRegex(ValueError, 'Numerical review is stale'):
            self.validate()

    def test_statement_hash_detects_changed_contract_even_if_file_hash_is_refreshed(self):
        path = self.root / self.file
        path.write_text(path.read_text().replace('0 < n', '1 < n'))
        self.review['input_sha256'][self.file] = numerical.digest(path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Changed numerical-review statement'):
            self.validate()

    def test_statement_hash_itself_must_match(self):
        self.review['declarations'][0]['statement_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Changed numerical-review statement'):
            self.validate()

    def test_public_source_must_be_bound_even_with_correct_statement_hash(self):
        del self.review['input_sha256'][self.file]
        with self.assertRaisesRegex(ValueError, 'Unbound public source'):
            self.validate()

    def test_bound_evidence_changes_are_rejected(self):
        self.add_obligation()
        (self.root / self.evidence).write_text('{"decision":"REJECT"}\n')
        with self.assertRaisesRegex(ValueError, 'Numerical review is stale'):
            self.validate()

    def test_local_evidence_cannot_be_removed_from_bound_inputs(self):
        self.add_obligation()
        del self.review['input_sha256'][self.evidence]
        with self.assertRaisesRegex(ValueError, '[Uu]nbound.*evidence|[Ee]vidence.*bound'):
            self.validate()

    def test_input_path_cannot_escape_review_root(self):
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / 'report.json'
            path.write_text('{}\n')
            self.review['input_sha256'][str(path)] = numerical.digest(path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Invalid numerical-review input'):
                self.validate()

    def test_numeric_model_changes_require_review(self):
        self.rows[0]['numeric_model'] = 'admitted_fp'
        with self.assertRaisesRegex(ValueError, 'Numerical model changed'):
            self.validate()

    def test_unreviewed_or_empty_basis_is_rejected(self):
        for key, value in (('review_status', 'pending'), ('review_basis', '')):
            with self.subTest(field=key):
                original = self.review['declarations'][0][key]
                self.review['declarations'][0][key] = value
                with self.assertRaisesRegex(ValueError, 'have not been reviewed'):
                    self.validate()
                self.review['declarations'][0][key] = original

    def test_duplicate_ids_and_unused_obligations_are_rejected(self):
        self.add_obligation()
        self.review['declarations'][0]['extra_numeric_ids'].append('add')
        with self.assertRaisesRegex(ValueError, 'duplicate numerical IDs'):
            self.validate()
        self.review['declarations'][0]['extra_numeric_ids'] = []
        with self.assertRaisesRegex(ValueError, 'Unused or missing numerical experiment entries'):
            self.validate()

    def test_unknown_kind_status_or_empty_required_field_is_rejected(self):
        item = self.add_obligation()
        for key, value in (('kind', 'universal_ieee_proof'),
                           ('evidence_status', 'PASS'), ('formula', ''),
                           ('domain', ''), ('profiles', {}), ('experiment', ''),
                           ('evidence', [])):
            with self.subTest(field=key):
                original = item[key]
                item[key] = value
                with self.assertRaises(ValueError):
                    self.validate()
                item[key] = original

    def test_policy_must_not_claim_unrequested_scope_acceptance(self):
        self.review['policy']['numeric_hypotheses'] = 'accepted'
        with self.assertRaisesRegex(ValueError, 'scope policy'):
            self.validate()

    def experiment_task(self, name='add-fp32', obligation='add',
                        status='existing_report_optional_replay'):
        return {
            'id': name, 'obligation_id': obligation, 'status': status,
            'formula': 'add_fp32(a, b) = add_fp32(b, a)',
            'profile': {'input': 'fp32', 'compute': 'fp32', 'output': 'fp32'},
            'domain': 'Finite actual operands.',
        }

    def test_experiment_tasks_preserve_all_four_evidence_categories(self):
        self.add_obligation('reported')
        self.add_obligation('pending', status='binding_validation_pending')
        self.add_obligation('identity', kind='explicit_numeric_law',
                            status='counterexample_to_unrestricted_form')
        self.add_obligation('shared', kind='shared_model_requirement',
                            status='shared_model_contract')
        self.review['experiment_tasks'] = [
            self.experiment_task('existing', 'reported', 'existing_report_optional_replay'),
            self.experiment_task('typed', 'pending', 'pending_exact_binding'),
            self.experiment_task('model-fix', 'identity', 'not_an_experiment'),
            self.experiment_task('rounding', 'shared', 'shared_model_check'),
        ]
        self.assertEqual(self.first_status(), 'requires_numeric_model_revision')

    def test_experiment_manifest_must_cover_every_obligation(self):
        self.add_obligation()
        self.review['experiment_tasks'] = []
        with self.assertRaises(ValueError):
            self.validate()

    def test_counterexample_cannot_be_presented_as_pending_experiment(self):
        self.add_obligation('identity', kind='explicit_numeric_law',
                            status='counterexample_to_unrestricted_form')
        self.review['experiment_tasks'] = [
            self.experiment_task(obligation='identity', status='pending_exact_binding')]
        with self.assertRaises(ValueError):
            self.validate()

    def test_reported_and_shared_tasks_cannot_be_presented_as_other_categories(self):
        for kind, evidence_status, task_status in (
                ('reported_atom', 'reported_evidence_available', 'pending_exact_binding'),
                ('shared_model_requirement', 'shared_model_contract',
                 'existing_report_optional_replay')):
            with self.subTest(kind=kind):
                self.review['obligations'] = []
                for row in self.review['declarations']:
                    row['extra_numeric_ids'] = []
                    row['shared_model_ids'] = []
                self.add_obligation(kind=kind, status=evidence_status)
                self.review['experiment_tasks'] = [self.experiment_task(status=task_status)]
                with self.assertRaises(ValueError):
                    self.validate()

    def test_pending_obligation_needs_a_pending_configuration_not_only_old_report(self):
        self.add_obligation(status='binding_validation_pending')
        self.review['experiment_tasks'] = [self.experiment_task()]
        with self.assertRaises(ValueError):
            self.validate()
        self.review['experiment_tasks'].append(
            self.experiment_task('typed-fp32', status='pending_exact_binding'))
        self.assertEqual(self.first_status(), 'awaiting_numeric_validation')

    def test_experiment_tasks_need_exact_formula_profile_and_domain(self):
        self.add_obligation()
        task = self.experiment_task()
        self.review['experiment_tasks'] = [task]
        for field in ('formula', 'profile', 'domain'):
            with self.subTest(field=field):
                original = task.pop(field)
                with self.assertRaises(ValueError):
                    self.validate()
                task[field] = ''
                with self.assertRaises(ValueError):
                    self.validate()
                task[field] = original

    def test_duplicate_or_unbound_experiment_task_is_rejected(self):
        self.add_obligation()
        task = self.experiment_task()
        self.review['experiment_tasks'] = [task, copy.deepcopy(task)]
        with self.assertRaisesRegex(ValueError, 'Duplicate or unbound experiment task'):
            self.validate()
        self.review['experiment_tasks'] = [self.experiment_task(obligation='unknown')]
        with self.assertRaisesRegex(ValueError, 'Duplicate or unbound experiment task'):
            self.validate()

    def test_unknown_experiment_status_is_rejected(self):
        self.add_obligation()
        self.review['experiment_tasks'] = [self.experiment_task(status='ACCEPT')]
        with self.assertRaises(ValueError):
            self.validate()

    def test_model_review_is_required_even_when_extra_law_review_is_complete(self):
        del self.review['declarations'][0]['fp_model_review']
        with self.assertRaisesRegex(ValueError, 'FP model review'):
            self.validate()

    def test_unknown_or_missing_model_kind_is_rejected(self):
        model = self.review['declarations'][0]['fp_model_review']
        for value in (None, '', 'native_function_correct', 'mathematical'):
            with self.subTest(kind=value):
                model['kind'] = value
                with self.assertRaisesRegex(ValueError, 'FP model review'):
                    self.validate()
        del model['kind']
        with self.assertRaisesRegex(ValueError, 'FP model review'):
            self.validate()

    def test_model_review_requires_explicit_basis(self):
        model = self.review['declarations'][0]['fp_model_review']
        model['review_basis'] = ''
        with self.assertRaisesRegex(ValueError, 'FP model review'):
            self.validate()
        del model['review_basis']
        with self.assertRaisesRegex(ValueError, 'FP model review'):
            self.validate()

    def test_model_review_requires_nonempty_structured_bound_evidence(self):
        model = self.review['declarations'][0]['fp_model_review']
        for evidence in (None, [], {}, self.file, [self.file], [{}],
                         [{'path': 'unbound/Structural.lean'}]):
            with self.subTest(evidence=evidence):
                model['evidence'] = evidence
                with self.assertRaisesRegex(ValueError, 'FP model review evidence'):
                    self.validate()
        del model['evidence']
        with self.assertRaisesRegex(ValueError, 'FP model review evidence'):
            self.validate()

    def test_separately_reviewed_model_source_must_be_bound_and_current(self):
        support = 'VeriTile/Float/SyntheticModel.lean'
        path = self.root / support
        path.parent.mkdir(parents=True)
        path.write_text('-- Synthetic model evidence, not an arithmetic implementation.\n')
        model = self.review['declarations'][0]['fp_model_review']
        model['evidence'] = [{'path': support}]
        with self.assertRaisesRegex(ValueError, 'Unbound FP model review evidence'):
            self.validate()
        self.review['input_sha256'][support] = numerical.digest(path.read_bytes())
        self.validate()
        path.write_text('-- Changed synthetic evidence.\n')
        with self.assertRaisesRegex(ValueError, 'Numerical review is stale'):
            self.validate()

    def test_admitted_fp_label_requires_review_of_its_relative_equivalence(self):
        for kind in ('mathematical_contract', 'rounding_model_contract',
                     'bit_interpreter_contract'):
            with self.subTest(kind=kind):
                self.set_model(kind, label='admitted_fp')
                with self.assertRaisesRegex(ValueError, 'relative equivalence'):
                    self.validate()
        self.set_model('backend_equivalence', label='admitted_fp')
        self.validate()

    def test_mathematical_contract_with_no_extra_law_still_lacks_fp_proof(self):
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'no_extra_numeric_law')
        self.assertEqual(task['fp_adequacy_status'], 'math_contract_only')
        self.assertEqual(task['claim_subjects'], ['mathematical_value_or_io'])
        self.assertEqual(task['backend_equivalence_declarations'], [])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')

    def test_reports_and_pending_laws_cannot_supply_a_missing_fp_model(self):
        for kind, adequacy in (('mathematical_contract', 'math_contract_only'),
                               ('rounding_model_contract', 'rounding_contract_only')):
            for status in ('reported_evidence_available', 'binding_validation_pending'):
                with self.subTest(kind=kind, evidence=status):
                    self.review['obligations'] = []
                    self.review['declarations'][0]['extra_numeric_ids'] = []
                    self.set_model(kind)
                    self.add_obligation(status=status)
                    reviewed, _ = self.validate()
                    task = self.task(reviewed)
                    self.assertEqual(task['fp_adequacy_status'], adequacy)
                    self.assertEqual(task['backend_equivalence_declarations'], [])
                    self.assertEqual(task['native_functional_correctness'],
                                     'not_established_by_this_review')

    def test_abstract_rounding_label_does_not_override_explicit_model_review(self):
        for kind, adequacy in (
                ('mathematical_contract', 'math_contract_only'),
                ('rounding_model_contract', 'rounding_contract_only'),
                ('backend_equivalence', 'backend_equivalence_no_extra_law'),
                ('bit_interpreter_contract', 'bit_interpreter_contract_only')):
            with self.subTest(kind=kind):
                self.set_model(kind, label='abstract_rounding')
                reviewed, _ = self.validate()
                self.assertEqual(self.task(reviewed)['fp_adequacy_status'], adequacy)

    def test_mathematical_label_alone_cannot_hide_reviewed_backend_equivalence(self):
        self.set_model('backend_equivalence', label='mathematical')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_no_extra_law')
        self.assertIn('candidate_vs_original', task['claim_subjects'])

    def test_empty_rules_backend_equivalence_is_relative_not_native_correctness(self):
        self.set_model('backend_equivalence', label='admitted_fp')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_no_extra_law')
        self.assertIn('candidate_vs_original', task['claim_subjects'])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')
        self.assertFalse(task['hardware_or_independent_replay_accepted'])

    def test_reported_backend_equivalence_is_relative_not_native_correctness(self):
        self.set_model('backend_equivalence', label='admitted_fp')
        self.add_obligation()
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_existing_reports')
        self.assertIn('candidate_vs_original', task['claim_subjects'])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')
        self.assertFalse(task['hardware_or_independent_replay_accepted'])

    def test_native_law_under_abstract_label_is_still_a_conditional_equivalence(self):
        self.set_model('backend_equivalence', label='abstract_rounding')
        self.add_obligation('native_commute', kind='explicit_numeric_law',
                            status='binding_validation_pending')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_needs_validation')
        self.assertIn('scope_accepted', task['scope_policy'])
        self.assertEqual(task['extra_numeric_ids'], ['native_commute'])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')

    def test_cast_store_rounding_contract_is_not_fp_arithmetic_equivalence(self):
        self.set_model('rounding_model_contract', label='abstract_rounding')
        self.add_obligation('idempotence', kind='shared_model_requirement',
                            status='shared_model_contract')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'no_extra_numeric_law')
        self.assertEqual(task['fp_adequacy_status'], 'rounding_contract_only')
        self.assertEqual(task['backend_equivalence_declarations'], [])
        self.assertEqual(task['shared_model_ids'], ['idempotence'])

    def test_mixed_models_keep_subjects_and_identify_exact_backend_declarations(self):
        self.set_model('rounding_model_contract', row=0, label='abstract_rounding')
        self.set_model('backend_equivalence', row=1, label='admitted_fp')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_no_extra_law')
        self.assertEqual(task['fp_model_kinds'], ['backend_equivalence', 'rounding_model_contract'])
        self.assertEqual(task['claim_subjects'], ['candidate_vs_original', 'rounding_model_value_or_io'])
        self.assertEqual(task['backend_equivalence_declarations'], [
            {'file': self.file, 'declaration': 'second'}])
        self.assertIn('omitted producers, profiles and stages are not thereby proved',
                      task['scope_note'])

    def test_mixed_non_backend_models_preserve_the_strongest_available_contract(self):
        self.set_model('mathematical_contract', row=0)
        self.set_model('rounding_model_contract', row=1)
        reviewed, _ = self.validate()
        self.assertEqual(self.task(reviewed)['fp_adequacy_status'], 'rounding_contract_only')
        self.set_model('bit_interpreter_contract', row=0)
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'bit_interpreter_contract_only')
        self.assertEqual(task['claim_subjects'], ['bit_interpreter_value_or_io',
                                                'rounding_model_value_or_io'])

    def test_pending_mathematical_companion_is_not_a_backend_premise(self):
        self.set_model('mathematical_contract', row=0)
        self.set_model('backend_equivalence', row=1, label='admitted_fp')
        self.add_obligation('math_pending', status='binding_validation_pending', row=0)
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'awaiting_numeric_validation')
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_no_extra_law')
        self.assertEqual(task['extra_numeric_ids'], ['math_pending'])
        self.add_obligation('backend_reported', row=1)
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['numerical_status'], 'awaiting_numeric_validation')
        self.assertEqual(task['fp_adequacy_status'], 'backend_equivalence_existing_reports')

    def test_counterexample_overrides_even_an_available_backend_equivalence(self):
        self.set_model('backend_equivalence', row=0, label='admitted_fp')
        self.set_model('rounding_model_contract', row=1, label='abstract_rounding')
        self.add_obligation('round_identity', kind='explicit_numeric_law', row=1,
                            status='counterexample_to_unrestricted_form')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'requires_numeric_model_revision')
        self.assertEqual(task['backend_equivalence_declarations'], [
            {'file': self.file, 'declaration': 'first'}])

    def test_bit_interpreter_contract_does_not_claim_hardware_correspondence(self):
        self.set_model('bit_interpreter_contract')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'bit_interpreter_contract_only')
        self.assertIn('bit_interpreter_value_or_io', task['claim_subjects'])
        self.assertEqual(task['backend_equivalence_declarations'], [])
        self.assertFalse(task['hardware_or_independent_replay_accepted'])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')

    def test_fixed_bit_example_cannot_promote_a_parametric_backend_contract(self):
        self.set_model('backend_value_contract', row=0, label='backend_fp')
        self.set_model('bit_interpreter_contract', row=1, label='bit_interpreter')
        self.review['declarations'][1]['fp_model_review']['role'] = 'fixed_nonempty_witness'
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_value_contract_only')
        self.assertEqual(task['parametric_fp_model_kinds'], ['backend_value_contract'])
        self.assertEqual(task['fixed_witness_declarations'], [
            {'file': self.file, 'declaration': 'second'}])

    def test_unknown_contract_role_is_rejected(self):
        self.review['declarations'][0]['fp_model_review']['role'] = 'native_proof'
        with self.assertRaisesRegex(ValueError, 'Invalid FP contract role'):
            self.validate()

    def test_backend_value_contract_is_neither_real_arithmetic_nor_relative_equivalence(self):
        self.set_model('backend_value_contract', label='backend_fp')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'backend_value_contract_only')
        self.assertIn('backend_per_operation_value_or_io', task['claim_subjects'])
        self.assertEqual(task['backend_equivalence_declarations'], [])
        self.assertEqual(task['backend_value_declarations'], [
            {'file': self.file, 'declaration': 'first'}])
        self.assertFalse(task['hardware_or_independent_replay_accepted'])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')

    def test_typed_value_labels_must_match_the_reviewed_model(self):
        for label, kind in (('backend_fp', 'backend_value_contract'),
                            ('bit_interpreter', 'bit_interpreter_contract')):
            with self.subTest(label=label):
                self.set_model('mathematical_contract', label=label)
                with self.assertRaisesRegex(ValueError, 'conflicts with reviewed subject'):
                    self.validate()
                self.set_model(kind, label=label)
                self.validate()

    def test_remaining_accepted_stages_are_separate_from_model_presence(self):
        self.set_model('bit_interpreter_contract', label='bit_interpreter')
        model = self.review['declarations'][0]['fp_model_review']
        model.update(accepted_scope_coverage='partial', scope_basis='Second accepted stage is still missing.')
        reviewed, _ = self.validate()
        task = self.task(reviewed)
        self.assertEqual(task['fp_adequacy_status'], 'bit_interpreter_contract_only')
        self.assertEqual(task['accepted_scope_fp_coverage'], 'partial')
        model['accepted_scope_coverage'] = 'unreviewed_success'
        with self.assertRaisesRegex(ValueError, 'Invalid accepted-scope'):
            self.validate()

    def test_missing_source_has_no_fp_adequacy_or_claim_subject(self):
        task = self.task({}, source=False, declarations=[])
        self.assertEqual(task['fp_adequacy_status'], 'missing_source')
        self.assertEqual(task['fp_model_kinds'], [])
        self.assertEqual(task['claim_subjects'], [])
        self.assertEqual(task['native_functional_correctness'],
                         'not_established_by_this_review')


if __name__ == '__main__':
    unittest.main()
