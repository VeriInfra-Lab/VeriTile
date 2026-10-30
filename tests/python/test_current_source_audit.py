"""Current-layout source gates; subprocess stubs are never proof evidence."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'bench'))
from tests.python import test_current_source_adapters as fixtures
from scripts.bench.adapter_common import declarations, hash_file, write_json
from scripts.bench.current_source_audit import command_gate, declaration_result, run, structural_checks


TYPED_RESULT = ('∃ final, FP.Structural.exec (backend rest) (source p) initial = some final ∧\n'
                '  final.mem = (written p initial).mem ∧ frame initial final')
TRACE_ALIAS = '''def ValueContract (source : List Command) (grid : Nat) (c : Config)
    (b : Backend V) (schedule : List Nat) (mem : Memory V) : Prop :=
  let events := controlTrace source c grid schedule mem.locks
  let reference := replay c events (referenceInitial mem.locks)
  ∃ final, run c b grid schedule (initial source b mem) = (events, final) ∧
    (∀ address, final.mem.data address = denote b mem.data (reference.data address)) ∧
    final.mem.locks = reference.locks ∧
    (∀ p name lane, final.regs p name lane = denote b mem.data (reference.regs p name lane))
'''


class CurrentSourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AdapterTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.catalog = self.fixture.root, self.fixture.catalog
        self.fixture.put('bench/tritonbench_g/proof_blockers.md', '')
        self.fixture.put('bench/tritonbench_g/completion_audit.md', '')
        self.replace(self.fixture.public,
            'import bench.tritonbench_g.t.Proofs.Source\n'
            'specification values : ComputeCorrect.Realizes typed reference := by exact proof\n')
        self.catalog['tasks'][0]['declarations'][0]['statement_sha256'] = \
            declarations(self.root / self.fixture.public)['values']['statement_sha256']

    def replace(self, path, text):
        self.fixture.put(path, text)
        digest = hash_file(self.root / path)
        self.catalog['build_input_sha256'][path] = digest
        for entry in self.catalog['tasks'][0]['files']:
            if entry['path'] == path:
                entry['observed_sha256'] = digest
                entry['review_expected_sha256'] = digest

    def reviewed_public(self, result=TYPED_RESULT, frontend='shared_triton_dsl',
                        prefix='', binders='', proof='by exact proof'):
        self.replace(self.fixture.public,
            'import bench.tritonbench_g.t.Proofs.Source\n' + prefix +
            f'specification values {binders} :\n{result} := {proof}\n')
        task = self.catalog['tasks'][0]
        claim = task['declarations'][0]
        claim.update(kind='specification', role='parametric_value_contract',
            model_kind='backend_value_contract', scope_relation='full_declared_stage',
            covers_stages=['full_kernel'], source_variant='original_source_translation',
            review_basis='Synthetic explicitly bound review; not a Lean certificate.',
            statement_sha256=declarations(self.root / self.fixture.public)['values']['statement_sha256'])
        for entry in task['files']:
            entry['review_authority'] = 'synthetic_explicit_review'
            entry['review_binding_matches'] = True
            entry['review_expected_sha256'] = entry['observed_sha256']
        self.fixture.view.update(frontend=frontend, authority='synthetic_explicit_review',
            translation_trust_boundary='Synthetic source binding; not native refinement.')
        if frontend == 'task_local_exact_ast':
            self.replace(self.fixture.source, 'def typed := [1, 2, 3]\n')
            self.fixture.view.update(ordinary_shared_dsl_admission=False,
                memory_semantics='sequentially_consistent_schedule')
        else:
            self.replace(self.fixture.source, 'def typed := triton {}\n')
            self.fixture.view.update(ordinary_shared_dsl_admission=True,
                memory_semantics='single_program_total_typed_memory')
        return task, claim

    def assert_missing_surface(self):
        result = structural_checks(self.root, self.catalog)
        self.assertEqual('failed', result['status'], result)
        self.assertTrue(any('missing named correctness surface' in e for e in result['errors']), result)
        self.assertEqual([], result['reviewed_named_surfaces'])

    def assert_surface(self, family):
        result = structural_checks(self.root, self.catalog)
        self.assertEqual('passed', result['status'], result)
        self.assertEqual([family], [s['surface'] for s in result['reviewed_named_surfaces']])
        return result['reviewed_named_surfaces'][0]

    def test_bound_explicit_layout_passes_structural_rules_only(self):
        result = structural_checks(self.root, self.catalog)
        self.assertEqual('passed', result['status'], result)
        self.assertEqual(3, result['checked_task_files'])
        self.assertIn('no compilation', result['boundary'])

    def test_review_hash_is_independent_of_observed_hash(self):
        self.catalog['tasks'][0]['files'][1]['review_expected_sha256'] = None
        result = structural_checks(self.root, self.catalog)
        self.assertEqual('failed', result['status'])
        self.assertTrue(any('unreviewed' in e for e in result['errors']))

    def test_placeholder_in_helper_is_not_skipped_by_public_file_selection(self):
        self.replace(self.fixture.helper, 'theorem hidden : False := by sorry\n')
        result = structural_checks(self.root, self.catalog)
        self.assertEqual('failed', result['status'])
        self.assertTrue(any('placeholder proof' in e for e in result['errors']))

    def test_halg_is_still_a_structural_blocker(self):
        self.replace(self.fixture.helper, 'def helper (hAlg : False) := hAlg\n')
        result = structural_checks(self.root, self.catalog)
        self.assertTrue(any('hAlg blocker' in e for e in result['errors']))

    def test_named_surface_is_not_supplied_by_comment(self):
        self.replace(self.fixture.public,
            'import bench.tritonbench_g.t.Proofs.Source\n'
            '-- ComputeCorrect.Realizes\nspecification values : P := by exact proof\n')
        self.catalog['tasks'][0]['declarations'][0]['statement_sha256'] = \
            declarations(self.root / self.fixture.public)['values']['statement_sha256']
        result = structural_checks(self.root, self.catalog)
        self.assertTrue(any('missing named correctness surface' in e for e in result['errors']))

    def test_reviewed_typed_execution_and_memory_result_is_a_named_surface(self):
        self.reviewed_public()
        witness = self.assert_surface('typed_execution_memory')
        self.assertEqual('t', witness['task'])  # No task-name exception list.
        self.assertEqual('original_source_translation', witness['source_variant'])

    def test_candidate_surface_retains_candidate_variant(self):
        task, claim = self.reviewed_public()
        claim['source_variant'] = 'compile_repaired_candidate'
        self.fixture.view.update(variant='compile_repaired_candidate',
            original_python='original.py', candidate_python='candidate.py')
        task['original_source_present'] = False
        self.assertEqual('compile_repaired_candidate', self.assert_surface('typed_execution_memory')['source_variant'])
        self.assertFalse(task['original_source_present'])

    def test_typed_surface_is_not_supplied_by_helper_comment_string_or_proof(self):
        for location in ('helper', 'comment', 'string', 'proof', 'hypothesis'):
            with self.subTest(location=location):
                prefix, binders, proof = '', '', 'by exact proof'
                self.replace(self.fixture.helper, 'def helper := 1\n')
                if location == 'helper':
                    self.replace(self.fixture.helper, 'theorem helper : ' + TYPED_RESULT + ' := by exact proof\n')
                elif location == 'comment':
                    prefix = '/- ' + TYPED_RESULT + ' -/\n'
                elif location == 'string':
                    prefix = 'def prose := "' + TYPED_RESULT.replace('\n', ' ') + '"\n'
                elif location == 'proof':
                    proof = 'by\n  have h : ' + TYPED_RESULT + ' := someProof\n  exact proof'
                else:
                    binders = '(assumed : ' + TYPED_RESULT + ')'
                self.reviewed_public('P', prefix=prefix, binders=binders, proof=proof)
                self.assert_missing_surface()

    def test_typed_execution_without_memory_result_or_only_as_antecedent_is_rejected(self):
        for result in ('∃ final, FP.Structural.exec b source initial = some final',
                       '(∃ final, FP.Structural.exec b source initial = some final ∧ final.mem = expected) → P',
                       '∃ final, FP.Structural.exec b source initial = some final → final.mem = expected',
                       TYPED_RESULT + ' → P', TYPED_RESULT + ' ∨ P',
                       TYPED_RESULT.replace('FP.Structural.exec', 'FP.Structural.exec.unrelated')):
            with self.subTest(result=result):
                self.reviewed_public(result)
                self.assert_missing_surface()

    def test_reviewed_surface_requires_value_role_full_stage_and_source_variant(self):
        changes = [{'role': 'memory_frame_contract', 'model_kind': 'memory_frame_contract'},
                   {'scope_relation': 'strict_specialization'}, {'covers_stages': []},
                   {'review_basis': ''}, {'source_variant': 'standalone_sc_model'}]
        for change in changes:
            with self.subTest(change=change):
                _, claim = self.reviewed_public()
                claim.update(change)
                self.assert_missing_surface()

    def test_reviewed_surface_requires_public_and_selected_source_review_bindings(self):
        for path in (self.fixture.public, self.fixture.source):
            for field, invalid in [('review_authority', None), ('review_binding_matches', False),
                                   ('review_expected_sha256', None)]:
                with self.subTest(path=path, field=field):
                    task, _ = self.reviewed_public()
                    for entry in task['files']:
                        entry['review_binding_matches'] = True
                    next(f for f in task['files'] if f['path'] == path)[field] = invalid
                    self.assert_missing_surface()

    def test_compiled_sc_memory_result_and_trace_result_are_named_surfaces(self):
        results = [('compileForward typed b c mem = forwardMemoryReference b c mem', 'sc_compiled_memory'),
                   ('compileFinal typed b c mem = finalMemoryReference b c mem', 'sc_compiled_memory'),
                   ('let events := controlTrace plan c schedule mem.locks\n'
                    '∃ final, compileAndRun typed b c schedule mem = some (events, final) ∧\n'
                    'final.mem = (replay b events (Reference.initial mem)).mem ∧ frame final',
                    'sc_compiled_trace_value'),
                   ('let rows := backwardReference b c inputs\n'
                    'let events := controlTrace c schedule words\n'
                    '∃ finalState, compileBackward typed b c inputs mem words schedule = some (events, finalState) ∧\n'
                    'Related b rows finalState.values (replay b c rows events (Reference.initial mem)) ∧ frame finalState',
                    'sc_compiled_trace_value')]
        for result, family in results:
            with self.subTest(family=family, result=result):
                self.reviewed_public(result, 'task_local_exact_ast')
                self.assert_surface(family)

    def test_sc_surface_requires_selected_source_frontend_and_independent_trace(self):
        bad_results = ['compileForward unselected b c mem = forwardMemoryReference b c mem',
                       'compileForward typed b c mem = compileForward typed b c mem',
                       'let events := arbitraryEvents\n'
                       '∃ final, compileAndRun typed b c schedule mem = some (events, final) ∧\n'
                       'final.mem = (replay b events (Reference.initial mem)).mem ∧ frame final',
                       'let events := controlTrace plan c schedule mem.locks\n'
                       '∃ final, compileAndRun typed b c schedule mem = some (events, final) ∧ P']
        for result in bad_results:
            with self.subTest(result=result):
                self.reviewed_public(result, 'task_local_exact_ast')
                self.assert_missing_surface()
        self.reviewed_public('compileForward typed b c mem = forwardMemoryReference b c mem')
        self.fixture.view['memory_semantics'] = 'single_program_total_typed_memory'
        self.assert_missing_surface()

    def test_direct_public_sc_trace_value_definition_is_checked(self):
        self.reviewed_public('ValueContract typed.first c.firstGrid c b schedule mem',
                             'task_local_exact_ast', prefix=TRACE_ALIAS)
        witness = self.assert_surface('sc_trace_value_definition')
        self.assertEqual('ValueContract', witness['referenced_definition'])

    def test_alias_name_or_unreferenced_helper_does_not_supply_sc_surface(self):
        prefixes = ['', 'def ValueContract (source : List Command) : Prop := P\n',
                    TRACE_ALIAS.replace('reference.data', 'unrelated.data'),
                    TRACE_ALIAS.replace('final.mem.locks = reference.locks', 'P'),
                    TRACE_ALIAS.replace('reference.regs', 'unrelated.regs')]
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                self.reviewed_public('ValueContract typed.first c.firstGrid c b schedule mem',
                                     'task_local_exact_ast', prefix=prefix)
                self.replace(self.fixture.helper, TRACE_ALIAS)
                self.assert_missing_surface()
        self.reviewed_public('ArbitraryBackendPredicate b c mem', 'task_local_exact_ast', prefix=TRACE_ALIAS)
        self.assert_missing_surface()
        self.reviewed_public('ValueContract typed.first c.firstGrid c b schedule mem → P',
                             'task_local_exact_ast', prefix=TRACE_ALIAS)
        self.assert_missing_surface()

    def test_protocol_surface_is_recognized_only_as_protocol(self):
        result = ('∃ events finalState, compileBackward typed b c inputs mem words schedule = '
                  'some (events, finalState) ∧ ProtocolInvariant c finalState.control')
        _, claim = self.reviewed_public(result, 'task_local_exact_ast')
        claim.update(role='protocol_contract', model_kind='protocol_contract',
                     scope_relation='not_an_execution_contract', covers_stages=[])
        self.assertEqual('protocol_contract', self.assert_surface('sc_source_protocol')['role'])
        claim.update(role='parametric_value_contract', model_kind='backend_value_contract',
                     scope_relation='full_declared_stage', covers_stages=['full_kernel'])
        self.assert_missing_surface()

    def test_result_parser_ignores_binder_colons_named_arguments_and_proof(self):
        text = ('specification values (p : Nat) (h : Hidden.exec p = expected) :\n'
                'Result (n := p) := by exact proof')
        self.assertEqual('Result (n := p)', declaration_result(text))

    def test_stale_blocker_registries_do_not_pass(self):
        self.fixture.put('bench/tritonbench_g/proof_blockers.md',
                         '## Correctness-Surface Blockers\n\n| `t` | historical |\n')
        result = structural_checks(self.root, self.catalog)
        self.assertTrue(any('stale correctness-surface registry' in e for e in result['errors']))

    def test_supplementary_surface_does_not_erase_historical_surface_blocker(self):
        self.reviewed_public(prefix='-- Correctness-surface blocker: selected real-model gap remains\n')
        self.fixture.put('bench/tritonbench_g/proof_blockers.md',
                         '## Correctness-Surface Blockers\n\n| `t` | selected real-model gap |\n')
        self.assert_surface('typed_execution_memory')

    def test_supplementary_surface_does_not_bypass_empty_or_unregistered_blocker(self):
        for marker, registered in [('', True), ('real-model gap', False)]:
            with self.subTest(marker=marker, registered=registered):
                self.reviewed_public(prefix='-- Correctness-surface blocker: ' + marker + '\n')
                self.fixture.put('bench/tritonbench_g/proof_blockers.md',
                    '## Correctness-Surface Blockers\n\n| `t` | gap |\n' if registered else '')
                result = structural_checks(self.root, self.catalog)
                self.assertEqual('failed', result['status'], result)
                self.assertTrue(any('registered blocker' in e for e in result['errors']), result)

    def test_missing_bound_python_is_not_silently_accepted(self):
        self.catalog['python_sources'] = {'bench/tritonbench_g/t/absent.py': {'sha256': 'a' * 64}}
        self.catalog['missing_bound_python_sources'] = ['bench/tritonbench_g/t/absent.py']
        result = structural_checks(self.root, self.catalog)
        self.assertTrue(any('missing bound Python' in e for e in result['errors']))

    def test_real_subprocess_failure_is_retained(self):
        output = self.root / 'gate-output'; output.mkdir()
        result = command_gate(self.root, output, 'fixture',
                              [sys.executable, '-c', 'print("actual failing fixture"); raise SystemExit(7)'])
        self.assertEqual('failed', result['status'])
        self.assertEqual(7, result['exit_code'])
        self.assertIn('actual failing fixture', (output / result['log']).read_text())

    def args(self, suffix):
        catalog = self.root / 'catalog.json'; write_json(catalog, self.catalog)
        return SimpleNamespace(root=self.root, catalog=catalog, output=self.root / suffix,
            baseline_package=self.root / 'package', source_map=self.root / 'map.json',
            legacy_inventory=self.root / 'legacy.json', evidence_root=self.root / 'evidence',
            review_manifest=self.root / 'review.json', scope_ledger=self.root / 'scope.json',
            supplement=None)

    def test_each_mandatory_command_failure_fails_global_result(self):
        gates = ['baseline', 'catalog_reproduction', 'scope_review', 'legacy_source',
                 'value_metadata', 'deferred_numerics', 'comparator_preflight', 'library_replay']
        for failed in gates:
            with self.subTest(gate=failed):
                calls = []
                def fixture_gate(root, output, name, command):
                    calls.append(name)
                    return {'status': 'failed' if name == failed else 'passed',
                            'exit_code': 9 if name == failed else 0}
                with patch('scripts.bench.current_source_audit.command_gate', side_effect=fixture_gate):
                    result = run(self.args(failed))
                self.assertEqual('failed', result['status'])
                self.assertEqual(gates, calls)
                self.assertEqual('not_run_by_global_runner', result['corpus_replay'])
                self.assertEqual('not_run_by_global_runner', result['integration_build'])
                self.assertEqual('not_established_by_global_gates_alone', result['whole_task_acceptance'])

    def test_global_postcheck_rejects_source_mutation(self):
        def fixture_gate(root, output, name, command):
            if name == 'library_replay':
                (self.root / self.fixture.helper).write_text('-- modified during worker\n')
            return {'status': 'passed', 'exit_code': 0}
        with patch('scripts.bench.current_source_audit.command_gate', side_effect=fixture_gate):
            result = run(self.args('changed-source'))
        self.assertEqual('failed', result['status'])
        self.assertEqual('failed', result['gates']['unchanged_inputs']['status'])

    def test_missing_rg_cannot_masquerade_as_no_source_matches(self):
        with patch('scripts.bench.current_source_audit.command_gate', return_value={'status': 'passed', 'exit_code': 0}), \
                patch('scripts.bench.current_source_audit.shutil.which', side_effect=lambda name: None if name == 'rg' else '/bin/' + name):
            result = run(self.args('missing-rg'))
        self.assertEqual('failed', result['status'])
        self.assertEqual(['rg'], result['gates']['source_tools']['missing'])

    def test_global_receipts_never_overwrite_previous_run(self):
        args = self.args('existing'); args.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh directory'):
            run(args)


if __name__ == '__main__':
    unittest.main()
