"""Synthetic filesystem tests only: fixture bytes are never Lean certificates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=='tests' else Path(__file__).resolve().parent))
from copy import deepcopy
import tempfile
import unittest
from scripts.bench.adapter_common import *
from scripts.bench.current_source_catalog import (validate_catalog, verify_source_view,
    validate_supplement, review_manifest, graph_stage_coverage, public_files, proof_files,
    trust_targets, validate_metadata_corrections)
from scripts.bench.current_acceptance import (make_build_receipt, verify_build, acceptance,
    REQUIRED_EXTERNAL_GATES, validate_gate, run_gate)

class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        self.root.mkdir()
        self.objects = Path(self.temp.name) / 'objects'
        self.objects.mkdir()
        (self.objects/'Init.olean').write_bytes(b'synthetic implicit Init')
        self.lean_binary=Path(self.temp.name)/'lean-fixture'
        self.lean_binary.write_bytes(b'synthetic compiler fixture; not executable')
        self.mods = ['bench.tritonbench_g.t.Proofs.Source',
                     'bench.tritonbench_g.t.FPCorrect',
                     'bench.tritonbench_g.t.Proofs.Helper']
        self.source = BASE + 't/Proofs/Source.lean'
        self.public = BASE + 't/FPCorrect.lean'
        self.helper = BASE + 't/Proofs/Helper.lean'
        self.put(self.source, 'def typed := triton {}\n')
        self.put(self.public, 'import bench.tritonbench_g.t.Proofs.Source\nspecification values : True := by trivial\n')
        self.put(self.helper, '-- helper with no source graph\ndef helper := 1\n')
        self.put('lakefile.toml', '\n'.join(
            '[[lean_lib]]\nname = "' + n + '"\nglobs = ["bench.tritonbench_g.t.+"]\n'
            for n in LIBRARIES))
        self.put('lean-toolchain', 'leanprover/lean4:v4.29.0\n')
        self.put('lake-manifest.json', '{}\n')
        regs = registered(self.root)
        binds = {p: hash_file(self.root / p) for p in [self.source, self.public, self.helper,
            'lakefile.toml', 'lean-toolchain', 'lake-manifest.json']}
        self.view = {'id': 'original', 'variant': 'original_source_translation',
            'frontend': 'shared_triton_dsl', 'memory_semantics': 'single_program_total_typed_memory',
            'selected_definitions': {self.source: ['typed']}}
        self.catalog = {'schema': 1, 'kind': 'current_source_catalog_not_completion_ledger',
            'expected_task_names': ['t'], 'registered_modules': regs,
            'build_input_sha256': binds, 'unregistered_task_modules': [], 'tasks': [{
                'task': 't', 'historical_source_entry': None, 'source_views': [self.view],
                'original_source_present': True,
                'declarations': [{'file': self.public, 'declaration': 'values',
                    'statement_sha256': declarations(self.root/self.public)['values']['statement_sha256']}],
                'files': [{'path': p, 'module': module_name(p), 'roles': roles,
                    'observed_sha256': binds[p], 'review_expected_sha256': binds[p],
                    'review_binding_matches': True, 'libraries': regs[module_name(p)]}
                    for p,roles in [(self.public,['public_contract']), (self.helper,['helper']),
                                    (self.source,['selected_source'])]]}]}
        self.records = []
        for module in self.mods:
            path = Path(*module.split('.')).with_suffix('.lean').as_posix()
            obj = self.objects / Path(*module.split('.')).with_suffix('.olean')
            obj.parent.mkdir(parents=True, exist_ok=True)
            obj.write_bytes(('synthetic object, no compiler claim: '+module).encode())
        for module in self.mods:
            path = Path(*module.split('.')).with_suffix('.lean').as_posix()
            log = 'logs/' + module + '.log'
            self.put(log, 'fixture succeeded\n')
            self.records.append({'module': module, 'source': path,
                'source_sha256': binds[path], 'exit_code': 0, 'log': log,
                'log_sha256': hash_file(self.root/log),
                'olean_sha256': hash_file(self.objects/Path(*module.split('.')).with_suffix('.olean')),
                'direct_import_olean_sha256': {d: hash_file(self.objects/Path(*d.split('.')).with_suffix('.olean'))
                    for d in imports((self.root/path).read_text())}})

    def put(self, path, text):
        p = self.root/path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def build(self, records=None):
        return make_build_receipt(self.root, self.catalog, self.records if records is None else records, [self.objects], lean_binary=self.lean_binary)

    def assertBadBuild(self, r):
        self.assertEqual('failed', verify_build(self.root, self.catalog, r, [self.objects],lean_binary=self.lean_binary)['status'])

    def test_explicit_fp_correct_only_layout(self):
        self.assertEqual('passed', validate_catalog(self.root, self.catalog)['catalog_integrity'])

    def test_trust_targets_include_public_source_and_every_helper(self):
        self.assertEqual(sorted([self.public, self.source, self.helper]),
                         trust_targets(self.root, self.catalog))
        self.assertEqual(trust_targets(self.root, self.catalog),
                         trust_targets(self.root, self.catalog, ['t']))

    def test_trust_targets_reject_unknown_and_duplicate_tasks(self):
        for tasks in (['absent'], ['t', 't']):
            with self.subTest(tasks=tasks), self.assertRaisesRegex(ValueError, 'trust audit task'):
                trust_targets(self.root, self.catalog, tasks)

    def test_trust_targets_fail_closed_for_uninventoried_helpers(self):
        self.put(BASE + 't/RawBitCorrect.lean', 'def omitted := 1\n')
        with self.assertRaisesRegex(ValueError, 'omits or adds'):
            trust_targets(self.root, self.catalog, ['t'])

    def test_trust_targets_fail_closed_for_changed_or_missing_source(self):
        self.put(self.helper, 'def changed := 2\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            trust_targets(self.root, self.catalog)
        (self.root / self.helper).unlink()
        with self.assertRaisesRegex(ValueError, 'omits or adds'):
            trust_targets(self.root, self.catalog)

    def correction(self):
        before = {'value_extension': {'dtype_profile': 'incorrect historical dtype text'}}
        corrections = [{'field_path': ['value_extension', 'dtype_profile'],
            'before': before['value_extension']['dtype_profile'],
            'after': 'fp32 storage; internal half casts; both tf32 branches',
            'reason': 'Actual typed source and public statement reviewed.'}]
        review = {'reviewer': 'root', 'status': 'reviewed',
                  'reviewed_metadata_corrections_sha256': {'t': sha(canonical(corrections))}}
        return before, corrections, review

    def test_metadata_correction_preserves_frozen_row(self):
        before, corrections, review = self.correction()
        frozen = deepcopy(before)
        self.assertEqual(corrections, validate_metadata_corrections('t', before, corrections, review))
        self.assertEqual(frozen, before)

    def test_metadata_correction_rejects_wrong_before(self):
        before, corrections, review = self.correction()
        before['value_extension']['dtype_profile'] = 'different frozen value'
        with self.assertRaisesRegex(ValueError, 'before value differs'):
            validate_metadata_corrections('t', before, corrections, review)

    def test_metadata_correction_rejects_unreviewed_after(self):
        before, corrections, review = self.correction()
        corrections[0]['after'] = 'claims all original host stages'
        with self.assertRaisesRegex(ValueError, 'exact root review'):
            validate_metadata_corrections('t', before, corrections, review)

    def test_metadata_correction_cannot_relabel_completion_status(self):
        before, corrections, review = self.correction()
        corrections[0]['field_path'] = ['integration_accepted']
        review['reviewed_metadata_corrections_sha256']['t'] = sha(canonical(corrections))
        with self.assertRaisesRegex(ValueError, 'unsupported/duplicate'):
            validate_metadata_corrections('t', before, corrections, review)

    def test_catalog_correction_remains_bound_to_current_review(self):
        before, corrections, review = self.correction()
        write_json(self.root / 'dtype-review.json', review)
        task = self.catalog['tasks'][0]
        task['value_registry_row'] = before
        task['reviewed_metadata_corrections'] = corrections
        task['metadata_correction_review'] = {'path': 'dtype-review.json',
                                              'sha256': hash_file(self.root / 'dtype-review.json')}
        validate_catalog(self.root, self.catalog)
        corrections[0]['after'] = 'unreviewed widened dtype domain'
        with self.assertRaisesRegex(ValueError, 'exact root review'):
            validate_catalog(self.root, self.catalog)

    def test_extra_helper_is_not_ignored(self):
        self.put(BASE+'t/Proofs/Forgotten.lean', 'def unrelated := 1\n')
        with self.assertRaisesRegex(ValueError, 'omits or adds'):
            validate_catalog(self.root, self.catalog)

    def test_helper_cannot_be_relabelled_source(self):
        self.catalog['tasks'][0]['files'][1]['roles'] = ['selected_source']
        with self.assertRaisesRegex(ValueError, 'role drift'):
            validate_catalog(self.root, self.catalog)

    def test_source_definition_must_exist(self):
        self.view['selected_definitions'][self.source] = ['absent']
        with self.assertRaisesRegex(ValueError, 'missing selected'):
            validate_catalog(self.root, self.catalog)

    def test_triton_in_comment_is_not_source(self):
        self.put(self.source, '-- triton {}\ndef typed := 1\n')
        with self.assertRaisesRegex(ValueError, 'no actual DSL'):
            verify_source_view(self.root,self.view,{self.source:hash_file(self.root/self.source)})

    def test_task_local_sc_requires_explicit_memory_boundary(self):
        self.view['frontend'] = 'task_local_exact_ast'
        with self.assertRaisesRegex(ValueError, 'task-local SC'):
            verify_source_view(self.root,self.view,{self.source:hash_file(self.root/self.source)})

    def test_task_local_sc_can_be_registered_without_shared_dsl(self):
        self.put(self.source, 'def typed := [1, 2, 3]\n')
        self.view.update(frontend='task_local_exact_ast', ordinary_shared_dsl_admission=False,
            memory_semantics='sequentially_consistent_schedule', translation_trust_boundary='external reviewed typed translator')
        verify_source_view(self.root,self.view,{self.source:hash_file(self.root/self.source)})

    def test_candidate_cannot_fill_original_presence(self):
        self.view.update(variant='compile_repaired_candidate', original_python='original.py', candidate_python='candidate.py')
        with self.assertRaisesRegex(ValueError, 'promoted to original'):
            validate_catalog(self.root,self.catalog)

    def test_candidate_remains_missing_original_in_acceptance(self):
        self.view.update(variant='compile_repaired_candidate', original_python='original.py', candidate_python='candidate.py')
        self.catalog['tasks'][0]['original_source_present'] = False
        result = acceptance(self.root,self.catalog)
        self.assertEqual(['t'],result['gates']['original_source_presence']['missing_original'])
        self.assertFalse(result['mechanical_gates_passed'])

    def test_candidate_cannot_use_same_original_path(self):
        self.view.update(variant='compile_repaired_candidate', original_python='original.py', candidate_python='original.py')
        with self.assertRaisesRegex(ValueError, 'replace original'):
            verify_source_view(self.root,self.view,{self.source:hash_file(self.root/self.source)})

    def test_unknown_task_directory_rejected(self):
        (self.root/BASE/'new_task').mkdir()
        with self.assertRaisesRegex(ValueError, 'membership'):
            validate_catalog(self.root,self.catalog)

    def test_unregistered_task_module_rejected(self):
        self.put('lakefile.toml',(self.root/'lakefile.toml').read_text().replace('bench.tritonbench_g.t.+','bench.tritonbench_g.t.FPCorrect'))
        with self.assertRaisesRegex(ValueError, 'registration changed'):
            validate_catalog(self.root,self.catalog)

    def test_changed_source_rejected(self):
        self.put(self.source,'def typed := triton { different = 2 }\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_catalog(self.root,self.catalog)

    def test_path_traversal_rejected(self):
        for name in ['../escape','a/../b','/tmp/file','a\\b','./file','a//b','']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                relative(name)

    def test_symlink_escape_rejected(self):
        outside=Path(self.temp.name)/'outside';outside.write_text('x')
        (self.root/'escape').symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'escaping'):
            within(self.root,'escape')

    def test_duplicate_json_rejected(self):
        p=self.put('bad.json','{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'duplicate JSON'):
            load(p)

    def test_build_positive_checks_all_modules(self):
        r=self.build()
        self.assertEqual('passed',r['status'])
        self.assertEqual(3,r['validation']['passed_modules'])

    def test_build_missing_record_is_not_run(self):
        r=self.build(self.records[:-1])
        self.assertBadBuild(r)
        self.assertEqual('not_run',r['records'][1]['status'])

    def test_failed_compile_cannot_use_existing_object(self):
        self.records[0]['exit_code']=1
        self.assertBadBuild(self.build())

    def test_boolean_exit_code_not_zero(self):
        self.records[0]['exit_code']=False
        self.assertBadBuild(self.build())

    def test_stale_successful_source_hash(self):
        self.records[0]['source_sha256']='0'*64
        self.assertBadBuild(self.build())

    def test_stale_successful_object_hash(self):
        self.records[0]['olean_sha256']='0'*64
        self.assertBadBuild(self.build())

    def test_changed_object_after_receipt(self):
        r=self.build()
        (self.objects/Path(*self.mods[0].split('.')).with_suffix('.olean')).write_bytes(b'changed')
        self.assertBadBuild(r)

    def test_stale_direct_dependency_from_before_compile(self):
        self.records[1]['direct_import_olean_sha256'][self.mods[0]]='0'*64
        self.assertBadBuild(self.build())

    def test_omitted_direct_dependency_rejected(self):
        self.records[1]['direct_import_olean_sha256']={}
        self.assertBadBuild(self.build())

    def test_extra_direct_dependency_rejected(self):
        self.records[1]['direct_import_olean_sha256'][self.mods[2]]=self.records[2]['olean_sha256']
        self.assertBadBuild(self.build())

    def test_fake_import_in_comment_not_a_dependency(self):
        self.assertEqual(['Real.Dependency'],imports('/- import Fake.One /- import Fake.Two -/ -/\nimport Real.Dependency\ndef x := "import Fake.Three"\n'))

    def test_multiline_import_is_not_omitted(self):
        self.assertEqual(['A.One','B.Two'],imports('import A.One\n  B.Two\nnamespace X\n'))

    def test_log_changed_after_success(self):
        r=self.build();self.put(self.records[0]['log'],'error\n')
        self.assertBadBuild(r)

    def test_failed_record_cannot_be_promoted_by_receipt_status(self):
        self.records[0]['exit_code']=1;r=self.build();r['status']='passed'
        self.assertBadBuild(r)

    def test_input_omission_rejected(self):
        r=self.build();r['input_sha256']=dict(r['input_sha256']);del r['input_sha256'][self.helper]
        with self.assertRaisesRegex(ValueError,'snapshot incomplete'):
            verify_build(self.root,self.catalog,r,[self.objects])

    def test_duplicate_build_attempts_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate module'):
            self.build(self.records+[self.records[0]])

    def test_unknown_build_schema_rejected(self):
        r=self.build();r['schema']=2
        with self.assertRaisesRegex(ValueError,'unknown integration'):
            verify_build(self.root,self.catalog,r,[self.objects])

    def test_new_object_search_shadow_rejected(self):
        r=self.build();shadow=Path(self.temp.name)/'shadow';shadow.mkdir()
        with self.assertRaisesRegex(ValueError,'root count'):
            verify_build(self.root,self.catalog,r,[shadow,self.objects])

    def test_current_hash_is_not_review_authority(self):
        self.catalog['tasks'][0]['files'][0]['review_expected_sha256']=None
        r=acceptance(self.root,self.catalog)
        self.assertEqual('failed',r['gates']['reviewed_source_bindings']['status'])

    def test_absent_gates_are_not_passed_by_metadata(self):
        r=acceptance(self.root,self.catalog,build=self.build(),object_roots=[self.objects])
        self.assertFalse(r['mechanical_gates_passed'])
        self.assertTrue(all(r['gates'][n]['status']=='not_run' for n in REQUIRED_EXTERNAL_GATES))

    def test_failed_gate_with_zero_exit_still_fails(self):
        gate={'schema':1,'gate':'value_metadata','status':'failed','exit_code':0}
        with self.assertRaisesRegex(ValueError,'failed or not run'):
            validate_gate(self.root,self.catalog,'value_metadata',gate)

    def test_stale_gate_cannot_be_reused(self):
        gate={'schema':1,'gate':'value_metadata','status':'passed','exit_code':0,
              'catalog_sha256':'0'*64,'input_sha256':self.catalog['build_input_sha256']}
        with self.assertRaisesRegex(ValueError,'different inputs'):
            validate_gate(self.root,self.catalog,'value_metadata',gate)

    def test_bound_review_hash_required(self):
        self.put('receipt.json','{}')
        self.put('review.json',__import__('json').dumps({'reviewer':'root','status':'reviewed',
            'receipt_sha256':hash_file(self.root/'receipt.json'),'final':{}}))
        manifest={'schema':1,'status':'accepted_final_source_bindings','files':{self.source:{'sha256':hash_file(self.root/self.source),
            'review':{'path':'review.json','sha256':hash_file(self.root/'review.json')},
            'receipt':{'path':'receipt.json','sha256':hash_file(self.root/'receipt.json')},
            'hash_document':'review','hash_pointer':['final']}}}
        with self.assertRaisesRegex(ValueError,'does not bind'):
            review_manifest(self.root,self.root,manifest)

    def supplement(self,kind='protocol_contract',role='protocol_contract',stages=None):
        hashes={self.source:hash_file(self.root/self.source),self.public:hash_file(self.root/self.public)}
        self.put('review.json',__import__('json').dumps({'reviewer':'root','status':'reviewed_final_source_bindings','reviewed_input_sha256':hashes}))
        return {'schema':1,'tasks':[{'task':'t','source_sha256':hashes,
            'review':{'path':'review.json','sha256':hash_file(self.root/'review.json'),'source_hashes_pointer':['reviewed_input_sha256']},
            'source_views':[self.view], 'declarations':[{'file':self.public,'declaration':'values',
                'statement_sha256':declarations(self.root/self.public)['values']['statement_sha256'],
                'source_variant':'original_source_translation','model_kind':kind,'role':role,'covers_stages':stages or []}]}]}

    def test_protocol_supplement_is_supported(self):
        r=validate_supplement(self.root,self.root,self.supplement(),{'t'})
        self.assertEqual('protocol_contract',r['t']['declarations'][0]['model_kind'])

    def test_protocol_cannot_fill_numeric_stage(self):
        with self.assertRaisesRegex(ValueError,'promoted to value'):
            validate_supplement(self.root,self.root,self.supplement(stages=['forward']),{'t'})

    def test_protocol_cannot_be_value_role(self):
        with self.assertRaisesRegex(ValueError,'promoted to value'):
            validate_supplement(self.root,self.root,self.supplement(role='parametric_value_contract'),{'t'})

    def test_supplement_unknown_task_rejected(self):
        with self.assertRaisesRegex(ValueError,'population'):
            validate_supplement(self.root,self.root,self.supplement(),{'other'})

    def test_supplement_missing_review_rejected(self):
        s=self.supplement();del s['tasks'][0]['review']
        with self.assertRaisesRegex(ValueError,'unreviewed'):
            validate_supplement(self.root,self.root,s,{'t'})

    def test_unrun_raw_result_with_zero_exit_cannot_pass(self):
        self.records[0]['status']='not_run'
        self.assertBadBuild(self.build())

    def test_failed_raw_result_with_zero_exit_cannot_pass(self):
        self.records[0]['status']='failed'
        self.assertBadBuild(self.build())

    def test_compiler_executable_drift_rejected(self):
        r=self.build();self.lean_binary.write_bytes(b'changed compiler')
        self.assertBadBuild(r)

    def test_implicit_init_drift_rejected(self):
        r=self.build();(self.objects/'Init.olean').write_bytes(b'changed Init')
        self.assertBadBuild(r)

    def test_missing_environment_does_not_pass(self):
        r=self.build();r['environment']=None
        self.assertBadBuild(r)

    def test_public_statement_omitted_rejected(self):
        self.catalog['tasks'][0]['declarations']=[]
        self.catalog['tasks'][0]['files'][0]['roles']=['helper']
        with self.assertRaisesRegex(ValueError,'public declaration'):
            validate_catalog(self.root,self.catalog)

    def test_public_statement_hash_lie_rejected(self):
        self.catalog['tasks'][0]['declarations'][0]['statement_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'public declaration'):
            validate_catalog(self.root,self.catalog)

    def test_added_python_source_is_not_silent(self):
        self.put(BASE+'t/candidate.py','x=1\n')
        with self.assertRaisesRegex(ValueError,'Python source inventory'):
            validate_catalog(self.root,self.catalog)

    def test_missing_pinned_original_python_blocks_acceptance(self):
        self.catalog['python_sources']={'frozen/original.py':{'sha256':'0'*64,'variant':'original'}}
        self.catalog['missing_bound_python_sources']=['frozen/original.py']
        r=acceptance(self.root,self.catalog)
        self.assertEqual('failed',r['gates']['python_source_bindings']['status'])

    def test_supplement_public_file_must_be_bound(self):
        s=self.supplement();del s['tasks'][0]['source_sha256'][self.public]
        with self.assertRaisesRegex(ValueError,'public file not bound'):
            validate_supplement(self.root,self.root,s,{'t'})

    def test_supplement_statement_change_rejected(self):
        s=self.supplement();s['tasks'][0]['declarations'][0]['statement_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'statement changed'):
            validate_supplement(self.root,self.root,s,{'t'})

    def test_callback_cannot_be_closed_software(self):
        s=self.supplement(kind='bit_interpreter_contract',role='parametric_value_contract')
        s['tasks'][0]['declarations'][0]['unclosed_numeric_callbacks']=['sqrt']
        with self.assertRaisesRegex(ValueError,'unclosed callback'):
            validate_supplement(self.root,self.root,s,{'t'})

    def test_proposed_migration_cannot_be_auto_accepted(self):
        with self.assertRaisesRegex(ValueError,'manifest schema'):
            review_manifest(self.root,self.root,{'schema':1,'status':'proposed_for_root_review_not_applied','files':{}})

    def test_all_checks_positive_is_still_not_native_correctness(self):
        self.put('gate.log','all synthetic mechanical fixture checks passed\n')
        gates={n:{'schema':1,'gate':n,'status':'passed','exit_code':0,
            'catalog_sha256':sha(canonical(self.catalog)),'input_sha256':self.catalog['build_input_sha256'],
            'command':['fixture-check',n],'log':'gate.log','log_sha256':hash_file(self.root/'gate.log')}
            for n in REQUIRED_EXTERNAL_GATES}
        r=acceptance(self.root,self.catalog,build=self.build(),object_roots=[self.objects],
            gates=gates,lean_binary=self.lean_binary)
        self.assertTrue(r['mechanical_gates_passed'])
        self.assertEqual('not_established_by_this_review',r['native_functional_correctness'])

    def test_relocated_objects_require_explicit_mapping(self):
        import shutil
        r=self.build();dest=Path(self.temp.name)/'new-objects';shutil.copytree(self.objects,dest)
        self.assertEqual('failed',verify_build(self.root,self.catalog,r,[dest],lean_binary=self.lean_binary)['status'])
        self.assertEqual('passed',verify_build(self.root,self.catalog,r,[dest],lean_binary=self.lean_binary,
            allow_root_relocation=True)['status'])

    def test_relocated_objects_still_require_exact_hash(self):
        import shutil
        r=self.build();dest=Path(self.temp.name)/'new-objects';shutil.copytree(self.objects,dest)
        (dest/Path(*self.mods[0].split('.')).with_suffix('.olean')).write_bytes(b'bad relocated object')
        self.assertEqual('failed',verify_build(self.root,self.catalog,r,[dest],lean_binary=self.lean_binary,
            allow_root_relocation=True)['status'])

    def test_multiple_public_files_are_explicit_not_sources(self):
        row=self.catalog['tasks'][0]
        row['declarations'].append({'file':BASE+'t/ProtocolCorrect.lean','declaration':'mutex'})
        self.assertEqual([self.public,BASE+'t/ProtocolCorrect.lean'],public_files(self.catalog))
        self.assertEqual(3,len(proof_files(self.catalog)))

    def test_host_final_software_does_not_cover_per_wave_prefix(self):
        task={'required_value_stages':['prefix','final'], 'declarations':[
            {'source_variant':'original_source_translation','model_kind':model,'role':'parametric_value_contract',
             'scope_relation':'full_declared_stage','covers_stages':stages}
            for model,stages in [('backend_value_contract',['prefix','final']),('bit_interpreter_contract',['final'])]]}
        result=graph_stage_coverage(task,'original_source_translation')
        self.assertTrue(result['all_required_value_stages']);self.assertFalse(result['all_required_software_stages'])

    def test_protocol_and_candidate_cannot_close_original_value_stage(self):
        task={'required_value_stages':['forward'], 'declarations':[
            {'source_variant':'original_source_translation','model_kind':'protocol_contract','role':'protocol_contract',
             'scope_relation':'not_an_execution_contract','covers_stages':[]},
            {'source_variant':'compile_repaired_candidate','model_kind':'bit_interpreter_contract','role':'parametric_value_contract',
             'scope_relation':'full_declared_stage','covers_stages':['forward']}]}
        self.assertFalse(graph_stage_coverage(task,'original_source_translation')['all_required_value_stages'])

    def test_legacy_block_boundary_drift_fails(self):
        from scripts.bench.legacy_source_adapter import source_blocks
        with self.assertRaisesRegex(ValueError,'boundaries changed'):
            source_blocks('echo a pretend PASS')

    def test_legacy_projection_excludes_unimported_new_typed_helper(self):
        from scripts.bench.legacy_source_adapter import legacy_projection
        row=deepcopy(self.catalog['tasks'][0]);row['historical_source_entry']=self.source
        self.assertEqual([self.source],list(legacy_projection(self.root,row)))
        self.assertIn(self.helper,proof_files(self.catalog))

    def test_legacy_projection_keeps_every_local_source_dependency(self):
        from scripts.bench.legacy_source_adapter import legacy_projection
        row=deepcopy(self.catalog['tasks'][0]);row['historical_source_entry']=self.public
        self.assertEqual({self.source,self.public},set(legacy_projection(self.root,row)))

    def test_legacy_projection_missing_import_rejected(self):
        from scripts.bench.legacy_source_adapter import legacy_projection
        row=deepcopy(self.catalog['tasks'][0]);row['historical_source_entry']=self.public
        row['files']=[f for f in row['files'] if f['path']!=self.source]
        with self.assertRaisesRegex(ValueError,'local import absent'):
            legacy_projection(self.root,row)

    def test_unplanned_failed_record_is_not_silently_discarded(self):
        r=self.build(self.records+[{'module':'Unplanned','exit_code':1}])
        self.assertEqual('failed',r['status'])

    def test_external_environment_inventory_tampering_rejected(self):
        r=self.build();r['environment']['external_direct_modules']=['Invented']
        self.assertBadBuild(r)

    def test_real_gate_runner_failure_cannot_pass(self):
        gate=run_gate(self.root,self.catalog,'axiom_audit',[sys.executable,'-c','raise SystemExit(3)'],
            gate_root=self.root,log='actual-gate.log')
        self.assertEqual('failed',gate['status']);self.assertEqual(3,gate['exit_code'])
        with self.assertRaisesRegex(ValueError,'failed or not run'):
            validate_gate(self.root,self.catalog,'axiom_audit',gate)

    def test_real_gate_runner_input_mutation_rejected(self):
        gate=run_gate(self.root,self.catalog,'axiom_audit',
            [sys.executable,'-c',"from pathlib import Path;Path('lean-toolchain').write_text('changed')"],
            gate_root=self.root,log='actual-gate.log')
        self.assertEqual('failed',gate['status']);self.assertEqual(['lean-toolchain'],gate['changed_inputs'])

    def test_real_gate_runner_success_binds_output(self):
        gate=run_gate(self.root,self.catalog,'axiom_audit',[sys.executable,'-c',"print('synthetic unit only')"],
            gate_root=self.root,log='actual-gate.log')
        validate_gate(self.root,self.catalog,'axiom_audit',gate)
        (self.root/'actual-gate.log').write_text('modified')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            validate_gate(self.root,self.catalog,'axiom_audit',gate)

    def test_supplement_generic_historical_input_map_rejected(self):
        s=self.supplement();s['tasks'][0]['review']['source_hashes_pointer']=['historical_inputs']
        with self.assertRaisesRegex(ValueError,'successful root final-source'):
            validate_supplement(self.root,self.root,s,{'t'})

    def test_supplement_failed_review_rejected(self):
        s=self.supplement();r=load(self.root/'review.json');r['status']='failed'
        write_json(self.root/'review.json',r)
        s['tasks'][0]['review']['sha256']=hash_file(self.root/'review.json')
        with self.assertRaisesRegex(ValueError,'successful root final-source'):
            validate_supplement(self.root,self.root,s,{'t'})

    def final_revision_manifest(self, field='after_sha256'):
        h=hash_file(self.root/self.source)
        write_json(self.root/'receipt.json',{'changed_sources':[{'file':self.source,'after_sha256':h,'before_sha256':h}]})
        write_json(self.root/'review.json',{'reviewer':'root','status':'compatibility_review_pass',
            'receipt_sha256':hash_file(self.root/'receipt.json')})
        return {'schema':1,'status':'accepted_final_source_bindings','files':{self.source:{
            'sha256':h,'review':{'path':'review.json','sha256':hash_file(self.root/'review.json')},
            'receipt':{'path':'receipt.json','sha256':hash_file(self.root/'receipt.json')},
            'hash_document':'receipt','hash_pointer':['changed_sources',0,field]}}}

    def test_explicit_changed_sources_final_revision_supported(self):
        self.assertIn(self.source,review_manifest(self.root,self.root,self.final_revision_manifest()))

    def test_before_revision_is_not_final_review_even_when_hash_matches(self):
        with self.assertRaisesRegex(ValueError,'final source revision'):
            review_manifest(self.root,self.root,self.final_revision_manifest('before_sha256'))

    def archive_fixture(self,name='safe/source.lean'):
        import zipfile
        package=Path(self.temp.name)/'metadata';package.mkdir()
        data=b'synthetic archive entry'
        with zipfile.ZipFile(package/'inputs.zip','w') as z:z.writestr(name,data)
        write_json(package/'input-sha256.json',{name:sha(data)})
        write_json(package/'receipt.json',{'schema':3,'summary':{'tasks':184},
            'artifact_sha256':{'input-sha256.json':hash_file(package/'input-sha256.json')},
            'inputs_zip_sha256':hash_file(package/'inputs.zip')})
        return package

    def test_baseline_extract_is_explicit_and_hash_bound(self):
        from scripts.bench.current_source_catalog import prepare_baseline
        package=self.archive_fixture();output=Path(self.temp.name)/'extracted'
        r=prepare_baseline(package,output)
        self.assertEqual(1,r['prepared_inputs']);self.assertTrue((output/'tree/safe/source.lean').is_file())

    def test_baseline_archive_escape_rejected_before_writing(self):
        from scripts.bench.current_source_catalog import prepare_baseline
        package=self.archive_fixture('../outside.lean');output=Path(self.temp.name)/'extracted'
        with self.assertRaisesRegex(ValueError,'invalid relative path'):prepare_baseline(package,output)
        self.assertFalse(output.exists())

    def test_baseline_extract_never_reuses_existing_tree(self):
        from scripts.bench.current_source_catalog import prepare_baseline
        package=self.archive_fixture();output=Path(self.temp.name)/'extracted';output.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh directory'):prepare_baseline(package,output)

    def candidate_fixture(self):
        self.put('original.py','x = (1,)\n');self.put('pkg/native/candidate.py','x = (1)\n')
        write_json(self.root/'pkg/native/receipt.json',{'original_source':'original.py',
            'original_source_sha256':hash_file(self.root/'original.py'),
            'artifacts':{'candidate.py':hash_file(self.root/'pkg/native/candidate.py')}})
        write_json(self.root/'pkg/receipt.json',{'native_compile':{'receipt':'native/receipt.json',
            'receipt_sha256':hash_file(self.root/'pkg/native/receipt.json')},
            'native_source':{'file':'original.py','sha256':hash_file(self.root/'original.py')}})
        write_json(self.root/'pkg/root-review.json',{'reviewer':'root','status':'review_pass',
            'receipt_sha256':hash_file(self.root/'pkg/receipt.json')})
        ext={'packages':[{'package':'pkg','receipt':'pkg/receipt.json','receipt_sha256':hash_file(self.root/'pkg/receipt.json'),
            'root_review':'pkg/root-review.json','root_review_sha256':hash_file(self.root/'pkg/root-review.json')}]}
        view={'id':'candidate','variant':'compile_repaired_candidate_graph','selected_definitions':{self.source:['typed']},
            'original_python':'original.py','candidate_python':'pkg/native/candidate.py',
            'candidate_python_binding':{'kind':'reviewed_native_compile_variant','package':'pkg','artifact':'candidate.py'}}
        return ext,view

    def test_candidate_source_view_preserves_two_python_variants_without_admission(self):
        from scripts.bench.current_source_catalog import source_views
        ext,view=self.candidate_fixture();actual=source_views('t',ext,None,{'t':[view]})[0]
        self.assertEqual('original.py',actual['original_python'])
        self.assertEqual('pkg/native/candidate.py',actual['candidate_python'])
        self.assertEqual('compile_repaired_candidate_graph',actual['variant'])

    def test_candidate_probe_chain_binds_exact_two_python_files(self):
        from scripts.bench.current_source_catalog import candidate_python_dependencies
        ext,view=self.candidate_fixture();deps,variants=candidate_python_dependencies(self.root,ext,view)
        self.assertEqual(5,len(deps));self.assertEqual('original',variants['original.py']['variant'])
        self.assertEqual('candidate',variants['pkg/native/candidate.py']['variant'])

    def test_candidate_python_content_drift_rejected(self):
        from scripts.bench.current_source_catalog import candidate_python_dependencies
        ext,view=self.candidate_fixture();self.put('pkg/native/candidate.py','changed\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):candidate_python_dependencies(self.root,ext,view)

    def test_candidate_probe_receipt_drift_rejected(self):
        from scripts.bench.current_source_catalog import candidate_python_dependencies
        ext,view=self.candidate_fixture();self.put('pkg/native/receipt.json','{}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):candidate_python_dependencies(self.root,ext,view)

    def test_candidate_outside_reviewed_package_rejected(self):
        from scripts.bench.current_source_catalog import candidate_python_dependencies
        ext,view=self.candidate_fixture();view['candidate_python_binding']['package']='other'
        with self.assertRaisesRegex(ValueError,'not in value registry'):candidate_python_dependencies(self.root,ext,view)

    def test_candidate_cannot_use_original_python_as_candidate(self):
        from scripts.bench.current_source_catalog import candidate_python_dependencies
        ext,view=self.candidate_fixture();view['candidate_python']='original.py'
        with self.assertRaisesRegex(ValueError,'path does not match'):candidate_python_dependencies(self.root,ext,view)

    def test_candidate_requires_explicit_review_binding(self):
        from scripts.bench.current_source_catalog import candidate_python_dependencies
        ext,view=self.candidate_fixture();del view['candidate_python_binding']
        with self.assertRaisesRegex(ValueError,'lacks reviewed'):candidate_python_dependencies(self.root,ext,view)

    def test_catalog_checks_candidate_review_artifact_drift(self):
        self.put('candidate-evidence.json','{}')
        self.catalog['source_variant_evidence_sha256']={'candidate-evidence.json':hash_file(self.root/'candidate-evidence.json')}
        self.put('candidate-evidence.json','changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):validate_catalog(self.root,self.catalog)

if __name__=='__main__':
    unittest.main()
