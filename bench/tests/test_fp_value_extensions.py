"""Failures that must not turn a partial/candidate proof into full acceptance."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bench'))
import fp_value_extensions as ex
from value_source_entries import validate_entries

class ValueExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry=json.loads((ROOT/ex.BASE/'fp_value_extensions.json').read_text())
        cls.entries=json.loads((ROOT/ex.BASE/'fp_value_source_entries.json').read_text())
    def task(self,name):return copy.deepcopy(next(t for t in self.registry['tasks'] if t['task']==name))
    def altered(self,mutate):
        data=copy.deepcopy(self.registry);mutate(data)
        path=ROOT/ex.BASE/'fp_value_extensions.json';original=Path.read_bytes
        def read(p):return (json.dumps(data,ensure_ascii=False)+'\n').encode() if p==path else original(p)
        with patch.object(Path,'read_bytes',read):return ex.collect(ROOT)
    def test_rational_scale_does_not_close_arbitrary_scale(self):
        summary=ex.stage_summary(self.task('token_attn_llama2'))
        self.assertTrue(summary['all_declared_stages_have_value_contract'])
        self.assertFalse(summary['all_declared_stages_have_closed_software_contract'])
    def test_relu_only_does_not_close_all_activation_strings(self):
        self.assertFalse(ex.stage_summary(self.task('triton_linear_activation'))['all_declared_stages_have_closed_software_contract'])
    def test_fixed_witness_cannot_cover_stage(self):
        t=self.task('iv_dependent_matmul')
        for d in t['declarations']:d['role']='fixed_nonempty_witness'
        self.assertFalse(ex.stage_summary(t)['all_declared_stages_have_value_contract'])
    def test_source_and_compiled_graph_cannot_be_combined(self):
        t=self.task('iv_dependent_matmul');t['required_value_stages']=['one','two']
        for d in t['declarations']:d['covers_stages']=['two' if d['source_variant']=='observed_compiled_graph' else 'one']
        self.assertEqual(ex.stage_summary(t)['uncovered_stages'],['two'])
    def test_phi_exp_callback_leaves_two_software_stages_open(self):
        s=ex.stage_summary(self.task('flash_decode2_phi'))
        self.assertEqual(s['closed_software_stages'],['copy','normalization'])
    def test_identity_cannot_fill_missing_stage(self):
        t=self.task('flash_decode2_phi');t['required_value_stages'].append('extra')
        d=next(d for d in t['declarations'] if d['role']=='recurrence_identity');d['covers_stages']=['extra']
        self.assertIn('extra',ex.stage_summary(t)['uncovered_stages'])
    def test_missing_whole_stage_is_reported_pending(self):
        def change(r):next(t for t in r['tasks'] if t['task']=='fp4_to_bf16_conversion')['required_value_stages'].append('missing')
        report=self.altered(change)
        t=next(t for t in report['tasks'] if t['task']=='fp4_to_bf16_conversion')
        self.assertEqual(t['current_value_status'],'value_contract_review_pending')
    def test_candidate_missing_source_binding_is_pending_first(self):
        def change(r):next(t for t in r['tasks'] if t['task']=='fp4_to_bf16_conversion')['packages']=[]
        report=self.altered(change)
        t=next(t for t in report['tasks'] if t['task']=='fp4_to_bf16_conversion')
        self.assertEqual(t['current_value_status'],'value_contract_review_pending')
        self.assertIn('unbound_final_public_source',t['value_extension']['review_pending'])
    def test_strict_specialization_cannot_fill_candidate(self):
        def change(r):
            for d in next(t for t in r['tasks'] if t['task']=='fp4_to_bf16_conversion')['declarations']:d['scope_relation']='strict_specialization'
        report=self.altered(change)
        self.assertEqual(next(t for t in report['tasks'] if t['task']=='fp4_to_bf16_conversion')['current_value_status'],'value_contract_review_pending')
    def test_closed_bit_label_with_callback_rejected(self):
        def change(r):next(t for t in r['tasks'] if t['task']=='softmax_reducev')['declarations'][1]['model_kind']='bit_interpreter_contract'
        with self.assertRaisesRegex(ValueError,'Unclosed callback'):self.altered(change)
    def test_changed_old41_fingerprint_rejected(self):
        with self.assertRaisesRegex(ValueError,'Stale input'):self.altered(lambda r:r['preserved_original_registry'].update(sha256='0'*64))
    def test_changed_public_header_hash_rejected(self):
        def change(r):r['tasks'][0]['declarations'][0]['statement_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'New public statement changed'):self.altered(change)
    def test_receipt_drift_rejected(self):
        def change(r):r['tasks'][0]['packages'][0]['receipt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Stale task/package binding'):self.altered(change)
    def test_bare_fp_correct_not_an_admission(self):
        entries=copy.deepcopy(self.entries);row=entries['entries'][0];del row['input_sha256'][row['bridge']['file']]
        with self.assertRaisesRegex(ValueError,'Incomplete source-entry evidence'):validate_entries(ROOT,entries)
    def test_candidate_cannot_be_registered_as_original(self):
        entries=copy.deepcopy(self.entries);next(r for r in entries['entries'] if r['task']=='fp4_to_bf16_conversion')['source_variant']='original'
        with self.assertRaisesRegex(ValueError,'Candidate incorrectly classified'):validate_entries(ROOT,entries)
    def test_original_laws_not_removed_by_new_conv2d_contract(self):
        report=ex.collect(ROOT)
        t=next(t for t in report['tasks'] if t['task']=='triton_conv2d_fwd')
        self.assertEqual(t['historical_numeric_status'],'requires_numeric_model_revision')
        self.assertEqual(t['current_value_status'],'full_declared_profile_software_value_contract')
        self.assertEqual(report['summary']['original_pending_exact_bindings'],5)
        self.assertEqual(len(report['historical_refuted_model_tasks']),3)
    def test_selected_dtype_profile_does_not_become_old_dtype_complete(self):
        t=self.task('matmul_triton_autotune');self.assertEqual(t['dtype_scope_relation'],'selected_explicit_profiles_only')
        self.assertTrue(ex.stage_summary(t)['all_declared_stages_have_closed_software_contract'])
    def test_retention_header_preservation_is_not_semantic_acceptance(self):
        report=ex.collect(ROOT)
        self.assertEqual(next(t for t in report['tasks'] if t['task']=='chunk_retention')['current_value_status'],'backend_equivalence_reference_revalidated')
        def change(r):r['known_pending_semantic_revisions'][0]['package']['receipt_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Unreviewed semantic revision'):self.altered(change)
    def test_original_and_candidate_cannot_jointly_fill_stages(self):
        t=self.task('iv_dependent_matmul');t['required_value_stages']=['one','two']
        for i,d in enumerate(t['declarations']):
            d['source_variant']='compile_repaired_candidate' if i%2 else 'original_source_translation'
            d['covers_stages']=['two' if i%2 else 'one']
        self.assertEqual(ex.stage_summary(t)['uncovered_stages'],['two'])
    def test_standalone_SC_model_cannot_replace_source_contract(self):
        t=self.task('spinning_lock_reduction')
        t['declarations']=[d for d in t['declarations'] if d['source_variant']=='standalone_sc_model']
        self.assertFalse(ex.stage_summary(t)['all_declared_stages_have_value_contract'])
    def test_frame_cannot_fill_value_stage(self):
        t=self.task('spinning_lock_reduction')
        t['declarations']=[d for d in t['declarations'] if d['role']=='memory_frame_contract']
        self.assertFalse(ex.stage_summary(t)['all_declared_stages_have_value_contract'])
    def test_task_local_source_is_not_shared_DSL_admission(self):
        entries=copy.deepcopy(self.entries);next(e for e in entries['entries'] if e['task']=='spinning_lock_reduction')['ordinary_shared_dsl_admission']=True
        with self.assertRaisesRegex(ValueError,'Task-local frontend promoted'):validate_entries(ROOT,entries)
    def test_multi_file_public_omission_rejected(self):
        def change(r):next(t for t in r['tasks'] if t['task']=='spinning_lock_reduction')['public_files']=['bench/tritonbench_g/spinning_lock_reduction/SourceCorrect.lean']
        with self.assertRaisesRegex(ValueError,'Invalid declaration model/scope/source'):self.altered(change)
    def test_aligned_scalar_bits_do_not_close_callbacks(self):
        self.assertFalse(ex.stage_summary(self.task('attention_kernel_aligned'))['all_declared_stages_have_closed_software_contract'])
    def test_unknown_dtype_coverage_relation_rejected(self):
        def change(r):r['tasks'][0]['dtype_scope_relation']='all_hardware_dtypes'
        with self.assertRaisesRegex(ValueError,'Invalid dtype coverage relation'):self.altered(change)
    def test_SC_mutex_is_not_a_numeric_value_contract(self):
        t=self.task('layer_norm_triton')
        t['declarations']=[d for d in t['declarations'] if d['role']=='protocol_contract']
        self.assertFalse(ex.stage_summary(t)['all_declared_stages_have_value_contract'])
        def change(r):
            t=next(t for t in r['tasks'] if t['task']=='layer_norm_triton')
            d=next(d for d in t['declarations'] if d['role']=='protocol_contract')
            d.update(role='parametric_value_contract',scope_relation='full_declared_stage',covers_stages=t['required_value_stages'])
        with self.assertRaisesRegex(ValueError,'Identity/frame/protocol promoted'):self.altered(change)
    def test_layernorm_backward_final_bits_do_not_close_forward_sqrt(self):
        s=ex.stage_summary(self.task('layer_norm_triton'))
        self.assertTrue(s['all_declared_stages_have_value_contract'])
        self.assertEqual(s['closed_software_stages'],['backward_dx_partial','final_dwdb'])
        self.assertFalse(s['all_declared_stages_have_closed_software_contract'])
    def test_streamk_host_final_contract_does_not_certify_intermediate_wave(self):
        s=ex.stage_summary(self.task('streamk_matmul'))
        self.assertTrue(s['all_declared_stages_have_value_contract'])
        self.assertEqual(s['closed_software_stages'],['ordered_host_launches'])
        self.assertFalse(s['all_declared_stages_have_closed_software_contract'])
    def test_layernorm_protocol_public_file_cannot_be_omitted(self):
        def change(r):
            t=next(t for t in r['tasks'] if t['task']=='layer_norm_triton')
            t['public_files'].remove('bench/tritonbench_g/layer_norm_triton/ProtocolCorrect.lean')
        with self.assertRaisesRegex(ValueError,'Invalid declaration model/scope/source'):self.altered(change)
    def test_all_three_task_local_frontends_remain_separate(self):
        e=validate_entries(ROOT,self.entries)
        names={k for k,v in e.items() if v.get('frontend_kind')=='task_local_exact_ast'}
        self.assertEqual(names,{'spinning_lock_reduction','layer_norm_triton','streamk_matmul'})
    def test_triton_attention_preprocess_bits_do_not_close_both_attention_loops(self):
        t = self.task('triton_attention')
        s = ex.stage_summary(t)
        self.assertTrue(s['all_declared_stages_have_value_contract'])
        self.assertEqual(s['closed_software_stages'], ['preprocess'])
        self.assertFalse(s['all_declared_stages_have_closed_software_contract'])
        self.assertEqual(sum(d['model_kind'] == 'bit_interpreter_contract'
                             for d in t['declarations']), 1)
        self.assertEqual(t['dtype_scope_relation'], 'selected_explicit_profiles_only')

    def test_triton_attention_scalar_bit_names_do_not_close_callbacks(self):
        for stage in ('forward', 'backward'):
            with self.subTest(stage=stage):
                def change(registry):
                    task = next(t for t in registry['tasks'] if t['task'] == 'triton_attention')
                    d = next(d for d in task['declarations']
                             if d['declaration'] == 'triton_attention_' + stage + '_scalar_bit_correctness')
                    self.assertTrue(d['unclosed_numeric_callbacks'])
                    d['model_kind'] = 'bit_interpreter_contract'
                with self.assertRaisesRegex(ValueError, 'Unclosed callback'):
                    self.altered(change)

    def test_triton_attention_other_graph_cannot_complete_original_backward(self):
        t = self.task('triton_attention')
        for d in t['declarations']:
            if d['covers_stages'] == ['backward']:
                d['source_variant'] = 'observed_compiled_graph'
        s = ex.stage_summary(t)
        self.assertEqual(s['uncovered_stages'], ['backward'])
        self.assertFalse(s['all_declared_stages_have_value_contract'])
        # A distinct graph can be reviewed on its own; it cannot patch the
        # original-source stage or justify universal dot+add = dotAcc/FMA.
        other = ex.stage_summary(t, source_variant='observed_compiled_graph')
        self.assertEqual(other['covered_value_stages'], ['backward'])

if __name__=='__main__':unittest.main()
