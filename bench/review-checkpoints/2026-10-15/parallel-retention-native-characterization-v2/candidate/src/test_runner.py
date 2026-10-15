import copy,json,os,shutil,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
m=types.ModuleType('pra_test_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m;exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)
v=types.ModuleType('pra_test_verifier');v.__file__=str(HERE/'verify_capture.py');sys.modules[v.__name__]=v;exec(compile((HERE/'verify_capture.py').read_bytes(),v.__file__,'exec'),v.__dict__)
REF=HERE.parent/'reference-v1';OFF=HERE.parent/'offline-v1';GPU=HERE.parent/'gpu-attempt-v1'
def fake_modules(case,fault=None):
    """CPU test double; temporary records are never retained as native evidence."""
    t=types.ModuleType('torch');t.__version__='CPU_TEST_DOUBLE';t.version=types.SimpleNamespace(cuda='MOCK');t.uint32='torch.uint32';t.float32='torch.float32';nextaddr=[4096]
    class Buf:
        def __init__(self,words,dtype,owner=None,off=0):
            self.owner=owner or self;self.off=off;self.dtype=dtype
            if owner is None:self.words=list(words);self.addr=nextaddr[0];nextaddr[0]+=1<<24
        def view(self,dtype):return Buf(None,dtype,self.owner,self.off)
        def __getitem__(self,s):return Buf(None,self.dtype,self.owner,self.off+s.start)
        def data_ptr(self):return self.owner.addr+4*self.off+(1 if fault=='origin' and self.off else 0)
        def numel(self):return len(self.owner.words)-self.off
        def cpu(self):return self
        def tolist(self):return self.owner.words[self.off:]
    t.tensor=lambda words,dtype,device:Buf(words,dtype);t.cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda d:None,current_device=lambda:0,get_device_properties=lambda i:types.SimpleNamespace(major=8,minor=0,uuid='MOCK',name='CPU_TEST_DOUBLE'),synchronize=lambda:None)
    tr=types.ModuleType('triton');tc=types.ModuleType('triton.compiler');tb=types.ModuleType('triton.backends');tbc=types.ModuleType('triton.backends.compiler')
    class ASTSource:
        def __init__(self,fn,sig,constants):self.fn=fn;self.sig=sig;self.constants=constants
    tc.ASTSource=ASTSource;tbc.GPUTarget=lambda *a:a
    class Kernel:
        def __init__(self,source,opts):self.s=source;self.metadata=opts;self.asm={k:'CPU_TEST_DOUBLE' for k in ['ttir','ttgir','llir','ptx','cubin']}
        def __getitem__(self,grid):
            def launch(*args):
                if fault=='launch':raise RuntimeError('mock launch failure')
                stage=self.s.fn.stage;lp=m.c.launch_plan(case,stage);values=dict(zip(self.s.sig,args));expect=[1,1,1] if stage=='backward_helper' else lp['grid'];assert tuple(expect)==grid
                if stage=='backward_helper':
                    expected={k:values[k] for k in ['i_bh','i_c','i_k','i_v','i_h']};assert expected in lp['helper_calls']
                for n,offsets in lp['write_offsets'].items():
                    for off in offsets:values[n].owner.words[off]=0x3f800000
                if fault=='frame':values[lp['output_regions'][0]].owner.words[0]^=1
            return launch
    tr.compile=lambda source,target,options:Kernel(source,options)
    return {'torch':t,'triton':tr,'triton.compiler':tc,'triton.backends':tb,'triton.backends.compiler':tbc}
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ref,cls.cases=m.validate(REF)
    def before(self,case,stage):
        p=m.c.launch_plan(case,stage);return {n:case['allocations'][n]['raw_words'].copy() for n in p['input_regions']+p['output_regions']}
    def mutate(self,fn,source=OFF):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'capture';shutil.copytree(source,q,ignore=shutil.ignore_patterns('cache','pycache'));r=m.u.load(q/'report.json');fn(q,r);m.u.dump(q/'report.json',r)
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):v.verify(q)
    def test_01_four_jits_exact_AST(self):
        jit,ex=m.probe.extract((REF/'original.py').read_bytes());self.assertEqual(jit,(REF/'original_jit.py').read_bytes());self.assertEqual(len(ex['functions']),4)
    def test_02_original_scope_not_shell_or_reduction(self):self.assertEqual(self.ref['accepted_stages'],['forward','backward_helper']);self.assertIn('host partial reductions',self.ref['outside_accepted_scope'])
    def test_03_full_forward_two_loops(self):
        p=m.c.launch_plan(self.cases['full_two_forward_chunks'],'forward');last=[x for x in p['programs'] if x['program']==[0,1,0]][0]
        self.assertEqual(last['offdiagonal_order'],[0,32,64,96]);self.assertEqual(last['diagonal_order'],[128,160,192,224]);self.assertEqual(len(p['write_offsets']['o']),2*256*64)
    def test_04_backward_descending_and_diagonal_order(self):
        p=m.c.launch_plan(self.cases['ragged_time_special'],'backward_helper');x=[x for x in p['programs'] if x['program'][2]==0]
        self.assertEqual(x[0]['offdiagonal_order'],[128,96,64]);self.assertEqual(x[1]['offdiagonal_order'],[128]);self.assertEqual(x[2]['offdiagonal_order'],[]);self.assertEqual(x[2]['diagonal_order'],[128,160])
    def test_05_complete_helper_arguments(self):
        p=m.c.launch_plan(self.cases['full_two_forward_chunks'],'backward_helper');self.assertEqual(len(p['helper_calls']),8);self.assertEqual({x['i_c'] for x in p['helper_calls']},{0,1,2,3});self.assertEqual({x['i_h'] for x in p['helper_calls']},{0,1})
    def test_06_preserve_all_NK_partials(self):
        c=self.cases['ragged_split_heads'];f=m.c.launch_plan(c,'forward');b=m.c.launch_plan(c,'backward_helper');self.assertEqual(c['NK'],2)
        self.assertEqual(len(f['write_offsets']['o']),2*2*65*96);self.assertEqual(len(b['write_offsets']['dv']),2*2*65*96);self.assertEqual(len(b['write_offsets']['dk']),2*65*160);self.assertFalse(f['host_reduction_performed'])
    def test_07_undef_explicit_not_zero(self):
        c=self.cases['ragged_split_heads'];p=m.c.launch_plan(c,'forward');self.assertGreater(p['undefined_load_lanes'],0);self.assertTrue(all(x['padding']=='unspecified' for x in p['load_descriptors']))
        b=self.before(c,'forward');r=m.c.check_capture(c,'forward',b,copy.deepcopy(b));self.assertFalse(r['undefined_values_measured']);self.assertEqual(r['numeric_comparison'],'NOT_ESTABLISHED')
    def test_08_dense_no_tail_undef(self):
        for stage in ['forward','backward_helper']:
            p=m.c.launch_plan(self.cases['full_two_forward_chunks'],stage);self.assertEqual(p['undefined_load_lanes'],0)
    def test_09_bwd_tail_undef_not_pruned_by_causal_mask(self):
        p=m.c.launch_plan(self.cases['ragged_time_special'],'backward_helper');self.assertTrue(any(x['phase']=='descending' and x['invalid_lanes']>0 for x in p['load_descriptors']))
    def test_10_shell_explicitly_outside_proof(self):
        c=self.cases['strided'];p=m.c.launch_plan(c,'backward_shell');self.assertFalse(p['accepted_scope']);self.assertIn('dq',p['output_regions']);b=self.before(c,'backward_shell');self.assertFalse(m.c.check_capture(c,'backward_shell',b,copy.deepcopy(b))['accepted_stage'])
    def test_11_raw_special_bits_not_decoded(self):
        c=self.cases['ragged_time_special'];words=set(c['allocations']['q']['raw_words']);self.assertTrue({0,0x80000000,1,0x7f800000,0x7fc01234,0x7f800001}.issubset(words))
    def test_12_nonwrite_frame_and_inputs(self):
        c=self.cases['strided'];b=self.before(c,'backward_helper');a=copy.deepcopy(b);a['dk'][0]^=1;a['q'][-1]^=1;r=m.c.check_capture(c,'backward_helper',b,a);self.assertEqual(len(r['frame_violations']),2)
    def test_13_any_active_value_not_numeric_pass(self):
        c=self.cases['strided'];b=self.before(c,'forward');a=copy.deepcopy(b);p=m.c.launch_plan(c,'forward')
        for i in p['write_offsets']['o']:a['o'][i]=0x7fffffff
        r=m.c.check_capture(c,'forward',b,a);self.assertEqual(r['frame_violations'],[]);self.assertFalse(r['numeric_acceptance']);self.assertEqual(len(r['active_outputs']['o']['raw_words']),2*96*64)
    def test_14_missing_output_rejected(self):
        c=self.cases['strided'];b=self.before(c,'backward_helper');a=copy.deepcopy(b);del a['dv']
        with self.assertRaises(ValueError):m.c.check_capture(c,'backward_helper',b,a)
    def test_15_initial_drift_or_short_buffer_rejected(self):
        c=self.cases['strided'];b=self.before(c,'forward');a=copy.deepcopy(b);a['o'].pop()
        with self.assertRaises(ValueError):m.c.check_capture(c,'forward',b,a)
        b['q'][0]^=1
        with self.assertRaises(ValueError):m.c.check_capture(c,'forward',b,b)
    def test_16_partial_selection_explicit(self):
        ids,stages=m.selection(self.cases,['strided'],['backward_helper']);self.assertEqual((ids,stages),(['strided'],['backward_helper']))
        with self.assertRaises(ValueError):m.selection(self.cases,['missing'],None)
        with self.assertRaises(ValueError):m.selection(self.cases,None,['dq'])
    def test_17_signature_preserves_runtime_scale_and_indices(self):
        args=self.ref['source_extraction']['arguments']['_parallel_retention_bwd_dkv'];sig=m.probe.signature(types.SimpleNamespace(arg_names=args),m.c.constants(self.cases['strided'],'backward_helper'))
        self.assertEqual(sig['scale'],'fp32');self.assertEqual(sig['i_h'],'i32');self.assertEqual(sig['q'],'*fp32')
    def test_18_exact_old_declarations_pinned(self):
        b=m.u.load(REF/'inputs/source-bindings.json');self.assertIn('FPCorrect.lean',b['task_source_sha256']);self.assertIn('ParallelRetentionAttention.lean',b['task_source_sha256'])
    def test_19_rebound_case_forgery_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);cs=m.u.load(q/'cases.json');cs['strided']['T']=32;m.u.dump(q/'cases.json',cs);r=m.u.load(q/'reference.json');r['cases_sha256']=m.u.sha(q/'cases.json');m.u.dump(q/'reference.json',r)
            with self.assertRaises(ValueError):m.validate(q)
    def test_20_24_actual_offline_profiles(self):self.assertEqual(v.verify(OFF)['records'],24)
    def test_21_missing_profile_rejected(self):self.mutate(lambda q,r:r['records'].pop())
    def test_22_duplicate_profile_rejected(self):self.mutate(lambda q,r:r['records'].__setitem__(1,copy.deepcopy(r['records'][0])))
    def test_23_offline_not_numeric(self):self.mutate(lambda q,r:r.__setitem__('numeric_acceptance',True))
    def test_24_offline_not_native(self):self.mutate(lambda q,r:r.__setitem__('native_execution',True))
    def test_25_compiler_artifact_drift(self):self.mutate(lambda q,r:(q/(r['records'][0]['label']+'.ptx')).write_text('changed'))
    def test_26_actual_gpu_unavailable(self):self.assertEqual(v.verify(GPU)['status'],'VERIFIED_NOT_RUN')
    def test_27_notrun_not_success(self):self.mutate(lambda q,r:r.__setitem__('exit_code',0),GPU)
    def runtime(self,fault=None):
        case=self.cases['strided']
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'mock';mods=fake_modules(case,fault);fns={}
            for name in m.probe.NAMES:
                stage=next((s for s,n in m.STAGES.items() if n==name),'outside');fns[name]=types.SimpleNamespace(stage=stage,arg_names=self.ref['source_extraction']['arguments'][name],src='CPU_TEST_DOUBLE',cache_key='MOCK')
            env={'file_sha256':{str(HERE/'test_runner.py'):m.u.sha(HERE/'test_runner.py')},'boundary':'CPU test only'};query=types.SimpleNamespace(args=m.DRIVER_QUERY_COMMAND,returncode=0,stdout='MOCK, mock, mock',stderr='')
            with patch.dict(sys.modules,mods),patch.object(m.u,'checked_module',return_value=types.SimpleNamespace(**fns)),patch.object(m.u,'compiler_environment',return_value=env),patch.object(m.subprocess,'run',return_value=query),patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
                code,_=m.collect(REF,out,'run',None,'cuda:0',['strided'],None)
            r=m.u.load(out/'report.json')
            if r['status']=='CHARACTERIZATION_NUMERICS_UNBOUND':self.assertEqual(v.verify(out)['records'],6)
            return code,r
    def test_28_synthetic_all_stages_complete_capture(self):
        code,r=self.runtime();self.assertEqual(code,0);self.assertEqual(len(r['records']),6);self.assertEqual(r['native_launches_completed'],12);self.assertFalse(r['numeric_acceptance'])
    def test_29_synthetic_frame_failure(self):
        code,r=self.runtime('frame');self.assertEqual(code,1);self.assertTrue(any(x['comparison']['frame_violations'] for x in r['records']))
    def test_30_synthetic_wrong_origin(self):
        code,r=self.runtime('origin');self.assertEqual(code,1);self.assertFalse(r['native_launch_attempted'])
    def test_31_synthetic_launch_failure(self):
        code,r=self.runtime('launch');self.assertEqual(code,1);self.assertEqual(r['status'],'EXECUTION_ERROR');self.assertEqual(r['native_launches_completed'],0)
    def test_32_requested_architecture_cannot_disappear(self):
        def change(q,r):
            r['resolved_plan']['archs']=[90];r['resolved_plan']['expected_records']=12
            r['records']=[x for x in r['records'] if x['target']['arch']==90]
        self.mutate(change)
    def test_33_requested_architecture_order_preserved(self):self.mutate(lambda q,r:r['resolved_plan'].__setitem__('archs',[90,80]))
    def test_34_offline_environment_binding_cannot_be_true(self):self.mutate(lambda q,r:r.__setitem__('environment_binding_complete',True))
    def test_35_notrun_environment_binding_cannot_be_true(self):self.mutate(lambda q,r:r.__setitem__('environment_binding_complete',True),GPU)
    def environment_report(self):
        return {'mode':'run','native_execution':True,'device_uuid':'MOCK',
          'driver_query':{'command':m.DRIVER_QUERY_COMMAND,'exit_code':0,'stdout':'MOCK, DRIVER, DEVICE\n'}}
    def test_36_native_environment_exact_uuid_valid(self):self.assertTrue(m.environment_binding_complete(self.environment_report()))
    def test_37_native_environment_wrong_uuid(self):
        r=self.environment_report();r['device_uuid']='DIFFERENT';self.assertFalse(m.environment_binding_complete(r))
    def test_38_native_environment_failed_query(self):
        r=self.environment_report();r['driver_query']['exit_code']=1;self.assertFalse(m.environment_binding_complete(r))
    def test_39_native_environment_wrong_query(self):
        r=self.environment_report();r['driver_query']['command']=['different'];self.assertFalse(m.environment_binding_complete(r))
    def test_40_native_environment_empty_driver(self):
        r=self.environment_report();r['driver_query']['stdout']='MOCK, , name';self.assertFalse(m.environment_binding_complete(r))
    def test_41_native_environment_requires_execution(self):
        r=self.environment_report();r['native_execution']=False;self.assertFalse(m.environment_binding_complete(r))
    def test_42_synthetic_runtime_environment_verified(self):
        code,r=self.runtime();self.assertEqual(code,0);self.assertTrue(r['environment_binding_complete'])
if __name__=='__main__':unittest.main(verbosity=2)
