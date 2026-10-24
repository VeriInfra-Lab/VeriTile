import copy,hashlib,json,os,shutil,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
m=types.ModuleType('rbe_tests_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m
exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)
v=types.ModuleType('rbe_tests_verifier');v.__file__=str(HERE/'verify_capture.py');sys.modules[v.__name__]=v
exec(compile((HERE/'verify_capture.py').read_bytes(),v.__file__,'exec'),v.__dict__)
REF=Path(os.environ.get('RBE_TEST_REFERENCE',str(HERE.parent/'reference-v1')));OFF=Path(os.environ.get('RBE_TEST_OFFLINE',str(HERE.parent/'offline-v1')));GPU=Path(os.environ.get('RBE_TEST_GPU',str(HERE.parent/'gpu-attempt-v1')))
def synthetic_modules(cases,fault=None):
    """CPU fixtures only, discarded by TemporaryDirectory; not native evidence."""
    t=types.ModuleType('torch');t.__version__='CPU_TEST_DOUBLE';t.version=types.SimpleNamespace(cuda='MOCK');t.uint16='torch.uint16';t.float16='torch.float16';address=[4096]
    class Buffer:
        def __init__(self,words,dtype,owner=None,offset=0):
            self.dtype=dtype;self.owner=owner or self;self.offset=offset
            if owner is None:self.words=list(words);self.address=address[0];address[0]+=1<<20
        def view(self,dtype):return Buffer(None,dtype,self.owner,self.offset)
        def __getitem__(self,s):return Buffer(None,self.dtype,self.owner,self.offset+s.start)
        def data_ptr(self):return self.owner.address+2*self.offset+(1 if fault=='origin' and self.offset else 0)
        def numel(self):return len(self.owner.words)-self.offset
        def cpu(self):return self
        def tolist(self):return self.owner.words[self.offset:]
    t.tensor=lambda words,dtype,device:Buffer(words,dtype)
    t.cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda d:None,current_device=lambda:0,get_device_properties=lambda i:types.SimpleNamespace(major=8,minor=0,uuid='MOCK_UUID',name='CPU TEST DOUBLE'),synchronize=lambda:None)
    tr=types.ModuleType('triton');tr.__version__='MOCK';tc=types.ModuleType('triton.compiler');tb=types.ModuleType('triton.backends');tbc=types.ModuleType('triton.backends.compiler')
    class ASTSource:
        def __init__(self,fn,signature,constants):self.fn=fn;self.signature=signature;self.constants=constants
    tc.ASTSource=ASTSource;tbc.GPUTarget=lambda *args:args
    class Kernel:
        def __init__(self,source,options):self.source=source;self.metadata={**options,'num_warps':4,'num_stages':3};self.asm={n:'MOCK CPU ONLY' for n in ['ttir','ttgir','llir','ptx','cubin']}
        def __getitem__(self,grid):
            def launch(*args):
                if fault=='launch':raise RuntimeError('mock launch error')
                values=dict(zip(self.source.signature,args));scalars={n:v for n,v in values.items() if not n.endswith('_ptr')}
                found=[c for c in cases.values() if c['runtime_args']==scalars and c['constexprs']==self.source.constants];assert len(found)==1;c=found[0];assert tuple(c['grid'])==grid
                pointer=values['out_ptr']
                for item in c['footprint']['accesses']:pointer.owner.words[item['output_word']]=0x3c00
                if fault=='frame':pointer.owner.words[0]^=1
            return launch
    tr.compile=lambda source,target,options:Kernel(source,options)
    return {'torch':t,'triton':tr,'triton.compiler':tc,'triton.backends':tb,'triton.backends.compiler':tbc}
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ref,cls.cases=m.validate(REF)
    def before(self,c):return {n:a['raw_words'].copy() for n,a in c['allocations'].items()}
    def mutate(self,fn,source=OFF):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'case';shutil.copytree(source,q,ignore=shutil.ignore_patterns('cache','pycache'));r=m.u.load(q/'report.json');fn(q,r);m.u.dump(q/'report.json',r)
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):v.verify(q)
    def test_01_ten_complete_cases(self):self.assertEqual(len(self.cases),10);self.assertEqual(set(c['profile_id'] for c in self.cases.values()),set(m.c.PROFILES))
    def test_02_default_four_positions_full_buffer(self):
        for pos in range(4):
            c=self.cases['wrapper_pos'+str(pos)];self.assertEqual(c['runtime_args']['start_token_position'],pos);self.assertEqual(c['grid'],[2,2,1]);self.assertEqual(c['footprint']['active_components'],8192)
    def test_03_distinct_tail_masks(self):
        c=self.cases['odd_tail_all_payloads'];self.assertEqual(c['footprint']['active_pairs_real'],2*3*513);self.assertEqual(c['footprint']['active_pairs_imag'],2*3*512);self.assertEqual(c['footprint']['masked_lanes']['real_only_pairs'],6)
    def test_04_nonunit_stride_imag_plus_one(self):
        c=self.cases['strided_dual_mask'];items=c['footprint']['accesses'];r=items[0];i=items[1]
        self.assertEqual(i['input_word']-r['input_word'],1);self.assertEqual(i['output_word']-r['output_word'],1);self.assertEqual(c['runtime_args']['stride_x_n'],3);self.assertEqual(c['runtime_args']['stride_out_n'],5)
    def test_05_all_payload_classes_not_finite_only(self):
        words=set(self.cases['odd_tail_all_payloads']['allocations']['x']['raw_words'])
        self.assertTrue({0,0x8000,1,0x8001,0x7c00,0xfc00,0x7e55,0xfe01,0x7d01}.issubset(words))
    def test_06_capture_preserves_nan_bits_without_decode(self):
        c=self.cases['odd_tail_all_payloads'];before=self.before(c);r=m.c.check_capture(c,before,copy.deepcopy(before));self.assertEqual(r['frame_violations'],[]);self.assertEqual(r['numeric_comparison'],'NOT_ESTABLISHED')
    def test_07_all_active_words_reported(self):
        c=self.cases['wrapper_pos0'];b=self.before(c);r=m.c.check_capture(c,b,copy.deepcopy(b));self.assertEqual(len(r['active_output_words']),8192);self.assertFalse(r['numeric_acceptance'])
    def test_08_arbitrary_active_output_never_numeric_pass(self):
        c=self.cases['wrapper_pos0'];b=self.before(c);a=copy.deepcopy(b)
        for n,offs in c['footprint']['write_offsets'].items():
            for o in offs:a[n][o]=0x7fff
        r=m.c.check_capture(c,b,a);self.assertEqual(r['frame_check'],'UNCHANGED_OUTSIDE_FOOTPRINT');self.assertFalse(r['numeric_acceptance']);self.assertEqual(r['status'],'CHARACTERIZATION_NUMERICS_UNBOUND')
    def test_09_guard_violation(self):
        c=self.cases['wrapper_pos0'];b=self.before(c);a=copy.deepcopy(b);a['out'][0]^=1;r=m.c.check_capture(c,b,a);self.assertEqual(r['frame_check'],'VIOLATION')
    def test_10_input_mutation_rejected_by_frame(self):
        c=self.cases['wrapper_pos0'];b=self.before(c);a=copy.deepcopy(b);a['x'][-1]^=1;self.assertTrue(m.c.check_capture(c,b,a)['frame_violations'])
    def test_11_inplace_writes_allowed_only_in_output_footprint(self):
        c=self.cases['inplace_same_storage'];self.assertEqual(len(c['allocations']),1);b=self.before(c);a=copy.deepcopy(b);a['x'][c['footprint']['write_offsets']['x'][0]]^=1
        self.assertEqual(m.c.check_capture(c,b,a)['frame_violations'],[]);a['x'][0]^=1;self.assertTrue(m.c.check_capture(c,b,a)['frame_violations'])
    def test_12_missing_capture_rejected(self):
        c=self.cases['wrapper_pos0'];b=self.before(c);a=copy.deepcopy(b);del a['x']
        with self.assertRaises(ValueError):m.c.check_capture(c,b,a)
    def test_13_invalid_payload_or_length_rejected(self):
        c=self.cases['wrapper_pos0'];b=self.before(c)
        for x in [65536,-1,False]:
            a=copy.deepcopy(b);a['out'][8]=x
            with self.assertRaises(ValueError):m.c.check_capture(c,b,a)
        a=copy.deepcopy(b);a['out'].pop()
        with self.assertRaises(ValueError):m.c.check_capture(c,b,a)
    def test_14_before_tamper_rejected(self):
        c=self.cases['wrapper_pos0'];b=self.before(c);b['x'][0]^=1
        with self.assertRaises(ValueError):m.c.check_capture(c,b,b)
    def test_15_missing_row_or_column_grid_rejected(self):
        c=copy.deepcopy(self.cases['strided_dual_mask']);c['grid'][1]-=1
        with self.assertRaises(ValueError):m.c.footprint(c)
    def test_16_output_collision_rejected_in_corpus(self):
        c=copy.deepcopy(self.cases['strided_dual_mask']);c['runtime_args']['stride_out_m']=0
        with self.assertRaises(ValueError):m.c.footprint(c)
    def test_17_boundaries_not_hidden_preconditions(self):
        c=self.cases['signed_position_wrap'];self.assertEqual(c['runtime_args']['start_token_position'],2147483647);self.assertFalse(c['finite_input_assumption_added_to_formal_theorem'])
        exact=c['runtime_args']['start_token_position']+1;wrapped=(exact+(1<<31))%(1<<32)-(1<<31);self.assertEqual((exact,wrapped),(2147483648,-2147483648))
    def test_18_two_source_AST_functions_exact(self):
        jit,info=m.probe.extract((REF/'original.py').read_bytes());self.assertEqual(jit,(REF/'original_jit.py').read_bytes());self.assertEqual(info['functions'],['get_freq_multi_tokens','rbe_triton']);self.assertTrue(info['function_AST_equal'])
    def test_19_supplied_trig_not_in_interface(self):
        self.assertNotIn('cos',self.ref['source_extraction']['kernel_arguments']);self.assertNotIn('sin',self.ref['source_extraction']['kernel_arguments']);self.assertEqual(self.ref['native_numeric_reference'],'NOT_ESTABLISHED')
    def test_20_reference_forged_cases_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);cs=m.u.load(q/'cases.json');cs['wrapper_pos0']['runtime_args']['M']=1;m.u.dump(q/'cases.json',cs)
            r=m.u.load(q/'reference.json');r['cases_sha256']=m.u.sha(q/'cases.json');m.u.dump(q/'reference.json',r)
            with self.assertRaises(ValueError):m.validate(q)
    def test_21_reference_parent_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);(q/'inputs/task/FPCorrect.lean').write_text('forged')
            with self.assertRaises(ValueError):m.validate(q)
    def test_22_actual_offline_sixteen_profiles(self):self.assertEqual(v.verify(OFF)['records'],16)
    def test_23_missing_record(self):self.mutate(lambda q,r:r['records'].pop())
    def test_24_duplicate_record(self):self.mutate(lambda q,r:r['records'].__setitem__(1,copy.deepcopy(r['records'][0])))
    def test_25_offline_cannot_be_numeric(self):self.mutate(lambda q,r:r.__setitem__('numeric_acceptance',True))
    def test_26_offline_cannot_be_native(self):self.mutate(lambda q,r:r.__setitem__('native_execution',True))
    def test_27_missing_compiler_artifact(self):self.mutate(lambda q,r:(q/(r['records'][0]['label']+'.ptx')).unlink())
    def test_28_actual_no_gpu_preserved(self):self.assertEqual(v.verify(GPU)['status'],'VERIFIED_NOT_RUN')
    def test_29_notrun_cannot_pass(self):self.mutate(lambda q,r:r.__setitem__('exit_code',0),GPU)
    def test_30_failed_capture_not_upgraded(self):self.mutate(lambda q,r:r.__setitem__('status','COMPILE_FAILED'))
    def runtime(self,fault=None):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'mock';mods=synthetic_modules(self.cases,fault);fn=types.SimpleNamespace(arg_names=self.ref['source_extraction']['kernel_arguments'],src='MOCK',cache_key='MOCK')
            env={'file_sha256':{str(HERE/'test_runner.py'):m.u.sha(HERE/'test_runner.py')},'boundary':'MOCK ONLY'}
            query=types.SimpleNamespace(args=m.DRIVER_QUERY_COMMAND,returncode=0,stdout='MOCK_UUID, mock, mock',stderr='')
            with patch.dict(sys.modules,mods),patch.object(m.u,'checked_module',return_value=types.SimpleNamespace(rbe_triton=fn,get_freq_multi_tokens=fn)),patch.object(m.u,'compiler_environment',return_value=env),patch.object(m.subprocess,'run',return_value=query),patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
                code,result=m.collect(REF,out,'run',None,'cuda:0')
            r=m.u.load(out/'report.json')
            if r['status']=='CHARACTERIZATION_NUMERICS_UNBOUND':self.assertEqual(v.verify(out)['records'],20)
            return code,r
    def test_31_synthetic_runtime_full20(self):
        code,r=self.runtime();self.assertEqual(code,0);self.assertEqual(len(r['records']),20);self.assertFalse(r['numeric_acceptance']);self.assertTrue(all(x['comparison']['numeric_comparison']=='NOT_ESTABLISHED' for x in r['records']))
    def test_32_synthetic_frame_failure(self):
        code,r=self.runtime('frame');self.assertEqual(code,1);self.assertTrue(all(x['comparison']['frame_violations'] for x in r['records']))
    def test_33_synthetic_wrong_origin_no_launch(self):
        code,r=self.runtime('origin');self.assertEqual(code,1);self.assertFalse(r['native_launch_attempted'])
    def test_34_synthetic_launch_failure_preserved(self):
        code,r=self.runtime('launch');self.assertEqual(code,1);self.assertEqual(r['status'],'EXECUTION_ERROR');self.assertFalse(r['native_execution'])
    def test_35_requested_architecture_cannot_disappear(self):
        def change(q,r):
            r['resolved_plan']['archs']=[90];r['resolved_plan']['expected_records']=8
            r['records']=[x for x in r['records'] if x['target']['arch']==90]
        self.mutate(change)
    def test_36_requested_architecture_order_preserved(self):
        self.mutate(lambda q,r:r['resolved_plan'].__setitem__('archs',[90,80]))
    def test_37_offline_environment_binding_cannot_be_true(self):
        self.mutate(lambda q,r:r.__setitem__('environment_binding_complete',True))
    def test_38_notrun_environment_binding_cannot_be_true(self):
        self.mutate(lambda q,r:r.__setitem__('environment_binding_complete',True),GPU)
    def environment_report(self):
        return {'mode':'run','native_execution':True,'device_uuid':'GPU-CPU-TEST',
          'driver_query':{'command':m.DRIVER_QUERY_COMMAND,'exit_code':0,
            'stdout':'GPU-CPU-TEST, TEST-DRIVER, CPU TEST DOUBLE\n'}}
    def test_39_native_environment_exact_uuid_valid(self):
        self.assertTrue(m.environment_binding_complete(self.environment_report()))
    def test_40_native_environment_wrong_uuid(self):
        r=self.environment_report();r['device_uuid']='GPU-DIFFERENT';self.assertFalse(m.environment_binding_complete(r))
    def test_41_native_environment_failed_query(self):
        r=self.environment_report();r['driver_query']['exit_code']=1;self.assertFalse(m.environment_binding_complete(r))
    def test_42_native_environment_wrong_query(self):
        r=self.environment_report();r['driver_query']['command']=['different'];self.assertFalse(m.environment_binding_complete(r))
    def test_43_native_environment_empty_driver(self):
        r=self.environment_report();r['driver_query']['stdout']='GPU-CPU-TEST, , name';self.assertFalse(m.environment_binding_complete(r))
    def test_44_native_environment_requires_execution(self):
        r=self.environment_report();r['native_execution']=False;self.assertFalse(m.environment_binding_complete(r))
    def test_45_synthetic_runtime_environment_verified(self):
        code,r=self.runtime();self.assertEqual(code,0);self.assertTrue(r['environment_binding_complete'])
if __name__=='__main__':unittest.main(verbosity=2)
