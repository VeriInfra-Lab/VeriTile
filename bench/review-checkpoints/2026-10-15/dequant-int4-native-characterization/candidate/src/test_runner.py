import collections,copy,hashlib,importlib.util,json,os,py_compile,shutil,subprocess,sys,tempfile,time,types,unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
m=types.ModuleType('checked_dequant_test_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m
exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)
v=types.ModuleType('checked_dequant_test_verifier');v.__file__=str(HERE/'verify_capture.py');sys.modules[v.__name__]=v
exec(compile((HERE/'verify_capture.py').read_bytes(),v.__file__,'exec'),v.__dict__)
REF=Path(os.environ.get('DEQUANT_TEST_REFERENCE',str(HERE.parent/'reference-final')));OFFLINE=Path(os.environ.get('DEQUANT_TEST_OFFLINE',str(HERE.parent/'offline-final')));NO_GPU=Path(os.environ.get('DEQUANT_TEST_NO_GPU',str(HERE.parent/'gpu-attempt-final')))
def synthetic_runtime(out,refs,fault=None):
    """CPU-only test double. Never retained as a GPU receipt."""
    torch=types.ModuleType('torch');torch.__version__='TEST_DOUBLE';torch.version=types.SimpleNamespace(cuda='TEST_DOUBLE')
    torch.uint16='torch.uint16';torch.int32='torch.int32';torch.float16='torch.float16'
    nextaddr=[4096];allocs=[]
    class Buffer:
        def __init__(self,words,dtype,owner=None,offset=0):
            self.dtype=dtype;self.offset=offset;self.owner=owner or self
            if owner is None:self.words=list(words);self.address=nextaddr[0];nextaddr[0]+=1<<20;allocs.append(self)
        def view(self,dtype):return Buffer(None,dtype,self.owner,self.offset)
        def __getitem__(self,sl):return Buffer(None,self.dtype,self.owner,self.offset+sl.start)
        def data_ptr(self):return self.owner.address+self.offset*self.element_size()+(1 if fault=='origin' and self.offset else 0)
        def numel(self):return len(self.owner.words)-self.offset
        def element_size(self):return 4 if self.dtype==torch.int32 else 2
        def cpu(self):return self
        def tolist(self):return self.owner.words[self.offset:]
    torch.tensor=lambda words,dtype,device:Buffer(words,dtype)
    torch.cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda d:None,current_device=lambda:0,
      get_device_properties=lambda i:types.SimpleNamespace(major=8,minor=0,uuid='MOCK-UUID',name='SYNTHETIC CPU TEST DOUBLE'),synchronize=lambda:None)
    triton=types.ModuleType('triton');triton.__version__='TEST_DOUBLE'
    compiler=types.ModuleType('triton.compiler');backends=types.ModuleType('triton.backends');backend=types.ModuleType('triton.backends.compiler')
    class ASTSource:
        def __init__(self,fn,signature,constants):self.fn=fn;self.signature=signature;self.constants=constants
    compiler.ASTSource=ASTSource;backend.GPUTarget=lambda *a:a
    class Kernel:
        def __init__(self,source,options):
            self.source=source;self.metadata=options;self.asm={k:'CPU-ONLY TEST DOUBLE' for k in ['ttir','ttgir','llir','ptx','cubin']}
        def __getitem__(self,grid):
            def launch(*args):
                if fault=='launch':raise RuntimeError('synthetic launch failure')
                assert grid==(1,1,1) and len(args)==19
                plan=json.loads((out/'preselected-plan.json').read_text());assert plan['selected_profile_by_fusion']
                cid='nogroups_true' if self.source.constants['NO_GROUPS'] else 'nogroups_false'
                profile=m.PROFILES[2] if self.metadata['enable_fp_fusion'] else m.PROFILES[1]
                c=args[2];c.owner.words[:]=refs[cid][profile]
                if fault=='input':args[0].owner.words[-1]^=1
                if fault=='guard':c.owner.words[0]^=1
            return launch
    triton.compile=lambda source,target,options:Kernel(source,options)
    return {'triton':triton,'triton.compiler':compiler,'triton.backends':backends,'triton.backends.compiler':backend,'torch':torch}

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.r,cls.cf,cls.allrefs=m.validate(REF)
    def setUp(self):
        self.cid='nogroups_false';self.case=copy.deepcopy(self.cf['cases'][self.cid]);self.refs=self.allrefs[self.cid]
        self.before={n:list(a['raw_words']) for n,a in self.case['allocations'].items()}
    def actual(self,profile):
        x=copy.deepcopy(self.before);x['c_ptr']=self.refs[profile].copy();return x
    def mutate_reference(self,fn):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);fn(q)
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):m.validate(q)
    def mutate_capture(self,fn,source=OFFLINE):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'capture';shutil.copytree(source,q,ignore=shutil.ignore_patterns('cache','pycache'));r=m.load(q/'report.json');fn(q,r);m.dump(q/'report.json',r)
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):v.verify(q)
    def test_01_all_six_full_C_vectors(self):
        expected={'nogroups_false':[0x5401,0x5401,0x5400],'nogroups_true':[0x5402,0x5402,0x5401]}
        for cid,case in self.cf['cases'].items():
            for profile,word in zip(m.PROFILES,expected[cid]):
                out=m.oracle.expected(case,profile);self.assertEqual(len(out),8208)
                self.assertEqual(out[8:8200],[word]*8192);self.assertEqual(out[:8]+out[8200:],[0x7e55]*16)
                self.assertEqual(out,case['expected_full_c_allocations_by_software_graph'][profile])
    def test_02_each_match_remains_characterization(self):
        for n in m.PROFILES:
            x=m.oracle.compare(self.case,self.before,self.actual(n),self.refs,n)
            self.assertEqual(x['status'],'CHARACTERIZATION');self.assertFalse(x['numeric_acceptance']);self.assertFalse(x['universal_native_equality_proved'])
            self.assertEqual(x['selected_profile_result'],'MATCH_ON_FIXED_CASE')
    def test_03_any_other_match_cannot_satisfy_selected(self):
        x=m.oracle.compare(self.case,self.before,self.actual(m.PROFILES[2]),self.refs,m.PROFILES[1])
        self.assertTrue(any(x['profile_matches_on_fixed_case'].values()));self.assertEqual(x['selected_profile_result'],'MISMATCH_ON_FIXED_CASE')
    def test_04_unfused_graphs_not_discriminated(self):
        x=m.oracle.compare(self.case,self.before,self.actual(m.PROFILES[0]),self.refs,m.PROFILES[0]);self.assertEqual(sum(x['profile_matches_on_fixed_case'].values()),2)
        self.assertIn('does not distinguish',x['limitation'])
    def test_05_prefix_and_suffix_frame_rejected(self):
        for i in [0,7,8200,8207]:
            x=self.actual(m.PROFILES[0]);x['c_ptr'][i]^=1
            r=m.oracle.compare(self.case,self.before,x,self.refs,m.PROFILES[0]);self.assertTrue(r['output_frame_violations']);self.assertFalse(any(r['profile_matches_on_fixed_case'].values()))
    def test_06_every_input_including_unused_capacity_checked(self):
        for name in self.before:
            if name=='c_ptr':continue
            for i in [0,len(self.before[name])-1]:
                x=self.actual(m.PROFILES[0]);x[name][i]^=1
                r=m.oracle.compare(self.case,self.before,x,self.refs,m.PROFILES[0]);self.assertTrue(r['input_mutations']);self.assertFalse(any(r['profile_matches_on_fixed_case'].values()))
    def test_07_wrong_before_rejected(self):
        b=copy.deepcopy(self.before);b['c_ptr'][0]^=1
        with self.assertRaises(ValueError):m.oracle.compare(self.case,b,self.actual(m.PROFILES[0]),self.refs,m.PROFILES[0])
    def test_08_missing_allocation_rejected(self):
        x=self.actual(m.PROFILES[0]);del x['zeros_ptr']
        with self.assertRaises(ValueError):m.oracle.compare(self.case,self.before,x,self.refs,m.PROFILES[0])
    def test_09_output_short_or_invalid_word_rejected(self):
        for change in ['pop','word']:
            x=self.actual(m.PROFILES[0])
            if change=='pop':x['c_ptr'].pop()
            else:x['c_ptr'][8]=1<<16
            with self.assertRaises(ValueError):m.oracle.compare(self.case,self.before,x,self.refs,m.PROFILES[0])
    def test_10_selection_required_both_branches(self):
        for selected in [None,{}, {'fusion_off':m.PROFILES[0]},{'fusion_off':'native','fusion_on':m.PROFILES[2]}]:
            with self.assertRaises(ValueError):m.checked_selection(selected)
        self.assertEqual(m.checked_selection({'fusion_off':m.PROFILES[0],'fusion_on':m.PROFILES[2]})['fusion_on'],m.PROFILES[2])
    def test_11_unknown_or_unselected_profile_rejected(self):
        with self.assertRaises(ValueError):m.oracle.expected(self.case,'native')
        with self.assertRaises(ValueError):m.oracle.compare(self.case,self.before,self.actual(m.PROFILES[0]),self.refs,None)
    def test_12_complete_matrix_not_sample(self):
        x=self.actual(m.PROFILES[0]);x['c_ptr'][8199]^=1
        r=m.oracle.compare(self.case,self.before,x,self.refs,m.PROFILES[0]);self.assertEqual(r['selected_profile_result'],'MISMATCH_ON_FIXED_CASE');self.assertEqual(r['compared_C_words'],8208)
    def test_13_scalar_fusion_discriminator(self):
        self.assertEqual(m.oracle._dequant(3,2,0x3c01,False),0x3c02)
        self.assertEqual(m.oracle._dequant(3,2,0x3c01,True),0x3c01)
    def test_14_rne_ties_and_subnormal(self):
        self.assertEqual(m.oracle.r32(Q((1<<24)+1)),0x4b800000);self.assertEqual(m.oracle.r32(Q((1<<24)+3)),0x4b800002)
        self.assertEqual(m.oracle.rhalf(Q(65504)),0x7bff);self.assertEqual(m.oracle.rhalf(Q(65520)),0x7c00)
        self.assertEqual(m.oracle.rhalf(Q(1,1<<25)),0);self.assertEqual(m.oracle.rhalf(Q(3,1<<25)),2);self.assertEqual(m.oracle.r32(Q(1,1<<150)),0)
    def test_15_finite_half_roundtrip(self):
        for b in range(65536):
            if (b>>10)&31==31 or b==0x8000:continue
            self.assertEqual(m.oracle.rhalf(m.oracle.half(b)),b)
    def test_16_nonfinite_oracle_not_claimed(self):
        for b in [0x7c00,0xfc00,0x7fff]:
            with self.assertRaises(ValueError):m.oracle.half(b)
    def test_17_raw_little_endian(self):self.assertEqual(m.raw_bytes([0x3c00,0x7e55],16),b'\x00<U~');self.assertEqual(m.raw_bytes([0x33333333],32),b'3333')
    def test_18_explicit_signature_original_order(self):
        sig=m.signature_for(self.r['source_extraction']['arguments'],self.case['launch']['constexprs'])
        self.assertEqual(len(sig),19);self.assertEqual(list(sig)[:5],list(m.POINTER_TYPES))
        ptrs={n:'PTR:'+n for n in m.POINTER_TYPES};vals=m.runtime_arguments(list(sig),ptrs,self.case['runtime_args'])
        self.assertEqual(vals[:5],['PTR:'+n for n in m.POINTER_TYPES]);self.assertEqual(len(vals),19)
        self.assertTrue(all(v=='i32' for n,v in sig.items() if n not in ptrs));self.assertIn('groupsize',sig)
    def test_19_runtime_signature_missing_rejected(self):
        with self.assertRaises(ValueError):m.runtime_arguments(['a_ptr'],{},self.case['runtime_args'])
    def test_20_source_AST_and_original_autotune_member(self):
        jit,loc=m.extract((REF/'original.py').read_bytes());self.assertEqual(jit,(REF/'original_jit.py').read_bytes());self.assertEqual(len(loc['configs']),17)
        chosen={'constants':{k:v for k,v in self.case['launch']['constexprs'].items() if k!='NO_GROUPS'},'options':{'num_warps':4,'num_stages':4}};self.assertIn(chosen,loc['configs'])
    def test_21_full_legal_geometry_two_blocks(self):
        for cid,c in self.cf['cases'].items():
            self.assertEqual((c['runtime_args']['M'],c['runtime_args']['N'],c['runtime_args']['K']),(64,128,64));self.assertEqual(c['launch']['grid'],[1,1,1])
            self.assertEqual(c['launch']['constexprs']['NO_GROUPS'],c['runtime_args']['groupsize']==64)
            self.assertTrue(all(a['logical_origin_elements']==8 for a in c['allocations'].values()))
    def test_22_changed_case_hash_rejected(self):
        def f(q):x=m.load(q/'case.json');x['cases'][self.cid]['runtime_args']['K']=32;m.dump(q/'case.json',x)
        self.mutate_reference(f)
    def test_23_rebound_forged_case_cannot_escape_proof(self):
        def f(q):
            x=m.load(q/'case.json');x['cases'][self.cid]['allocations']['a_ptr']['raw_words'][8]=0
            m.dump(q/'case.json',x);m.dump(q/'evidence/experiment-case.json',x)
            r=m.load(q/'reference.json');r['case_sha256']=m.sha(q/'case.json');r['evidence_sha256']['evidence/experiment-case.json']=m.sha(q/'case.json');m.dump(q/'reference.json',r)
        self.mutate_reference(f)
    def test_24_missing_parent_receipt_rejected(self):self.mutate_reference(lambda q:(q/'evidence/parents/compiled_graphs/receipt.json').unlink())
    def test_25_root_replay_drift_rejected(self):self.mutate_reference(lambda q:(q/'root-replay-receipt.json').write_text('{}'))
    def test_26_reference_cannot_claim_gpu(self):
        def f(q):r=m.load(q/'reference.json');r['native_execution']=True;m.dump(q/'reference.json',r)
        self.mutate_reference(f)
    def test_27_bytecode_poison_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'mod.py';good=b'VALUE = 123\n';bad=b'VALUE = 999\n';stamp=int(time.time())-10
            q.write_bytes(bad);os.utime(q,(stamp,stamp));py_compile.compile(str(q),doraise=True);q.write_bytes(good);os.utime(q,(stamp,stamp))
            spec=importlib.util.spec_from_file_location('poison_control',q);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);self.assertEqual(control.VALUE,999)
            self.assertEqual(m.checked_module(q,'safe_checked',hashlib.sha256(good).hexdigest()).VALUE,123)
    def test_28_after_load_oracle_drift_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);shutil.copy2(HERE/'run.py',q/'run.py');shutil.copy2(HERE/'oracle.py',q/'oracle.py')
            code="import run; from pathlib import Path; p=Path(run.__file__).parent/'oracle.py'; p.write_bytes(p.read_bytes()+b'\\n# changed\\n'); run.tools_binding()"
            x=subprocess.run([sys.executable,'-B','-c',code],cwd=q,capture_output=True,text=True);self.assertNotEqual(x.returncode,0);self.assertIn('oracle changed after load',x.stderr)
    def test_29_eight_real_offline_captures(self):self.assertEqual(v.verify(OFFLINE)['records'],8)
    def test_30_missing_profile_record_rejected(self):self.mutate_capture(lambda q,r:r['records'].pop())
    def test_31_duplicate_profile_rejected(self):self.mutate_capture(lambda q,r:r['records'].__setitem__(1,copy.deepcopy(r['records'][0])))
    def test_32_offline_cannot_be_native(self):self.mutate_capture(lambda q,r:r.__setitem__('native_execution',True))
    def test_33_cannot_claim_numeric_acceptance(self):self.mutate_capture(lambda q,r:r.__setitem__('numeric_acceptance',True))
    def test_34_compiler_artifact_tamper_rejected(self):
        self.mutate_capture(lambda q,r:(q/next(x for x in r['records'][0]['artifact_sha256'] if x.endswith('.ptx'))).write_text('forged'))
    def test_35_selection_cannot_change_after_capture(self):self.mutate_capture(lambda q,r:r['records'][0].__setitem__('selected_profile',m.PROFILES[2]))
    def test_36_no_gpu_is_not_run(self):
        x=v.verify(NO_GPU);self.assertEqual(x['status'],'VERIFIED_NOT_RUN');self.assertFalse(x['native_execution']);self.assertEqual(x['records'],0)
    def test_37_notrun_cannot_be_pass(self):self.mutate_capture(lambda q,r:r.__setitem__('exit_code',0),NO_GPU)
    def test_38_failed_compile_not_upgraded(self):self.mutate_capture(lambda q,r:r.__setitem__('status','COMPILE_FAILED'))
    def test_39_comparison_exit_uses_preselected_profile(self):
        compare=m.oracle.compare(self.case,self.before,self.actual(m.PROFILES[2]),self.refs,m.PROFILES[0])
        row={'status':'CHARACTERIZATION','native_execution':True,'comparison':compare}
        self.assertEqual(m.selected_exit([row]),1)
        row['comparison']=m.oracle.compare(self.case,self.before,self.actual(m.PROFILES[0]),self.refs,m.PROFILES[0]);self.assertEqual(m.selected_exit([row]),0)
        with self.assertRaises(ValueError):m.selected_exit([])
    def test_40_metadata_named_fields(self):
        N=collections.namedtuple('N','enable_fp_fusion num_warps');self.assertEqual(m.serial(N(False,4)),{'enable_fp_fusion':False,'num_warps':4})
    def runtime_fixture(self,fault=None,selected=None):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'synthetic';mods=synthetic_runtime(out,self.allrefs,fault)
            fn=types.SimpleNamespace(arg_names=self.r['source_extraction']['arguments'],src=(REF/'original_jit.py').read_text().split('@triton.jit\n',1)[1],cache_key='SYNTHETIC')
            env={'file_sha256':{str(HERE/'test_runner.py'):m.sha(HERE/'test_runner.py')},'boundary':'CPU test double only'}
            query=types.SimpleNamespace(args=['MOCK'],returncode=0,stdout='MOCK-UUID, mockdriver, mockdevice',stderr='')
            selected=selected or {'fusion_off':m.PROFILES[1],'fusion_on':m.PROFILES[2]}
            with patch.dict(sys.modules,mods),patch.object(m,'checked_module',return_value=types.SimpleNamespace(matmul4_kernel=fn)),patch.object(m,'compiler_environment',return_value=env),patch.object(m.subprocess,'run',return_value=query),patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
                code,result=m.collect(REF,out,'run',None,selected,'cuda:0')
            report=m.load(out/'report.json')
            if report['status']=='CHARACTERIZATION':self.assertEqual(v.verify(out)['exit_code'],code)
            return code,result,report
    def test_41_synthetic_complete_runtime_branch(self):
        code,result,r=self.runtime_fixture();self.assertEqual(code,0);self.assertEqual(result['records'],4);self.assertEqual(r['status'],'CHARACTERIZATION');self.assertFalse(r['numeric_acceptance'])
        for row in r['records']:
            self.assertEqual(row['comparison']['compared_C_words'],8208);self.assertEqual(len(row['raw_captures']),5)
    def test_42_synthetic_wrong_selected_graph_not_rescued(self):
        code,_,r=self.runtime_fixture(selected={'fusion_off':m.PROFILES[2],'fusion_on':m.PROFILES[2]});self.assertEqual(code,1)
        self.assertTrue(any(row['comparison']['selected_profile_result']=='MISMATCH_ON_FIXED_CASE' and any(row['comparison']['profile_matches_on_fixed_case'].values()) for row in r['records']))
    def test_43_synthetic_input_mutation_and_guard(self):
        for fault in ['input','guard']:
            code,_,r=self.runtime_fixture(fault);self.assertEqual(code,1);self.assertEqual(r['status'],'CHARACTERIZATION')
            self.assertTrue(all(row['comparison']['selected_profile_result']=='MISMATCH_ON_FIXED_CASE' for row in r['records']))
    def test_44_synthetic_wrong_origin_stops_before_launch(self):
        code,_,r=self.runtime_fixture('origin');self.assertEqual(code,1);self.assertFalse(r['native_launch_attempted']);self.assertNotEqual(r['status'],'CHARACTERIZATION')
    def test_45_synthetic_launch_error_not_notrun(self):
        code,_,r=self.runtime_fixture('launch');self.assertEqual(code,1);self.assertTrue(r['native_launch_attempted']);self.assertEqual(r['status'],'EXECUTION_ERROR');self.assertFalse(r['native_execution'])
if __name__=='__main__':unittest.main(verbosity=2)
