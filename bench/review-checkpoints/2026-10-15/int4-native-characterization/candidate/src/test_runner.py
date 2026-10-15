import collections,copy,hashlib,importlib.util,json,os,py_compile,shutil,struct,subprocess,sys,tempfile,time,types,unittest
from fractions import Fraction as Q
from pathlib import Path
HERE=Path(__file__).resolve().parent
m=types.ModuleType('checked_int4_runner');m.__file__=str(HERE/'run.py');sys.modules[m.__name__]=m
exec(compile((HERE/'run.py').read_bytes(),m.__file__,'exec'),m.__dict__)
REF=HERE.parent/'reference-v3'
class Tests(unittest.TestCase):
    def setUp(self):
        self.r,self.case,self.refs=m.validate(REF)
        self.before={n:list(a['raw_words']) for n,a in self.case['allocations'].items()}
    def actual(self,profile):
        x=copy.deepcopy(self.before);x['c_ptr']=self.refs[profile].copy();return x
    def test_01_three_independent_oracle_words(self):
        self.assertEqual({n:m.oracle.expected(self.case,n)[8] for n in m.PROFILES},{'ascending_fma':0,'descending_fma':0x4000,'source_separate_dot_add':0x3c00})
    def test_02_each_match_remains_characterization(self):
        for n in m.PROFILES:
            x=m.oracle.compare(self.case,self.before,self.actual(n),self.refs,n)
            self.assertEqual(x['status'],'CHARACTERIZATION');self.assertFalse(x['numeric_acceptance']);self.assertFalse(x['universal_native_equality_proved'])
            self.assertEqual(sum(x['profile_matches_on_fixed_case'].values()),1);self.assertEqual(x['selected_profile_result'],'MATCH_ON_FIXED_CASE')
    def test_03_wrong_selected_profile_not_pass(self):
        x=m.oracle.compare(self.case,self.before,self.actual('descending_fma'),self.refs,'ascending_fma');self.assertEqual(x['selected_profile_result'],'MISMATCH_ON_FIXED_CASE')
    def test_04_frame_violation_rejected(self):
        x=self.actual('ascending_fma');x['c_ptr'][0]^=1
        r=m.oracle.compare(self.case,self.before,x,self.refs);self.assertTrue(r['output_frame_violations']);self.assertFalse(any(r['profile_matches_on_fixed_case'].values()))
    def test_05_input_mutation_rejected(self):
        x=self.actual('ascending_fma');x['a_ptr'][1]^=1
        r=m.oracle.compare(self.case,self.before,x,self.refs);self.assertTrue(r['input_mutations']);self.assertFalse(any(r['profile_matches_on_fixed_case'].values()))
    def test_06_wrong_before_rejected(self):
        b=copy.deepcopy(self.before);b['c_ptr'][0]^=1
        with self.assertRaises(ValueError):m.oracle.compare(self.case,b,self.actual('ascending_fma'),self.refs)
    def test_07_missing_allocation_rejected(self):
        x=self.actual('ascending_fma');del x['bzp_ptr']
        with self.assertRaises(ValueError):m.oracle.compare(self.case,self.before,x,self.refs)
    def test_08_length_rejected(self):
        x=self.actual('ascending_fma');x['c_ptr'].pop()
        with self.assertRaises(ValueError):m.oracle.compare(self.case,self.before,x,self.refs)
    def test_09_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):m.oracle.expected(self.case,'native')
        with self.assertRaises(ValueError):m.oracle.compare(self.case,self.before,self.actual('ascending_fma'),self.refs,'native')
    def test_10_rne_ties(self):
        self.assertEqual(m.oracle.r32(Q((1<<24)+1)),0x4b800000)
        self.assertEqual(m.oracle.r32(Q((1<<24)+3)),0x4b800002)
        self.assertEqual(m.oracle.rhalf(Q(65504)),0x7bff)
        self.assertEqual(m.oracle.rhalf(Q(65520)),0x7c00)
    def test_11_subnormal_rounding(self):
        self.assertEqual(m.oracle.rhalf(Q(1,1<<25)),0)
        self.assertEqual(m.oracle.rhalf(Q(3,1<<25)),2)
        self.assertEqual(m.oracle.r32(Q(1,1<<150)),0)
    def test_12_finite_half_roundtrip(self):
        for b in range(65536):
            if (b>>10)&31==31 or b==0x8000:continue
            self.assertEqual(m.oracle.rhalf(m.oracle.half(b)),b)
    def test_13_all_nibble_differences(self):
        for n in range(-15,16):self.assertEqual(m.oracle.half(m.oracle.rhalf(n)),n)
    def test_14_nonfinite_oracle_not_claimed(self):
        for b in [0x7c00,0xfc00,0x7fff]:
            with self.assertRaises(ValueError):m.oracle.half(b)
    def test_15_raw_word_endianness(self):
        self.assertEqual(m.raw_bytes([0x3c00,0xec00],16),b'\x00<\x00\xec')
        self.assertEqual(m.raw_bytes([-1,0x11111111],32),b'\xff'*4+b'\x11'*4)
    def test_16_namedtuple_metadata(self):
        N=collections.namedtuple('N','enable_fp_fusion num_warps');self.assertEqual(m.serial(N(False,4)),{'enable_fp_fusion':False,'num_warps':4})
    def test_17_explicit_signature_no_specialization(self):
        sig=m.signature_for(self.r['source_extraction']['arguments'],self.case['launch']['constexprs'])
        self.assertEqual(len(sig),19);self.assertEqual(sum(x.startswith('*') for x in sig.values()),5)
        self.assertEqual(list(sig)[:5],['a_ptr','b_ptr','c_ptr','bs_ptr','bzp_ptr'])
        ptrs={n:'PTR:'+n for n in m.POINTER_TYPES};vals=m.runtime_arguments(list(sig),ptrs,self.case['runtime_args'])
        self.assertEqual(vals[:5],['PTR:'+n for n in list(sig)[:5]]);self.assertEqual(len(vals),19)
        self.assertTrue(all(v=='i32' for n,v in sig.items() if n not in ptrs))
    def test_18_all_nonzero_origins(self):
        for a in self.case['allocations'].values():self.assertEqual(a['logical_origin_elements'],8)
    def test_19_exact_source_and_member(self):
        jit,loc=m.extract((REF/'original.py').read_bytes());self.assertEqual(jit,(REF/'original_jit.py').read_bytes());self.assertEqual(len(loc['configs']),36)
        chosen={'constants':self.case['launch']['constexprs'],'options':{'num_warps':4,'num_stages':4}};self.assertIn(chosen,loc['configs'])
    def test_20_unbound_case_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);x=m.load(q/'case.json');x['runtime_args']['K']=32;m.dump(q/'case.json',x)
            with self.assertRaises(ValueError):m.validate(q)
    def test_21_bytecode_poison_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'testmod.py';good=b'VALUE = 123\n';bad=b'VALUE = 999\n';stamp=int(time.time())-10
            q.write_bytes(bad);os.utime(q,(stamp,stamp));py_compile.compile(str(q),doraise=True)
            q.write_bytes(good);os.utime(q,(stamp,stamp))
            spec=importlib.util.spec_from_file_location('poison_control',q);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
            self.assertEqual(control.VALUE,999)
            safe=m.checked_module(q,'safe_checked',hashlib.sha256(good).hexdigest());self.assertEqual(safe.VALUE,123)
    def test_22_oracle_after_load_drift(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);shutil.copy2(HERE/'run.py',q/'run.py');shutil.copy2(HERE/'oracle.py',q/'oracle.py')
            code="import run; from pathlib import Path; p=Path(run.__file__).parent/'oracle.py'; p.write_bytes(p.read_bytes()+b'\\n# changed\\n'); run.tools_binding()"
            x=subprocess.run([sys.executable,'-B','-c',code],cwd=q,capture_output=True,text=True)
            self.assertNotEqual(x.returncode,0);self.assertIn('oracle changed after load',x.stderr)
    def test_23_oracle_old_pyc_cannot_replace_comparator(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td);shutil.copy2(HERE/'run.py',q/'run.py');source=(HERE/'oracle.py').read_bytes();p=q/'oracle.py';stamp=int(time.time())-10
            evil=b"raise RuntimeError('poisoned oracle pyc')\n";p.write_bytes(evil+b' '*(len(source)-len(evil)));os.utime(p,(stamp,stamp));py_compile.compile(str(p),doraise=True)
            p.write_bytes(source);os.utime(p,(stamp,stamp))
            control=subprocess.run([sys.executable,'-c','import oracle'],cwd=q,capture_output=True,text=True);self.assertNotEqual(control.returncode,0);self.assertIn('poisoned oracle pyc',control.stderr)
            code=f"import run; run.validate(__import__('pathlib').Path({str(REF)!r})); print('checked oracle')"
            x=subprocess.run([sys.executable,'-c',code],cwd=q,capture_output=True,text=True);self.assertEqual(x.returncode,0,x.stderr);self.assertIn('checked oracle',x.stdout)

    def test_24_rebound_forged_case_cannot_escape_parent(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q)
            x=m.load(q/'case.json');x['allocations']['a_ptr']['raw_words'][8]=0
            refs={n:m.oracle.expected(x,n) for n in m.PROFILES};x['expected_full_c_allocations_by_software_graph']=refs
            m.dump(q/'case.json',x);m.dump(q/'evidence/experiment-case.json',x);m.dump(q/'oracle-reference.json',refs)
            r=m.load(q/'reference.json');r['case_sha256']=m.sha(q/'case.json');r['oracle_reference_sha256']=m.sha(q/'oracle-reference.json')
            r['evidence_sha256']['evidence/experiment-case.json']=m.sha(q/'evidence/experiment-case.json');m.dump(q/'reference.json',r)
            with self.assertRaisesRegex(ValueError,'not bound by proof receipt'):m.validate(q)
    def test_25_displayed_reference_cannot_lie(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);r=m.load(q/'reference.json');r['software_profile_outputs']['ascending_fma']=0x3c00;m.dump(q/'reference.json',r)
            with self.assertRaisesRegex(ValueError,'displayed reference'):m.validate(q)
    def test_26_cpu_reference_cannot_claim_gpu(self):
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'ref';shutil.copytree(REF,q);r=m.load(q/'reference.json');r['original_gpu_execution']=True;m.dump(q/'reference.json',r)
            with self.assertRaisesRegex(ValueError,'unsupported claim'):m.validate(q)
if __name__=='__main__':unittest.main(verbosity=2)

