import copy,json,os,shutil,tempfile,unittest
from pathlib import Path
import oracle,run
HERE=Path(__file__).resolve().parent
REFERENCE=Path(os.environ.get('FP4_REFERENCE_DIR',HERE/'reference'))
class OracleTests(unittest.TestCase):
    def test_compiler_named_metadata_keeps_field_names(self):
        from collections import namedtuple
        Metadata=namedtuple('Metadata','num_warps target shared')
        target=Metadata(4,'cuda:90',0)
        self.assertEqual(run.serial(target),{'num_warps':4,'target':'cuda:90','shared':0})
    def test_high_nibble_first(self):self.assertEqual(oracle.decode_byte(0x21),[0x3f80,0x3f00])
    def test_both_zero_signs(self):self.assertEqual(oracle.decode_byte(0x80),[0x8000,0]);self.assertEqual(oracle.decode_byte(0x08),[0,0x8000])
    def test_largest_signed_values(self):self.assertEqual(oracle.decode_byte(0xf7),[0xc0c0,0x40c0])
    def test_bool_not_a_byte(self):
        with self.assertRaises(ValueError):oracle.decode_byte(True)
    def test_invalid_byte(self):
        for b in [-1,256,1.0]:
            with self.assertRaises(ValueError):oracle.decode_byte(b)
    def test_every_byte_matches_lean_producer(self):
        table=json.loads((REFERENCE/'lean-table.json').read_text())
        self.assertEqual([r['input'] for r in table['rows']],list(range(256)))
        for r in table['rows']:
            self.assertEqual(oracle.decode_byte(r['input']),r['bf16']);self.assertEqual(r['software_bf16'],r['bf16'])
    def test_profiles_cover_every_byte_each_block(self):
        for block in [16,512]:
            rows=[c for c in oracle.cases() if c['block']==block]
            seen={b for c in rows for b in c['input_words'][c['input_origin']:c['input_origin']+c['count']]}
            self.assertEqual(seen,set(range(256)))
            self.assertTrue(any(c['count']>block and c['count']%block for c in rows))
            self.assertTrue(any(c['count']==0 for c in rows))
    def test_full_masked_zero_count_preserves_frame(self):
        c=next(c for c in oracle.cases() if c['count']==0)
        self.assertEqual(oracle.expected(c),c['output_initial_words'])
    def test_correct_execution_has_no_mismatches(self):
        for c in oracle.cases():self.assertEqual(oracle.compare(c,c['input_words'],oracle.expected(c)),[])
    def test_negative_zero_loss_is_value_failure(self):
        c=next(c for c in oracle.cases() if c['count']==256);out=oracle.expected(c)
        at=c['output_origin']+2*0x80;self.assertEqual(out[at],0x8000);out[at]=0
        self.assertEqual(oracle.compare(c,c['input_words'],out)[0]['kind'],'value')
    def test_tail_write_detected(self):
        c=next(c for c in oracle.cases() if c['count']==17);out=oracle.expected(c);out[c['output_origin']+34]^=1
        self.assertEqual(oracle.compare(c,c['input_words'],out)[0]['kind'],'frame')
    def test_input_write_detected(self):
        c=oracle.cases()[0];src=list(c['input_words']);src[0]^=1
        self.assertEqual(oracle.compare(c,src,oracle.expected(c))[0]['region'],'input')
    def test_missing_guard_rejected(self):
        c=copy.deepcopy(oracle.cases()[0]);c['output_initial_words']=[]
        with self.assertRaises(ValueError):oracle.expected(c)
    def test_short_native_output_rejected(self):
        c=oracle.cases()[0]
        with self.assertRaises(ValueError):oracle.compare(c,c['input_words'],[])
    def test_altered_profile_rejected(self):
        c=copy.deepcopy(oracle.cases()[0]);c['block']=32
        with self.assertRaises(ValueError):oracle.expected(c)
    def test_original_jit_keeps_exact_decorated_segments(self):
        data=(REFERENCE/'original.py').read_bytes()
        jit,locations=run.extract_jit(data)
        self.assertEqual(run.sha_bytes(data),run.SOURCE_SHA)
        self.assertNotIn(b'result_gold',jit);self.assertNotIn(b'test_triton',jit)
        for loc in locations:
            chunk=''.join(data.decode().splitlines(keepends=True)[loc['first_line']-1:loc['last_line']]).encode()
            self.assertIn(chunk,jit);self.assertEqual(run.sha_bytes(chunk),loc['source_segment_sha256'])
    def test_missing_original_definition_rejected(self):
        with self.assertRaises(ValueError):run.extract_jit(b'pass\n')

class ReferenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.ref=Path(self.temp.name)/'reference';shutil.copytree(REFERENCE,self.ref)
    def edit(self,name,mutate):
        p=self.ref/name;x=json.loads(p.read_text());mutate(x);run.dump(p,x)
    def test_delivered_reference_validates(self):run.validate(self.ref)
    def test_timestamp_valid_unbound_bytecode_is_ignored(self):
        import py_compile
        p=self.ref/'original_jit.py';original=p.read_bytes();stat=p.stat()
        marker=b'raise RuntimeError("UNBOUND_PYC")\n'
        p.write_bytes(marker+b' '*(len(original)-len(marker)))
        os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        py_compile.compile(str(p),doraise=True)
        p.write_bytes(original);os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        run.validate(self.ref)
        loaded=run.module(self.ref)
        self.assertEqual(loaded.triton_f4_to_bf16_kernel.__name__,'triton_f4_to_bf16_kernel')
    def test_timestamp_valid_oracle_cache_cannot_hide_frame_failure(self):
        import py_compile,subprocess,sys
        tools=self.ref.parent/'tools';tools.mkdir()
        for name in ['run.py','oracle.py','ModelExport.lean']:shutil.copyfile(HERE/name,tools/name)
        p=tools/'oracle.py';data=p.read_bytes();stat=p.stat()
        old=b'    return changes\n';new=b'    return []     \n'
        self.assertEqual(data.count(old),1);self.assertEqual(len(old),len(new))
        p.write_bytes(data.replace(old,new));os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        py_compile.compile(str(p),doraise=True)
        p.write_bytes(data);os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        code=('import run;from pathlib import Path;import sys;'
              'run.validate(Path(sys.argv[1]));c=run.oracle.cases()[0];'
              'out=run.oracle.expected(c);out[0]^=1;'
              'assert run.oracle.compare(c,c["input_words"],out)')
        r=subprocess.run([sys.executable,'-c',code,str(self.ref)],cwd=tools,text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    def test_expected_value_tampering_rejected(self):
        self.edit('reference.json',lambda x:x['cases'][0]['expected_output_words'].__setitem__(0,0))
        with self.assertRaisesRegex(ValueError,'Case corpus'):run.validate(self.ref)
    def test_original_body_edit_rejected(self):
        p=self.ref/'original_jit.py';p.write_bytes(p.read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'JIT body'):run.validate(self.ref)
    def test_missing_table_byte_rejected_even_with_new_hashes(self):
        self.edit('lean-table.json',lambda x:x['rows'].pop())
        newsha=run.sha(self.ref/'lean-table.json')
        self.edit('lean-export-receipt.json',lambda x:x.update(table_sha256=newsha))
        self.edit('reference.json',lambda x:x.update(lean_table_sha256=newsha,lean_export_receipt_sha256=run.sha(self.ref/'lean-export-receipt.json')))
        with self.assertRaisesRegex(ValueError,'Incomplete Lean table'):run.validate(self.ref)
    def test_fake_native_producer_rejected_even_with_new_hash(self):
        self.edit('lean-export-receipt.json',lambda x:x.update(native_execution=True))
        self.edit('reference.json',lambda x:x.update(lean_export_receipt_sha256=run.sha(self.ref/'lean-export-receipt.json')))
        with self.assertRaisesRegex(ValueError,'CPU Lean export'):run.validate(self.ref)
    def test_false_native_reference_claim_rejected(self):
        self.edit('reference.json',lambda x:x.update(native_execution=True))
        with self.assertRaisesRegex(ValueError,'falsely claims'):run.validate(self.ref)
if __name__=='__main__':unittest.main()
