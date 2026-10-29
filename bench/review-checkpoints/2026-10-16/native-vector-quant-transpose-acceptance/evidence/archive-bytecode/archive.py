from pathlib import Path
import copy,json,sys,shutil
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance'
sys.path.insert(0,str(R/'bench'));from adapter_common import load,hash_file,write_json
assert not (B/'validation.json').exists()
p=B/'evidence/producer/quant-transpose/evidence/pre-quant-baseline/__pycache__/reviewed_numerical_extensions.cpython-312.pyc'
rel=p.relative_to(B).as_posix();index=load(B/'evidence-index.json')
expected=index['retained_exact_artifacts'][rel];assert hash_file(p)==expected
receipt=load(B/'evidence/producer/quant-transpose/evidence/receipt.json')
assert receipt['artifacts']['evidence/pre-quant-baseline/__pycache__/reviewed_numerical_extensions.cpython-312.pyc']==expected
D=B/'evidence/archive-bytecode';assert not D.exists();D.mkdir()
for src,name in [(T/'finalize_native_vector_quant_integration.py','failed-finalize-attempt001.py'),(T/'native-vector-quant-finalize.log','failed-finalize-attempt001.log'),(B/'evidence/archive-adapter-review.json','archive-adapter-review-before.json'),(B/'evidence-index.json','evidence-index-before.json'),(R/'bench/tritonbench_g/deferred_numerical_experiments.json','numerical-queue-before.json')]:shutil.copyfile(src,D/name)
G=T/'native-vector-quant-current-global'
for src in [G/'receipt.json',G/'value-metadata.json',*G.glob('*.log')]:
 dest=D/'global-before'/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
dest=p.with_name(p.name+'.bin');assert not dest.exists();p.rename(dest);assert hash_file(dest)==expected
del index['retained_exact_artifacts'][rel];index['retained_exact_artifacts'][rel+'.bin']=expected
index['archived_producer_path_mapping'][rel]=rel+'.bin'
index['note']+=' One receipt-bound historical Python bytecode file is likewise archived as .pyc.bin, with unchanged exact bytes.'
write_json(B/'evidence-index.json',index)
adapter=R/'bench/quant_transpose_numerical_extension.py';before=hash_file(adapter);s=adapter.read_text()
old="('.olean', '.olean.private', '.olean.server', '.ilean', '.ir')";new="('.olean', '.olean.private', '.olean.server', '.ilean', '.ir', '.pyc')"
assert s.count(old)==1;adapter.write_text(s.replace(old,new))
a=load(B/'evidence/archive-adapter-review.json');assert a['quant_current_adapter_sha256']==before
a['quant_current_adapter_sha256']=hash_file(adapter)
a['change']+=' The receipt-bound pre-quant baseline Python bytecode cache uses the same exact-byte .bin relocation; no validation predicate changes.'
write_json(B/'evidence/archive-adapter-review.json',a)
shutil.copyfile(Path(__file__),D/'archive.py')
write_json(D/'receipt.json',{'reviewer':'root','reason':'The no-active-bytecode finalizer found a historical receipt-bound pyc. Preserve its bytes under a non-importable suffix rather than deleting it.','old_path':rel,'new_path':rel+'.bin','unchanged_sha256':expected,'adapter_before_sha256':before,'adapter_after_sha256':hash_file(adapter),'all_numerical_cases_rules_and_provenance_unchanged':True,'followup':'Regenerate queue input path bindings, rebind its ledger file hash, rerun relevant archive tampering test and current global gates. Earlier successful gates/tests are retained with their actual pre-relocation input hashes.'})
test=R/'bench/tests/test_exact_numerical_bindings.py';s=test.read_text()
extra='''
 def test_transpose_archived_python_bytecode_is_pinned(self):
  target=QUANT_TRANSPOSE_PACKAGE+'/evidence/pre-quant-baseline/__pycache__/reviewed_numerical_extensions.cpython-312.pyc.bin'
  with tempfile.TemporaryDirectory() as temp:
   root=clone_inputs(temp);path=root/target
   self.assertTrue(path.is_file())
   self.assertFalse(path.with_suffix('').exists())
   path.write_bytes(path.read_bytes()+b'changed archived bytecode')
   with self.assertRaises(ValueError):derive_authority(root)
'''
assert "if __name__=='__main__':" in s
test.write_text(s.replace("if __name__=='__main__':",extra+"\nif __name__=='__main__':"))
print(json.dumps({'archived_path':rel+'.bin','sha256':expected,'adapter_sha256':hash_file(adapter)},indent=2))
