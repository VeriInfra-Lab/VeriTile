"""Freeze exact inputs and all successful/failed attempts once, without repo writes."""
from pathlib import Path
import hashlib,json,re,sys
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
modules=load(root/'evidence/module-layout.json')['modules']
def after():
 snapshots=[root/'evidence/dependencies-before.json']+sorted((root/'evidence').glob('dependencies-added-before-*.json'))
 data={};project=set();external=set()
 for p in snapshots:
  d=load(p)
  for mod,r in d.get('project',{}).items():
   assert sha(Path(r['source']))==r['sha256'],mod
   assert sha(Path(r['olean']))==r['olean_sha256'],mod
   project.add(mod)
  for mod,r in d.get('external_objects',{}).items():
   assert sha(Path(r['olean']))==r['sha256'],mod
   external.add(mod)
  for f,h in d.get('input_sha256',{}).items():assert sha(Path(f))==h,f
  data[str(p.relative_to(root))]={'sha256':sha(p),'verified_unchanged':True}
 result={'captures':data,'project_sources_and_objects':len(project),'external_objects':len(external),'all_unchanged':True}
 dump(root/'evidence/dependencies-after.json',result);return result

def finalize():
 receipt=root/'evidence/receipt.json';assert not receipt.exists(),'A frozen receipt must never be overwritten'
 dependencies=after();attempts=[]
 inherited_receipt=root/'inherited-v1/evidence/receipt.json'
 inherited=load(inherited_receipt)
 assert sha(inherited_receipt)=='11bf46d3b53848860556a2852a96e8a5556bb37d55b0d083dfca658b7f914386'
 peer_receipt=root/'peer-review/native-review-receipt.json'
 assert sha(peer_receipt)=='30218217496d7351a19c53c30ed3032874863f6c6c896b441c99011f1686b61e'
 peer=load(peer_receipt)
 assert peer['gpu_executions']==0
 for module in modules:
  (root/(module+'.txt')).write_bytes((root/module).read_bytes())
 for p in sorted((root/'evidence/attempts').glob('*.json')):
  r=load(p);snapshot=p.with_suffix('.lean.txt');log=root/r['log']
  assert snapshot.exists() and sha(snapshot)==r['source_sha256']
  assert sha(log)==r['log_sha256'] and r['source_unchanged']
  attempts.append({**r,'record':str(p.relative_to(root)),'record_sha256':sha(p),'source_snapshot':str(snapshot.relative_to(root)),'snapshot_sha256':sha(snapshot)})
 final=attempts[-len(modules):]
 assert [str(Path(r['source']).relative_to(root)) for r in final]==modules
 for f,r in zip(modules,final):
  assert r['exit_code']==0 and r['source_sha256']==sha(root/f)
  assert not re.search(r'\berror:|warning:|sorryAx|DISALLOWED',(root/r['log']).read_text()),f
 dump(root/'evidence/verification-attempts.json',attempts)
 dump(root/'evidence/verification-results.json',final)
 delta=load(root/'evidence/numerical-authority-delta.json')
 for p in delta['provenance'].values():assert p['lean_source_sha256']=={f:sha(root/f) for f in modules}
 assert load(root/'evidence/adapter-check.json')['status']=='passed'
 assert load(root/'evidence/proposal-validation.json')['status']=='passed'
 artifacts={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink() and p.suffix not in ['.olean','.ilean'] and p!=receipt and str(p.relative_to(root)) not in modules}
 dump(receipt,{'schema_version':1,'revision':1,'frozen':True,'acceptance_increment':0,'candidate':'complete finite source-scope proposal, independent review/integration pending',
  'modules':{f:sha(root/f) for f in modules},'checkpoint_module_source_paths':{f:f+'.txt' for f in modules},
  'module_objects':{str(Path(f).with_suffix('.olean')):sha((root/f).with_suffix('.olean')) for f in modules},
  'artifacts':artifacts,'verification_attempt_count':len(attempts),'failed_attempts_preserved':sum(r['exit_code']!=0 for r in attempts),'final_clean_attempts':final,
  'dependencies':dependencies,'numerical_authority_summary':delta['summary'],
  'supersedes':{'receipt':'inherited-v1/evidence/receipt.json','sha256':sha(inherited_receipt),'preserved_unmodified':True,'was_accepted':False},
  'inherited_verification_attempt_count':inherited['verification_attempt_count'],
  'inherited_failed_attempts_preserved':inherited['failed_attempts_preserved'],
  'total_verification_attempt_count':len(attempts)+inherited['verification_attempt_count'],
  'total_failed_attempts_preserved':sum(r['exit_code']!=0 for r in attempts)+inherited['failed_attempts_preserved'],
  'additional_peer_native_evidence':{'receipt':'peer-review/native-review-receipt.json','sha256':sha(peer_receipt),
    'compiled_invocations':sum(r['status']=='compiled' for r in peer['records']),
    'rejected_invocations':sum(r['status']!='compiled' for r in peer['records']),
    'gpu_executions':0,'counts_may_overlap_inherited_profile_names':True},
  'public_specification':'QuantTransposeFinite.quant_transpose_finite_acceptance',
  'actual_address_interpreter':'QuantTransposeFinite.executeNativeAPI',
  'independent_actual_byte_result':'QuantTransposeFinite.NativeResult',
  'logical_formula_condition':'QuantTransposeFinite.ArithmeticRefines',
  'regression_examples':['QuantTransposeFinite.singleton_all_masks','QuantTransposeFinite.safe_wrapped_active_addresses',
    'QuantTransposeFinite.active_wrapping_not_refinement','QuantTransposeFinite.unsafe_wrapped_active_address'],
  'independent_review_status':'pending root substantive review and exact queue/catalog/ledger integration','numerical_conformance_validated':False,'native_backend_compiled':True,'native_compiled_profiles':18,'native_rejected_profiles':10,'gpu_executed':False})
 print(json.dumps({'receipt':str(receipt),'receipt_sha256':sha(receipt),'modules':len(modules),'attempts':len(attempts),'failed':sum(r['exit_code']!=0 for r in attempts),'authority':delta['summary']}))
if __name__=='__main__':
 if sys.argv[1]=='after':print(json.dumps(after()))
 elif sys.argv[1]=='finalize':finalize()
 else:raise SystemExit('expected after or finalize')
