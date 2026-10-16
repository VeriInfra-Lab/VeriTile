import sys; sys.dont_write_bytecode = True
"""Exercise the adapter in a synthetic integration fixture, never an approval."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,shutil,sys,tempfile
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical
from exact_scope_gate import validate_exact_case_links
from exact_numerical_bindings import collect,extend_queue,validate_exact_queue
from authority_adapter import extend_triton_mul2_authority
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
delta=load(W/'evidence/numerical-authority-delta.json');deps=load(W/'evidence/dependency-closure.json')
modules=next(iter(delta['provenance'].values()))['lean_source_sha256']
legacy,previous,_=collect(R);before=deepcopy(previous)
with tempfile.TemporaryDirectory(prefix='authority-fixture-',dir=W/'evidence') as temp:
 root=Path(temp);package=root/'checkpoint';package.mkdir()
 def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 for rel in deps['project']:copy(R/rel,root/rel)
 copy(R/'bench/tritonbench_g/triton_mul2/triton_mul2.py',root/'bench/tritonbench_g/triton_mul2/triton_mul2.py')
 for rel in modules:
  copy(W/'after'/rel,root/rel);copy(W/'after'/rel,package/(rel+'.txt'))
 artifacts={}
 needed={'evidence/numerical-authority-delta.json','evidence/dependency-closure.json',
  'evidence/pointer-execution-probe.json','evidence/grid-case-audit.json',
  'evidence/case-mapping-v1-v2.json','evidence/revision-lineage.json',
  'prior-v1/evidence/receipt.json','peer-review/peer-review.json'}
 for p in delta['provenance'].values():
  for field in ['raw_ttir','frontend_receipt','producer_scalar_inventory']:needed.add(p[field]['path'])
 for rel in needed:copy(W/rel,package/rel);artifacts[rel]=sha(W/rel)
 receipt={'task':'triton_mul2','revision':2,'original_source':{'path':'bench/tritonbench_g/triton_mul2/triton_mul2.py','sha256':sha(R/'bench/tritonbench_g/triton_mul2/triton_mul2.py')},'frozen':True,'modules':modules,'checkpoint_module_source_paths':{p:p+'.txt' for p in modules},'artifacts':artifacts}
 rp=package/'evidence/receipt.json';rp.write_text(json.dumps(receipt));rh=sha(rp)
 review={'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','producer_receipt_sha256':rh,
  'synthetic_fixture_only':True,'not_an_actual_review':True}
 rv=root/'synthetic-review.json';rv.write_text(json.dumps(review));vh=sha(rv)
 extended=extend_triton_mul2_authority(root,previous,'checkpoint',rh,'synthetic-review.json',vh)
 assert previous==before
 proposed=extend_queue(legacy,extended);full=validate_exact_queue(proposed,extended,legacy)
 rejected=[]
 for name,args in [('wrong_receipt',('0'*64,vh)),('wrong_review',(rh,'0'*64))]:
  try:extend_triton_mul2_authority(root,previous,'checkpoint',args[0],'synthetic-review.json',args[1])
  except ValueError:rejected.append(name)
  else:raise AssertionError(name)
 for name,path in [('changed_helper',root/list(modules)[0]),
  ('changed_pointer_setup_helper',root/'bench/tritonbench_g/triton_mul2/Proofs/FiniteHost.lean'),
  ('changed_mapping_contract',root/'bench/tritonbench_g/triton_mul2/Proofs/FiniteMemory.lean'),
  ('changed_pointer_boundary_proof',root/'bench/tritonbench_g/triton_mul2/Proofs/FinitePointerBoundary.lean'),
  ('changed_frozen_pointer_setup_helper',package/'bench/tritonbench_g/triton_mul2/Proofs/FiniteHost.lean.txt'),
  ('changed_pointer_evidence',package/'evidence/pointer-execution-probe.json'),
  ('changed_grid_audit',package/'evidence/grid-case-audit.json'),
  ('changed_historical_receipt',package/'prior-v1/evidence/receipt.json'),
  ('changed_raw_ttir',package/next(iter(delta['provenance'].values()))['raw_ttir']['path'])]:
  old=path.read_bytes();path.write_bytes(old+b'\n')
  try:extend_triton_mul2_authority(root,previous,'checkpoint',rh,'synthetic-review.json',vh)
  except ValueError:rejected.append(name)
  else:raise AssertionError(name)
  path.write_bytes(old)
 # Even caller-supplied matching hashes cannot turn an older or unknown
 # revision into this extension's reviewed revision2 contract.
 receipt_bytes=rp.read_bytes();review_bytes=rv.read_bytes()
 for revision in [1,3]:
  altered=deepcopy(receipt);altered['revision']=revision;rp.write_text(json.dumps(altered))
  altered_review=deepcopy(review);altered_review['producer_receipt_sha256']=sha(rp);rv.write_text(json.dumps(altered_review))
  try:extend_triton_mul2_authority(root,previous,'checkpoint',sha(rp),'synthetic-review.json',sha(rv))
  except ValueError:rejected.append('wrong_revision_'+str(revision))
  else:raise AssertionError('wrong revision passed')
 rp.write_bytes(receipt_bytes);rv.write_bytes(review_bytes)
 bad=deepcopy(proposed);bad['rules'][-1]['required_validation']='changed'
 try:validate_exact_queue(bad,extended,legacy)
 except ValueError:rejected.append('changed_queue')
 else:raise AssertionError('changed queue passed')
 # A review payload with the right hash but wrong reviewer also fails.
 review['reviewer']='producer';rv.write_text(json.dumps(review))
 try:extend_triton_mul2_authority(root,previous,'checkpoint',rh,'synthetic-review.json',sha(rv))
 except ValueError:rejected.append('producer_self_review')
 else:raise AssertionError('self-review passed')
 # Candidate payload is useful for review; final repo: pins are deliberately not
 # represented as established by this temporary fixture.
 save_payload={k:v for k,v in proposed.items() if k!='exact_numerical_inputs'}
 save_payload['exact_numerical_inputs_pending_independent_review']=True
 (W/'evidence/queue-payload-proposal.json').write_text(json.dumps(save_payload,indent=2)+'\n')

table=delta['requirements'][0];key=(table['declaration']['file'],table['declaration']['declaration'])
exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in table['cases']}}
links={}
for c in table['cases']:
 for ref in c['required_binding_refs']:links.setdefault(ref['callback'],set()).add((ref['assumption_id'],ref['scope_binding_id']))
missing=0
for case,c in exact[key].items():
 refs={(r['assumption_id'],r['scope_binding_id']) for r in c['required_binding_refs']}
 validate_exact_case_links(exact,key,case,refs,links)
 for omitted in refs:
  try:validate_exact_case_links(exact,key,case,refs-{omitted},links)
  except ValueError:missing+=1
  else:raise AssertionError('missing port passed')
assert missing==64 and len(rejected)==15
new_inputs=set(extended['input_sha256'])-set(previous['input_sha256'])
assert all(k.startswith('repo:') and '/tmp/' not in k and not k.endswith('.olean') for k in new_inputs)
result={'status':'passed','synthetic_review_fixture_only':True,'independent_review':'pending','case_links_checked':len(table['cases']),
 'omitted_port_mutations_rejected':missing,'adapter_mutations_rejected':rejected,'old_authority_unchanged':True,
 'new_input_keys_portable_repo_paths':True,'revision_downgrade_and_unknown_revision_fail_closed':True,
 'unchanged_whole_queue_equality_test':full,'final_current_repository_integration':'not performed','acceptance_increment':0}
(W/'evidence/authority-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
