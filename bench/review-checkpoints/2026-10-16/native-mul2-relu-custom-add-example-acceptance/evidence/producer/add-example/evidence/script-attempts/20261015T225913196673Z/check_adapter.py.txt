"""Isolated synthetic integration tests; no actual root approval is supplied."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,shutil,tempfile,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from authority_adapter import extend_authority
from exact_numerical_bindings import collect,extend_queue,validate_exact_queue
from exact_scope_gate import validate_exact_case_links
from adapter_common import canonical
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
delta=load(W/'evidence/numerical-authority-delta.json');modules=next(iter(delta['provenance'].values()))['lean_source_sha256']
legacy,previous,_=collect(R);baseline=deepcopy(previous);rejected=[]
with tempfile.TemporaryDirectory(prefix='add-example-adapter-',dir='/tmp') as tmp:
 root=Path(tmp);package=root/'checkpoint';package.mkdir()
 retained={'evidence/source-check.json':sha(W/'evidence/source-check.json'),'evidence/source-prefix.py':sha(W/'evidence/source-prefix.py'),'evidence/inherited/typed-host-numerical-proposal.json':sha(W/'evidence/inherited/typed-host-numerical-proposal.json'),'evidence/numerical-authority-delta.json':sha(W/'evidence/numerical-authority-delta.json'),'evidence/dependencies-after.json':sha(W/'evidence/dependencies-after.json')}
 for ref in delta['dependency_snapshots']:
  retained[ref['path']]=ref['sha256']
  snapshot=load(W/ref['path'])
  for mod,row in snapshot['project'].items():copy(Path(row['source']),root/(mod.replace('.','/')+'.lean'))
 for ref in [e['receipt'] for e in delta['native_evidence']]:
  retained[ref['path']]=ref['sha256'];nr=load(W/ref['path'])
  for p,h in nr['artifacts'].items():retained[(Path(ref['path']).parent/p).as_posix()]=h
 for p,h in modules.items():copy(W/p,root/p);copy(W/p,package/(p+'.txt'))
 copy(R/delta['original_source']['path'],root/delta['original_source']['path'])
 for p in retained:copy(W/p,package/p)
 receipt={'frozen':True,'modules':modules,'checkpoint_module_source_paths':{p:p+'.txt' for p in modules},'artifacts':retained}
 receiptpath=package/'evidence/receipt.json';reviewpath=root/'synthetic-root-review.json'
 review={'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','synthetic_fixture_only':True,'not_an_actual_review':True}
 def repin():
  dump(receiptpath,receipt);review['producer_receipt_sha256']=sha(receiptpath);dump(reviewpath,review);return sha(receiptpath),sha(reviewpath)
 rh,vh=repin()
 def extend(prior=previous,receipt_hash=None,review_hash=None):return extend_authority(root,prior,'checkpoint',receipt_hash or rh,'synthetic-root-review.json',review_hash or vh)
 out=extend();assert previous==baseline
 proposed=extend_queue(legacy,out);validate_exact_queue(proposed,out,legacy)
 def reject(name,f):
  try:f()
  except (ValueError,KeyError,OSError,AssertionError):rejected.append(name)
  else:raise AssertionError('tamper accepted: '+name)
 reject('wrong_producer_pin',lambda:extend(receipt_hash='0'*64))
 reject('wrong_review_pin',lambda:extend(review_hash='0'*64))
 for name,p in {'changed_current_module':root/next(iter(modules)),'changed_checkpoint_module':package/(next(iter(modules))+'.txt'),'changed_python':root/delta['original_source']['path'],'changed_native_ttir':package/next(p for p in retained if p.endswith('.ttir')),'changed_dependency_snapshot':package/'evidence/dependencies-before.json','changed_imported_source':root/'bench/tritonbench_g/vector_addition/Proofs/FiniteRoutes.lean'}.items():
  saved=p.read_bytes();p.write_bytes(saved+b'\n');reject(name,extend);p.write_bytes(saved)
 conflicting=deepcopy(previous);conflicting['input_sha256']['repo:'+next(iter(modules))]='0'*64
 reject('conflicting_existing_input',lambda:extend(prior=conflicting))
 saved=deepcopy(review);review['reviewer']='producer';dump(reviewpath,review);reject('producer_cannot_approve',lambda:extend(review_hash=sha(reviewpath)));review=saved;dump(reviewpath,review)
 for name,mutate in [('missing_binding',lambda q:q['rules'][-1]['scope_bindings'].pop()),('altered_callback',lambda q:q['rules'][-1]['scope_bindings'][0].update(callbacks=['unrelated.callback']))]:
  changed=deepcopy(proposed);mutate(changed);reject(name,lambda:validate_exact_queue(changed,out,legacy))
 new=set(out['input_sha256'])-set(previous['input_sha256']);assert all(p.startswith('repo:') and '/.lake/' not in p for p in new)
 payload={k:v for k,v in proposed.items() if k!='exact_numerical_inputs'}
 payload['exact_numerical_inputs_pending_independent_review']=True
 dump(W/'evidence/queue-payload-proposal.json',payload)
 missing=0
 for table in delta['requirements']:
  proof=(table['declaration']['file'],table['declaration']['declaration'])
  exact={proof:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in table['cases']}}
  links={}
  for case in table['cases']:
   for ref in case['required_binding_refs']:links.setdefault(ref['callback'],set()).add((ref['assumption_id'],ref['scope_binding_id']))
  for key,case in exact[proof].items():
   applicable={(r['assumption_id'],r['scope_binding_id']) for r in case['required_binding_refs']}
   validate_exact_case_links(exact,proof,key,applicable,links)
   for omitted in applicable:
    try:validate_exact_case_links(exact,proof,key,applicable-{omitted},links)
    except ValueError:missing+=1
    else:raise AssertionError('omitted exact port accepted')
 assert missing==len(delta['provenance'])
 dump(W/'evidence/adapter-check.json',{'status':'passed','positive':'exact unchanged queue checker accepts synthetic reviewed integration','rejected_tampers':rejected,'review_fixture':'isolated synthetic fixture only; not actual approval','portable_new_authority_inputs':len(new),'omitted_port_mutations_rejected':missing,'case_links_checked':delta['summary']['declaration_cases'],'completion_increment':0})
 print(json.dumps({'status':'passed','negative_checks':len(rejected),'ports':len(delta['provenance']),'cases':delta['summary']['declaration_cases']}))
