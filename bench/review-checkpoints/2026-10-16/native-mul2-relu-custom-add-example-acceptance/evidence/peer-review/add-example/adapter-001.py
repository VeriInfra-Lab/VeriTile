from pathlib import Path
from copy import deepcopy
import sys,json,hashlib,shutil,importlib.util
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/add-example-finite-acceptance-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');F=W/'adapter-fixture';F.mkdir()
sys.path.insert(0,str(R/'bench'))
from exact_numerical_bindings import extend_queue,validate_exact_queue
spec=importlib.util.spec_from_file_location('reviewed_relu_adapter',P/'authority_adapter.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
d=load(P/'evidence/numerical-authority-delta.json');r=load(P/'evidence/receipt.json')
for ref in d['dependency_snapshots']:
 for mod,row in load(P/ref['path'])['project'].items():copy(Path(row['source']),F/(mod.replace('.','/')+'.lean'))
copy(R/d['original_source']['path'],F/d['original_source']['path'])
for rel in r['modules']:copy(P/rel,F/rel);copy(P/rel,F/'checkpoint'/(rel+'.txt'))
for rel in r['artifacts']:copy(P/rel,F/'checkpoint'/rel)
copy(P/'evidence/receipt.json',F/'checkpoint/evidence/receipt.json')
rp=F/'checkpoint/evidence/receipt.json';rh=sha(rp);assert rh=='83920c47c558019d291230261b96f0827376831ae091430ce6971e37a640d8f5'
review={'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','producer_receipt_sha256':rh,'synthetic_fixture_only':True,'not_an_actual_review':True}
rv=F/'synthetic-review.json';rv.write_text(json.dumps(review));vh=sha(rv)
prev={'input_sha256':{},'rules':[],'provenance':{},'requirements':[],'independent_source_reviews':[],'summary':{},'unmigrated_descriptive_followups':{}}
def extend(previous=prev,receipt=rh,reviewhash=vh,reviewpath='synthetic-review.json'):
 return a.extend_authority(F,previous,'checkpoint',receipt,reviewpath,reviewhash)
baseline=deepcopy(prev);out=extend();assert prev==baseline
legacy={'schema':1,'rules':[],'tasks':[{'task':'add_example','rule_ids':[]}],'summary':{}}
queue=extend_queue(legacy,out);validate_exact_queue(queue,out,legacy)
rejected=[]
def reject(name,fn):
 try:fn()
 except ValueError:rejected.append(name)
 else:raise AssertionError('tamper accepted: '+name)
reject('wrong_producer_pin',lambda:extend(receipt='0'*64));reject('wrong_review_pin',lambda:extend(reviewhash='0'*64))
paths={
 'changed_public_theorem':F/'bench/tritonbench_g/add_example/FiniteCorrect.lean',
 'changed_host_primitive_contract':F/'bench/tritonbench_g/add_example/Proofs/FiniteHost.lean',
 'changed_byte_mapping':F/'bench/tritonbench_g/add_example/Proofs/FinitePhysical.lean',
 'changed_frozen_module':F/'checkpoint/bench/tritonbench_g/add_example/Proofs/FiniteIndices.lean.txt',
 'changed_original_python':F/d['original_source']['path'],
 'changed_imported_source':F/'bench/tritonbench_g/vector_addition/Proofs/FiniteRoutes.lean',
 'changed_raw_ttir':F/'checkpoint'/next(iter(d['provenance'].values()))['raw_ttir']['path'],
 'changed_dependency_snapshot':F/'checkpoint/evidence/dependencies-before.json',
 'changed_runtime_driver':F/'checkpoint/evidence/runtime-boundary/driver.c',
 'changed_scope_cases':F/'checkpoint/evidence/scope-cases.json'}
for name,path in paths.items():
 old=path.read_bytes();path.write_bytes(old+b'\n');reject(name,extend);path.write_bytes(old)
badprev=deepcopy(prev);badprev['input_sha256']['repo:'+next(iter(r['modules']))]='0'*64;reject('conflicting_prior_pin',lambda:extend(previous=badprev))
for name,mut in [('producer_self_review',lambda x:x.update(reviewer='producer')),('wrong_review_subject',lambda x:x.update(producer_receipt_sha256='0'*64)),('unapproved_review_status',lambda x:x.update(status='pending'))]:
 old=rv.read_bytes();value=deepcopy(review);mut(value);rv.write_text(json.dumps(value));reject(name,lambda:extend(reviewhash=sha(rv)));rv.write_bytes(old)
reject('escaping_review_path',lambda:extend(reviewpath='../synthetic-review.json'))
for name,mut in [('missing_numeric_binding',lambda x:x['rules'][-1]['scope_bindings'].pop()),('altered_numeric_callback',lambda x:x['rules'][-1]['scope_bindings'][0].update(callbacks=['wrong.callback'])),('changed_numeric_rule',lambda x:x['rules'][-1].update(required_validation='tampered'))]:
 bad=deepcopy(queue);mut(bad);reject(name,lambda:validate_exact_queue(bad,out,legacy))
result={'status':'passed','synthetic_review_only':True,'not_an_approval':True,'frozen_actual_producer_receipt_used':rh,'frozen_artifacts_copied_and_checked':len(r['artifacts']),'previous_authority_unchanged':True,'mutations_rejected':rejected,'whole_queue_equality':'passed on isolated additive fixture; repository integration remains root responsibility'}
(W/'adapter-tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
