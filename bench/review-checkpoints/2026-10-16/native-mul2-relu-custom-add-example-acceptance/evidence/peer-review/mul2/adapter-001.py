from pathlib import Path
from copy import deepcopy
import sys,json,hashlib,shutil,importlib.util
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/triton-mul2-finite-acceptance-revision2-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');F=W/'adapter-fixture';F.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'bench'))
from exact_numerical_bindings import extend_queue,validate_exact_queue
spec=importlib.util.spec_from_file_location('reviewed_mul_adapter',P/'authority_adapter.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
d=load(P/'evidence/numerical-authority-delta.json');r=load(P/'evidence/receipt.json');dep=load(P/'evidence/dependency-closure.json')
for rel in dep['project']:copy(R/rel,F/rel)
copy(R/r['original_source']['path'],F/r['original_source']['path'])
for rel in r['modules']:copy(P/'after'/rel,F/rel);copy(P/'after'/rel,F/'checkpoint'/(rel+'.txt'))
needed={'evidence/numerical-authority-delta.json','evidence/dependency-closure.json','evidence/pointer-execution-probe.json','evidence/grid-case-audit.json','prior-v1/evidence/receipt.json'}
for e in d['provenance'].values():
 for field in ['raw_ttir','frontend_receipt','producer_scalar_inventory']:needed.add(e[field]['path'])
for rel in needed:copy(P/rel,F/'checkpoint'/rel)
synth={'revision':2,'task':'triton_mul2','original_source':r['original_source'],'frozen':True,'modules':r['modules'],'checkpoint_module_source_paths':{p:p+'.txt' for p in r['modules']},'artifacts':{p:sha(P/p) for p in needed},'synthetic_test_receipt':True}
rp=F/'checkpoint/evidence/receipt.json';rp.write_text(json.dumps(synth));rh=sha(rp)
review={'reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','producer_receipt_sha256':rh,'synthetic_fixture_only':True,'not_an_actual_review':True}
rv=F/'synthetic-review.json';rv.write_text(json.dumps(review));vh=sha(rv)
prev={'input_sha256':{},'rules':[],'provenance':{},'requirements':[],'independent_source_reviews':[],'summary':{},'unmigrated_descriptive_followups':{}}
before=deepcopy(prev);extended=a.extend_triton_mul2_authority(F,prev,'checkpoint',rh,'synthetic-review.json',vh);assert prev==before
legacy={'schema':1,'rules':[],'tasks':[{'task':'triton_mul2','rule_ids':[]}],'summary':{}}
queue=extend_queue(legacy,extended);validate_exact_queue(queue,extended,legacy)
rejected=[]
def reject(label,call):
 try:call()
 except ValueError:rejected.append(label)
 else:raise AssertionError(label)
reject('wrong_receipt',lambda:a.extend_triton_mul2_authority(F,prev,'checkpoint','0'*64,'synthetic-review.json',vh))
reject('wrong_review',lambda:a.extend_triton_mul2_authority(F,prev,'checkpoint',rh,'synthetic-review.json','0'*64))
paths=[('changed_helper',F/next(iter(r['modules']))),
 ('changed_pointer_setup_helper',F/'bench/tritonbench_g/triton_mul2/Proofs/FiniteHost.lean'),
 ('changed_mapping_contract',F/'bench/tritonbench_g/triton_mul2/Proofs/FiniteMemory.lean'),
 ('changed_pointer_boundary_proof',F/'bench/tritonbench_g/triton_mul2/Proofs/FinitePointerBoundary.lean'),
 ('changed_frozen_pointer_setup_helper',F/'checkpoint/bench/tritonbench_g/triton_mul2/Proofs/FiniteHost.lean.txt'),
 ('changed_pointer_evidence',F/'checkpoint/evidence/pointer-execution-probe.json'),
 ('changed_grid_audit',F/'checkpoint/evidence/grid-case-audit.json'),
 ('changed_historical_receipt',F/'checkpoint/prior-v1/evidence/receipt.json'),
 ('changed_raw_ttir',F/'checkpoint'/next(iter(d['provenance'].values()))['raw_ttir']['path'])]
for label,path in paths:
 original=path.read_bytes();path.write_bytes(original+b'\n');reject(label,lambda:a.extend_triton_mul2_authority(F,prev,'checkpoint',rh,'synthetic-review.json',vh));path.write_bytes(original)
for revision in [1,3]:
 original_r=rp.read_bytes();original_v=rv.read_bytes()
 trial=deepcopy(synth);trial['revision']=revision;rp.write_text(json.dumps(trial));newrh=sha(rp)
 trial_v=deepcopy(review);trial_v['producer_receipt_sha256']=newrh;rv.write_text(json.dumps(trial_v))
 reject('wrong_revision_'+str(revision),lambda:a.extend_triton_mul2_authority(F,prev,'checkpoint',newrh,'synthetic-review.json',sha(rv)))
 rp.write_bytes(original_r);rv.write_bytes(original_v)
bad=deepcopy(queue);bad['rules'][-1]['required_validation']='tampered';reject('changed_queue',lambda:validate_exact_queue(bad,extended,legacy))
review['reviewer']='producer';rv.write_text(json.dumps(review));reject('producer_self_review',lambda:a.extend_triton_mul2_authority(F,prev,'checkpoint',rh,'synthetic-review.json',sha(rv)))
assert len(rejected)==15
(W/'adapter-tests.json').write_text(json.dumps({'status':'passed','synthetic_fixture_only':True,'not_an_approval':True,'previous_authority_unchanged':True,'mutations_rejected':rejected,'whole_queue_equality':'passed on isolated additive fixture; repository integration remains root responsibility'},indent=2)+'\n')
print('15 adapter tamper mutations rejected; isolated whole-queue equality passed')
