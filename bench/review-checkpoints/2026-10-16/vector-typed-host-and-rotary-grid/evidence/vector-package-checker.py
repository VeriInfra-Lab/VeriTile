from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,hashlib,json,re,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'vector-typed-host-wip'
sys.path.insert(0,str(R/'bench'))
from adapter_common import declarations

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
checked={}
def exact(p,h):assert sha(p)==h,str(p);checked[str(p)]=h
exact(P/'evidence/receipt.json','15aa1d46cac625669b79d29430bacb1d386b35be351080ac35edce804c986bc4')
r=load(P/'evidence/receipt.json')
for n,h in {**r['modules'],**r['module_objects'],**r['artifacts']}.items():exact(P/n,h)
deps=load(P/'evidence/dependencies-after.json');assert deps==load(P/'evidence/dependencies-before.json')
assert len(deps['project'])==194 and len(deps['external_objects'])==37
for d in deps['project'].values():
 exact(d['source'],d['sha256'])
 if 'olean' in d:exact(d['olean'],d['olean_sha256'])
 else:assert d['package'] and Path(d['source']).is_relative_to(P)
for d in deps['external_objects'].values():exact(d['olean'],d['sha256'])
for n,h in deps['input_sha256'].items():exact(n,h)
a=load(P/'evidence/verification-attempts.json');assert len(a)==14 and sum(t['exit_code']!=0 for t in a)==4
assert a[-5:]==r['final_clean_attempts']
for t in a:exact(P/t['log'],t['log_sha256']);exact(P/t['source_snapshot'],t['snapshot_sha256'])
for t in r['final_clean_attempts']:
 assert t['exit_code']==0 and t['source_sha256_before']==t['source_sha256_after']==r['modules'][t['module']]
 log=(P/t['log']).read_text();assert not re.search(r'\berror:|sorryAx|DISALLOWED|warning:',log)
route=load(P/'evidence/source-route-check.json');exact(route['original_source'],route['source_sha256']);exact(route['inherited_admission_audit'],route['inherited_audit_sha256'])
functions={n.name:n for n in ast.parse(Path(route['original_source']).read_text()).body if isinstance(n,ast.FunctionDef)}
for name,statements in route['source_statements'].items():assert [ast.unparse(n) for n in functions[name].body]==statements
rows=load(Path(route['inherited_admission_audit']).parent/'rows.json');assert len(rows)==len({(t['x'],t['y']) for t in rows})==289
assert Counter(t['finite_lean_model']['stage'] for t in rows)=={'admitted':189,'signature':64,'addition':36}
n=load(P/'evidence/numerical-bindings-proposal.json');assert n['integrated_in_repository_queue'] is False and n['native_numerical_validation'] is False
assert {(p['x_dtype'],p['y_dtype']) for p in n['profiles']}=={(p['x'],p['y']) for p in rows if p['finite_lean_model']['stage']=='admitted'}
assert Counter(p['family'] for p in n['profiles'])=={'integer':81,'floating':25,'mixed32':18,'mixedSmall':36,'wide':29}
assert len(n['profiles'])==189 and sum(len(p['operations']) for p in n['profiles'])==308
assert sorted({op['port'] for p in n['profiles'] if p['family']=='wide' for op in p['operations']})==n['fp64_callback_families'];assert len(n['fp64_callback_families'])==15
for p in n['profiles']:
 for key in ['execution_binding','host_binding']:
  bind=p[key];exact(P/bind['module'],bind['sha256'])
 for op in p['operations']:exact(op['source'],op['source_sha256'])
pub={k:v for k,v in declarations(P/'bench/tritonbench_g/vector_addition/TypedHostCorrect.lean').items() if v['kind']=='specification'};assert len(pub)==4
assert all(not (R/n).exists() for n in r['modules'])
record={'schema':1,'kind':'vector_typed_host_frozen_root_review','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'root','status':'substantive_partial_proof_review_passed_pending_integration','checked_file_sha256':checked,'source_modules':r['modules'],'public_statement_sha256':{k:v['statement_sha256'] for k,v in pub.items()},'proof_defects':[],'semantic_review':'Root read original eight-statement kernel and six-statement wrapper, retained HostWrapper, and all five new modules. Dispatch preserves all 17x17 first frontend cases; 189 admitted pairs select exactly the inherited 81 integer, 25 float, 18 mixed-fp32, 36 mixed-small and 29 fp64 actual grids. Kernel success and final values are derived, not assumed as a launch oracle. Branch-specific carrier projections, initial allocated-memory formulas, both memory frames and unselected state preservation are explicit. EmptyLikeContract binds returned metadata, freshness and old-input preservation without incorrectly copying strides. Public success/outcome require finite typed input views and defined conversions; nine fp64 integer/bool routes keep OutputDefined, and 20 floating-output routes have the inherited automatic defined-domain proof. Undefined-value policy independence is proved only in its explicit domain. Host allocation/CUDA/rejection order follows source. Native rollback, arbitrary views/finite-address applicability and whole-program/native success are not established. SourceDomain is an applicability premise, not yet a finite-address proof. The two early-error specifications are protocol contracts, with zero value-stage coverage.', 'remaining_finite_obligations':load(P/'evidence/scope-review.json')['remaining_finite_obligations'],'numerical_proposal_review':'All 189 profile identities, 308 recorded operations and 15 fp64 callback-family inventory were checked against bound source/dispatch evidence. Proposal remains unregistered and is not an independently accepted numerical law or native validation. Exact numerical queue integration remains required.','acceptance_increment':0,'new_dtype_route_increment':0,'new_numerical_registrations':0,'native_runs':0}
out=T/'vector-typed-host-root-review.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'checked_files':len(checked),'publics':len(pub),'sha256':sha(out)}))
