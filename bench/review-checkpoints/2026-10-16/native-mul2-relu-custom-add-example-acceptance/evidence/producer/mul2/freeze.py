import sys
sys.dont_write_bytecode=True
from pathlib import Path
import datetime,hashlib,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
rp=W/'evidence/receipt.json';assert not rp.exists(),'Already frozen'
compiles=load(W/'evidence/final-compiles.json');assert len(compiles['modules'])==9
for m in compiles['modules']:
 for field,hkey in [('source','source_sha256'),('receipt','receipt_sha256'),('log','log_sha256'),('object','object_sha256')]:assert sha(W/m[field])==m[hkey]
before=load(W/'evidence/dependencies-before.json');assert before==load(W/'evidence/dependencies-after.json')
for e in before['project'].values():assert sha(e['source'])==e['sha256'] and sha(e['olean'])==e['olean_sha256']
for e in before['external_objects'].values():assert sha(e['olean'])==e['sha256']
modules={p:sha(W/'after'/p) for p in load(W/'evidence/staging-map.json').values()};artifacts={}
for p in sorted(W.rglob('*')):
 if p.is_symlink() or not p.is_file():continue
 rel=p.relative_to(W)
 if '__pycache__' in rel.parts or 'cache' in rel.parts or rel.parts[0]=='after' or p.name in ['task.md','receipt.sha256'] or rel==Path('evidence/receipt.json'):continue
 if rel.parts[0] in ['bench','VeriTile'] and not p.name.endswith('.lean.txt'):continue
 if p.suffix in ['.olean','.ilean','.ir']:continue
 artifacts[str(rel)]=sha(p)
receipt={'schema':1,'task':'triton_mul2','revision':2,'kind':'immutable_finite_source_acceptance_candidate','frozen':True,'status':'frozen_pending_independent_root_review','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'original_source':{'path':'bench/tritonbench_g/triton_mul2/triton_mul2.py','sha256':sha(R/'bench/tritonbench_g/triton_mul2/triton_mul2.py')},'repository_and_git_modified_by_this_package':False,'previous_frozen_packages_modified':False,
 'modules':modules,'producer_module_source_paths':{p:'after/'+p for p in modules},'checkpoint_module_source_paths':{p:p+'.txt' for p in modules},'artifacts':artifacts,'public_contracts':load(W/'evidence/public-contracts.json'),'revision_lineage':load(W/'evidence/revision-lineage.json'),'repaired_blockers':['MUL2-ACCESS-OWNERSHIP','MUL2-NATIVE-POINTER-TRANSLATION','MUL2-I32-MINIMUM-B1-GRID-CASE'],
 'named_axiom_checks':compiles['named_axiom_checks'],'independent_reference_checks':compiles['noncircular_checks'],'lean_attempts':compiles['lean_attempts'],'failed_lean_attempts':compiles['failed_lean_attempts'],'project_dependency_count':len(before['project']),'external_dependency_object_count':len(before['external_objects']),'dependency_object_count':len(before['project'])+len(before['external_objects']),
 'scope_summary':load(W/'evidence/numerical-authority-delta.json')['summary'],'native_evidence':load(W/'evidence/native-evidence-summary.json'),'authority_test':load(W/'evidence/authority-check.json'),
 'revision_evidence':{p:{'path':p,'sha256':sha(W/p)} for p in ['evidence/case-mapping-v1-v2.json','evidence/grid-case-audit.json','evidence/pointer-execution-probe.json','evidence/pointer-replay-context.json']},
 'integration_requires':'independent substantive root review pinned by receipt and review hash; regenerate exact queue through reviewed authority extension','native_functional_correctness':'not_established','native_backend_compiles':30,'native_executions':0,'acceptance_increment':0,
 'excluded_generated_or_duplicate_paths':['dependency-overlay symlinks','working active source copies captured by after modules','after/** captured by modules','__pycache__/**','native-attempts/**/cache/**','task.md','compiled objects hash-bound in final-compiles.json']}
rp.write_text(json.dumps(receipt,indent=2)+'\n');digest=sha(rp);(W/'evidence/receipt.sha256').write_text(digest+'  receipt.json\n')
print(json.dumps({'receipt_sha256':digest,'artifacts':len(artifacts),'modules':len(modules),'status':receipt['status']}))
