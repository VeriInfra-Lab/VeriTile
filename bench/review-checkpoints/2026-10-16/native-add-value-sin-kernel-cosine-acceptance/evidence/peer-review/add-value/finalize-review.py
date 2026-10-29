from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,sysconfig
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/add-value-finite-acceptance-revision1-wip')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
expected='4ad8aff1121ae9a10d70952456aaf75f31c26c408ecc37bb8f641a9c712ef774'
assert sha(P/'evidence/receipt.json')==expected
r=load(P/'evidence/receipt.json');assert len(r['artifacts'])==434 and len(r['modules'])==7
for rel,h in r['artifacts'].items():assert sha(P/rel)==h,rel
for rel,h in r['modules'].items():assert sha(P/'after'/rel)==h,rel
final={a['module']:a for a in load(P/'evidence/current-final-rebuild.json')}
for rel,a in final.items():
 attempt=load(P/a['attempt']);assert attempt['exit_code']==0 and attempt['source_unchanged'] and attempt['source_sha256']==r['modules'][rel]
 assert sha(P/a['log'])==a['log_sha256']==attempt['log_sha256'] and sha(P/a['object'])==a['object_sha256']
s=load(W/'dependencies-before.json');assert sha(W/'dependencies-before.json')==sha(W/'dependencies-after.json')
for a in s['project'].values():assert sha(a['source'])==a['sha256'] and sha(a['olean'])==a['olean_sha256']
for a in s['external_objects'].values():assert sha(a['olean'])==a['sha256']
build=load(W/'independent-build-receipt.json')
for a in build['modules']:
 assert a['exit_code']==0 and sha(W/a['log'])==a['log_sha256']
 assert sha(W/'isolated'/a['source'])==a['sha256']==r['modules'][a['source']]
 assert sha(W/'isolated'/a['source'].replace('.lean','.olean'))==a['object_sha256']==final[a['source']]['object_sha256']
axioms=sum((W/a['log']).read_text().count('axiom footprint ⊆ standard base') for a in build['modules']);noncirc=sum((W/a['log']).read_text().count('non-circular ✓') for a in build['modules']);assert (axioms,noncirc)==(30,6)
for seq,code in [('001',1),('002',1),('003',0)]:
 a=load(W/f'regression-{seq}.json');assert a['exit_code']==code and a['source_sha256']==sha(W/f'regression-{seq}.lean.txt') and a['log_sha256']==sha(W/f'regression-{seq}.log')
assert (W/'regression-003.log').read_text().count('axiom footprint ⊆ standard base')==10
admission=W/load(W/'recorded-admission/latest.json')['receipt'];assert load(admission)['status']=='passed' and load(admission)['exact_dtype_profiles_checked']==22
evidence=['independent-build-receipt.json','dependencies-before.json','dependencies-after.json','case-and-port-review.json','adapter-tests.json','replay-comparison.json','scalar-dataflow.json','exact-extractors/receipt.json','cpu-wrapper-review.json','replay/evidence/semantics/receipt.json','replay/evidence/setup-order/receipt.json',str(admission.relative_to(W))]
for rel in evidence:
 if 'dependencies-' not in rel:assert load(W/rel)['status']=='passed',rel
assert len(load(W/'exact-extractors/receipt.json')['i32_profiles'])==16 and load(W/'exact-extractors/receipt.json')['runtime_fp32_bits']==0x41200000
assert load(W/'cpu-wrapper-review.json')['cases']==49
native=W/'replay'/load(W/'replay/evidence/native-latest.json')['receipt'];abi=W/'replay'/load(W/'replay/evidence/abi-latest.json')['path']
evidence += [str(native.relative_to(W)),str(abi.relative_to(W))]
for p in [native,abi]:
 for rel,h in load(p)['artifacts'].items():assert sha(p.parent/rel)==h,(p,rel)
assert sha(W/'replay/evidence/inherited/annotated-i32-scope-review/driver.c.txt')==sha('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')
toolpaths=[Path('/tmp/veritile-lean-toolchain/bin/lean'),Path(sys.executable).resolve(),Path(shutil.which('cc')).resolve(),Path(sysconfig.get_path('include'))/'Python.h']
cc=str(Path(shutil.which('cc')).resolve())
for flag in ['-print-prog-name=cc1','-print-prog-name=as','-print-prog-name=ld']:
 value=subprocess.check_output([cc,flag],text=True).strip();p=Path(value)
 if not p.is_absolute():p=Path(shutil.which(value) or value)
 if p.is_file():toolpaths.append(p.resolve())
import _ctypes
toolpaths += [Path(_ctypes.__file__),Path('/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py'),Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py'),Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')]
(W/'tool-dependencies.json').write_text(json.dumps({'python':sys.version,'cc_version':subprocess.check_output([cc,'--version'],text=True),'tools':{str(p):sha(p) for p in toolpaths},'dependency_closure':'dependencies-before.json','native_versions_in':str(native.relative_to(W)),'GPU_launches':0},indent=2)+'\n')
delta=load(P/'evidence/numerical-authority-delta.json')
(W/'deferred-numerical-families.json').write_text(json.dumps({'status':'explicitly_deferred_unvalidated','families':[{'id':q['id'],'required_validation':q['required_validation'],'port_count':len(q['scope_bindings'])} for q in delta['rules']],'exact_numeric_ports':41,'producer_delta_sha256':sha(P/'evidence/numerical-authority-delta.json'),'native_scalar_validation':False},indent=2)+'\n')
evidence += ['tool-dependencies.json','deferred-numerical-families.json']
for seq in ['001','002','003']:evidence += [f'regression-{seq}.json',f'regression-{seq}.lean.txt',f'regression-{seq}.log']
evidence += ['rebuild-001.py','rebuild-001.log','rebuild-002.py','rebuild-002.log']
failed=[{'kind':'reviewer_build_script','attempt':'001','source':'rebuild-001.py','log':'rebuild-001.log','reason':'Expected future archive .lean.txt paths instead of existing bound after/ sources; failed before compilation','resolved_by':'independent-build-receipt.json'},{'kind':'reviewer_regression','attempt':'001','source':'regression-001.lean.txt','log':'regression-001.log','reason':'Opened Numerics name ambiguous with TypedHost.Numerics','resolved_by':'regression-003.json'},{'kind':'reviewer_regression','attempt':'002','source':'regression-002.lean.txt','log':'regression-002.log','reason':'Boolean-only branch needed reduction before decide to remove the unused Numerics variable','resolved_by':'regression-003.json'}]
report={'schema':1,'reviewer':'independent_peer','agent':'/root/mul2_independent_review','task':'add_value','revision':'actual-i32-source-revision1','status':'substantively_reviewed_for_finite_scope_integration','producer_receipt_sha256':expected,'public_statement_sha256':load(W/'case-and-port-review.json')['public_statement_sha256'],'candidate':str(P),'blockers':[],'recommendation':'Finite original-source integration supported with 15 exact scalar conformance families deferred. Root review, exact pins and repository gates remain required.','acceptance_increment':0,'schedule_boundary':'Public host uses List.range(grid n); underlying grid theorem supports arbitrary pid lists. Disjoint stores commute bytewise. No all-native-schedulers or scheduler-implementation claim.','verification':{'modules_rebuilt':7,'producer_object_hashes_reproduced':7,'candidate_named_clean_axiom_checks':30,'candidate_noncircular_checks':6,'independent_regression_axiom_checks':10,'recorded_admission_axiom_checks':1,'project_dependencies':201,'direct_external_objects':37,'producer_source_object_log_record_sets_verified':7,'frozen_candidate_artifacts_unchanged':434,'original_dtype_profiles':22,'dtype_backend_compiles':15,'binder_rejections':5,'architecture_rejections':2,'extent_backend_profiles':14,'equal_IR_artifacts':145,'scalar_dataflow_profiles':15,'compiled_C_pointer_checks':1425408,'exact_extractI32_profiles':16,'exact_extractFP32_argument10_bits':0x41200000,'signed64_overflow_rejected':True,'exact_wrapper_AST_stub_cases':18,'small_CPU_metadata_cases':6,'actual_CPU_wrapper_allocation_assertion_cases':49,'exact_signed_grid_parser_checks':7,'scope_cases':238,'exact_numeric_ports':41,'numeric_families':15,'omission_mutations_rejected':41,'adapter_mutations_rejected':20,'gpu_launches':0,'large_allocations':0,'native_scalar_validation':False},'source_review':{'path':'source-review.md','sha256':sha(W/'source-review.md')},'primary_evidence':{p:sha(W/p) for p in evidence},'retained_failed_review_attempts':failed,'candidate_and_repository_dependencies_unchanged':True}
rp=W/'peer-review.json';assert not rp.exists();rp.write_text(json.dumps(report,indent=2)+'\n')
receipt_path=W/'review-receipt.json';assert not receipt_path.exists()
artifacts={str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and not p.is_symlink() and p!=receipt_path and not any(part in ['cache','tmp','__pycache__','adapter-fixture'] for part in p.relative_to(W).parts)}
receipt={'schema':1,'task':'add_value','revision':'actual-i32-source-revision1','kind':'immutable_independent_source_review','frozen':True,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':report['status'],'producer_receipt_sha256':expected,'review':{'path':'peer-review.json','sha256':sha(rp)},'artifacts':artifacts,'excluded_generated_paths':['read-only dependency object symlinks','**/cache/**','**/tmp/**','**/__pycache__/**','adapter-fixture/** (duplicate producer inputs and synthetic fixture; scripts/report retained)'],'gpu_launches':0,'acceptance_increment':0}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
for rel,h in artifacts.items():assert sha(W/rel)==h,rel
print(json.dumps({'review':str(rp),'review_sha256':sha(rp),'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'artifacts':len(artifacts),'blockers':0}))
