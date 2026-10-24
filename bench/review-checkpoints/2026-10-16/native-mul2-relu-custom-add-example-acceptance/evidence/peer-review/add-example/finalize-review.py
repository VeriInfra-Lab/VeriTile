from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,sysconfig
W=Path(__file__).resolve().parent
P=Path('/tmp/veritile-standardization-20261015/add-example-finite-acceptance-wip')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
expected='83920c47c558019d291230261b96f0827376831ae091430ce6971e37a640d8f5'
assert sha(P/'evidence/receipt.json')==expected
r=load(P/'evidence/receipt.json')
for field in ['artifacts','modules','module_objects']:
 for rel,h in r[field].items():assert sha(P/rel)==h,(field,rel)
assert len(r['artifacts'])==3302 and len(r['modules'])==9
for entry in r['final_clean_attempts']:
 a=load(P/entry['record'])
 assert a['exit_code']==0 and a['source_unchanged'] and a['source_sha256']==r['modules'][entry['module']]
 assert sha(P/a['log'])==a['log_sha256']
s=load(W/'dependencies-before.json')
assert sha(W/'dependencies-before.json')==sha(W/'dependencies-after.json')
for d in s['project'].values():assert sha(d['source'])==d['sha256'] and sha(d['olean'])==d['olean_sha256']
for d in s['external_objects'].values():assert sha(d['olean'])==d['sha256']
build=load(W/'independent-build-receipt.json')
for m in build['modules']:
 assert m['exit_code']==0 and sha(W/m['log'])==m['log_sha256']
 assert m['object_sha256']==r['module_objects'][m['source'].replace('.lean','.olean')]
 assert sha(W/'isolated'/m['source'])==m['sha256']==r['modules'][m['source']]
 assert sha(W/'isolated'/m['source'].replace('.lean','.olean'))==m['object_sha256']
axioms=sum((W/m['log']).read_text().count('axiom footprint ⊆ standard base') for m in build['modules'])
noncirc=sum((W/m['log']).read_text().count('non-circular ✓') for m in build['modules'])
assert (axioms,noncirc)==(46,7)
for attempt,expected_exit in [('001',1),('002',0)]:
 a=load(W/f'regression-{attempt}.json')
 assert a['exit_code']==expected_exit
 assert a['log_sha256']==sha(W/f'regression-{attempt}.log') and a['source_sha256']==sha(W/f'regression-{attempt}.lean.txt')
assert (W/'regression-002.log').read_text().count('axiom footprint ⊆ standard base')==9
admission=W/load(W/'recorded-admission/latest.json')['receipt']
assert load(admission)['status']=='passed' and load(admission)['exact_pairs_checked']==484
for f in ['independent-build-receipt.json','case-and-port-review.json','adapter-tests.json','replay-comparison.json','driver-extractor-review.json','driver-parser-001/receipt.json','binder-001/receipt.json']:
 assert load(W/f)['status']=='passed',f
assert len(load(W/'driver-extractor-review.json')['records'])==16
assert load(W/'driver-parser-001/receipt.json')['actual_parser_tests']==7
assert len(load(W/'scalar-fingerprints.json')['records'])==189
shared=load(W/'shared-scalar-reference/receipt.json')
assert sha(W/'shared-scalar-reference/receipt.json')==load(W/'scalar-fingerprints.json')['shared_vector_receipt_sha256']
for p in (W/'shared-scalar-reference').glob('*.raw.ttir'):assert sha(p)==shared['artifacts'][p.name]
assert sha(P/'evidence/runtime-boundary/driver.c')==sha('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')

# Capture the actual executable tools and Python/C bridge headers used here.
toolpaths=[Path('/tmp/veritile-lean-toolchain/bin/lean'),Path(sys.executable).resolve(),Path(shutil.which('cc')).resolve(),Path(sysconfig.get_path('include'))/'Python.h']
cc=str(Path(shutil.which('cc')).resolve())
for flag in ['-print-prog-name=cc1','-print-prog-name=as','-print-prog-name=ld']:
 value=subprocess.check_output([cc,flag],text=True).strip();p=Path(value)
 if not p.is_absolute():p=Path(shutil.which(value) or value)
 if p.is_file():toolpaths.append(p.resolve())
import _ctypes
toolpaths.append(Path(_ctypes.__file__))
toolpaths.extend([Path('/usr/local/lib/python3.12/site-packages/triton/runtime/jit.py'),Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py'),Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')])
toolinfo={'python':sys.version,'cc_version':subprocess.check_output([cc,'--version'],text=True),'python_include':sysconfig.get_path('include'),'tools':{str(p):sha(p) for p in toolpaths},'dependency_closure':'dependencies-before.json','native_tool_versions':'fresh frontend/backend/host receipt files under replay/evidence','GPU_launches':0}
(W/'tool-dependencies.json').write_text(json.dumps(toolinfo,indent=2)+'\n')
checks=load(W/'case-and-port-review.json')
delta=load(P/'evidence/numerical-authority-delta.json')
deferred={'status':'explicitly_deferred_unvalidated','families':[{'id':q['id'],'required_validation':q['required_validation'],'port_count':len(q['scope_bindings'])} for q in delta['rules']],'exact_numeric_ports':296,'producer_delta_sha256':sha(P/'evidence/numerical-authority-delta.json'),'native_scalar_validation':False}
(W/'deferred-numerical-families.json').write_text(json.dumps(deferred,indent=2)+'\n')
evidence=['independent-build-receipt.json','dependencies-before.json','dependencies-after.json','case-and-port-review.json','adapter-tests.json','replay-comparison.json','scalar-fingerprints.json','replay/source-identity.json','driver-extractor-review.json','driver-parser-001/receipt.json','binder-001/receipt.json','tool-dependencies.json','deferred-numerical-families.json',str(admission.relative_to(W)),'regression-001.json','regression-001.lean.txt','regression-001.log','regression-002.json','regression-002.lean.txt','regression-002.log','recorded-admission-001.py','recorded-admission-001.log','recorded-admission-002.py','recorded-admission-002.log']
failed=[{'kind':'reviewer_regression','attempt':'001','source':'regression-001.lean.txt','log':'regression-001.log','reason':'Ambiguous opened MemoryPrimitives namespace, nodup_range API invocation, and opaque classical decision instance','resolved_by':'regression-002.json'}, {'kind':'reviewer_recorded_admission_script','attempt':'001','source':'recorded-admission-001.py','log':'recorded-admission-001.log','reason':'Requested absent backend-latest.json before compilation; package uses timestamped backend receipts','resolved_by':str(admission.relative_to(W))}]
report={'schema':1,'reviewer':'independent_peer','agent':'/root/mul2_independent_review','task':'add_example','revision':1,'status':'substantively_reviewed_for_finite_scope_integration','producer_receipt_sha256':expected,'public_statement_sha256':checks['public_statement_sha256'],'candidate':str(P),'blockers':[],'recommendation':'Finite original-source integration is supported with 37 exact scalar conformance families deferred. Root substantive review, exact pins and repository integration gates remain required.','acceptance_increment':0,'verification':{'modules_rebuilt':9,'producer_object_hashes_reproduced':9,'candidate_named_clean_axiom_checks':axioms,'candidate_noncircular_checks':noncirc,'independent_regression_axiom_checks':9,'recorded_admission_axiom_checks':1,'recorded_admission_pairs':484,'project_dependencies':len(s['project']),'direct_external_objects':len(s['external_objects']),'producer_source_object_log_record_sets_verified':9,'frozen_candidate_artifacts_unchanged':3302,'frontend_dtype_pairs_replayed':484,'backend_profiles_replayed':189,'native_backend_compiled':183,'backend_rejected':6,'equal_IR_artifacts':921,'scalar_fingerprints_compared':189,'real_JIT_binder_dtype_pairs':484,'real_extractI32_extent_checks':16,'real_extractI32_signed64_overflow_rejection':True,'real_signed_grid_parser_checks':7,'exact_wrapper_CPU_stub_cases':73,'zero_allocations_returned':72,'actual_allocation_errors':1,'scope_cases':1700,'exact_numeric_ports':296,'numeric_families':37,'omission_mutations_rejected':296,'adapter_mutations_rejected':20,'gpu_launches':0,'native_scalar_validation':False},'source_review':{'path':'source-review.md','sha256':sha(W/'source-review.md')},'primary_evidence':{p:sha(W/p) for p in evidence},'retained_failed_review_attempts':failed,'candidate_and_repository_dependencies_unchanged':True}
rp=W/'peer-review.json';assert not rp.exists();rp.write_text(json.dumps(report,indent=2)+'\n')
receipt_path=W/'review-receipt.json';assert not receipt_path.exists()
artifacts={str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and not p.is_symlink() and p!=receipt_path and not any(part in ['cache','__pycache__','adapter-fixture'] for part in p.relative_to(W).parts)}
receipt={'schema':1,'task':'add_example','revision':1,'kind':'immutable_independent_source_review','frozen':True,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':report['status'],'producer_receipt_sha256':expected,'review':{'path':'peer-review.json','sha256':sha(rp)},'artifacts':artifacts,'excluded_generated_paths':['read-only dependency object symlinks','**/cache/**','**/__pycache__/**','adapter-fixture/** (duplicated frozen producer inputs and synthetic test fixture; scripts and test report retained)'],'gpu_launches':0,'acceptance_increment':0}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
for rel,h in artifacts.items():assert sha(W/rel)==h,rel
print(json.dumps({'review':str(rp),'review_sha256':sha(rp),'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'artifacts':len(artifacts),'blockers':0}))
