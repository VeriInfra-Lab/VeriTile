from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,sysconfig
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/sin-kernel-finite-acceptance-wip')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
expected='faf9aaf762de78aadfb536e11e5655af31075637fd536d3d6c04dc8bda226ddb'
assert sha(P/'evidence/receipt.json')==expected
r=load(P/'evidence/receipt.json');assert len(r['artifacts'])==535 and len(r['modules'])==8
for rel,h in r['artifacts'].items():assert sha(P/rel)==h,rel
for rel,h in r['modules'].items():
 assert sha(P/'after'/rel)==h and sha(P/r['checkpoint_module_source_paths'][rel])==h,rel
for rel,h in r['module_objects'].items():assert sha(P/'build-overlay'/rel)==h,rel
for entry in r['final_clean_attempts']:
 a=load(P/entry['record']);assert a['exit_code']==0 and a['source_unchanged'] and a['source_sha256']==r['modules'][entry['module']]
 assert sha(P/a['log'])==a['log_sha256'] and sha(P/a['log'].replace('.log','.lean.txt'))==a['source_sha256']
s=load(W/'dependencies-before.json');assert sha(W/'dependencies-before.json')==sha(W/'dependencies-after.json')
for a in s['project'].values():assert sha(a['source'])==a['sha256'] and sha(a['olean'])==a['olean_sha256']
for a in s['external_objects'].values():assert sha(a['olean'])==a['sha256']
build=load(W/'independent-build-receipt.json')
for a in build['modules']:
 assert a['exit_code']==0 and sha(W/a['log'])==a['log_sha256']
 assert sha(W/'isolated'/a['source'])==a['sha256']==r['modules'][a['source']]
 assert sha(W/'isolated'/a['source'].replace('.lean','.olean'))==a['object_sha256']==r['module_objects'][a['source'].replace('.lean','.olean')]
axioms=sum((W/a['log']).read_text().count('axiom footprint ⊆ standard base') for a in build['modules']);noncirc=sum((W/a['log']).read_text().count('non-circular ✓') for a in build['modules']);assert (axioms,noncirc)==(40,7)
a=load(W/'regression-001.json');assert a['exit_code']==0 and a['source_sha256']==sha(W/'regression-001.lean.txt') and a['log_sha256']==sha(W/'regression-001.log')
assert (W/'regression-001.log').read_text().count('axiom footprint ⊆ standard base')==10
admission=W/load(W/'recorded-admission/latest.json')['receipt'];assert load(admission)['status']=='passed' and load(admission)['exact_dtype_profiles_checked']==22
evidence=['independent-build-receipt.json','dependencies-before.json','dependencies-after.json','case-and-port-review.json','adapter-tests.json','replay-comparison.json','scalar-dataflow.json','binder-001/receipt.json','replay/evidence/setup-order/receipt.json','replay/evidence/source-lowering-audit.json',str(admission.relative_to(W))]
for rel in evidence:
 if 'dependencies-' not in rel:assert load(W/rel)['status']=='passed',rel
native=W/'replay'/load(W/'replay/evidence/native-latest.json')['receipt'];host=W/'replay'/load(W/'replay/evidence/host-latest.json')['receipt'];[abi]=list((W/'replay/evidence/abi-profiles').glob('*/receipt.json'))
for p in [native,host,abi]:
 for rel,h in load(p)['artifacts'].items():assert sha(p.parent/rel)==h,(p,rel)
 evidence.append(str(p.relative_to(W)))
assert sha(native.with_name('driver.c'))==sha('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c')
for path,h in load(P/'evidence/retained-proof-review.json')['retained_original_files_unchanged'].items():assert sha(Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')/path)==h
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
(W/'deferred-numerical-families.json').write_text(json.dumps({'status':'explicitly_deferred_unvalidated','families':[{'id':q['id'],'required_validation':q['required_validation'],'port_count':len(q['scope_bindings'])} for q in delta['rules']],'exact_numeric_ports':2,'producer_delta_sha256':sha(P/'evidence/numerical-authority-delta.json'),'native_scalar_validation':False},indent=2)+'\n')
evidence += ['tool-dependencies.json','deferred-numerical-families.json','regression-001.json','regression-001.lean.txt','regression-001.log','replay/source-identity.json']
report={'schema':1,'reviewer':'independent_peer','agent':'/root/mul2_independent_review','task':'sin_kernel','revision':1,'status':'substantively_reviewed_for_finite_scope_integration','producer_receipt_sha256':expected,'public_statement_sha256':load(W/'case-and-port-review.json')['public_statement_sha256'],'candidate':str(P),'blockers':[],'recommendation':'Finite original-source integration supported with two exact same-width scalar sine conformance families deferred. Root substantive review, exact pins and repository gates remain required.','acceptance_increment':0,'pointer_memory_boundary':'Actual extracted pointers and mapped allocation metadata, with memory bytes addressed in current launch space. No separate native-to-canonical conversion theorem is claimed.','verification':{'modules_rebuilt':8,'producer_object_hashes_reproduced':8,'candidate_named_clean_axiom_checks':40,'candidate_noncircular_checks':7,'independent_regression_axiom_checks':10,'recorded_admission_axiom_checks':1,'project_dependencies':204,'direct_external_objects':37,'producer_source_object_log_record_sets_verified':8,'frozen_candidate_artifacts_unchanged':535,'retained_original_files_unchanged':11,'original_dtype_profiles':22,'dtype_backend_compiles':2,'binder_rejections':5,'architecture_rejections':2,'sine_type_rejections':13,'extent_backend_profiles':30,'equal_IR_artifacts':160,'scalar_dataflow_profiles':2,'actual_JIT_binder_dtype_profiles':22,'binder_extent_profiles':10,'exact_i32_extractor_cases':15,'CPU_wrapper_launch_recorder_cases':75,'tensor_metadata_records':150,'exact_wrapper_AST_stub_cases':18,'exact_signed_grid_parser_checks':7,'scope_cases':52,'exact_numeric_ports':2,'numeric_families':2,'omission_mutations_rejected':2,'adapter_mutations_rejected':20,'gpu_launches':0,'large_nonempty_allocations':0,'native_scalar_validation':False},'source_review':{'path':'source-review.md','sha256':sha(W/'source-review.md')},'primary_evidence':{p:sha(W/p) for p in evidence},'retained_failed_review_attempts':[],'retained_review_warnings':[{'log':'regression-001.log','reason':'unused simp argument dtype; proof and axiom checks passed'}],'candidate_and_repository_dependencies_unchanged':True}
rp=W/'peer-review.json';assert not rp.exists();rp.write_text(json.dumps(report,indent=2)+'\n')
receipt_path=W/'review-receipt.json';assert not receipt_path.exists()
artifacts={str(p.relative_to(W)):sha(p) for p in sorted(W.rglob('*')) if p.is_file() and not p.is_symlink() and p!=receipt_path and not any(part in ['cache','tmp','__pycache__','adapter-fixture'] for part in p.relative_to(W).parts)}
receipt={'schema':1,'task':'sin_kernel','revision':1,'kind':'immutable_independent_source_review','frozen':True,'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':report['status'],'producer_receipt_sha256':expected,'review':{'path':'peer-review.json','sha256':sha(rp)},'artifacts':artifacts,'excluded_generated_paths':['read-only dependency object symlinks','**/cache/**','**/tmp/**','**/__pycache__/**','adapter-fixture/** (duplicate producer inputs and synthetic fixture; scripts/report retained)'],'gpu_launches':0,'acceptance_increment':0}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
for rel,h in artifacts.items():assert sha(W/rel)==h,rel
print(json.dumps({'review':str(rp),'review_sha256':sha(rp),'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),'artifacts':len(artifacts),'blockers':0}))
