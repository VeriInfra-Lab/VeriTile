import hashlib,json,re,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parent;repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
evidence=root/'evidence';run=evidence/'attempt-03'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def write(p,d):
 value=json.dumps(d,indent=2)+'\n'
 if p.exists():assert p.read_text()==value,str(p)
 else:p.write_text(value)
assert not (evidence/'receipt.json').exists()
results=read(run/'verification-results.json');assert len(results)==2 and all(r['exit_code']==0 for r in results)
modules={r['module']:r['source_sha256_after'] for r in results}
deps=read(run/'dependencies-before.json');assert deps==read(run/'dependencies-after.json')
for rel,r in deps['files'].items():
 assert sha(Path(r['source_path']))==r['sha256']
 if 'olean_path' in r:assert sha(Path(r['olean_path']))==r['olean_sha256']
for r in deps['external_direct_import_oleans'].values():assert sha(Path(r['olean_path']))==r['olean_sha256']
for p,h in deps['toolchain_binary_sha256'].items():assert sha(Path(p))==h
logs=[]
for r in results:
 assert r['source_sha256_before']==r['source_sha256_after']==sha(root/r['module'])
 assert r['log_sha256']==sha(root/r['log']);logs.append((root/r['log']).read_text())
s='\n'.join(logs);assert s.count('axiom footprint ⊆ standard base ✓')==17 and s.count('non-circular ✓')==0
assert not re.search(r'error:|warning:|sorryAx|DISALLOWED',s)
assert sum((root/p).read_text().count('#auditModuleAxioms') for p in modules)==2
out=evidence/'source-audit-final.log'
if not out.exists():
 with out.open('w') as f:audit_result=subprocess.run([sys.executable,str(root/'audit_source.py')],stdout=f,stderr=subprocess.STDOUT)
 assert audit_result.returncode==0,out.read_text()
 write(evidence/'source-audit-execution.json',{'command':[sys.executable,str(root/'audit_source.py')],'exit_code':audit_result.returncode,'log_sha256':sha(out),'audit_sha256':sha(evidence/'source-audit/audit-result.json')})
else:
 prior=read(evidence/'source-audit-execution.json');assert prior['exit_code']==0 and prior['log_sha256']==sha(out) and prior['audit_sha256']==sha(evidence/'source-audit/audit-result.json')
audit=read(evidence/'source-audit/audit-result.json')
sys.path.insert(0,str(repo/'bench'));from adapter_common import declarations
publicpath='bench/tritonbench_g/vector_addition/DTypeAdmissionCorrect.lean'
parsed=declarations(root/publicpath)
names=['vector_addition_sm90_admission_model_partition','vector_addition_fnuz_first_stage_model','vector_addition_sm90_add_rejection_model','vector_addition_sm90_promotion_stage_cross_model']
assert set(parsed)==set(names)
public=[]
for name in names:
 assert parsed[name]['kind']=='specification'
 public.append({'file':publicpath,'parser_name':name,'declaration':'VeriTile.Bench.TritonBenchG.VectorAddition.DTypeAdmissionCorrect.'+name,
  'kind':'specification','statement_sha256':parsed[name]['statement_sha256'],'model_kind':'protocol_contract','role':'protocol_contract',
  'covers_stages':[],'scope_relation':'not_an_execution_contract','native_binding_status':'not_established',
  'source_semantics_refinement_proved':False,'original_program_accepted':False,'unclosed_numeric_callbacks':[],
  'meaning':'Finite selected dtype-gate model; no claim of original host/JIT/compiler exception or successful execution.'})
counts={'canonical_scalar_signatures':17,'ordered_pairs_in_model':289,'selected_dtype_gates_pass':189,
 'signature_rejection_model_pairs':64,'addition_rejection_model_pairs':36,
 'standalone_promotion_success_pairs':217,'standalone_promotion_failure_pairs':72,
 'signature_rejection_and_standalone_promotion_success':28,'signature_rejection_and_standalone_promotion_failure':36,
 'public_model_contracts':4,'new_execution_contracts':0,'new_value_dtype_profiles':0,
 'new_frontend_runs':0,'native_backend_runs':0,'gpu_runs':0,'whole_original_program_completion_increment':0}
gaps=[{'id':'VA-ADMISSION-S'+str(i+1),'gap':g,'deferrable_as_numerical_assumption':False} for i,g in enumerate(audit['structural_remaining'])]
scope={'task':'vector_addition frozen sm90 finite dtype admission and first-stage model','standard':'Cover all original stages, dtypes and branches; only explicitly registered numerical assumptions may be deferred.',
 'original_program_accepted':False,'whole_original_program_accepted':False,'completed_original_program_increment':0,
 'source':audit['files']['original_python'],'target':audit['target'],'counts':counts,'public_contracts':public,
 'structural_remaining':gaps,'numerical_assumptions_added':[],
 'source_binding_status':'28 frozen Python AST/text units with 24 source-shape and observation checks; not formal Python-to-model refinement.',
 'observations_are_proof_premises':False,'prior_289_frontend_observations_reused_without_execution':True,
 'model_scope':'Fresh ASTSource kernel frontend selected gates only, 17 canonical signatures, n_elements i32, BLOCK_SIZE 1024, default frozen sm90 options. It does not claim the original wrapper reaches these gates.',
 'stage_order':'Unsupported fnuz signatures reject before kernel body; 36 of the 72 standalone promotion failures therefore never reach addition on this audited source route.',
 'value_count_rule':'All 100 rejection-model cases remain separate from executed value profiles. The 28 fnuz promotion-success cases add zero value profiles.',
 'inherited_value_inventory_note':audit['value_inventory_note'],'independent_review':'pending independent source/receipt review'}
write(evidence/'scope-review.json',scope)
(root/'README.md').write_text('''# Finite sm90 dtype admission model

Two Lean proof modules and four public protocol-model specifications classify all
289 ordered pairs of 17 canonical scalar signatures. They prove the selected-gate
partition: 189 pass, 64 reject at signature serialization, 36 reject at addition
promotion. Standalone promotion has 217 successes and 72 failures. Of the 64
signature rejections, 28 have successful standalone promotion and 36 overlap its
failures; those 36 do not reach addition on the audited source route.

This adds four model contracts, zero execution contracts, zero value dtype
profiles and zero accepted original programs. All 100 rejected pairs, including
the 28 promotion-success fnuz pairs, remain separate from value execution.

Actual final compilation exits are 0/0: 17 named axiom checks, two module audits,
zero warnings and zero noncircular-command checks. Earlier real failures and
source snapshots are retained under evidence/attempt-01 and attempt-02. A separate
successful executable export compares all 289 model rows to prior frozen probes.

The source audit freezes 28 exact Python units and performs 24 checks, including
builder.options assignment, argument serialization before body traversal, the
sm90 supported set and tensor promotion branches. Prior probe bytes are preserved.
No original Python module, frontend compiler, native backend or GPU was executed
for this batch. The four public statements have protocol_contract metadata and
covers_stages=[] because they do not establish original-program execution.

Python/AST-to-model semantic refinement is still a structural gap. Host allocation,
CUDA assertions, JIT/cache/launch paths, module/harness effects, other dtype families,
architecture or option overrides, exception wrapping, lowering/debug/runtime
behavior, physical memory and GPU execution remain open. These gaps are not
numerical assumptions. This package adds no numerical assumptions.
''')
checkroot=Path(tempfile.mkdtemp(prefix='veritile-admission-whitespace-'))
paths=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.suffix in {'.olean','.pyc'} or '__pycache__' in p.parts:continue
 rel=p.relative_to(root);target=checkroot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);paths.append(str(rel))
subprocess.run(['git','init','--quiet',str(checkroot)],check=True)
subprocess.run(['git','add','--',*paths],cwd=checkroot,check=True)
check=subprocess.run(['git','diff','--cached','--check'],cwd=checkroot,capture_output=True,text=True)
write(evidence/'whitespace-check.json',{'kind':'actual_isolated_staged_whitespace_check','exit_code':check.returncode,
 'command':['git','diff','--cached','--check'],'temporary_index_root':str(checkroot),'staged_files':len(paths),
 'stdout':check.stdout,'stderr':check.stderr})
assert check.returncode==0,check.stdout+check.stderr
verification={'proof_modules':2,'actual_exit_codes':[r['exit_code'] for r in results],'named_axiom_checks':17,
 'module_axiom_audit_commands':2,'modules_audited':2,'noncircularity_checks':0,'warnings':0,
 'dependency_sources_and_anchors':len(deps['files']),'project_oleans_guarded':sum('olean_path' in r for r in deps['files'].values()),
 'external_direct_import_oleans_guarded':len(deps['external_direct_import_oleans']),'external_library_build_rechecked':False,
 'source_and_dependency_hashes_unchanged':True,'source_audit_execution_exit_code':0,'frozen_source_units':28,'source_branch_audit_checks':24,
 'actual_model_row_export_exit_code':read(evidence/'model-rows-result.json')['exit_code'],'model_rows_compared':289,
 'actual_staged_whitespace_check_exit_code':check.returncode,'preserved_failed_compile_attempts':2,
 'verification_results':'evidence/attempt-03/verification-results.json'}
artifacts={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean','.pyc'} and '__pycache__' not in p.parts and p!=evidence/'receipt.json'}
receipt={'task':scope['task'],'package_root':str(root),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
 'original_program_accepted':False,'whole_original_program_accepted':False,'original_program_completion_increment':0,
 'modules':modules,'public_contracts':public,'verification':verification,'coverage_summary':counts,
 'acceptance_limits':gaps,'artifacts':dict(sorted(artifacts.items()))}
write(evidence/'receipt.json',receipt)
print(json.dumps({'receipt_sha256':sha(evidence/'receipt.json'),'scope_sha256':sha(evidence/'scope-review.json'),
 'source_audit_sha256':sha(evidence/'source-audit/audit-result.json'),'modules':modules,'verification':verification,'counts':counts},indent=2))
