from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');M=T/'matrix-transpose-acceptance-wip';L=T/'relu-fp64-source-wip'
sys.path.insert(0,str(R/'bench'))
from adapter_common import declarations,canonical
from check_original_program_scope import source_inventory,declaration_scope,case_set

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checked={}
def exact(p,h):assert sha(p)==h,str(p);checked[str(p)]=h
exact(M/'receipt.json','f628338734a010f0e17a073655144005d9a3cce1b6077b7b9753ad6c6e9ae144')
mr=read(M/'receipt.json')
for n,h in mr['artifact_sha256'].items():exact(M/n,h)
for n,h in read(M/'current-input-sha256.json').items():exact(R/n,h)
mi=read(M/'integration-inputs.json')
mp=M/'after'/mi['new_source']['path'];exact(mp,mi['new_source']['sha256'])
assert declarations(mp)[mi['public_statement']['declaration']]['statement_sha256']==mi['public_statement']['statement_sha256']
ms=read(M/'catalog-supplement-task.proposed.json');old=read(R/mi['current_supplement']['path']);oldm=next(t for t in old['tasks'] if t['task']=='matrix_transpose');assert ms['declarations'][:-1]==oldm['declarations']
inv=source_inventory(R,{'bench/tritonbench_g/matrix_transpose/matrix_transpose.py':sha(R/'bench/tritonbench_g/matrix_transpose/matrix_transpose.py')})
assert inv==read(M/'scope-row.proposed.json')['source_inventory'];declScope=declaration_scope(ms['declarations'][-1],inv)
cases=read(M/'scope-cases.json');assert len(cases['successes'])==231 and len(cases['rejections'])==9 and len(cases['primitive_errors'])==4
pairset={(c['configuration']['SIZE_M'],c['configuration']['D_HEAD']) for c in cases['successes']}
assert pairset=={(2**a,2**b) for a in range(21) for b in range(21) if a+b<=20}
for c in cases.values():case_set(c)
wb=read(M/'evidence/wrapper-binding.json');assert wb['status']=='passed' and len(wb['admitted_metadata_cases'])==231 and len(wb['source_failure_cases'])==96 and len(wb['invalid_shape_boundary_cases'])==8
for row in wb['admitted_metadata_cases']:
 m,d=row['m'],row['d'];assert (m,d) in pairset;assert row['output_shape']==[d,m] and row['output_stride']==[m,1] and row['output_dtype']=='torch.float16' and row['return_identity'];assert row['trace']==['randn','zeros','launch','returned']
fn=next(n for n in ast.parse((R/'bench/tritonbench_g/matrix_transpose/matrix_transpose.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='wrapper');assert hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==wb['wrapper_ast_sha256']
# The precise symbolic rejection and failure partitions were independently read
# with the source validators and final Lean source, not inferred from probe counts.
exact(T/'matrix-frontend-root-validation.json','ca905f31039d64b5710bb1ce6d4e6e17ef2990a2e8449788df7bf6c731899928')
exact(L/'evidence/receipt.json','66365aadb6a087fee0cdf9ab6976f4418ea6f1f07c4da9cc99ddf75f9a3e042b')
lr=read(L/'evidence/receipt.json')
for n,h in {**lr['modules'],**lr['module_objects'],**lr['artifacts']}.items():exact(L/n,h)
dep=read(L/'evidence/dependencies-after.json');assert dep==read(L/'evidence/dependencies-before.json')
for row in dep['project'].values():
 exact(Path(row['source']),row['sha256'])
 if 'olean' in row:exact(Path(row['olean']),row['olean_sha256'])
 else:assert row['package'] and str(row['source']).startswith(str(L))
for row in dep['external_objects'].values():exact(Path(row['olean']),row['sha256'])
for n,h in dep['input_sha256'].items():exact(Path(n),h)
attempts=read(L/'evidence/verification-attempts.json');assert len(attempts)==15 and sum(t['exit_code']!=0 for t in attempts)==6
for row in attempts:
 exact(L/row['log'],row['log_sha256']);exact(L/row['source_snapshot'],row['snapshot_sha256'])
for row in lr['final_clean_attempts']:
 assert row['exit_code']==0 and row['source_sha256_after']==lr['modules'][row['module']]
logtext=''.join((L/r['log']).read_text() for r in lr['final_clean_attempts']);assert 'error:' not in logtext
route=read(L/'evidence/source-route-check.json');exact(Path(route['source']),route['source_sha256'])
py=next(n for n in ast.parse(Path(route['source']).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='relu_kernel');assert [ast.unparse(n) for n in py.body]==route['source_statements']
raw=next(Path(p) for p in dep['input_sha256'] if p.endswith('/float64.raw.ttir'));exact(raw,route['ttir_sha256']);ttir=raw.read_text();assert [ttir.index(t) for t in route['ttir_order_tokens']]==route['ttir_order_positions'];assert 'tensor<1024xf64>' in ttir and 'tensor<1024xf32>' in ttir
assert len(route['admitted_profiles'])==13 and len(route['comparison_rejections'])==2 and len(route['signature_rejections'])==2
proposal=read(L/'evidence/numerical-bindings-proposal.json');assert proposal['integrated_in_repository_queue'] is False and len(proposal['bindings'])==3
for n,h in lr['modules'].items():assert not (R/n).exists()
assert not (R/mi['new_source']['path']).exists()
review={'schema':1,'kind':'matrix_and_relu_frozen_package_root_review','reviewer':'root','status':'substantive_review_complete_pending_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'checked_file_sha256':checked,'matrix':{'producer_receipt_sha256':sha(M/'receipt.json'),'scope_semantically_ready_for_schema2':True,'proof_defects':[],'metadata_integration_corrections':['Add missing required dtype_profile string to new supplement declaration; producer correctly labels metadata checks proposal-structure only. Replace proposed review with root source/scope review.'],'semantic_review':'Root read exact original six-statement source, shared typed quotation, complete FPBits interpreter, Host allocation/execution/frame proof, final FullScope admission/outcome theorem, pinned frontend validators, all symbolic case partitions, wrapper meta/error probe and independent full-domain evidence validation. All natural shape dimensions are classified in source order; 231 accepted shapes exhaust the success domain. The one-program source transposes arbitrary fp16 bits without a numerical callback. Torch allocation/shape conversion and launch runtime are named trusted primitives; successful kernel values are derived by execution, never an oracle. Returned metadata and complete memory frame are preserved. Natural domain is explicit, negative/non-shape callers remain API failures rather than falsely generalized classifications. No native/GPU correctness claim.'},'relu':{'producer_receipt_sha256':sha(L/'evidence/receipt.json'),'scope_semantically_ready_for_schema2':False,'proof_defects':[],'semantic_review':'Root read all four modules and matched seven AST statements plus frozen f64 SSA order. All pids compute 64-bit load/ordered compare/select before pid-zero store narrowing; inactive loads use supplied arbitrary words. Complete ordered source execution, trace, i32 offset/mask links, independent initial output/frame and unchanged tails are proved. Host dispatch calls the unchanged twelve routes and new fp64 grid; four fp8 rejection routes stop return. Product carriers have explicit branch projections and unselected carrier preservation. Native narrow/compare/literal correspondence is only proposed, not registered/validated. Remaining finite host metadata/view/configuration bindings and schema2 integration remain pending; known N>1024 full-ReLU claim stays false.'},'new_numerical_registrations':0,'whole_program_acceptance_increment_before_integration':0,'native_backend_runs_by_root':0,'gpu_runs':0}
out=T/'matrix-relu-packages-root-review.json';out.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'checked_files':len(checked),'matrix_scope_semantically_ready':True,'relu_partial':True}))
