"""Integrate independently reviewed matrix scope and ReLU fp64 source additions."""
from pathlib import Path
import copy,hashlib,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');M=T/'matrix-transpose-acceptance-wip';L=T/'relu-fp64-source-wip'
OLDREL='2026-10-16/quantization-and-mixed-softmax';NEWREL='2026-10-16/matrix-scope-and-relu-fp64';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,hash_file,write_json
from current_value_metadata import value_scope

def read(p):return json.loads(p.read_text())
def sha(p):return hash_file(Path(p))
def git(*a):return subprocess.check_output(['git',*a],cwd=R)
def digest(b):return hashlib.sha256(b).hexdigest()
def hashes(p):return {q.relative_to(p).as_posix():sha(q) for q in p.rglob('*') if q.is_file()}
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip()
reviewPath=T/'matrix-relu-packages-root-review.json';assert sha(reviewPath)=='ba38cdaf88f55d9eda0cacfe0bc71cee454777b56e6362b65fb3c55148602d1b'
review=read(reviewPath)
for p,h in review['checked_file_sha256'].items():assert sha(p)==h,p
assert review['matrix']['scope_semantically_ready_for_schema2'] and not review['relu']['scope_semantically_ready_for_schema2']
mr=read(M/'receipt.json');lr=read(L/'evidence/receipt.json');mi=read(M/'integration-inputs.json')
sources={p:(L/p).read_bytes() for p in lr['modules']};sources[mi['new_source']['path']]=(M/'after'/mi['new_source']['path']).read_bytes();assert len(sources)==5
assert all(not (R/p).exists() for p in sources)
supp=copy.deepcopy(read(OLD/'catalog-supplement.json'));oldSupp=copy.deepcopy(supp);oldTasks={t['task']:t for t in supp['tasks']};ledger=read(R/'bench/original_program_scope.json');rows={t['task']:t for t in ledger['tasks']}
mt=read(M/'catalog-supplement-task.proposed.json');assert mt['declarations'][:-1]==oldTasks['matrix_transpose']['declarations'];mt['declarations'][-1]['dtype_profile']='Complete original wrapper fp16 input/output and all BitVec16 payloads. Canonical natural shape API; 231 exact successful shapes m=2^a,d=2^b,a+b<=20, nine exhaustive frontend rejection classes and four primitive error partitions. Signed32 arange/stride offsets bounded by admitted tensor count; one program, fresh contiguous allocation and exact return metadata.'
mt['declarations'][-1]['review_basis']=review['matrix']['semantic_review']
lt=copy.deepcopy(oldTasks['relu_triton_kernel']);newDecls=[]
rel='bench/tritonbench_g/relu_triton_kernel/FP64Correct.lean'
stages=['complete_fp64_pid_zero_source_grid','fp64_source_wrapper_success']
residuals=read(L/'evidence/scope-review.json')['remaining_finite_bindings']
for name,stage in zip(['relu_fp64_source_grid_correctness','relu_fp64_host_source_correctness'],stages):
 public=declarations(L/rel)[name]
 newDecls.append({'file':rel,'declaration':name,'kind':'specification','statement_sha256':public['statement_sha256'],'model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':[stage],'dtype_profile':'Binary64 input/load/ordered compare/select with binary32 output store narrowing; all input bit patterns and arbitrary inactive words. Fixed block1024, 0<=N<2^31, complete ceil-grid permutations, fresh typed input/output views. Known pid-zero-only stores and unchanged tail retained.','dtype_scope_relation':'selected_explicit_profiles_only','unclosed_numeric_callbacks':['FP64Source.Numerics.narrow64to32 : BitVec 64 -> BitVec 32'],'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':review['relu']['semantic_review'],'original_scope':{'standard_accepted':False,'covered':stage,'remaining_structural_obligations':residuals,'remaining_native_numerical_bindings':['Proposed exact fp64 ordered comparison, zero literals and f64-to-f32 narrowing; no integrated exact registrations for these statements.'],'known_source_bug':'N>1024 leaves tail unchanged; not a whole-tensor mathematical ReLU theorem.','whole_program_completion_increment':0}})
lt['declarations']+=newDecls;lt['required_value_stages']+=stages
lt['source_views'].append({'id':'fp64_source_and_host_dispatch','variant':'original_source_translation','frontend':'task_local_exact_ast','ordinary_shared_dsl_admission':False,'memory_semantics':'total_separate_typed_f64_f32_views_and_sequential_source_programs','translation_trust_boundary':review['relu']['semantic_review'],'original_python':'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py','candidate_python':None,'selected_definitions':{'bench/tritonbench_g/relu_triton_kernel/Proofs/FP64Source.lean':['source']},'entry_aliases':{'bench/tritonbench_g/relu_triton_kernel/Proofs/FP64Host.lean':['logicalLaunch']}})
inputs={}
for task in [mt,lt]:
 name=task['task'];prefix='bench/tritonbench_g/'+name+'/'
 for group in rows[name]['input_sha256'].values():inputs.update(group)
 task['source_sha256'].update({p:digest(b) for p,b in sources.items() if p.startswith(prefix)})
 inputs.update(task['source_sha256'])
# Bind every actual dependency read during review as well as all task files.
for p,h in review['checked_file_sha256'].items():
 q=Path(p)
 if q.is_relative_to(R) and q.suffix in {'.lean','.py'} and '/.lake/' not in p:inputs[q.relative_to(R).as_posix()]=h
inputs.update({p:digest(b) for p,b in sources.items()})
for p,h in inputs.items():
 if p not in sources:assert sha(R/p)==h,p
baseline=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline';baselineHashes=hashes(baseline);assert baselineHashes['receipt.json']=='207d0fc6f6fd80c3dd15b54908e87c4c72a5fc6e7bf5ebac08d9344ed890ddd3'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':baselineHashes,'new_source_sha256':{p:digest(b) for p,b in sources.items()},'scope_ledger_before_sha256':sha(R/'bench/original_program_scope.json'),'scope_markdown_before_sha256':sha(R/'bench/original_program_scope.md'),'workflow_before_sha256':sha(R/'.github/workflows/bench-audit.yml'),'changed_tasks':['matrix_transpose','relu_triton_kernel'],'root_package_review_sha256':sha(reviewPath),'prior_publics':942,'prior_supplemental_declarations':49,'new_publics':3}
# All immutable evidence checks precede writes.
for p,data in sources.items():(R/p).write_bytes(data)
B.mkdir(parents=True);(B/'evidence').mkdir();retained={};excluded={}
packages=[('matrix',M,{**mr['artifact_sha256'],'receipt.json':sha(M/'receipt.json')}),('relu',L,{**lr['modules'],**lr['artifacts'],'evidence/receipt.json':sha(L/'evidence/receipt.json')})]
for a in read(L/'evidence/verification-attempts.json'):
 for field,hf in [('log','log_sha256'),('source_snapshot','snapshot_sha256')]:packages[1][2][a[field]]=a[hf]
for name,pkg,files in packages:
 for p,h in files.items():
  assert sha(pkg/p)==h
  if Path(p).suffix=='.olean':excluded[name+'/'+p]=h;continue
  relp='evidence/producer/'+name+'/'+(p+'.txt' if p.endswith('.lean') else p);dest=B/relp;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(pkg/p,dest);retained[relp]=h
for p,name in [(reviewPath,'package-review.json'),(T/'review_matrix_relu_packages.py','package-checker.py'),(T/'matrix-frontend-root-validation.json','matrix-frontend-root-validation.json'),(T/'check_matrix_frontend_root.py','matrix-frontend-root-checker.py'),(Path(__file__),'prepare.py')]:
 shutil.copyfile(p,B/'evidence'/name);retained['evidence/'+name]=sha(p)
rootReview={'schema':1,'kind':'matrix_scope_and_relu_fp64_extension_root_review','reviewer':'root','status':'reviewed_source_execution_and_finite_matrix_scope','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{t['task']:digest(canonical(value_scope(t))) for t in [mt,lt]},'reviewed_metadata_corrections_sha256':{},'semantic_review':{'matrix_transpose':review['matrix']['semantic_review'],'relu_triton_kernel':review['relu']['semantic_review']},'new_public_contracts':{'matrix_transpose':[mt['declarations'][-1]],'relu_triton_kernel':newDecls},'whole_stage_residuals':{'matrix_transpose':[],'relu_triton_kernel':residuals},'metadata_baseline':read(OLD/'root-review.json')['metadata_baseline'],'numerical_registration_boundary':'29 registered rules unchanged. New ReLU numeric bindings remain proposals; matrix has no numerical callbacks.','package_review':{'path':'evidence/package-review.json','sha256':sha(reviewPath)},'acceptance':'Matrix source-scope may be accepted only after final schema2 review and gate. ReLU pending. Global/build/comparator results remain separate.'}
write_json(B/'root-review.json',rootReview)
for task in [mt,lt]:
 task['review']={'path':NEWREL+'/root-review.json','sha256':sha(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']}
 for i,t in enumerate(supp['tasks']):
  if t['task']==task['task']:supp['tasks'][i]=task;break
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==52
shutil.copyfile(OLD/'numerical-followup.json',B/'numerical-followup.json');shutil.copyfile(R/'bench/original_program_scope.json',B/'evidence/ledger-before.json')
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'compiled_artifacts_not_copied':excluded,'new_source_sha256':state['new_source_sha256']})
workflow=R/'.github/workflows/bench-audit.yml';text=workflow.read_text();assert text.count(OLDREL)==3;workflow.write_text(text.replace(OLDREL,NEWREL))
(B/'README.md').write_text('''# Matrix transpose finite source scope and ReLU fp64 execution

Five additive modules add three public execution contracts. Matrix transpose
composes exact source frontend admission, trusted host primitives and the complete
bit-exact fp16 interpreter with returned metadata and full memory frame. Its
231 successful shape pairs and all natural-shape rejection classes are explicit.
No numerical assumption is needed for copying arbitrary fp16 words.

ReLU adds explicit fp64 load/ordered comparison/select and f32 store narrowing,
complete sequential grid traces and finite index links, and host dispatch using
all twelve existing routes plus fp64. The four canonical fp8 rejected cases stop
before return. The original pid-zero-only tail bug is preserved. ReLU remains
pending: finite host metadata/view/configuration bindings, exact numerical
registrations and complete schema2 review are still required.

The original baseline, every prior public statement, all frozen checkpoints and
29 numerical rules are preserved. Three new ReLU numerical bindings are retained
as proposals, not claimed as integrated or validated. Build, scope and global
results will be recorded after actual integration checks. No GPU run is claimed.
''')
print(json.dumps({'checkpoint':str(B),'new_modules':5,'new_publics':3,'supplemental_declarations':52,'prepared_acceptance_increment':0},indent=2))
