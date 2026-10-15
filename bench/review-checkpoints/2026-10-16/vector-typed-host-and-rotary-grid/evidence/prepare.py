from pathlib import Path
import copy,hashlib,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');V=T/'vector-typed-host-wip';Q=T/'rotary-emb-host-grid-wip/frozen-v2'
OLDREL='2026-10-16/matrix-scope-and-relu-fp64';NEWREL='2026-10-16/vector-typed-host-and-rotary-grid';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,hash_file,load,sha,write_json
from current_value_metadata import value_scope

def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(Path(p))==h,str(p)
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip()
vp=T/'vector-typed-host-root-review.json';qp=T/'rotary-host-grid-root-review.json';vm=T/'rotary-root-validation/validation.json'
exact(vp,'e9014773c713685593b646de9b9eed918c05736efce790c908b1f370ff4045c9');exact(qp,'f52932d73800e1a846e3c146240227fb4d2e6f6e513a40af6ac33e0e203b7a7e');exact(vm,'274b9b08a6ec7424f0918e5e2b25b3bdd559cae1f8bd9767569b055854d0efbe')
vr,qr,qm=load(vp),load(qp),load(vm);vreceipt=load(V/'evidence/receipt.json');qreceipt=load(Q/'receipt.json')
for p,h in vr['checked_file_sha256'].items():exact(p,h)
for p,h in qm['frozen_local_artifact_sha256'].items():exact(Q/p,h)
sources={p:(V/p).read_bytes() for p in vreceipt['modules']}
for p,h in qr['integration_sources'].items():exact(Q/'overlay'/p,h);sources[p]=(Q/'overlay'/p).read_bytes()
public='bench/tritonbench_g/rotary_emb/HostCorrect.lean';before=sources[public];assert before.count(b'theorem rotary_emb_host_grid_correctness\n')==1
sources[public]=before.replace(b'theorem rotary_emb_host_grid_correctness\n',b'specification rotary_emb_host_grid_correctness\n');assert len(sources)==12
assert all(not (R/p).exists() for p in sources)
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'));priorSupp=copy.deepcopy(supp);st={t['task']:t for t in supp['tasks']};ledger=load(R/'bench/original_program_scope.json');rows={t['task']:t for t in ledger['tasks']}
vector=st['vector_addition'];rotary={'task':'rotary_emb','required_value_stages':[],'source_sha256':{},'source_views':[],'declarations':[]}
inputs={}
for name,task in [('vector_addition',vector),('rotary_emb',rotary)]:
 for group in rows[name]['input_sha256'].values():inputs.update(group)
 prefix='bench/tritonbench_g/'+name+'/'
 task['source_sha256'].update({p:sha(data) for p,data in sources.items() if p.startswith(prefix)})
 inputs.update(task['source_sha256'])
# Preserve task-local review scope while also binding the exact repo dependencies read.
for p,h in vr['checked_file_sha256'].items():
 path=Path(p)
 if path.is_relative_to(R) and path.suffix in {'.lean','.py'} and '/.lake/' not in p:inputs[path.relative_to(R).as_posix()]=h
for d in load(Q/'dependencies.json'):
 path=Path(d['source']['path'])
 if path.is_relative_to(R) and path.suffix=='.lean' and '/.lake/' not in str(path):inputs[path.relative_to(R).as_posix()]=d['source']['sha256']
for p,h in inputs.items():
 if p not in sources:exact(R/p,h)
baseline=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':hashes(baseline),'new_source_sha256':{p:sha(b) for p,b in sources.items()},'changed_tasks':['vector_addition','rotary_emb'],'rebound_verified_task':'matrix_transpose','prior_publics':945,'new_publics':5,'prior_supplemental_declarations':52,'queue_sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json'),'source_transformations':[{'file':public,'producer_sha256':sha(before),'integrated_sha256':sha(sources[public]),'exact_change':'theorem keyword to specification; identical parameters, proposition and proof'}]}
# Checked immutable preconditions precede all repository writes.
for p,data in sources.items():(R/p).write_bytes(data)
B.mkdir(parents=True);(B/'evidence').mkdir();retained={};excluded={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dst=B/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);retained[rel]=h
vfiles={**vreceipt['modules'],**vreceipt['artifacts'],'evidence/receipt.json':hash_file(V/'evidence/receipt.json')}
for a in load(V/'evidence/verification-attempts.json'):
 for field,hf in [('log','log_sha256'),('source_snapshot','snapshot_sha256')]:vfiles[a[field]]=a[hf]
for p,h in vfiles.items():retain(V/p,'evidence/producer/vector/'+p+('.txt' if p.endswith('.lean') else ''),h)
excluded.update({'vector/'+p:h for p,h in vreceipt['module_objects'].items()})
for p,h in qm['frozen_local_artifact_sha256'].items():retain(Q/p,'evidence/producer/rotary/'+p+('.txt' if p.endswith('.lean') else ''),h)
# Retain the earlier overlay namespace failure and its inputs/logs; never copy support symlinks.
first=Q.parent/'frozen'
for p in first.rglob('*'):
 if p.is_file() and not p.is_symlink() and not any(q.is_symlink() for q in p.parents if q!=first) and p.suffix not in {'.olean','.ilean'} and '.olean.' not in p.name:
  rel=p.relative_to(first).as_posix();retain(p,'evidence/producer/rotary-first-overlay/'+rel+('.txt' if rel.endswith('.lean') else ''))
for src,rel in [(vp,'vector-package-review.json'),(qp,'rotary-package-review.json'),(vm,'rotary-mechanical-validation.json'),(T/'review_vector_typed_host.py','vector-package-checker.py'),(T/'review_rotary_frozen.py','rotary-package-checker.py'),(T/'rotary-root-validation/evidence/HostProbe-source-ast.json','rotary-replayed-probes.json'),(T/'rotary-root-validation/probe.log','rotary-replayed-probes.log'),(Path(__file__),'prepare.py')]:retain(src,'evidence/'+rel)
# Final source statements are discovered only after the one recorded keyword edit.
vfile='bench/tritonbench_g/vector_addition/TypedHostCorrect.lean';vnew=[]
for name,stage,protocol in [('vector_addition_typed_host_success','typed_host_success_all189',False),('vector_addition_typed_host_outcome','typed_host_canonical_outcomes289',False),('vector_addition_typed_host_allocation_error',None,True),('vector_addition_typed_host_cuda_error',None,True)]:
 d=declarations(R/vfile)[name]
 row={'file':vfile,'declaration':name,'kind':'specification','statement_sha256':d['statement_sha256'],'model_kind':'protocol_contract' if protocol else 'backend_value_contract','role':'protocol_contract' if protocol else 'parametric_value_contract','scope_relation':'not_an_execution_contract' if protocol else 'full_declared_stage','source_variant':'original_source_translation','covers_stages':[] if protocol else [stage],'dtype_profile':'17 canonical scalar types / 289 ordered dtype pairs: 189 actual inherited source-grid routes, 64 first signature rejections and 36 addition rejections. Explicit safe/defined typed-value domain, fresh output metadata and frames. Early host allocation/CUDA failures carry no kernel value coverage.','dtype_scope_relation':'selected_explicit_profiles_only','unclosed_numeric_callbacks':[] if protocol else ['TypedHost.Numerics.rest floating casts/addition','TypedHost.Numerics.cfg software-to-native numeric correspondence',*load(V/'evidence/numerical-bindings-proposal.json')['fp64_callback_families']],'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':vr['semantic_review'],'original_scope':{'standard_accepted':False,'covered':stage or 'source_order_host_early_failure_protocol','remaining_structural_obligations':vr['remaining_finite_obligations'],'numerical_registration':'189 profiles / 308 operations retained as unregistered candidates, not numerical validation.','whole_program_completion_increment':0}}
 vnew.append(row)
 if stage:vector['required_value_stages'].append(stage)
vector['declarations']+=vnew
rd=declarations(R/public)['rotary_emb_host_grid_correctness'];stage='original_real_host_complete_grid'
rnew={'file':public,'declaration':'rotary_emb_host_grid_correctness','kind':'specification','statement_sha256':rd['statement_sha256'],'model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':[stage],'dtype_profile':'Original exact-real-erased source at symbolic Q/K/Cos/Sin storage selections; BH4 BS16 Q-derived full grid, both warp branches and partial/odd/extra-K tails under explicit nonnegative layout and ownership domain. This is not rounded typed dtype-domain coverage.','dtype_scope_relation':'selected_explicit_profiles_only','unclosed_numeric_callbacks':[],'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':qr['semantic_review'],'original_scope':{'standard_accepted':False,'covered':'Complete original real-erased sequential host/grid execution and initial-memory Q/K formulas, source-trace safety and full raw-cell frame.','remaining_structural_obligations':qr['remaining_finite_obligations'],'numerical_registration':'No floating equality deferral: exact-real semantics is explicit and rounded execution remains open.','whole_program_completion_increment':0}}
rotary['declarations']=[rnew];rotary['required_value_stages']=[stage];supp['tasks'].append(rotary)
rootreview={'schema':1,'kind':'vector_typed_host_rotary_grid_root_review','reviewer':'root','status':'reviewed_partial_source_host_grid_extensions','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{t['task']:sha(canonical(value_scope(t))) for t in [vector,rotary]},'reviewed_metadata_corrections_sha256':{},'semantic_review':{'vector_addition':vr['semantic_review'],'rotary_emb':qr['semantic_review']},'new_public_contracts':{'vector_addition':vnew,'rotary_emb':[rnew]},'whole_stage_residuals':{'vector_addition':vr['remaining_finite_obligations'],'rotary_emb':qr['remaining_finite_obligations']},'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'source_transformations':state['source_transformations'],'numerical_registration_boundary':'29 registered rules unchanged. Vector 308 numerical bindings remain proposals; rotary is explicitly real-erased, with no false real=floating deferral.','acceptance':'No increment; both tasks pending. Matrix verified scope is preserved by exact unchanged-source/evidence review rebinding to current catalog.'}
write_json(B/'root-review.json',rootreview)
for task in [vector,rotary]:task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']}
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==57
shutil.copyfile(OLD/'numerical-followup.json',B/'numerical-followup.json');shutil.copyfile(R/'bench/original_program_scope.json',B/'evidence/ledger-before.json')
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'compiled_artifacts_not_copied':excluded,'new_source_sha256':state['new_source_sha256']})
f=R/'.github/workflows/bench-audit.yml';text=f.read_text();assert text.count(OLDREL)==3;f.write_text(text.replace(OLDREL,NEWREL))
(B/'README.md').write_text('''# Vector typed host and rotary real source grid

Twelve additive modules introduce five public specifications. Vector addition
composes the 189 existing typed execution routes with source-order allocation,
CUDA assertion, canonical dtype rejection and returned metadata/frames. Two
success/outcome contracts cover execution; two early-error contracts are protocol
only. Safe views, finite addresses, undefined conversion outcomes and exact
numerical/schema2 registration remain open. Its 308 numerical binding candidates
are retained without claiming registration or native validation.

RotaryEmb now composes every original 34-statement source program on preceding
memory, with the exact Q-derived grid, initial-memory formulas, trace safety and
raw frame for odd/partial channels and extra K heads. This is exact-real erasure;
rounded typed execution and source-grounded dtype/admission/address domains remain
open. No false real/floating equality is deferred. The frozen theorem is retained
and its active keyword changed to specification; dependency-inventory output is
retained as review instrumentation rather than an active build module.

Both tasks remain pending. Matrix transpose remains the one verified source
scope among 183. All earlier public statements, frozen baselines and 29 numerical
rules are preserved. Integration results will be recorded after actual checks.
No GPU execution or official comparator success is claimed.
''')
print(json.dumps({'checkpoint':str(B),'new_modules':12,'new_publics':5,'supplement_execution':51,'supplement_protocol':6,'scope_increment':0,'retained_artifacts':len(retained)},indent=2))
