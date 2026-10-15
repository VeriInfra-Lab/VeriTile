from pathlib import Path
import copy,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'softmax-flaggems-host-grid-wip';V=T/'softmax-root-validation'
OLDREL='2026-10-16/relu-finite-source-acceptance';NEWREL='2026-10-16/softmax-source-host-grid';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,hash_file,load,sha,write_json
from current_value_metadata import value_scope

def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(Path(p))==h,str(p)
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip()
assert git('branch','--show-current').decode().strip()=='fp-examples'
exact(P/'evidence/receipt.json','b6252c7ca19fde05651ea9dcce730046452641bd52cd3ab22a911acf2f5f1cf7')
exact(V/'review.json','462b5f4784e41879dc2f4fa8fdb4e8137d111e4a19e3767fe5fbda2d28aeffe3')
review=load(V/'review.json');receipt=load(P/'evidence/receipt.json')
for p,h in review['checked_file_sha256'].items():exact(p,h)
for p,h in review['root_replay_artifact_sha256'].items():exact(V/p,h)
sources={p:(P/'after'/p).read_bytes() for p in review['integration_sources']}
assert len(sources)==9 and all(not (R/p).exists() for p in sources)
for p,data in sources.items():assert sha(data)==review['integration_sources'][p]
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'));assert all(t['task']!='softmax_flaggems' for t in supp['tasks'])
ledger=load(R/'bench/original_program_scope.json');row=next(t for t in ledger['tasks'] if t['task']=='softmax_flaggems');assert row['status']=='pending'
inputs={p:h for group in row['input_sha256'].values() for p,h in group.items()};inputs.update(review['integration_sources'])
for p,h in review['checked_file_sha256'].items():
 path=Path(p)
 if path.is_relative_to(R) and path.suffix in {'.lean','.py'} and '/.lake/' not in p:inputs[path.relative_to(R).as_posix()]=h
for p,h in inputs.items():
 if p not in sources:exact(R/p,h)
BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':hashes(BASE),'new_source_sha256':review['integration_sources'],'changed_tasks':['softmax_flaggems'],'rebound_accepted_tasks':['matrix_transpose','relu_triton_kernel'],'prior_publics':951,'new_publics':2,'prior_supplemental_declarations':58,'queue_sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json'),'source_transformations':[]}
# Every immutable package and dependency precondition is checked before writes.
for p,data in sources.items():(R/p).write_bytes(data)
B.mkdir(parents=True);(B/'evidence').mkdir();retained={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dst=B/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);retained[rel]=h
for p,h in {**receipt['frozen_files'],'evidence/receipt.json':hash_file(P/'evidence/receipt.json')}.items():
 assert not p.endswith(('.olean','.ilean'))
 retain(P/p,'evidence/producer/'+p+('.txt' if p.endswith('.lean') else ''),h)
for p,h in review['root_replay_artifact_sha256'].items():retain(V/p,'evidence/root-replay/'+p,h)
for src,rel in [(V/'review.json','package-root-review.json'),(T/'review_softmax_host_grid.py','package-root-checker.py'),(Path(__file__),'prepare.py')]:retain(src,'evidence/'+rel)
for i in (1,2,3):
 for prefix,suffix in [('review_softmax_host_grid_attempt','.py.txt'),('softmax-root-review-attempt','.log')]:
  p=T/(prefix+str(i)+suffix)
  if p.exists():retain(p,'evidence/root-review-attempts/'+p.name)
task={'task':'softmax_flaggems','required_value_stages':[],'source_sha256':review['integration_sources'],'source_views':[],'declarations':[]}
for public,stage in zip(review['new_public_contracts'],['original_real_host_grid_composition','original_real_host_grid_admitted']):
 d=declarations(R/public['file'])[public['declaration']];assert d['statement_sha256']==public['statement_sha256'] and d['kind']=='specification'
 meta={'file':public['file'],'declaration':public['declaration'],'kind':'specification','statement_sha256':d['statement_sha256'],'model_kind':'backend_value_contract','role':'parametric_value_contract','scope_relation':'full_declared_stage','source_variant':'original_source_translation','covers_stages':[stage],'dtype_profile':'Original exact-real-erased forward/backward source execution, complete inner/noninner grids, source heuristics and six autotune configurations. The admitted contract preserves cold/cache-hit fp64 first frontend rejections; the helper is dtype-erased mathematical execution including frontend-rejected profiles. Rounded typed values and finite scalar/pointer/view domains remain open.','dtype_scope_relation':'selected_explicit_profiles_only','unclosed_numeric_callbacks':[],'extra_numeric_law_ids':[],'native_binding_status':'not_established','review_basis':review['semantic_review'],'original_scope':{'standard_accepted':False,'covered':stage,'remaining_structural_obligations':review['remaining_finite_obligations'],'numerical_registration':'No numerical registrations: real erasure is explicit; no false real=floating equality is deferred.','whole_program_completion_increment':0}}
 task['required_value_stages'].append(stage);task['declarations'].append(meta)
rootreview={'schema':1,'kind':'softmax_source_host_grid_root_review','reviewer':'root','status':'reviewed_partial_real_source_host_grid_extension','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{'softmax_flaggems':sha(canonical(value_scope(task)))},'reviewed_metadata_corrections_sha256':{},'semantic_review':review['semantic_review'],'new_public_contracts':{'softmax_flaggems':task['declarations']},'whole_stage_residuals':{'softmax_flaggems':review['remaining_finite_obligations']},'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'evidence_limitations':review['evidence_limitations'],'retained_final_linter_warning':review['retained_final_linter_warning'],'producer_receipt_sha256':review['producer_receipt_sha256'],'source_transformations':[],'numerical_registration_boundary':'The 41 registered rules remain unchanged; real-erased softmax adds no numerical deferrals.','acceptance':'No increment; softmax remains pending. Both previously accepted scopes retain all source/coverage/numerical evidence, with catalog-only review rebinding.'}
write_json(B/'root-review.json',rootreview);task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']};supp['tasks'].append(task)
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==60
shutil.copyfile(OLD/'numerical-followup.json',B/'numerical-followup.json');shutil.copyfile(R/'bench/original_program_scope.json',B/'evidence/ledger-before.json')
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'compiled_artifacts_not_copied':receipt['excluded_generated_build_artifacts'],'new_source_sha256':state['new_source_sha256']})
f=R/'.github/workflows/bench-audit.yml';txt=f.read_text();assert txt.count(OLDREL)==3;f.write_text(txt.replace(OLDREL,NEWREL))
(B/'README.md').write_text('''# Softmax original real source host/grid composition

Nine additive modules introduce two public exact-real-erased execution
specifications for the original softmax forward/backward producer-consumer chain.
Every original kernel invocation executes on the preceding memory, including
arbitrary finite autotune trial sequences and the final launch. The proofs cover
the four complete grids, full logical frames, saved-output/gradient aliasing,
actual Torch primitive-produced strides and source dispatch/rejection order.

The primary admitted contract preserves cold/cache-hit fp64 fallback rejection.
The mathematical helper also has real-erased frontend-rejected instances, which
are not source-admitted native executions. Undefined backward padding remains
arbitrary. K=0 unsafe behavior, M/N zero errors and the unused tl.cidv helper
rejection are recorded without inventing successful guards. Rounded typed
arithmetic and finite scalar/pointer/view domains remain structural obligations.
No numerical assumptions or source acceptance increment are introduced.

Root independently replayed 960 host, 352 heuristic, seven boundary, 80 frontend,
252 autotune, 24 alias and 24 layout cases, plus the helper probe. All 42 producer
compilation attempts, including 12 failures, are preserved. The final benign
HostGridTuning simpa linter warning remains visible. Historical external object
hashes and raw IR for every profile were not retained by the producer; the root
review records that limitation and a full active integration build is required.

Both accepted source scopes remain separate from the 181 pending programs.
All previous public statements, 41 numerical rules and the original metadata
baseline are preserved. Integration results will be recorded after actual checks.
No GPU execution, official comparator success or whole-corpus CI is claimed.
''')
print(json.dumps({'checkpoint':str(B),'new_modules':9,'new_publics':2,'supplement_execution':54,'supplement_protocol':6,'scope_increment':0,'retained_artifacts':len(retained)},indent=2))
