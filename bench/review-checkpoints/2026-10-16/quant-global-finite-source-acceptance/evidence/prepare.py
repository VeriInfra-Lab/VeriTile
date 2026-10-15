from pathlib import Path
import copy,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'quant-global-finite-acceptance-wip/frozen-v2';V=T/'quant-global-finite-root-validation'
OLDREL='2026-10-16/vector-finite-source-acceptance';NEWREL='2026-10-16/quant-global-finite-source-acceptance';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from current_value_metadata import value_scope

def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(Path(p))==h,str(p)
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip()
assert git('branch','--show-current').decode().strip()=='fp-examples'
exact(P/'evidence/receipt.json','e1edc3543369fd52ed4d66dda58e6ea81778ea7312437adf7ebd2ed83c5e15e6')
exact(V/'review.json','676cd2d3cb245d600dacb2c2734ef38cd0fbe0f78fd9817feb490c85437af1ef')
review=load(V/'review.json');r=load(P/'evidence/receipt.json');sources=r['modules']
for p,h in review['checked_file_sha256'].items():exact(p,h)
for p,h in review['root_replay_artifact_sha256'].items():exact(V/p,h)
assert len(sources)==11 and sources==review['integration_sources'] and all(not(R/p).exists() for p in sources)
for p,h in sources.items():exact(P/'overlay'/p,h)
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'));task=next(t for t in supp['tasks'] if t['task']=='quantize_global');decl=load(P/'evidence/declaration-proposal.json');decl['review_basis']=review['semantic_review']
assert decl['native_binding_status']=='not_established';task['declarations'].append(decl);task['source_sha256'].update(sources);task['required_value_stages'] += [s for s in decl['covers_stages'] if s not in task['required_value_stages']]
ledger=load(R/'bench/original_program_scope.json');oldrow=next(t for t in ledger['tasks'] if t['task']=='quantize_global');inputs={**oldrow['input_sha256']['lean'],**oldrow['input_sha256']['python'],**sources}
for p,row in load(P/'evidence/dependencies-project.json')['project'].items():inputs[p]=row['sha256']
BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':hashes(BASE),'new_source_sha256':sources,'previous_queue_sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json'),'producer_receipt_sha256':hash_file(P/'evidence/receipt.json'),'prior_catalog_publics':954,'previous_supplement_declarations':61,'prior_source_scope':{'verified':1,'verified_with_deferred_numerics':2,'pending':180}}
# All immutable authority inputs are checked before repository writes.
for p,h in sources.items():(R/p).write_bytes((P/'overlay'/p).read_bytes())
B.mkdir(parents=True);(B/'evidence').mkdir();retained={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dst=B/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);retained[rel]=h
prefix='evidence/producer/quant-global/'
for p,h in sources.items():retain(P/'overlay'/p,prefix+r['checkpoint_module_source_paths'][p],h)
artifacts={**r['artifacts'],'evidence/receipt.json':hash_file(P/'evidence/receipt.json')}
for p,h in artifacts.items():retain(P/p,prefix+p,h)
for p in [V/'review.json',V/'review.log',*[V/p for p in review['root_replay_artifact_sha256']]]:
 if p.is_file():retain(p,'evidence/root-validation/'+p.relative_to(V).as_posix())
retain(T/'review_quant_global_finite.py','evidence/review-package.py');retain(Path(__file__),'evidence/prepare.py')
write_json(B/'evidence/ledger-before.json',ledger);shutil.copyfile(R/'bench/tritonbench_g/deferred_numerical_experiments.json',B/'evidence/numerical-queue-before.json')
rootreview={'schema':1,'kind':'quant_global_finite_source_integration_root_review','reviewer':'root','status':'reviewed_for_finite_scope_acceptance_with_registered_numerical_deferrals','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{'quantize_global':sha(canonical(value_scope(task)))},'reviewed_metadata_corrections_sha256':{},'semantic_review':review['semantic_review'],'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'new_public_contracts':[decl],'producer_receipt_sha256':state['producer_receipt_sha256'],'numeric_registration':{'families':16,'ports':420,'cases':2400,'native_validation':False},'native_execution':False,'acceptance_boundary_policy_sha256':hash_file(R/'bench/original_program_acceptance.md')}
write_json(B/'root-review.json',rootreview);task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']}
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==62
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'compiled_artifacts_not_copied':[obj for build in r['builds'] for obj in build['compiled_artifacts']],'new_source_sha256':sources})
f=R/'.github/workflows/bench-audit.yml';text=f.read_text();assert text.count(OLDREL)==3;f.write_text(text.replace(OLDREL,NEWREL))
(B/'README.md').write_text('# Quantize global finite source acceptance\n\nEleven additive modules compose the original logical abs/max producer, reciprocal,\nallocation/assertion and typed two-configuration Triton execution. All 22 canonical\ndtype families and original first rejections remain, including fp16/bf16 llrint\ncompilation failures. Eight admitted storage types cover all finite input words:\nreturned values or explicitly undefined active conversions, without a Safe or\nsuccessful-launch premise. Both source configs and arbitrary finite autotune\ntrials execute on evolving memory.\n\nThe physical memory relation composes actual host scalar writes, every successful\nkernel byte store, finite signed indices/pointers, access permissions, original\nlogical-view reduction versus flat kernel reads, input/current allocation devices,\nreturn metadata and frames. N=2^31 is safe with an i64 n argument; larger signed32\noffset wrap and flat-view overrun remain source hazard observations. No invented\nnative trap, rollback or postfailure output is asserted.\n\nThe public contract binds 2,400 cases, with 420 exact numerical ports in 16\nfamilies. Native numeric conformance remains deferred. Four additive dtype labels\npreserve exact rejections; no validator predicate is weakened. GPU correctness,\ncompiler implementation correctness and official comparator results remain separate.\n')
print(json.dumps({'checkpoint':str(B),'modules':11,'publics':1,'supplement_declarations':62,'root_review_sha256':hash_file(B/'root-review.json'),'producer_files_retained':len(retained)},indent=2))
