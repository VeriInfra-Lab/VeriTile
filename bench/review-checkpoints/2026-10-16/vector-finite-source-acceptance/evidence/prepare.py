from pathlib import Path
import copy,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'vector-finite-acceptance-wip';V=T/'vector-finite-root-validation'
OLDREL='2026-10-16/softmax-source-host-grid';NEWREL='2026-10-16/vector-finite-source-acceptance';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from current_value_metadata import value_scope

def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(Path(p))==h,str(p)
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip()
assert git('branch','--show-current').decode().strip()=='fp-examples'
exact(P/'evidence/receipt.json','8349ecfd8051f74130acf58fb8f8aa0b62d4760203188aa392a5f8670c0ec7d8')
exact(V/'review.json','8f1b98f79b4df50ebb065bc31153218590808f576220d6fb2fc26aa3db4b4840')
review=load(V/'review.json');r=load(P/'evidence/receipt.json');sources=r['modules']
for p,h in review['checked_file_sha256'].items():exact(p,h)
for p,h in review['root_replay_artifact_sha256'].items():exact(V/p,h)
assert len(sources)==6 and sources==review['integration_sources'] and all(not(R/p).exists() for p in sources)
for p,h in sources.items():exact(P/r['producer_module_source_paths'][p],h)
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'));task=next(t for t in supp['tasks'] if t['task']=='vector_addition');decl=load(P/'evidence/declaration-proposal.json');decl['review_basis']=review['semantic_review']
assert decl['native_binding_status']=='not_established';task['declarations'].append(decl);task['source_sha256'].update(sources);task['required_value_stages'] += [s for s in decl['covers_stages'] if s not in task['required_value_stages']]
ledger=load(R/'bench/original_program_scope.json');oldrow=next(t for t in ledger['tasks'] if t['task']=='vector_addition');inputs={**oldrow['input_sha256']['lean'],**oldrow['input_sha256']['python'],**sources}
for p,row in load(P/'evidence/dependencies-before.json')['project'].items():inputs[p]=row['sha256']
BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':hashes(BASE),'new_source_sha256':sources,'previous_queue_sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json'),'producer_receipt_sha256':hash_file(P/'evidence/receipt.json'),'prior_catalog_publics':953,'previous_supplement_declarations':60,'prior_source_scope':{'verified':1,'verified_with_deferred_numerics':1,'pending':181}}
# All immutable authority inputs are checked before repository writes.
for p,h in sources.items():(R/p).write_bytes((P/r['producer_module_source_paths'][p]).read_bytes())
B.mkdir(parents=True);(B/'evidence').mkdir();retained={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dst=B/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);retained[rel]=h
prefix='evidence/producer/vector/'
for p,h in sources.items():retain(P/r['producer_module_source_paths'][p],prefix+r['checkpoint_module_source_paths'][p],h)
artifacts={**r['artifacts'],'evidence/receipt.json':hash_file(P/'evidence/receipt.json')}
# The producer excludes every basename receipt.json from its artifact table.
# The nested frontend receipt is independently bound by each scalar provenance.
delta=load(P/'evidence/numerical-authority-delta.json')
for port in delta['provenance'].values():
 ref=port['frontend_receipt'];assert artifacts.get(ref['path'],ref['sha256'])==ref['sha256'];artifacts[ref['path']]=ref['sha256']
for p,h in artifacts.items():retain(P/p,prefix+p,h)
for p in V.rglob('*'):
 if p.is_file():retain(p,'evidence/root-validation/'+p.relative_to(V).as_posix())
retain(T/'review_vector_finite.py','evidence/review-package.py');retain(Path(__file__),'evidence/prepare.py')
write_json(B/'evidence/ledger-before.json',ledger);shutil.copyfile(R/'bench/tritonbench_g/deferred_numerical_experiments.json',B/'evidence/numerical-queue-before.json')
rootreview={'schema':1,'kind':'vector_finite_source_integration_root_review','reviewer':'root','status':'reviewed_for_finite_scope_acceptance_with_registered_numerical_deferrals','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{'vector_addition':sha(canonical(value_scope(task)))},'reviewed_metadata_corrections_sha256':{},'semantic_review':review['semantic_review'],'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'new_public_contracts':[decl],'producer_receipt_sha256':state['producer_receipt_sha256'],'numeric_registration':{'families':43,'ports':616,'cases':2601,'native_validation':False},'native_execution':False,'acceptance_boundary_policy_sha256':hash_file(R/'bench/original_program_acceptance.md')}
write_json(B/'root-review.json',rootreview);task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']}
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==61
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'compiled_artifacts_not_copied':r['excluded_generated_or_duplicate_paths'],'new_source_sha256':sources})
f=R/'.github/workflows/bench-audit.yml';text=f.read_text();assert text.count(OLDREL)==3;f.write_text(text.replace(OLDREL,NEWREL))
(B/'README.md').write_text('''# Vector addition finite source acceptance

Six additive modules connect the original eight-statement kernel and its wrapper
to existing typed execution, physical storage views, finite indices and pointers,
load-before-store memory effects, frames and complete schedule invariance.
The canonical 17 by 17 dtype inventory retains all 189 admitted profiles and 100
actual frontend rejections. Source n_elements is i32, with 0 <= N < 2^31 and
BLOCK_SIZE=1024. Physical y backing capacity suffices even when y.numel < N.

The source returns flat physical addition under actual preserve-format output
metadata. Arbitrary strided inputs are not falsely described as logical Torch
elementwise addition. Undefined active float-to-integer conversions remain an
explicit structural defined-domain restriction, never a numerical deferral or
an invented native trap/rollback result. Allocation and CUDA failures retain
their source order. Input views may alias; fresh output is disjoint.

The public contract binds 2,601 cases and 616 exact numerical ports in 43 new
families. Numerical conformance remains deferred. Source acceptance requires
the active integration build and schema2/exact-authority checks below; it does
not establish compiler implementation correctness or native GPU correctness.
''')
print(json.dumps({'checkpoint':str(B),'modules':6,'publics':1,'supplement_declarations':61,'root_review_sha256':hash_file(B/'root-review.json'),'producer_files_retained':len(retained)},indent=2))
