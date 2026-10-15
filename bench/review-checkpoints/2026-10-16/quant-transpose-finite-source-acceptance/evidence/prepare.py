from pathlib import Path
import copy,json,shutil,subprocess,sys,importlib.util
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'quant-transpose-finite-acceptance-revision1-wip';V=T/'quant-transpose-revision1-root-validation';PEER=T/'quant-transpose-revision1-matrix-review'
OLDREL='2026-10-16/quant-global-finite-source-acceptance';NEWREL='2026-10-16/quant-transpose-finite-source-acceptance';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'));from adapter_common import canonical,hash_file,load,sha,write_json
from current_value_metadata import value_scope

def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(Path(p))==h,str(p)
def archival(p):return p+'.txt' if p.endswith('.lean') else p
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip();assert base.startswith('09eb0985');assert git('branch','--show-current').decode().strip()=='fp-examples'
exact(P/'evidence/receipt.json','63392bac5342812d467c2bc499e7870e2609d49fa74faaf01731c1391b1dcf83')
exact(V/'review.json','9e0c4b360363f94670bc2477b8783477d4a93aa89ac93a2e48b050b066e87f77')
review=load(V/'review.json');r=load(P/'evidence/receipt.json');sources=r['modules']
for p,h in review['checked_file_sha256'].items():exact(p,h)
for p,h in review['root_replay_artifact_sha256'].items():exact(V/p,h)
assert len(sources)==18 and sources==review['integration_sources'] and all(not(R/p).exists() for p in sources)
for p,h in sources.items():exact(P/p,h)
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'));task=next(t for t in supp['tasks'] if t['task']=='quant_transpose_kernel');decl=load(P/'evidence/declaration-proposal.json');decl['review_basis']=review['semantic_review'];task['declarations'].append(decl);task['source_sha256'].update(sources);task['required_value_stages'] += [s for s in decl['covers_stages'] if s not in task['required_value_stages']]
ledger=load(R/'bench/original_program_scope.json');oldrow=next(t for t in ledger['tasks'] if t['task']=='quant_transpose_kernel');inputs={**oldrow['input_sha256']['lean'],**oldrow['input_sha256']['python'],**sources}
for p in r['dependencies']['captures']:
 for v in load(P/p).get('project',{}).values():inputs[Path(v['source']).relative_to(R).as_posix()]=v['sha256']
BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':hashes(BASE),'new_source_sha256':sources,'previous_queue_sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json'),'previous_ledger_sha256':hash_file(R/'bench/original_program_scope.json'),'producer_receipt_sha256':hash_file(P/'evidence/receipt.json'),'prior_catalog_publics':955,'previous_supplement_declarations':62,'prior_source_scope':{'verified':1,'verified_with_deferred_numerics':1,'pending':181},'historical_large_json_preservation':'Previous ledger/queue retained exactly at base_commit; no redundant full copy is created.'}
for p,h in sources.items():dest=R/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/p,dest)
B.mkdir(parents=True);(B/'evidence').mkdir();retained={};relocations={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dest=B/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);retained[rel]=h
prefix='evidence/producer/quant-transpose/'
for p,h in sources.items():retain(P/p,prefix+r['checkpoint_module_source_paths'][p],h)
for p,h in {**r['artifacts'],'evidence/receipt.json':state['producer_receipt_sha256']}.items():
 dest=archival(p);retain(P/p,prefix+dest,h)
 if dest!=p:relocations[p]=dest
for p,h in review['root_replay_artifact_sha256'].items():retain(V/p,'evidence/root-validation/'+archival(p),h)
retain(V/'review.json','evidence/root-validation/review.json');retain(T/'quant-transpose-revision1-review-attempt003.log','evidence/root-validation/review-attempt003.log');retain(T/'review_quant_transpose_revision1.py','evidence/review-package.py');retain(Path(__file__),'evidence/prepare.py')
peerReceipt=load(PEER/'review-receipt.json');retain(PEER/'review-receipt.json','evidence/peer-validation/review-receipt.json')
for p,h in peerReceipt['artifacts'].items():
 if not p.endswith(('.olean','.olean.private','.olean.server','.ilean','.ir')):retain(PEER/p,'evidence/peer-validation/'+archival(p),h)
rootreview={'schema':1,'kind':'quant_transpose_finite_source_integration_root_review','reviewer':'root','status':'reviewed_for_finite_scope_acceptance_with_registered_numerical_deferrals','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{'quant_transpose_kernel':sha(canonical(value_scope(task)))},'reviewed_metadata_corrections_sha256':{},'semantic_review':review['semantic_review'],'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'new_public_contracts':[decl],'producer_receipt_sha256':state['producer_receipt_sha256'],'numeric_registration':{'families':16,'ports':140,'cases':863,'native_validation':False},'native_execution':False,'acceptance_boundary_policy_sha256':hash_file(R/'bench/original_program_acceptance.md')}
write_json(B/'root-review.json',rootreview);task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']};write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==63
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'new_source_sha256':sources,'archived_producer_path_mapping':relocations,'note':'Frozen historical Lean bytes retain exact hashes with .txt archival suffix; receipt bytes are unchanged. Generated Lean objects are retained by hash outside Git.'})
f=R/'.github/workflows/bench-audit.yml';s=f.read_text();assert s.count(OLDREL)==3;f.write_text(s.replace(OLDREL,NEWREL))
(B/'README.md').write_text('# Quantize transpose finite source acceptance\n\nEighteen additive modules model the original logical abs/max producer, reciprocal,\nrank/allocation/assertion order, both autotune configurations and actual typed\ninteger/byte execution. The main public theorem includes actual wrapped masks\nand uint64 addresses without a Safe or ArithmeticRefines hypothesis. It classifies\nactual grid-limit, access/ownership, integer-division and undefined-conversion\nobservations separately from returned memory. Observations assert no native trap,\nrollback or postfailure result. A mathematical transpose formula is conditional\non the separately stated arithmetic refinement.\n\nAll22 dense CPU/CUDA dtype families retain their original first rejections;\neight admitted types and both configs bind16 numerical families/140 exact ports\nacross863 cases. All finite autotune trials execute on evolving memory. Fresh\noutput metadata, independent last-write byte results, exterior frames and returned\nmaximum/inverse scalar bytes are explicit. Native numeric correspondence remains\ndeferred; no GPU execution or official comparator success is claimed.\n\nProducer and independent peer evidence, root rebuild/review and failed attempts\nare preserved. The producer omitted one direct external import object from its\nhistorical snapshot; root captured Mathlib.Data.List.Range before its independent\nrebuild and verified its current hash unchanged, without rewriting that history.\nFrozen historical Lean sources use a .txt archival suffix with exact bytes; the\nactive18 sources are registered separately. Integration results follow below.\n')
print(json.dumps({'checkpoint':str(B),'modules':18,'publics':1,'supplement_declarations':63,'root_review_sha256':hash_file(B/'root-review.json'),'retained_files':len(retained),'historical_lean_relocations':len(relocations)},indent=2))
