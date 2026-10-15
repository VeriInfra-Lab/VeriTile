from pathlib import Path
import copy,json,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'relu-finite-acceptance-wip';V=T/'relu-finite-root-validation';OLDREL='2026-10-16/vector-typed-host-and-rotary-grid';NEWREL='2026-10-16/relu-finite-source-acceptance';OLD=R/'bench/review-checkpoints'/OLDREL;B=R/'bench/review-checkpoints'/NEWREL
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from current_value_metadata import value_scope

def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(p)==h,str(p)
assert not git('status','--porcelain');assert not B.exists();base=git('rev-parse','HEAD').decode().strip()
exact(V/'review.json','30c9a722e03c76521da3daf7ec370c8151fb18dd3dbcd424ad185451bc99c909');review=load(V/'review.json')
for p,h in review['checked_file_sha256'].items():exact(Path(p),h)
for p,h in review['root_replay_artifact_sha256'].items():exact(V/p,h)
r=load(P/'evidence/receipt.json');sources=r['modules'];assert len(sources)==5 and all(not(R/p).exists() for p in sources)
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'));task=next(t for t in supp['tasks'] if t['task']=='relu_triton_kernel');decl=load(P/'evidence/declaration-proposal.json');decl['review_basis']=review['semantic_review']
assert decl['native_binding_status']=='not_established';task['declarations'].append(decl);task['source_sha256'].update(sources);task['required_value_stages']+= [s for s in decl['covers_stages'] if s not in task['required_value_stages']]
ledger=load(R/'bench/original_program_scope.json');oldrow=next(t for t in ledger['tasks'] if t['task']=='relu_triton_kernel');inputs={**oldrow['input_sha256']['lean'],**oldrow['input_sha256']['python'],**sources}
for p,row in load(P/'evidence/dependencies-before.json')['project'].items():inputs[p]=row['sha256']
basepath=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'metadata_baseline_sha256':hashes(basepath),'new_source_sha256':sources,'previous_queue_sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json'),'producer_receipt_sha256':hash_file(P/'evidence/receipt.json'),'prior_catalog_publics':950,'previous_supplement_declarations':57,'prior_source_scope':{'verified':1,'pending':182}}
for p,h in sources.items():exact(P/p,h);(R/p).write_bytes((P/p).read_bytes())
B.mkdir(parents=True);(B/'evidence').mkdir();retained={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dst=B/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);retained[rel]=h
for p,h in {**r['modules'],**r['artifacts'],'evidence/receipt.json':hash_file(P/'evidence/receipt.json')}.items():retain(P/p,'evidence/producer/relu/'+p+('.txt' if p.endswith('.lean') else ''),h)
for p in V.rglob('*'):
 if p.is_file():retain(p,'evidence/root-validation/'+p.relative_to(V).as_posix())
retain(T/'review_relu_finite.py','evidence/review-package.py');retain(Path(__file__),'evidence/prepare.py')
write_json(B/'evidence/ledger-before.json',ledger);shutil.copyfile(R/'bench/tritonbench_g/deferred_numerical_experiments.json',B/'evidence/numerical-queue-before.json')
rootreview={'schema':1,'kind':'relu_finite_source_integration_root_review','reviewer':'root','status':'reviewed_for_finite_scope_acceptance_with_registered_numerical_deferrals','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':{'relu_triton_kernel':sha(canonical(value_scope(task)))},'reviewed_metadata_corrections_sha256':{},'semantic_review':review['semantic_review'],'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'new_public_contracts':[decl],'producer_receipt_sha256':state['producer_receipt_sha256'],'numeric_registration':{'families':12,'ports':84,'cases':340,'native_validation':False},'native_execution':False,'acceptance_boundary_policy_sha256':review['finite_acceptance_policy_sha256']}
write_json(B/'root-review.json',rootreview);task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']}
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==58
write_json(B/'evidence/preparation-state.json',state);write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'compiled_artifacts_not_copied':r['module_objects'],'new_source_sha256':sources})
f=R/'.github/workflows/bench-audit.yml';text=f.read_text();assert text.count(OLDREL)==3;f.write_text(text.replace(OLDREL,NEWREL))
(B/'README.md').write_text('''# ReLU finite source acceptance\n\nFive additive modules connect the original ReLU wrapper and every canonical\ndtype route to actual typed/fp64 grid execution, returned metadata, source views,\nfinite pointers/masks and frames. The safe CUDA sm90 domain includes N=2^31;\nlarger index overflow remains explicit. All 17 profiles, 340 outcome cases and\n12 numerical families / 84 ports are bound to the public execution statement.\nNumerical conformance remains deferred, and GPU correctness is not established.\n\nThe source writes only pid0. For N>1024 the remaining output stays arbitrary\nallocation contents. Safe noncontiguous views have physical flat-prefix behavior;\nthis is not a false logical elementwise ReLU claim. All original failure evidence\nand the historical baseline are preserved. Integration results follow the actual\ncurrent catalog, schema2, exact numerical and build checks.\n''')
print(json.dumps({'checkpoint':str(B),'modules':5,'publics':1,'supplement_declarations':58,'root_review_sha256':hash_file(B/'root-review.json'),'producer_files_retained':len(retained)},indent=2))
