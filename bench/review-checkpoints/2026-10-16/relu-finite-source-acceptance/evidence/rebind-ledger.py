from pathlib import Path
import copy,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance';OLD=R/'bench/review-checkpoints/2026-10-16/vector-typed-host-and-rotary-grid'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings,validate_task
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
from exact_numerical_bindings import collect,validate_exact_queue
cat=load(B/'current-source-catalog.json');validate_catalog(R,cat);supp=load(B/'catalog-supplement.json');validate_supplement(R,R/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,R/'bench/review-checkpoints')
ledger=load(R/'bench/original_program_scope.json');before=load(B/'evidence/ledger-before.json');assert ledger==before
crows={t['task']:t for t in cat['tasks']};oldrows={t['task']:t for t in before['tasks']};srows={t['task']:t for t in supp['tasks']}
legacy,authority,queue=collect(R);assert queue==load(R/'bench/tritonbench_g/deferred_numerical_experiments.json');assert len(queue['rules'])==41
priorQueue=load(B/'evidence/numerical-queue-before.json');assert queue['rules'][:29]==priorQueue['rules'];assert authority['requirements'][:3]==priorQueue['exact_callback_requirements']
assert all(queue['exact_binding_provenance'][k]==v for k,v in priorQueue['exact_binding_provenance'].items())
mrow=copy.deepcopy(oldrows['matrix_transpose']);assert mrow['input_sha256']==task_bindings(R,cat,crows['matrix_transpose'])
mreview=load(R/mrow['review']['path']);assert hash_file(R/mrow['review']['path'])==mrow['review']['sha256']
assert crows['matrix_transpose']==next(t for t in load(OLD/'current-source-catalog.json')['tasks'] if t['task']=='matrix_transpose')
mreview['catalog_sha256']=sha(canonical(cat));mreview['catalog_only_rebind']={'previous_review':mrow['review'],'reason':'Only finite ReLU modules/numerical registrations and supplemental metadata changed; all matrix source/coverage/cases/assumptions and its task catalog row remain identical.','matrix_task_catalog_row_sha256':sha(canonical(crows['matrix_transpose']))}
mp=B/'matrix-original-scope-review-rebound.json';write_json(mp,mreview);mrow['review']={'path':mp.relative_to(R).as_posix(),'sha256':hash_file(mp)};mrow['accepted_scope_progress']['scope_review']=mrow['review'];assert validate_task(R,cat,crows['matrix_transpose'],mrow,queue)=='verified'
row=load(B/'evidence/producer/relu/evidence/scope-row-proposal.json');name='relu_triton_kernel';assert row['source_inventory']==oldrows[name]['source_inventory'];row.pop('proposed_status_after_independent_root_review')
row['status']='verified_with_deferred_numerics';row['input_sha256']=task_bindings(R,cat,crows[name]);row['acceptance_boundary_review'].update(status='independently_reviewed',queue_integration='12 reviewed scalar families / 84 exact ports registered; whole-queue equality passes; numerical validation remains deferred')
review={'schema':1,'scope_schema':2,'kind':'original_program_scope_review','task':name,'reviewer':'root','status':'reviewed','catalog_sha256':sha(canonical(cat)),'input_sha256':row['input_sha256'],'source_inventory_sha256':sha(canonical(row['source_inventory'])),'coverage_sha256':sha(canonical(row['coverage'])),'deferred_numerical_sha256':sha(canonical(row['deferred_numerical_assumptions'])),'callback_assumptions_sha256':sha(canonical(row['callback_assumptions'])),'structural_gaps':[],'semantic_review':load(B/'root-review.json')['semantic_review'],'finite_acceptance_policy':{'path':'bench/original_program_acceptance.md','sha256':hash_file(R/'bench/original_program_acceptance.md')},'evidence':{key:{'path':(B/path).relative_to(R).as_posix(),'sha256':hash_file(B/path)} for key,path in [('producer_receipt','evidence/producer/relu/evidence/receipt.json'),('root_package_review','evidence/root-validation/review.json'),('root_value_review','root-review.json')]},'boundary':'Finite source-scope acceptance under explicit metadata/frontend/typed-memory primitives; numerical conformance is exactly deferred. Known pid0-only source behavior remains. Official comparator and GPU/native correctness are independent and not established.'}
rp=B/'relu-original-scope-review.json';write_json(rp,review);row['review']={'path':rp.relative_to(R).as_posix(),'sha256':hash_file(rp)};row['accepted_scope_progress']={'status':'finite_source_scope_accepted_with_deferred_numerics','scope_review':row['review'],'previous_partial_progress':oldrows[name].get('partial_progress'),'closed_structural_gaps':oldrows[name]['structural_gaps'],'source_cases':340,'numerical_families':12,'numerical_ports':84,'known_source_bug':'Only pid0 stores; N>1024 output tail retains arbitrary allocation contents.','native_execution':False}
assert validate_task(R,cat,crows[name],row,queue)=='verified_with_deferred_numerics'
for i,t in enumerate(ledger['tasks']):
 if t['task']==name:ledger['tasks'][i]=row
 elif t['task']=='matrix_transpose':ledger['tasks'][i]=mrow
 else:assert t==oldrows[t['task']] and t['input_sha256']==task_bindings(R,cat,crows[t['task']])
ledger['catalog_sha256']=sha(canonical(cat));ledger['deferred_queue']={'path':'bench/tritonbench_g/deferred_numerical_experiments.json','sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json')};ledger['progress_checkpoint']=(B/'README.md').relative_to(R).as_posix()
write_json(R/'bench/original_program_scope.json',ledger)
md=R/'bench/original_program_scope.md';text=md.read_text().replace('`vector-typed-host-and-rotary-grid`','`relu-finite-source-acceptance`')
text=text.replace('Matrix transpose has passed its schema2 source-scope check; 182 scopes\nremain pending.','Matrix transpose has passed its schema2 source-scope check. ReLU has passed\nwith exactly registered numerical deferrals; 181 scopes remain pending.')
text=text.replace('contains 51\nexecution contracts and','contains 52\nexecution contracts and')
text+='\nReLU now has complete finite source coverage for all 17 canonical CUDA profiles,\n340 outcome cases and the safe index domain through N=2^31. Its 12 numerical\nfamilies / 84 exact ports remain unvalidated. Source pointer/view permissions,\nfresh allocation metadata and pid0-only stores are explicit. For N>1024 this\naccepts the original physical prefix behavior, with arbitrary untouched output\ntail; it does not assert whole-tensor mathematical ReLU or native GPU correctness.\n'
md.write_text(text)
print(json.dumps({'verified':1,'verified_with_deferred_numerics':1,'pending':181,'accepted_total':2,'catalog_public_declarations':sum(len(t['declarations']) for t in cat['tasks']),'new_numerical_rules':12,'total_numerical_rules':41,'matrix_rebound':True,'unrelated_scope_rows_unchanged':181,'exact_queue':validate_exact_queue(queue,authority,legacy)},indent=2))
