from pathlib import Path
import copy,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance';OLD=R/'bench/review-checkpoints/2026-10-16/vector-finite-source-acceptance'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings,validate_task
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
from exact_numerical_bindings import collect,validate_exact_queue
cat=load(B/'current-source-catalog.json');validate_catalog(R,cat);supp=load(B/'catalog-supplement.json');validate_supplement(R,R/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,R/'bench/review-checkpoints')
ledger=load(R/'bench/original_program_scope.json');before=load(B/'evidence/ledger-before.json');assert ledger==before
crows={t['task']:t for t in cat['tasks']};oldrows={t['task']:t for t in before['tasks']};oldcat={t['task']:t for t in load(OLD/'current-source-catalog.json')['tasks']}
legacy,authority,queue=collect(R);assert queue==load(R/'bench/tritonbench_g/deferred_numerical_experiments.json');assert len(queue['rules'])==100
priorQueue=load(B/'evidence/numerical-queue-before.json');assert queue['rules'][:84]==priorQueue['rules'];assert authority['requirements'][:5]==priorQueue['exact_callback_requirements']
assert all(queue['exact_binding_provenance'][k]==v for k,v in priorQueue['exact_binding_provenance'].items())
accepted={}
for name in ['matrix_transpose','relu_triton_kernel','vector_addition']:
 row=copy.deepcopy(oldrows[name]);assert row['input_sha256']==task_bindings(R,cat,crows[name]);assert crows[name]==oldcat[name]
 review=load(R/row['review']['path']);assert hash_file(R/row['review']['path'])==row['review']['sha256']
 review['catalog_sha256']=sha(canonical(cat));review['catalog_only_rebind']={'previous_review':row['review'],'reason':'Only quantize_global finite sources and numerical registrations changed. All task sources, coverage, cases, assumptions and the task catalog row are unchanged.','task_catalog_row_sha256':sha(canonical(crows[name]))}
 path=B/(name+'-original-scope-review-rebound.json');write_json(path,review);row['review']={'path':path.relative_to(R).as_posix(),'sha256':hash_file(path)};row['accepted_scope_progress']['scope_review']=row['review'];assert validate_task(R,cat,crows[name],row,queue)==oldrows[name]['status'];accepted[name]=row
row=load(B/'evidence/producer/quant-global/evidence/scope-row-proposal.json');name='quantize_global';assert row['source_inventory']==oldrows[name]['source_inventory'];row.pop('proposed_status_after_independent_root_review')
row['status']='verified_with_deferred_numerics';row['input_sha256']=task_bindings(R,cat,crows[name]);row['acceptance_boundary_review'].update(status='independently_reviewed',queue_integration='16 reviewed scalar families / 420 exact ports registered; whole-queue equality passes; numerical validation remains deferred')
review={'schema':1,'scope_schema':2,'kind':'original_program_scope_review','task':name,'reviewer':'root','status':'reviewed','catalog_sha256':sha(canonical(cat)),'input_sha256':row['input_sha256'],'source_inventory_sha256':sha(canonical(row['source_inventory'])),'coverage_sha256':sha(canonical(row['coverage'])),'deferred_numerical_sha256':sha(canonical(row['deferred_numerical_assumptions'])),'callback_assumptions_sha256':sha(canonical(row['callback_assumptions'])),'structural_gaps':[],'semantic_review':load(B/'root-review.json')['semantic_review'],'finite_acceptance_policy':{'path':'bench/original_program_acceptance.md','sha256':hash_file(R/'bench/original_program_acceptance.md')},'evidence':{key:{'path':(B/path).relative_to(R).as_posix(),'sha256':hash_file(B/path)} for key,path in [('producer_receipt','evidence/producer/quant-global/evidence/receipt.json'),('root_package_review','evidence/root-validation/review.json'),('root_value_review','root-review.json')]},'boundary':'Finite source acceptance under explicit dense strided CPU/CUDA metadata/primitive boundary. All22 dtypes, two configs, host/trial execution, physical writes and source hazards remain. All finite words either return independent results or reach undefined active llrint, without a Safe premise. Numerical conformance is exactly deferred; native/comparator correctness remains separate.'}
rp=B/'quant-global-original-scope-review.json';write_json(rp,review);row['review']={'path':rp.relative_to(R).as_posix(),'sha256':hash_file(rp)};row['accepted_scope_progress']={'status':'finite_source_scope_accepted_with_deferred_numerics','scope_review':row['review'],'previous_partial_progress':oldrows[name].get('partial_progress'),'closed_structural_gaps':oldrows[name]['structural_gaps'],'source_cases':2400,'numerical_families':16,'numerical_ports':420,'source_semantics':'Logical-view abs/max and reciprocal feed actual flat typed kernel reads and int8 byte stores through both configs and all trials; exact return metadata, frames, signed32 wrap/view hazards and undefined llrint are retained.','native_execution':False}
assert validate_task(R,cat,crows[name],row,queue)=='verified_with_deferred_numerics';accepted[name]=row
for i,t in enumerate(ledger['tasks']):
 if t['task'] in accepted:ledger['tasks'][i]=accepted[t['task']]
 else:assert t==oldrows[t['task']] and t['input_sha256']==task_bindings(R,cat,crows[t['task']])
ledger['catalog_sha256']=sha(canonical(cat));ledger['deferred_queue']={'path':'bench/tritonbench_g/deferred_numerical_experiments.json','sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json')};ledger['progress_checkpoint']=(B/'README.md').relative_to(R).as_posix()
write_json(R/'bench/original_program_scope.json',ledger)
md=R/'bench/original_program_scope.md';text=md.read_text()
replacements={
 '**3 source scopes accepted and 180 pending**':'**4 source scopes accepted and 179 pending**',
 '`relu_triton_kernel` and `vector_addition` are verified with exactly registered':'`relu_triton_kernel`, `vector_addition` and `quantize_global` are verified with exactly registered',
 '`vector-finite-source-acceptance`':'`quant-global-finite-source-acceptance`',
 'current queue has 84 rules, with 786 exact bindings across 2,999 cases in five\ndeclarations.':'reviewed quantize_global extension adds 16 families and 420 ports in 2,400 cases.\nThe current queue has 100 rules, with 1,206 exact bindings across 5,399 cases in\nsix declarations.',
 'vector addition have passed with exactly registered numerical deferrals;\n180 scopes remain pending.':'vector addition and quantize_global have passed with exactly registered numerical\ndeferrals; 179 scopes remain pending.',
 'contains 55\nexecution contracts':'contains 56\nexecution contracts',
 'six protocol contracts across 16 tasks':'six protocol contracts across 17 tasks',
}
for old,new in replacements.items():assert old in text,old;text=text.replace(old,new)
text+='\nQuantize_global now covers all 22 canonical dense dtype families, both source\nconfigs, host allocation and logical abs/max, typed linear kernel execution,\nall autotune trials, finite byte-memory effects and frames. Its 2,400 source\ncases include exact first rejections, source hazards and returned-or-undefined\nconversion outcomes without a Safe premise. The 16 numerical families / 420\nports remain unvalidated; no native/GPU or comparator success is claimed.\n'
md.write_text(text)
print(json.dumps({'verified':1,'verified_with_deferred_numerics':3,'pending':179,'accepted_total':4,'catalog_public_declarations':sum(len(t['declarations']) for t in cat['tasks']),'new_numerical_rules':16,'total_numerical_rules':100,'accepted_reviews_rebound':list(accepted)[:3],'unrelated_scope_rows_unchanged':179,'exact_queue':validate_exact_queue(queue,authority,legacy)},indent=2))
