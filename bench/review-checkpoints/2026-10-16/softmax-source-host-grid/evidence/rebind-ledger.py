from pathlib import Path
import copy,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');B=R/'bench/review-checkpoints/2026-10-16/softmax-source-host-grid';OLD=R/'bench/review-checkpoints/2026-10-16/relu-finite-source-acceptance'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings,validate_task
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
cat=load(B/'current-source-catalog.json');validate_catalog(R,cat);supp=load(B/'catalog-supplement.json');validate_supplement(R,R/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,R/'bench/review-checkpoints')
ledger=load(R/'bench/original_program_scope.json');before=load(B/'evidence/ledger-before.json');assert ledger==before
crows={t['task']:t for t in cat['tasks']};oldrows={t['task']:t for t in before['tasks']};srows={t['task']:t for t in supp['tasks']};review=load(B/'root-review.json');priorcat={t['task']:t for t in load(OLD/'current-source-catalog.json')['tasks']};queue=load(R/ledger['deferred_queue']['path'])
changed={}
for name,status in [('matrix_transpose','verified'),('relu_triton_kernel','verified_with_deferred_numerics')]:
 row=copy.deepcopy(oldrows[name]);assert row['input_sha256']==task_bindings(R,cat,crows[name]);assert crows[name]==priorcat[name]
 oldreview=load(R/row['review']['path']);assert hash_file(R/row['review']['path'])==row['review']['sha256']
 for field,key in [('source_inventory_sha256','source_inventory'),('coverage_sha256','coverage'),('deferred_numerical_sha256','deferred_numerical_assumptions'),('callback_assumptions_sha256','callback_assumptions')]:assert oldreview[field]==sha(canonical(row[key]))
 oldreview['catalog_sha256']=sha(canonical(cat));oldreview['catalog_only_rebind']={'previous_review':row['review'],'reason':'Only softmax additive real source host/grid modules and supplementary contracts changed; this accepted task retains identical inputs, public statements, coverage, cases, numerical assumptions and substantive review.','task_catalog_row_sha256':sha(canonical(crows[name]))}
 path=B/(name+'-original-scope-review-rebound.json');write_json(path,oldreview);row['review']={'path':path.relative_to(R).as_posix(),'sha256':hash_file(path)};row['accepted_scope_progress']['scope_review']=row['review'];assert validate_task(R,cat,crows[name],row,queue)==status;changed[name]=row
name='softmax_flaggems';row=copy.deepcopy(oldrows[name]);assert row['status']=='pending' and row['review'] is None
row['input_sha256']=task_bindings(R,cat,crows[name]);row['structural_gaps']=[{'kind':'remaining_original_program_scope','finding':g,'can_defer_as_numeric_assumption':False} for g in review['whole_stage_residuals'][name]]
row['partial_progress']={'status':'reviewed_partial_real_host_grid_extension_not_whole_program_acceptance','review':srows[name]['review'],'new_declarations':review['new_public_contracts'][name],'previous_partial_progress':row.get('partial_progress'),'previous_structural_gaps':oldrows[name]['structural_gaps'],'completion_increment':0,'new_numerical_registrations':0,'native_execution':False,'semantics':'exact_real_erasure_only','grid_paths':4,'autotune_configs':6,'host_ast_probes':960,'autotune_route_probes':252,'known_source_behavior':'Cold fp64 fallback CompilationError escapes before later selection; saved-output/gradient aliasing and noncanonical singleton strides retained; backward undefined padding arbitrary; K=0 unsafe inner source path not a new guard.'};changed[name]=row
for i,row in enumerate(ledger['tasks']):
 if row['task'] in changed:ledger['tasks'][i]=changed[row['task']]
 else:assert row==oldrows[row['task']] and row['input_sha256']==task_bindings(R,cat,crows[row['task']])
assert sum(t['status']=='verified' for t in ledger['tasks'])==1 and sum(t['status']=='verified_with_deferred_numerics' for t in ledger['tasks'])==1 and sum(t['status']=='pending' for t in ledger['tasks'])==181
ledger['catalog_sha256']=sha(canonical(cat));ledger['progress_checkpoint']=(B/'README.md').relative_to(R).as_posix()
md=R/'bench/original_program_scope.md';txt=md.read_text().replace('`relu-finite-source-acceptance`','`softmax-source-host-grid`').replace('contains 52\nexecution contracts and\nsix protocol contracts across 15 tasks','contains 54\nexecution contracts and\nsix protocol contracts across 16 tasks');assert 'contains 54' in txt and 'six protocol contracts across 16 tasks' in txt
txt+='\nSoftmax now composes its original forward/backward real-erased execution across\nall four grids, host metadata and arbitrary finite autotune trials, preserving\nfirst compiler rejection and saved-output/gradient aliasing. Its two new public\ncontracts remain partial progress: typed rounded arithmetic and finite scalar/\npointer/view domains still prevent whole-source acceptance. No additional\nnumerical registrations or completed programs are counted.\n'
write_json(R/'bench/original_program_scope.json',ledger);md.write_text(txt)
print(json.dumps({'scope_verified':1,'scope_verified_with_deferred_numerics':1,'pending':181,'partial_rows_updated':['softmax_flaggems'],'accepted_catalog_reviews_rebound':['matrix_transpose','relu_triton_kernel'],'unrelated_rows_unchanged':180,'catalog_sha256':ledger['catalog_sha256']},indent=2))
