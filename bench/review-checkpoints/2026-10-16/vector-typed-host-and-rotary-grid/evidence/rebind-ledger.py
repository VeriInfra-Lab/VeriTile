from pathlib import Path
import copy,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/vector-typed-host-and-rotary-grid'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings,validate_task
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
cat=load(B/'current-source-catalog.json');validate_catalog(R,cat);supp=load(B/'catalog-supplement.json');validate_supplement(R,R/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,R/'bench/review-checkpoints')
ledger=load(R/'bench/original_program_scope.json');before=load(B/'evidence/ledger-before.json');assert ledger==before
crows={t['task']:t for t in cat['tasks']};oldrows={t['task']:t for t in before['tasks']};srows={t['task']:t for t in supp['tasks']};rootReview=load(B/'root-review.json')
# Scope semantics/coverage/source hashes are unchanged for the accepted matrix.
# The required review catalog hash is re-bound in a new record, never by editing
# the previous frozen scope review.
mrow=copy.deepcopy(oldrows['matrix_transpose']);assert mrow['input_sha256']==task_bindings(R,cat,crows['matrix_transpose'])
mreview=load(R/mrow['review']['path']);assert hash_file(R/mrow['review']['path'])==mrow['review']['sha256']
for name,key in [('source_inventory_sha256','source_inventory'),('coverage_sha256','coverage'),('deferred_numerical_sha256','deferred_numerical_assumptions'),('callback_assumptions_sha256','callback_assumptions')]:assert mreview[name]==sha(canonical(mrow[key]))
mreview['catalog_sha256']=sha(canonical(cat));mreview['catalog_only_rebind']={'previous_review':mrow['review'],'reason':'Only vector/rotary modules and supplementary contracts changed; matrix inputs, public statements, coverage, cases, numerical assumptions and substantive acceptance review are identical.','matrix_task_catalog_row_sha256':sha(canonical(crows['matrix_transpose']))}
priorCat=load(R/'bench/review-checkpoints/2026-10-16/matrix-scope-and-relu-fp64/current-source-catalog.json');assert crows['matrix_transpose']==next(t for t in priorCat['tasks'] if t['task']=='matrix_transpose')
write_json(B/'matrix-original-scope-review-rebound.json',mreview);mrow['review']={'path':(B/'matrix-original-scope-review-rebound.json').relative_to(R).as_posix(),'sha256':hash_file(B/'matrix-original-scope-review-rebound.json')};mrow['accepted_scope_progress']['scope_review']=mrow['review']
queue=load(R/ledger['deferred_queue']['path']);assert validate_task(R,cat,crows['matrix_transpose'],mrow,queue)=='verified'
changed={}
for name in ['vector_addition','rotary_emb']:
 row=copy.deepcopy(oldrows[name]);assert row['status']=='pending' and row['review'] is None
 row['input_sha256']=task_bindings(R,cat,crows[name]);row['structural_gaps']=[{'kind':'remaining_original_program_scope','finding':finding,'can_defer_as_numeric_assumption':False} for finding in rootReview['whole_stage_residuals'][name]]
 row['partial_progress']={'status':'reviewed_partial_host_grid_extension_not_whole_program_acceptance','review':srows[name]['review'],'new_declarations':rootReview['new_public_contracts'][name],'previous_partial_progress':row.get('partial_progress'),'previous_structural_gaps':oldrows[name]['structural_gaps'],'completion_increment':0,'new_numerical_registrations':0,'native_execution':False}
 if name=='vector_addition':row['partial_progress'].update(canonical_pairs=289,admitted_routes_composed=189,signature_rejections=64,addition_rejections=36,unregistered_numerical_operation_bindings=308,defined_domain='Nine fp64 integer/bool OutputDefined routes and shared mixed conversions retained. Physical source pointer/view/address binding and unsafe outcomes remain pending.')
 else:row['partial_progress'].update(semantics='exact_real_erasure_only',original_host_ast_probes=63,complete_source_statements_per_program=34,known_source_behavior='Q-derived head grid leaves extra K heads unchanged; Cos strides reused for both trig inputs; odd and partial tails raw-framed.')
 changed[name]=row
for i,row in enumerate(ledger['tasks']):
 name=row['task']
 if name=='matrix_transpose':ledger['tasks'][i]=mrow
 elif name in changed:ledger['tasks'][i]=changed[name]
 else:assert row==oldrows[name] and row['input_sha256']==task_bindings(R,cat,crows[name])
assert sum(t['status']=='verified' for t in ledger['tasks'])==1 and sum(t['status']=='pending' for t in ledger['tasks'])==182
ledger['catalog_sha256']=sha(canonical(cat));ledger['progress_checkpoint']=(B/'README.md').relative_to(R).as_posix()
md=R/'bench/original_program_scope.md';text=md.read_text().replace('`matrix-scope-and-relu-fp64`','`vector-typed-host-and-rotary-grid`').replace('contains 48\nexecution contracts and\nfour protocol contracts across 14 tasks','contains 51\nexecution contracts and\nsix protocol contracts across 15 tasks')
assert 'contains 51' in text and 'six protocol contracts across 15 tasks' in text
text+='\nThe latest partial extensions compose vector addition\'s 189 existing typed\nroutes through source-order host allocation/assertions and canonical rejection\ncases, and RotaryEmb\'s original real-erased source through the complete Q-derived\ngrid with initial-memory formulas and tail frames. Vector safe-view/address and\nexact numerical registration gaps remain; rotary rounded execution and dtype/\nadmission/address-domain gaps remain. Neither task is counted as accepted.\n'
write_json(R/'bench/original_program_scope.json',ledger);md.write_text(text)
print(json.dumps({'scope_verified':1,'pending':182,'partial_rows_updated':list(changed),'matrix_catalog_review_rebound':True,'unrelated_rows_unchanged':180,'catalog_sha256':ledger['catalog_sha256']},indent=2))
