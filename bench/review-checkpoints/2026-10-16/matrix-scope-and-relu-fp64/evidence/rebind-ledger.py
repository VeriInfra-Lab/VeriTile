from pathlib import Path
import copy,json,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');B=R/'bench/review-checkpoints/2026-10-16/matrix-scope-and-relu-fp64';M=T/'matrix-transpose-acceptance-wip'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings,validate_task
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
cat=load(B/'current-source-catalog.json');validate_catalog(R,cat);supp=load(B/'catalog-supplement.json');validate_supplement(R,R/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,R/'bench/review-checkpoints')
ledger=load(R/'bench/original_program_scope.json');before=load(B/'evidence/ledger-before.json');assert ledger==before
crows={t['task']:t for t in cat['tasks']};oldrows={t['task']:t for t in before['tasks']};srows={t['task']:t for t in supp['tasks']};rootReview=load(B/'root-review.json')
proposal=load(M/'scope-row.proposed.json');mrow=copy.deepcopy(oldrows['matrix_transpose'])
for key in ['scope_schema','source_inventory','coverage','deferred_numerical_assumptions','callback_assumptions','structural_gaps']:mrow[key]=copy.deepcopy(proposal[key])
mrow['input_sha256']=task_bindings(R,cat,crows['matrix_transpose']);mrow['status']='verified';mrow['structural_gaps']=[]
mreview={'schema':1,'scope_schema':2,'kind':'original_program_scope_review','task':'matrix_transpose','reviewer':'root','status':'reviewed','catalog_sha256':sha(canonical(cat)),'input_sha256':mrow['input_sha256'],'source_inventory_sha256':sha(canonical(mrow['source_inventory'])),'coverage_sha256':sha(canonical(mrow['coverage'])),'deferred_numerical_sha256':sha(canonical(mrow['deferred_numerical_assumptions'])),'callback_assumptions_sha256':sha(canonical(mrow['callback_assumptions'])),'structural_gaps':[],'semantic_review':rootReview['semantic_review']['matrix_transpose'],'finite_acceptance_boundary':ledger['acceptance_boundary_review'],'evidence':{'producer_receipt':{'path':(B/'evidence/producer/matrix/receipt.json').relative_to(R).as_posix(),'sha256':hash_file(B/'evidence/producer/matrix/receipt.json')},'root_package_review':{'path':(B/'evidence/package-review.json').relative_to(R).as_posix(),'sha256':hash_file(B/'evidence/package-review.json')},'root_frontend_check':{'path':(B/'evidence/matrix-frontend-root-validation.json').relative_to(R).as_posix(),'sha256':hash_file(B/'evidence/matrix-frontend-root-validation.json')}},'acceptance_boundary':'Original source-scope review under explicit primitive contracts. Native GPU correctness, whole-corpus CI, official comparator and integration results are separate and are not asserted by this scope status.'}
# Prepare source review; the unchanged validator then independently checks all bindings.
write_json(B/'matrix-original-scope-review.json',mreview)
mrow['review']={'path':(B/'matrix-original-scope-review.json').relative_to(R).as_posix(),'sha256':hash_file(B/'matrix-original-scope-review.json')}
mrow['accepted_scope_progress']={'status':'verified_full_original_source_scope','previous_structural_gaps':oldrows['matrix_transpose']['structural_gaps'],'scope_review':mrow['review'],'numerical_deferrals':0,'native_correctness_established':False}
queue=load(R/ledger['deferred_queue']['path']);assert validate_task(R,cat,crows['matrix_transpose'],mrow,queue)=='verified'
lrow=copy.deepcopy(oldrows['relu_triton_kernel']);lrow['input_sha256']=task_bindings(R,cat,crows['relu_triton_kernel'])
lrow['structural_gaps']=[{'kind':'remaining_original_program_scope','finding':finding,'can_defer_as_numeric_assumption':False} for finding in rootReview['whole_stage_residuals']['relu_triton_kernel']]
lrow['partial_progress']={'status':'reviewed_fp64_source_grid_and_finite_host_dispatch_not_whole_program_acceptance','review':srows['relu_triton_kernel']['review'],'new_declarations':rootReview['new_public_contracts']['relu_triton_kernel'],'successful_logical_dtype_routes':13,'canonical_dtype_cases':17,'known_rejected_profiles':4,'previous_partial_progress':lrow.get('partial_progress'),'previous_structural_gaps':oldrows['relu_triton_kernel']['structural_gaps'],'exact_numeric_registration':'Three proposed correspondences retained; no exact new statement/case/port queue registration.','known_source_bug':'N>1024 retains initial tail because only pid0 stores; full mathematical tensor ReLU is false.'}
assert lrow['status']=='pending' and lrow['review'] is None
for i,row in enumerate(ledger['tasks']):
 if row['task']=='matrix_transpose':ledger['tasks'][i]=mrow
 elif row['task']=='relu_triton_kernel':ledger['tasks'][i]=lrow
 else:assert row['input_sha256']==task_bindings(R,cat,crows[row['task']])
assert sum(t['status']=='verified' for t in ledger['tasks'])==1 and sum(t['status']=='pending' for t in ledger['tasks'])==182
assert all(t==oldrows[t['task']] for t in ledger['tasks'] if t['task'] not in ['matrix_transpose','relu_triton_kernel'])
ledger['catalog_sha256']=sha(canonical(cat));ledger['progress_checkpoint']=(B/'README.md').relative_to(R).as_posix()
md=R/'bench/original_program_scope.md';text=md.read_text();text=text.replace('has all 183 tasks\n**pending**, bound to the current extension catalog.','has **1 source scope verified and 182 pending**, bound to the current extension\ncatalog. The verified scope is `matrix_transpose`, under explicit primitive\ncontracts and without numerical deferrals. Official comparator and whole-corpus\nCI results remain separate.')
text=text.replace('four gap lists and leaves every task pending.','four gap lists and introduced no acceptance by itself.')
text=text.replace('`quantization-and-mixed-softmax`','`matrix-scope-and-relu-fp64`')
text=text.replace('The full-scope review now covers all 183 original programs. Each retains\nconcrete unresolved structural obligations, so all 183 remain pending. No\nprogram is left without a completed scope-gap review. The\ncumulative supplemental proof inventory contains 45 execution contracts and','The full-scope review covers all 183 original programs, with no unreviewed\nprograms. Matrix transpose has passed its schema2 source-scope check; 182 scopes\nremain pending. This is not a claim that whole-corpus CI or official comparator\nreplay has passed. The cumulative supplemental proof inventory contains 48\nexecution contracts and')
assert '**1 source scope verified and 182 pending**' in text and 'contains 48' in text
write_json(R/'bench/original_program_scope.json',ledger);md.write_text(text)
print(json.dumps({'verified_source_scopes':1,'pending':182,'changed_rows':['matrix_transpose','relu_triton_kernel'],'unchanged_rows':181,'catalog_sha256':ledger['catalog_sha256']},indent=2))
