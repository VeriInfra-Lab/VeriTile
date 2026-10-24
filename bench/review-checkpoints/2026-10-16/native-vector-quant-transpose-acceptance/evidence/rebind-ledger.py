from pathlib import Path
import copy,json,sys,subprocess,shutil
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
B=R/'bench/review-checkpoints/2026-10-16/native-vector-quant-transpose-acceptance';OLD=R/'bench/review-checkpoints/2026-10-16/quant-global-finite-source-acceptance'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from check_original_program_scope import task_bindings,validate_task
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
from exact_numerical_bindings import collect,validate_exact_queue
s=load(B/'evidence/preparation-state.json')
def before(path):return json.loads(subprocess.check_output(['git','show',s['base_commit']+':'+path],cwd=R))
cat=load(B/'current-source-catalog.json');validate_catalog(R,cat);supp=load(B/'catalog-supplement.json');validate_supplement(R,R/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,R/'bench/review-checkpoints')
ledger=load(R/'bench/original_program_scope.json');assert hash_file(R/'bench/original_program_scope.json')==s['previous_working_ledger_sha256']
prior=before('bench/original_program_scope.json');crows={t['task']:t for t in cat['tasks']};oldrows={t['task']:t for t in prior['tasks']};workingrows={t['task']:t for t in ledger['tasks']};oldcat={t['task']:t for t in load(OLD/'current-source-catalog.json')['tasks']}
legacy,authority,queue=collect(R);assert queue==load(R/'bench/tritonbench_g/deferred_numerical_experiments.json');assert len(queue['rules'])==153
previous=before('bench/tritonbench_g/deferred_numerical_experiments.json')
assert queue['rules'][:100]==previous['rules'] and authority['requirements'][:6]==previous['exact_callback_requirements']
assert all(queue['exact_binding_provenance'][k]==v for k,v in previous['exact_binding_provenance'].items())
assert validate_exact_queue(queue,authority,legacy)=={'status':'passed','exact_bindings':1642,'declaration_cases':11041,'native_executions':0,'structural_acceptance':False}
accepted={}
for name in ['matrix_transpose','quantize_global']:
 row=copy.deepcopy(oldrows[name]);assert row['input_sha256']==task_bindings(R,cat,crows[name]) and crows[name]==oldcat[name]
 review=load(R/row['review']['path']);assert hash_file(R/row['review']['path'])==row['review']['sha256']
 review['catalog_sha256']=sha(canonical(cat));review['catalog_only_rebind']={'previous_review':row['review'],'reason':'Only vector_addition and quant_transpose_kernel gain reviewed native-domain contracts; this task source, coverage, cases, assumptions and catalog row remain unchanged.','task_catalog_row_sha256':sha(canonical(crows[name]))}
 path=B/(name+'-original-scope-review-rebound.json');write_json(path,review);row['review']={'path':path.relative_to(R).as_posix(),'sha256':hash_file(path)};row['accepted_scope_progress']['scope_review']=row['review']
 assert validate_task(R,cat,crows[name],row,queue)==oldrows[name]['status'];accepted[name]=row
for item in s['packages']:
 name=item['task'];prefix='evidence/producer/'+item['tag'];rootprefix='evidence/root-validation/'+item['tag']
 row=load(B/prefix/'evidence/scope-row-proposal.json');assert row['source_inventory']==oldrows[name]['source_inventory']
 assert row.pop('proposed_status_after_independent_root_review')=='verified_with_deferred_numerics'
 row['status']='verified_with_deferred_numerics';row['input_sha256']=task_bindings(R,cat,crows[name])
 row['acceptance_boundary_review'].update(status='independently_reviewed',queue_integration=f"{item['families']} reviewed numerical families / {item['ports']} exact ports; whole-queue equality checked; native numerical conformance deferred")
 # Preserve the exact earlier reopening/withdrawal history, including never-committed recommendations.
 for key in ['accepted_scope_history','scope_reopening']:
  if key in workingrows[name]:row[key]=copy.deepcopy(workingrows[name][key])
 semantic=load(B/'root-review.json')['semantic_review'][name]
 review={'schema':1,'scope_schema':2,'kind':'original_program_scope_review','task':name,'reviewer':'root','status':'reviewed','catalog_sha256':sha(canonical(cat)),'input_sha256':row['input_sha256'],'source_inventory_sha256':sha(canonical(row['source_inventory'])),'coverage_sha256':sha(canonical(row['coverage'])),'deferred_numerical_sha256':sha(canonical(row['deferred_numerical_assumptions'])),'callback_assumptions_sha256':sha(canonical(row['callback_assumptions'])),'structural_gaps':[],'semantic_review':semantic,'finite_acceptance_policy':{'path':'bench/original_program_acceptance.md','sha256':hash_file(R/'bench/original_program_acceptance.md')},'evidence':{key:{'path':(B/path).relative_to(R).as_posix(),'sha256':hash_file(B/path)} for key,path in [('producer_receipt',prefix+'/evidence/receipt.json'),('root_package_review',rootprefix+'/review.json'),('root_value_review','root-review.json')]},'boundary':'Complete declared finite source stages, canonical dtype families and branches under documented trusted primitive semantics. Actual typed integer/mask/address behavior and source-first failure ordering are modeled. Numerical correspondence remains exactly deferred; no native GPU or official comparator success.'}
 rp=B/(item['tag']+'-original-scope-review.json');write_json(rp,review);row['review']={'path':rp.relative_to(R).as_posix(),'sha256':hash_file(rp)}
 row['accepted_scope_progress']={'status':'finite_source_scope_accepted_with_deferred_numerics','scope_review':row['review'],'previous_partial_progress':oldrows[name].get('partial_progress'),'closed_structural_gaps':workingrows[name]['structural_gaps'],'source_cases':item['cases'],'numerical_families':item['families'],'numerical_ports':item['ports'],'native_execution':False,'source_semantics':semantic}
 assert validate_task(R,cat,crows[name],row,queue)=='verified_with_deferred_numerics';accepted[name]=row
for i,t in enumerate(ledger['tasks']):
 if t['task'] in accepted:ledger['tasks'][i]=accepted[t['task']]
 else:assert t==oldrows[t['task']] and t['input_sha256']==task_bindings(R,cat,crows[t['task']])
ledger['catalog_sha256']=sha(canonical(cat));ledger['deferred_queue']={'path':'bench/tritonbench_g/deferred_numerical_experiments.json','sha256':hash_file(R/'bench/tritonbench_g/deferred_numerical_experiments.json')};ledger['progress_checkpoint']=(B/'README.md').relative_to(R).as_posix()
write_json(R/'bench/original_program_scope.json',ledger)
f=R/'bench/original_program_scope.md';text=f.read_text()
replacements={
 '**2 source scopes accepted and 181 pending**':'**4 source scopes accepted and 179 pending**',
 '`quantize_global` is verified with exactly registered':'`quantize_global`, `vector_addition` and `quant_transpose_kernel` are verified with exactly registered',
 'The current queue has 116 rules, with 1,346 exact bindings across 6,262 cases in\nseven declarations.':'The current queue has 153 rules, with 1,642 exact bindings across 11,041 cases in\neight declarations.',
 '`quant-transpose-finite-source-acceptance` batch':'`native-vector-quant-transpose-acceptance` batch',
 'Quantize_global has passed with exactly registered numerical\ndeferrals; 181 scopes remain pending.':'Quantize_global, vector addition and quant_transpose_kernel have passed with exactly registered numerical\ndeferrals; 179 scopes remain pending.',
 'contains 57\nexecution contracts':'contains 58\nexecution contracts',
 "Vector addition's earlier acceptance was withdrawn after an actual runtime-i32 source route was found outside its finite theorem.":"Vector addition's earlier acceptance was withdrawn after an actual runtime-i32 source route was found outside its finite theorem; its reviewed native revision2 now covers that route.",
 'Vector addition and ReLU are pending again, and the custom-vector/add-value candidates remain pending until actual runtime arithmetic and outcomes are covered.':'That review reopened vector addition and ReLU. Vector addition is now accepted through its reviewed native revision2; ReLU and the custom-vector/add-value candidates remain pending until their revised contracts are integrated.'}
for a,b in replacements.items():assert a in text,a;text=text.replace(a,b)
# Replace the contradictory, never-committed draft summary; exact old prose is retained in evidence.
pos=text.index('\nQuant_transpose_kernel now covers all22')
text=text[:pos]+'''
Quant_transpose_kernel revision2 covers all 22 canonical dense dtype families,
both autotune configurations, logical-view abs/max and reciprocal, actual
constexpr/i32/i64 expression wraps, signed masks and uint64 addresses. It
composes all programs and finite trials with independent last-write byte results,
frames and preserved returned maximum/inverse bytes. Its 1,144 cases bind 140
ports in 16 numerical families. Source-first launch admission checks context,
signed grid parsing, entry hook, pointer extraction, launch and exit hook in
order, including CPU oversized-grid rejection before CPU pointer failure.
The mathematical transpose formula remains conditional on arithmetic refinement.

Vector addition native revision2 covers the full nonnegative signed64 host
extent domain, actual annotated i32 narrowing/wrapping, signed masks and uint64
addresses. It has no smaller-extent, same-device, equal-shape or defined-conversion
premise. The exact N=2^31 route returns the unchanged allocated output without
input-access or numerical assumptions. Actual extracted ABI pointers, launch
permissions, complete x-before-y loads, stores, independent final bytes and
allocator metadata are explicit. Its 4,498 cases bind 296 ports in 37 families.

The uncommitted quant-transpose revision1 recommendation was withdrawn because
CPU pointer rejection preceded the actual oversized-grid parser error. Its
proof/build/test evidence and the
[withdrawal record](review-checkpoints/2026-10-16/quant-transpose-finite-source-acceptance/withdrawal.json)
remain unchanged. Reviewed revision2 repairs this ordering; the older
dimension-free admission is historical small-profile evidence only.
Neither new source acceptance establishes native numerical conformance, GPU
correctness or official comparator success.
'''
f.write_text(text);shutil.copyfile(Path(__file__),B/'evidence/rebind-ledger.py')
print(json.dumps({'verified':1,'verified_with_deferred_numerics':3,'pending':179,'accepted_total':4,'public_declarations':sum(len(t['declarations']) for t in cat['tasks']),'total_numeric_rules':153,'unrelated_scope_rows_unchanged':179,'exact_queue':validate_exact_queue(queue,authority,legacy),'supplement_nonempty_tasks':sum(bool(t['declarations']) for t in supp['tasks'])},indent=2))
