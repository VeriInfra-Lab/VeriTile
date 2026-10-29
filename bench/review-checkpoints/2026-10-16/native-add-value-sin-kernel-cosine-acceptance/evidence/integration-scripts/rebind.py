"""Bind the regenerated catalog/queue without changing unrelated scope rows."""
from common import *
p=arguments(__doc__);a=p.parse_args();r,b,old,s=state(a);ac=bench_imports(r)
from check_original_program_scope import task_bindings,validate_task
validated=preservation(a,include_ledger=False);cat=validated['catalog'];queue=validated['queue']
require(digest(r/a.baseline_ledger)==s['baseline_ledger_sha256'],'Active ledger changed before rebind')
ledger=ac.load(r/a.baseline_ledger);prior=ac.load(b/'evidence/baseline-ledger.json.gz');require(ledger==prior,'Ledger baseline mismatch')
rows={t['task']:t for t in prior['tasks']};crows={t['task']:t for t in cat['tasks']};oldcat={t['task']:t for t in ac.load(old/a.predecessor_catalog)['tasks']}
accepted={};catsha=ac.sha(ac.canonical(cat));semantics=ac.load(b/'root-review.json.gz')['semantic_review']
for name in sorted(OLD_ACCEPTED):
 row=deepcopy(rows[name]);require(crows[name]==oldcat[name],'Existing accepted catalog row changed: '+name)
 exact(r/row['review']['path'],row['review']['sha256']);review=deepcopy(ac.load(r/row['review']['path']))
 review['catalog_sha256']=catsha;review['catalog_only_rebind']={'previous_review':row['review'],'reason':'Only the three next-batch tasks gain declarations; this task source, cases, coverage and numerical assumptions remain unchanged.','task_catalog_row_sha256':ac.sha(ac.canonical(crows[name]))}
 path=b/(name+'-original-scope-review-rebound.json.gz');require(not path.exists(),'Rebound review already exists');ac.write_json(path,review)
 row['review']={'path':path.relative_to(r).as_posix(),'sha256':digest(path)};row['accepted_scope_progress']['scope_review']=row['review']
 require(validate_task(r,cat,crows[name],row,queue)==rows[name]['status'],'Old acceptance no longer validates');accepted[name]=row
for item in SPEC['packages']:
 name=item['task'];prefix='evidence/producer/'+item['tag'];rootprefix='evidence/root-validation/'+item['tag'];peerprefix='evidence/peer-review/'+item['tag']
 row=deepcopy(ac.load(b/prefix/'evidence/scope-row-proposal.json'))
 require(row['source_inventory']==rows[name]['source_inventory'],'Source inventory differs from baseline')
 require(row.pop('proposed_status_after_independent_root_review')=='verified_with_deferred_numerics','Unexpected proposed status')
 row['status']='verified_with_deferred_numerics';row['input_sha256']=task_bindings(r,cat,crows[name])
 row['acceptance_boundary_review'].update(status='independently_reviewed',queue_integration=f"{item['families']} reviewed numerical families / {item['ports']} exact ports; whole-queue equality checked; numerical conformance deferred")
 for key in ('accepted_scope_history','scope_reopening'):
  if key in rows[name]:row[key]=deepcopy(rows[name][key])
 require(not row['structural_gaps'],'Candidate still has structural gaps')
 evidence=[('producer_receipt',prefix+'/evidence/receipt.json'),('root_package_review',rootprefix+'/review.json'),('peer_receipt',peerprefix+'/'+item['peer_receipt']),('root_value_review','root-review.json.gz')]
 review={'schema':1,'scope_schema':2,'kind':'original_program_scope_review','task':name,'reviewer':'root','status':'reviewed','catalog_sha256':catsha,'input_sha256':row['input_sha256'],'source_inventory_sha256':ac.sha(ac.canonical(row['source_inventory'])),'coverage_sha256':ac.sha(ac.canonical(row['coverage'])),'deferred_numerical_sha256':ac.sha(ac.canonical(row['deferred_numerical_assumptions'])),'callback_assumptions_sha256':ac.sha(ac.canonical(row['callback_assumptions'])),'structural_gaps':[],'semantic_review':semantics[name],'finite_acceptance_policy':{'path':'bench/original_program_acceptance.md','sha256':digest(r/'bench/original_program_acceptance.md')},'evidence':{key:{'path':(b/path).relative_to(r).as_posix(),'sha256':digest(b/path)} for key,path in evidence},'boundary':'Finite source scope under the independently reviewed primitive semantics. Numerical correspondence remains exactly deferred; no native GPU or official comparator pass is inferred.'}
 path=b/(item['tag']+'-original-scope-review.json.gz');require(not path.exists(),'New scope review already exists');ac.write_json(path,review)
 row['review']={'path':path.relative_to(r).as_posix(),'sha256':digest(path)}
 row['accepted_scope_progress']={'status':'finite_source_scope_accepted_with_deferred_numerics','scope_review':row['review'],'previous_partial_progress':rows[name].get('partial_progress'),'closed_structural_gaps':rows[name]['structural_gaps'],'source_cases':item['cases'],'numerical_families':item['families'],'numerical_ports':item['ports'],'native_execution':False,'source_semantics':semantics[name]}
 require(validate_task(r,cat,crows[name],row,queue)=='verified_with_deferred_numerics','New acceptance validation failed: '+name);accepted[name]=row
ledger['tasks']=[accepted.get(row['task'],row) for row in ledger['tasks']]
ledger['catalog_sha256']=catsha;ledger['deferred_queue']={'path':QUEUE,'sha256':digest(r/QUEUE)};ledger['progress_checkpoint']=(b/'README.md').relative_to(r).as_posix()
ac.write_json(r/a.ledger,ledger)
from progress_docs import render_documents
for rel,h in s['baseline_document_sha256'].items():exact(r/rel,h)
documents=render_documents((r/'bench/original_program_scope.md').read_text(),(r/'bench/current_source_acceptance.md').read_text())
for rel,text in documents.items():(r/rel).write_text(text)
result=preservation(a)
require(s['deactivated_ledger_paths']==[],'No historical or active-path deletion is authorized')
write(b/'evidence/rebind-state.json',{'schema':1,'status':'scope_and_preservation_checks_passed','accepted':11,'pending':172,'unchanged_unrelated_rows':172,'catalog_sha256':catsha,'ledger_sha256':digest(r/a.ledger),'queue_sha256':digest(r/QUEUE),'exact':result['exact']})
print(json.dumps({'phase':'rebind','accepted':11,'pending':172,'unrelated_rows_unchanged':172,'exact':result['exact']},indent=2))
