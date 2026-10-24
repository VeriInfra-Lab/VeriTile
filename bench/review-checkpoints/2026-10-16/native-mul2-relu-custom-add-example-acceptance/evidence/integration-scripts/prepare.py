"""Install frozen sources/evidence and prepare the explicit gzip supplement."""
from common import *
p=arguments(__doc__);a=p.parse_args();r,b,old=context(a)
require(not b.exists(),'Checkpoint already exists; preserve failed attempts and inspect partial state')
require(not git(r,'status','--porcelain=v1','--untracked-files=all'),'Preparation requires a clean committed worktree/index')
require(a.ledger.endswith('.json.gz'),'Active ledger must explicitly use .json.gz')
for rel in [a.baseline_ledger,*PROTECTED]:
 require(git(r,'show','HEAD:'+rel)==(r/rel).read_bytes(),'Baseline path is not committed: '+rel)
require((old/'validation.json').is_file() and (old/'staging-review.json').is_file(),'Predecessor lacks final/staging receipts')
tracked=set(git(r,'ls-files','--',str(old.relative_to(r))).decode().splitlines())
require(tracked=={p.relative_to(r).as_posix() for p in old.rglob('*') if p.is_file()},'Predecessor has uncommitted ignored files')
exact(HERE/'gzip-storage-receipt.json',GZIP_RECEIPT_SHA256)
gzip_receipt=load(HERE/'gzip-storage-receipt.json')
exact(r/'bench/adapter_common.py',gzip_receipt['candidate_files']['bench/adapter_common.py'])
exact(r/STORAGE_CHECKPOINT/'producer-receipt.json',GZIP_RECEIPT_SHA256)
require(a.baseline_ledger==a.ledger or not (r/a.ledger).exists(),'New active ledger already exists')
ac=bench_imports(r)
# Exercise the installed helper before copying any repository file.
probe=HERE/('gzip-runtime-probe-'+str(os.getpid()));probe.mkdir()
for filename in ('a.json.gz','b.json.gz'):ac.write_json(probe/filename,{'probe':[1,'α']})
require((probe/'a.json.gz').read_bytes()==(probe/'b.json.gz').read_bytes(),'gzip is not deterministic')
require(ac.load(probe/'a.json.gz')=={'probe':[1,'α']},'Explicit gzip load/write unsupported')
(probe/'duplicate.json.gz').write_bytes(gzip.compress(b'{"x":1,"x":2}'))
try:ac.load(probe/'duplicate.json.gz')
except ValueError:pass
else:raise ValueError('gzip load lost duplicate-key rejection')
packages=package_inputs(a);estimate=estimated_queue(r,packages)
require(estimate['under_limit_with_reserve'],'Queue estimate exceeds 100 MiB; root judgment required before integration')
ledger=ac.load(r/a.baseline_ledger);queue=ac.load(r/QUEUE);cat=ac.load(old/a.predecessor_catalog);supp=deepcopy(ac.load(old/a.predecessor_supplement))
require({t['task'] for t in ledger['tasks'] if t['status']!='pending'}==OLD_ACCEPTED,'Baseline acceptance differs from reviewed 4/183')
require(len(ledger['tasks'])==183 and len(queue['rules'])==153 and len(queue['exact_binding_provenance'])==1642 and len(queue['exact_callback_requirements'])==8,'Baseline ledger/queue totals differ')
require(cat['summary']['public_claims']==957,'Baseline public count differs')
oldrows={t['task']:t for t in ledger['tasks']};sources={};inputs={};semantics={};decls=[];scope_hashes={};source_view_additions={}
from current_value_metadata import value_scope
for package in packages:
 item=package['spec'];name=item['task'];receipt=package['receipt'];review=package['review'];producer=package['producer']
 for rel,(source,h) in package['sources'].items():
  require(not (r/rel).exists(),'New module already exists: '+rel)
  require(rel not in sources,'Candidate module collision');sources[rel]=h
 declaration=deepcopy(ac.load(producer/'evidence/declaration-proposal.json'));semantics[name]=prose(review['semantic_review']);declaration['review_basis']=semantics[name];decls.append(declaration)
 task=next((t for t in supp['tasks'] if t['task']==name),None)
 if task is None:
  proposal=producer/'evidence/catalog-supplement-task.proposed.json'
  if proposal.exists():
   task=deepcopy(ac.load(proposal));require(len(task['declarations'])==1,'Unexpected new task declaration inventory');task['declarations']=[]
  else:
   require(name=='add_example','Missing reviewed supplement task proposal')
   prefix='bench/tritonbench_g/add_example/'
   view={'id':'add_example_finite_original_source','variant':'original_source_translation','frontend':'task_local_exact_ast','ordinary_shared_dsl_admission':False,'memory_semantics':'sequentially_consistent_schedule','original_python':prefix+'add_example.py','candidate_python':None,'selected_definitions':{prefix+'Kernels.lean':['add_kernel'],prefix+'Proofs/FiniteSource.lean':['lowerBody','runProgram'],prefix+'Proofs/FiniteHost.lean':['sourceHost','sourceAllocated']},'translation_trust_boundary':semantics[name]}
   task={'task':name,'required_value_stages':[],'source_sha256':{},'source_views':[view],'declarations':[],'review':None};source_view_additions[name]=view
  supp['tasks'].append(task)
 task['declarations'].append(declaration)
 for binding in [oldrows[name]['input_sha256']['lean'],oldrows[name]['input_sha256']['python'],receipt['modules'],review.get('reviewed_existing_repo_source_sha256',{})]:
  for rel,h in binding.items():
   require(rel not in inputs or inputs[rel]==h,'Conflicting reviewed source binding');inputs[rel]=h
 task['source_sha256'].update(oldrows[name]['input_sha256']['lean']);task['source_sha256'].update(oldrows[name]['input_sha256']['python']);task['source_sha256'].update(receipt['modules'])
 for rel,h in task['source_sha256'].items():
  require(rel not in inputs or inputs[rel]==h,'Conflicting supplemental source hash');inputs[rel]=h
 task['required_value_stages'] += [stage for stage in declaration['covers_stages'] if stage not in task['required_value_stages']]
 scope_hashes[name]=ac.sha(ac.canonical(value_scope(task)))
 # Validate every new explicit source name before installing it.
 for view in task['source_views']:
  for rel,names in view['selected_definitions'].items():
   path=package['sources'][rel][0] if rel in package['sources'] else r/rel
   require(set(names)<=set(ac.declarations(path)),'Selected source definition missing: '+rel)
for rel,h in inputs.items():
 if rel not in sources:exact(r/rel,h)
workflow=r/'.github/workflows/bench-audit.yml';oldtext=workflow.read_text();newtext=oldtext
oldprefix='bench/review-checkpoints/'+SPEC['predecessor']+'/'
newprefix='bench/review-checkpoints/'+SPEC['intended_checkpoint']+'/'
for before,after in [(a.predecessor_catalog,'current-source-catalog.json.gz'),(a.predecessor_supplement,'catalog-supplement.json.gz')]:
 require(oldprefix+before in newtext,'Workflow predecessor reference missing: '+before)
 newtext=newtext.replace(oldprefix+before,newprefix+after)
require(a.baseline_ledger in newtext,'Workflow baseline ledger reference missing')
newtext=newtext.replace(a.baseline_ledger,a.ledger)
state_record={'schema':1,'base_commit':git(r,'rev-parse','HEAD').decode().strip(),'base_branch':git(r,'branch','--show-current').decode().strip(),'specs_sha256':digest(HERE/'specs.json'),'script_sha256':{p.name:digest(p) for p in sorted(HERE.glob('*.py'))},'protected_sha256':{rel:digest(r/rel) for rel in PROTECTED},'predecessor_sha256':hashes(old),'metadata_baseline_sha256':hashes(r/BASE),'baseline_ledger_sha256':digest(r/a.baseline_ledger),'baseline_ledger_path':a.baseline_ledger,'baseline_ledger_archive':'evidence/baseline-ledger.original'+('.json.gz' if a.baseline_ledger.endswith('.json.gz') else '.json'),'deactivated_ledger_paths':[] if a.baseline_ledger==a.ledger else [a.baseline_ledger],'storage_checkpoint_sha256':hashes(r/STORAGE_CHECKPOINT),'gzip_storage_receipt_sha256':GZIP_RECEIPT_SHA256,'baseline_queue_sha256':digest(r/QUEUE),'ledger_path':a.ledger,'predecessor_catalog':a.predecessor_catalog,'predecessor_supplement':a.predecessor_supplement,'predecessor_review':a.predecessor_review,'baseline_document_sha256':{p:digest(r/p) for p in ['bench/original_program_scope.md','bench/current_source_acceptance.md']},'baseline_document_archives':{'bench/original_program_scope.md':'evidence/baseline-progress.md','bench/current_source_acceptance.md':'evidence/baseline-current-source-acceptance.md'},'new_source_sha256':sources,'packages':SPEC['packages'],'queue_size_preflight':estimate,'add_example_source_view':source_view_additions}
# All prechecks above are read-only except the /tmp helper probe. Mutation starts here.
b.mkdir(parents=True);retained={};mapping={}
for package in packages:
 for rel,(source,h) in package['sources'].items():
  dest=r/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);exact(dest,h)
 for source,rel,h in package['records']:
  dest=b/rel
  if dest.exists():exact(dest,h)
  else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);exact(dest,h)
  retained[rel]=h
  if str(source).endswith(('.lean','.olean','.olean.private','.olean.server','.ilean','.ir','.pyc')):mapping[str(source)]=rel
review={'schema':1,'kind':'native_mul2_relu_custom_add_example_source_integration_review','reviewer':'root','status':'reviewed_for_finite_scope_acceptance_with_registered_numerical_deferrals','base_commit':state_record['base_commit'],'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':scope_hashes,'reviewed_metadata_corrections_sha256':{},'semantic_review':semantics,'metadata_baseline':ac.load(old/a.predecessor_review)['metadata_baseline'],'new_public_contracts':decls,'packages':SPEC['packages'],'native_execution':False,'acceptance_boundary_policy_sha256':digest(r/'bench/original_program_acceptance.md'),'added_source_view_metadata':source_view_additions}
ac.write_json(b/'root-review.json.gz',review)
for task in supp['tasks']:
 if task['task'] in NEW_TASKS:task['review']={'path':SPEC['intended_checkpoint']+'/root-review.json.gz','sha256':digest(b/'root-review.json.gz'),'source_hashes_pointer':['reviewed_input_sha256']}
ac.write_json(b/'catalog-supplement.json.gz',supp)
shutil.copyfile(r/a.baseline_ledger,b/state_record['baseline_ledger_archive']);exact(b/state_record['baseline_ledger_archive'],state_record['baseline_ledger_sha256'])
ac.write_json(b/'evidence/baseline-ledger.json.gz',ledger);ac.write_json(b/'evidence/baseline-queue.json.gz',queue)
for rel,destination in state_record['baseline_document_archives'].items():
 shutil.copyfile(r/rel,b/destination);exact(b/destination,state_record['baseline_document_sha256'][rel])
for p in [*HERE.glob('*.py'),HERE/'specs.json',HERE/'gzip-storage-receipt.json']:
 dest=b/'evidence/integration-scripts'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
write(b/'evidence/preparation-state.json',state_record)
write(b/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'new_source_sha256':sources,'archived_path_mapping':mapping,'policy':'Frozen producer/root/peer JSON bytes unchanged; historical Lean .txt, objects and Python bytecode .bin.'})
workflow.write_text(newtext)
(b/'README.md').write_text('# Mul2, ReLU, custom vector and add_example integration\n\nFour frozen, independently peer and root reviewed finite source candidates. Integration checks are pending. Numerical conformance, native GPU execution and official comparator success are not established by these source packages. Frozen receipts preserve exact bytes, including earlier failures. The active catalog, supplement, combined review and ledger explicitly use .json.gz; the numerical queue remains plain JSON.\n')
print(json.dumps({'phase':'prepare','base_commit':state_record['base_commit'],'new_modules':len(sources),'archive_files':len(retained),'status':'prepared_checks_pending'},indent=2))
