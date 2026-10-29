from pathlib import Path
import datetime,hashlib,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
rp=W/'evidence/receipt.json';assert not rp.exists(),'Already frozen'
compiles=load(W/'evidence/final-compiles.json');assert len(compiles['modules'])==6
for m in compiles['modules']:
 assert sha(W/m['source'])==m['source_sha256'] and sha(W/m['receipt'])==m['receipt_sha256'] and sha(W/m['log'])==m['log_sha256']
deps=load(W/'evidence/dependencies-before.json')
for rel,entry in deps['project'].items():assert sha(R/rel)==entry['sha256'],rel
assert deps==load(W/'evidence/dependencies-after.json')
modules={p:sha(W/'after'/p) for p in load(W/'evidence/staging-map.json').values()}
artifacts={}
for p in sorted(W.rglob('*')):
 if p.is_symlink() or not p.is_file():continue
 rel=p.relative_to(W)
 if rel.parts[0] in ['build-overlay','VectorFinite','after'] or '__pycache__' in rel.parts or p.name in ['task.md','receipt.json','receipt.sha256']:continue
 if p.suffix in ['.olean','.ilean','.ir']:continue
 artifacts[str(rel)]=sha(p)
receipt={'schema':1,'task':'vector_addition','kind':'immutable_finite_source_acceptance_candidate','frozen':True,
 'status':'frozen_pending_independent_root_review','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'original_source':{'path':'bench/tritonbench_g/vector_addition/vector_addition.py','sha256':sha(R/'bench/tritonbench_g/vector_addition/vector_addition.py')},
 'repository_and_git_modified_by_this_package':False,'previous_frozen_packages_modified':False,
 'modules':modules,'producer_module_source_paths':{p:'after/'+p for p in modules},
 'checkpoint_module_source_paths':{p:p+'.txt' for p in modules},'artifacts':artifacts,
 'public_contracts':load(W/'evidence/public-contracts.json'),'named_axiom_checks':compiles['named_axiom_checks'],
 'independent_reference_checks':compiles['noncircular_checks'],'project_dependency_count':len(deps['project']),
 'scope_summary':load(W/'evidence/numerical-authority-delta.json')['summary'],
 'probes':load(W/'evidence/view-and-schedule-probe.json')['counts'],
 'exact_authority_test':'synthetic fixture only; unchanged whole-queue equality;616 omitted ports and6 tamper/self-review mutations rejected',
 'integration_requires':'independent substantive root review pinned by receipt and review hash; regenerate exact queue through reviewed authority extension',
 'native_functional_correctness':'not_established','native_executions':0,'acceptance_increment':0,
 'excluded_generated_or_duplicate_paths':['build-overlay/**','VectorFinite/**','after/** captured by modules','__pycache__/**','task.md']}
rp.write_text(json.dumps(receipt,indent=2)+'\n');digest=sha(rp);(W/'evidence/receipt.sha256').write_text(digest+'  receipt.json\n')
print(json.dumps({'receipt_sha256':digest,'artifacts':len(artifacts),'modules':len(modules),'status':receipt['status']}))
