"""Validate observed runs and preservation, then freeze an integration receipt."""
from pathlib import Path as _Path
import sys as _sys
_OVERRIDE_ROOT = _Path(__file__).resolve().parent
_ORIGINAL_SCRIPTS = _Path('/tmp/veritile-standardization-20261015/next-three-source-integration-scripts-revision2')
_sys.path.insert(0, str(_ORIGINAL_SCRIPTS))
from common import *
from unittest_summary import observed_unittest_count
from datetime import datetime,timezone
import re
p=arguments(__doc__);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--selected-verify',type=Path,required=True);p.add_argument('--finalizer-correction-sha256',required=True);a=p.parse_args();r,b,old,s=state(a)
correction=verify_scripts(_OVERRIDE_ROOT,a.finalizer_correction_sha256)
exact(_ORIGINAL_SCRIPTS/'finalize.py','d61f18481c105485fffc7dc735f39a275e7e631caad5e1c6506d9e975f9d0663');ac=bench_imports(r)
require(not (b/'validation.json').exists(),'Final validation receipt already exists')
result=preservation(a);cat=result['catalog'];supp=result['supplement']
for rel,h in ac.load(b/'evidence-index.json')['retained_exact_artifacts'].items():exact(b/rel,h)
for rel,h in ac.load(b/'evidence/install-state.json')['installed_sha256'].items():exact(r/rel,h)
require(not any(p.is_file() and archive(p.relative_to(b))!=p.relative_to(b).as_posix() for p in b.rglob('*')),'Active Lean/object/bytecode file in historical checkpoint')
runsroot=a.runs_root.resolve();selected=a.selected_verify.resolve()
require(runsroot.is_relative_to(Path('/tmp')) and selected.parent==runsroot,'Selected verification must be an immediate child of /tmp runs root')
runs=[]
for folder in sorted(runsroot.iterdir()):
 if not folder.is_dir():continue
 require((folder/'run.json').is_file(),'Run directory lacks actual-run manifest: '+str(folder))
 rec=load(folder/'run.json');require(rec['base_commit']==s['base_commit'],'Run used another HEAD')
 require(rec['state']!='running','Incomplete run; preserve and record interruption before finalizing: '+str(folder))
 for command in rec['commands']:
  exact(folder/command['log'],command['log_sha256'])
  require(command['cwd']==str(r),'Run cwd mismatch')
 runs.append((folder,rec))
require(any(folder==selected for folder,_ in runs),'Selected run was not recorded')
observed=next(rec for folder,rec in runs if folder==selected)
require(observed['phase']=='verify' and observed['state']=='completed' and observed.get('inputs_unchanged') is True,'Selected verification incomplete')
require(observed['input_sha256_before']==observed['input_sha256_after']==verification_inputs(a),'Verification inputs differ from current reviewed state')
commands={row['name']:row for row in observed['commands']}
require(set(commands)=={'build','bench-tests','shard-tests','global'} and len(observed['commands'])==4,'Required checks missing or duplicated')
require(commands['build']['command']==['lake','build',*TARGETS],'Build did not use all eight required targets')
py=observed['python_executable']
require(commands['bench-tests']['command']==[py,'-B','-m','unittest','discover','-s','bench/tests','-p','test_*.py','-v'],'Benchmark suite command differs')
require(commands['shard-tests']['command']==[py,'-B','scripts/test_audit_shards.py'],'Shard suite command differs')
for name in ['build','bench-tests','shard-tests']:require(commands[name]['exit_code']==0 and commands[name]['state']=='completed','Required command failed: '+name)
actual_tests={}
for name in ['bench-tests','shard-tests']:
 text=(selected/commands[name]['log']).read_text()
 actual_tests[name]=observed_unittest_count(text,commands[name]['exit_code'])
build_text=(selected/commands['build']['log']).read_text();jobs=re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build_text)
# Job counts are observed metadata, never a hardcoded success threshold.
g=ac.load(selected/'global/receipt.json');require(g['catalog_sha256']==ac.sha(ac.canonical(cat)),'Global audit catalog differs')
expected_gates={'source_tools','structure','baseline','catalog_reproduction','scope_review','legacy_source','value_metadata','deferred_numerics','comparator_preflight','library_replay','unchanged_inputs'}
require(set(g['gates'])==expected_gates,'Global gate inventory differs')
for name,gate in g['gates'].items():
 if 'log' in gate:exact(selected/'global'/relative(gate['log']),gate['log_sha256'])
 if 'exit_code' in gate:require((gate['exit_code']==0)==(gate['status']=='passed'),'Global status/exit mismatch: '+name)
 if name not in {'scope_review','comparator_preflight','library_replay'}:require(gate['status']=='passed','Unexpected failing global gate: '+name)
scope=load(selected/'global/scope_review.log')
require(scope['counts']=={'verified':1,'verified_with_deferred_numerics':10,'not_accepted':172} and scope['errors']==['172 original-program scopes remain unaccepted'],'Scope global failure is not exactly the expected 172 pending rows')
require(g['gates']['scope_review']['status']=='failed','172 pending scopes must remain reported as a failing completeness gate')
for name in ['comparator_preflight','library_replay']:
 gate=g['gates'][name]
 if gate['status']!='passed':require('Official comparator is required' in (selected/'global'/gate['log']).read_text(),'Unexpected comparator failure: '+name)
expected_global=[py,'-B','bench/current_source_audit.py','--root',str(r),'--catalog',str(b/'current-source-catalog.json.gz'),'--baseline-package',str(r/BASE),'--source-map',str(r/SOURCE_MAP),'--legacy-inventory',str(r/LEGACY),'--evidence-root',str(r/'bench/review-checkpoints'),'--review-manifest',str(r/REVIEW),'--supplement',str(b/'catalog-supplement.json.gz'),'--scope-ledger',str(r/a.ledger),'--output',str(selected/'global')]
require(commands['global']['command']==expected_global,'Global audit command differs')
require(commands['global']['exit_code']==1,'Global runner exit must reflect incomplete scopes')
require(any(rec.get('generated_sha256')=={str(path.relative_to(r)):digest(path) for path in [b/'current-source-catalog.json.gz',r/QUEUE,r/'bench/tritonbench_g/deferred_numerical_experiments.md']} and rec['phase']=='generate' and rec['state']=='completed' and {c['name'] for c in rec['commands']}=={'prepare-baseline','catalog','queue'} and all(c['exit_code']==0 for c in rec['commands']) for _,rec in runs),'No complete recorded metadata generation')
# Retain the exact reviewed finalizer correction without altering frozen scripts/runs.
for rel,expected in {'candidate-receipt.json':a.finalizer_correction_sha256,**correction['artifacts']}.items():
 source=_OVERRIDE_ROOT/relative(rel);dest=b/'evidence/finalizer-summary-correction'/archive(rel)
 require(not dest.exists(),'Finalizer correction archive already exists')
 dest.parent.mkdir(parents=True,exist_ok=True);exact(source,expected);shutil.copyfile(source,dest);exact(dest,expected)
# Retain every run, including failed attempts. Archive generated baseline sources/objects too.
retained={};relocations={};attempts=[]
for folder,rec in runs:
 for source in sorted(folder.rglob('*')):
  if not source.is_file():continue
  require(not source.is_symlink(),'Run evidence must not depend on symlinks: '+str(source))
  raw='evidence/integration-runs/'+folder.name+'/'+source.relative_to(folder).as_posix();rel=archive(raw);dest=b/rel
  require(source.stat().st_size<LIMIT,'Run artifact exceeds 100 MiB: '+str(source))
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);retained[rel]=digest(source);exact(dest,retained[rel])
  if raw!=rel:relocations[raw]=rel
 attempts.append({'run':folder.name,'phase':rec['phase'],'state':rec['state'],'actual_exits':{c['name']:c['exit_code'] for c in rec['commands']},'manifest':{'path':'evidence/integration-runs/'+folder.name+'/run.json','sha256':digest(folder/'run.json')}})
write(b/'evidence/integration-runs-index.json',{'retained_exact_artifacts':retained,'archived_path_mapping':relocations,'selected_verify':selected.name,'attempts':attempts})
comparator=all(g['gates'][name]['status']=='passed' for name in ['comparator_preflight','library_replay'])
readme=b/'README.md';readme.write_text(readme.read_text()+f'\n## Observed integration checks\n\nThe ledger accepts 11/183 legacy full-source scopes: one verified and ten with registered numerical deferrals; 172 remain pending. These are not Real/FP32/BF16 completion counts. All eight Lean targets exited successfully. The benchmark suite ran {actual_tests["bench-tests"]} tests and the sharding suite ran {actual_tests["shard-tests"]} tests, each exiting 0. The global audit retains its actual exit 1 because scope completeness remains open; comparator gate statuses are recorded individually below. All attempts, including any failures before the selected run, are retained. Existing 961 public records, 239 numerical rules, 2,366 exact bindings and twelve callback tables remain identical, as do 172 unrelated ledger rows.\n\nGlobal gates: '+json.dumps({k:v['status'] for k,v in g['gates'].items()},sort_keys=True)+'.\n')
# Freeze every active/checkpoint input except this receipt's self-reference.
require(all(not (r/rel).exists() for rel in s['deactivated_ledger_paths']),'Competing prior active ledger remains')
paths={r/rel for rel in (active_paths(a,s)|set(PROTECTED))-set(s['deactivated_ledger_paths'])}|checkpoint_files(b)
require(all(path.stat().st_size<LIMIT for path in paths),'Per-file 100 MiB limit exceeded')
v={'schema':1,'kind':'native_add_value_sin_cosine_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':s['base_commit'],'status':'eleven_legacy_source_scopes_accepted_with_registered_numerical_deferrals','scope':{'requested':183,'accepted':11,'verified':1,'verified_with_deferred_numerics':10,'pending':172},'inventory':{'public_declarations':EXPECTED['public_declarations'],'task_modules':EXPECTED['task_modules'],'registered_modules':EXPECTED['registered_modules'],'supplement_execution':EXPECTED['supplemental_execution_contracts'],'supplement_protocol':EXPECTED['supplemental_protocol_contracts'],'supplement_nonempty_tasks':sum(bool(t['declarations']) for t in supp['tasks'])},'exact_queue':result['exact'],'actual_test_counts':actual_tests,'observed_build_jobs':jobs,'actual_runs':attempts,'selected_verify':selected.name,'global_gates':{k:v['status'] for k,v in g['gates'].items()},'deleted_paths':s['deactivated_ledger_paths'],'checked_input_sha256':{p.relative_to(r).as_posix():digest(p) for p in sorted(paths)},'preservation':{'prior_public_records':961,'prior_numeric_rules':239,'prior_exact_bindings':2366,'prior_callback_tables':12,'unrelated_scope_rows':172,'metadata_baseline_unchanged':True,'predecessor_unchanged':True,'validation_predicates_unchanged':True},'numerical_conformance_validated':False,'current_comparator_success':comparator,'native_gpu_execution':False}
v['finalizer_correction']={'receipt_path':'evidence/finalizer-summary-correction/candidate-receipt.json','receipt_sha256':a.finalizer_correction_sha256,'original_finalizer_sha256':'d61f18481c105485fffc7dc735f39a275e7e631caad5e1c6506d9e975f9d0663','reason':'Accept exact unittest success summary followed by buffered stdout; preserve actual exits/log bytes.'}
write(b/'validation.json',v);print(json.dumps({'phase':'finalize','validation_sha256':digest(b/'validation.json'),'scope':v['scope'],'actual_tests':actual_tests,'global':v['global_gates']},indent=2))
