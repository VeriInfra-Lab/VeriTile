"""Finalize only actual current integration outcomes and preserved source bindings."""
from datetime import datetime,timezone
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP=Path('/tmp/veritile-standardization-20261015')
BATCH=ROOT/'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics'
OLD=ROOT/'bench/review-checkpoints/2026-10-16/fp64-admission-relu-access'
EXACT=ROOT/'bench/review-checkpoints/2026-10-16/exact-numerical-bindings'
PKG=TMP/'numerical-exact-binding-repo-adapter-wip'
GLOBAL=TMP/'relu-source-grid-current-global'
BASE=ROOT/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import canonical,hash_file,load,module_name,sha,write_json
from check_original_program_scope import task_bindings
from current_source_catalog import validate_catalog,validate_supplement
from current_value_metadata import check_supplement_scope
def require(ok,message):
 if not ok:raise ValueError(message)
def hashes(p):return {f.relative_to(p).as_posix():hash_file(f) for f in sorted(p.rglob('*')) if f.is_file()}
def exact(p,h):require(hash_file(p)==h,'hash changed: '+str(p))
require(not (BATCH/'validation.json').exists(),'refuse final evidence overwrite')
state=load(BATCH/'evidence/preparation-state.json')
require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==state['base_commit'],'HEAD drift')
require(hashes(OLD)==state['predecessor_file_sha256'],'prior checkpoint drift')
require(hashes(BASE)==state['metadata_baseline_sha256'],'original baseline drift')
for p,h in state['original_baseline_input_map'].items():exact(ROOT/p,h)
catalog,prior=load(BATCH/'current-source-catalog.json'),load(OLD/'current-source-catalog.json')
validate_catalog(ROOT,catalog)
supp,oldsupp=load(BATCH/'catalog-supplement.json'),load(OLD/'catalog-supplement.json')
validate_supplement(ROOT,ROOT/'bench/review-checkpoints',supp,set(catalog['expected_task_names']))
check_supplement_scope(supp,ROOT/'bench/review-checkpoints')
oldrows={t['task']:t for t in oldsupp['tasks']};newrows={t['task']:t for t in supp['tasks']}
require(set(oldrows)==set(newrows),'supplement task drift')
for name,old in oldrows.items():
 new=newrows[name]
 if name!='relu_triton_kernel':require(new==old,'unrelated supplement drift: '+name);continue
 require(new['declarations'][:-2]==old['declarations'],'prior ReLU declarations drift')
 require(new['source_views']==old['source_views'],'source view drift')
 require(all(new['source_sha256'].get(p)==h for p,h in old['source_sha256'].items()),'prior source binding drift')
 require(new['required_value_stages']==old['required_value_stages']+[s for d in new['declarations'][-2:] for s in d['covers_stages']],'stage drift')
 allowed={'review','source_sha256','declarations','required_value_stages'}
 require({k:v for k,v in new.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed},'other supplement metadata drift')
all_decls=[d for t in supp['tasks'] for d in t['declarations']]
require(len(all_decls)==45 and sum(d['role']=='parametric_value_contract' for d in all_decls)==41
        and sum(d['role']=='protocol_contract' for d in all_decls)==4,'execution/protocol counts')
def claims(c):return {(t['task'],d['file'],d['declaration']):d for t in c['tasks'] for d in t['declarations']}
oldclaims,newclaims=claims(prior),claims(catalog)
require(all(newclaims.get(k)==v for k,v in oldclaims.items()),'prior catalog declaration drift')
added={('relu_triton_kernel',d['file'],d['declaration']) for d in newrows['relu_triton_kernel']['declarations'][-2:]}
require(set(newclaims)-set(oldclaims)==added,'new public membership')
require(set(catalog['registered_modules'])-set(prior['registered_modules'])=={module_name(p) for p in state['new_source_sha256']},'registered module delta')
require(all(catalog['registered_modules'].get(k)==v for k,v in prior['registered_modules'].items()),'prior registration drift')
for field in ('build_input_sha256','python_input_sha256'):
 require(all(catalog[field].get(k)==v for k,v in prior[field].items()),'prior task input drift')
 for p,h in catalog[field].items():exact(ROOT/p,h)
inventory={'tasks':catalog['summary']['tasks'],'public_declarations':catalog['summary']['public_claims'],
           'task_modules':catalog['summary']['task_modules'],'registered_modules':len(catalog['registered_modules'])}
require(inventory=={'tasks':184,'public_declarations':938,'task_modules':3549,'registered_modules':3980},'catalog counts')
review=load(BATCH/'root-review.json')
for p,h in review['reviewed_input_sha256'].items():exact(ROOT/p,h)
index=load(BATCH/'evidence-index.json')
for p,h in index['retained_exact_artifacts'].items():exact(BATCH/p,h)
require(not list(BATCH.rglob('*.lean')) and not list(BATCH.rglob('*.olean')),'evidence registered as live modules')
exact(BATCH/'numerical-followup.json',hash_file(OLD/'numerical-followup.json'))
require(len(load(BATCH/'numerical-followup.json')['rules'])==35,'descriptive followup count')

adapter_receipt=load(EXACT/'receipt.json');manifest=load(EXACT/'integration-manifest.json')
exact(EXACT/'receipt.json','0f8ee46effc304a9b9e70a900a5c95b8c300a4cd7f2d34fdf80bbd944e6c0b76')
exact(EXACT/'integration-manifest.json','3720edfdef74296b4aa42b0ecf6004ea68e4739f875e6565ad692f2c9545b540')
for rel,h in adapter_receipt['artifacts'].items():
 if rel.startswith('bench/review-checkpoints/'):exact(ROOT/rel,h)
for row in manifest['production_files']:exact(ROOT/row['destination'],adapter_receipt['artifacts'][row['source']])
source_guard=load(EXACT/'evidence/validation/completed-test-source-guard.json')
for p,h in source_guard['adapter_files'].items():exact(ROOT/p,h)
baseline=load(EXACT/'evidence/source-hashes.json')['baseline_repository_files']
replaced={row['destination'] for row in manifest['production_files']}
for p,h in baseline.items():
 if p not in replaced:exact(ROOT/p,h)
tests={p.relative_to(ROOT).as_posix() for p in (ROOT/'bench/tests').glob('test_*.py')}
require(tests==set(adapter_receipt['existing_python_test_files_unchanged'])|{'bench/tests/test_exact_numerical_bindings.py'},'test membership drift')
queue=load(ROOT/'bench/tritonbench_g/deferred_numerical_experiments.json')
legacy=json.loads(subprocess.check_output(['git','show',state['base_commit']+':bench/tritonbench_g/deferred_numerical_experiments.json'],cwd=ROOT))
require(queue['rules'][:11]==legacy['rules'] and queue['input_sha256']==legacy['input_sha256'],'legacy numeric drift')
require(len(queue['rules'])==29 and len(queue['exact_binding_provenance'])==86,'exact numeric count')
require([len(r['cases']) for r in queue['exact_callback_requirements']]==[29,20,9],'exact case count')
exact(TMP/'relu-source-grid-exact-current/deferred_numerical_experiments.json',hash_file(ROOT/'bench/tritonbench_g/deferred_numerical_experiments.json'))
exact(TMP/'relu-source-grid-exact-current/authority.json',hash_file(EXACT/'evidence/validation/generated/authority.json'))

ledger,before=load(ROOT/'bench/original_program_scope.json'),load(BATCH/'evidence/ledger-before.json')
original_ledger_bytes=subprocess.check_output(['git','show',state['base_commit']+':bench/original_program_scope.json'],cwd=ROOT)
require(sha(original_ledger_bytes)==state['scope_ledger_before_sha256'],'original ledger byte binding drift')
require(json.loads(original_ledger_bytes)==before,'retained ledger snapshot semantic drift')
require(len(ledger['tasks'])==183 and all(t['status']=='pending' for t in ledger['tasks']),'scope acceptance drift')
require(sum(t['initial_assessment']=='structural_gaps_confirmed' for t in ledger['tasks'])==66,'scope review count drift')
require(ledger['catalog_sha256']==sha(canonical(catalog)),'ledger catalog drift')
oldledger={t['task']:t for t in before['tasks']};catalogrows={t['task']:t for t in catalog['tasks']}
for row in ledger['tasks']:
 require(row['input_sha256']==task_bindings(ROOT,catalog,catalogrows[row['task']]),'task binding drift')
 old=oldledger[row['task']]
 if row['task']!='relu_triton_kernel':require(row==old,'unrelated scope review drift: '+row['task']);continue
 allowed={'input_sha256','structural_gaps','initial_assessment','partial_progress'}
 require({k:v for k,v in row.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed},'ReLU prior scope metadata drift')
 require(row['structural_gaps']==[{'kind':'remaining_original_program_scope','finding':item,'can_defer_as_numeric_assumption':False}
                                 for item in review['whole_stage_residuals']['relu_triton_kernel']],'ReLU gap drift')
allowed={'tasks','catalog_sha256','progress_checkpoint','deferred_queue'}
require({k:v for k,v in ledger.items() if k not in allowed}=={k:v for k,v in before.items() if k not in allowed},'ledger metadata drift')
exact(ROOT/ledger['deferred_queue']['path'],ledger['deferred_queue']['sha256'])

receipt=load(GLOBAL/'receipt.json')
require(receipt['status']=='failed' and receipt['catalog_sha256']==sha(canonical(catalog)),'global receipt binding')
failures={name for name,row in receipt['gates'].items() if row['status']!='passed'}
require(failures=={'scope_review','comparator_preflight','library_replay'},'unexpected global failures: '+repr(failures))
for name,row in receipt['gates'].items():
 if 'log' in row:exact(GLOBAL/row['log'],row['log_sha256'])
 if 'exit_code' in row:require((row['exit_code']==0)==(row['status']=='passed'),'gate status mismatch')
scope=load(GLOBAL/'scope_review.log')
require(scope['errors']==['183 original-program scopes remain unaccepted'] and scope['counts']=={'not_accepted':183},'unexpected scope failure')
for name in ('comparator_preflight','library_replay'):
 require('Official comparator is required' in (GLOBAL/receipt['gates'][name]['log']).read_text(),'unexpected comparator failure')
build=TMP/'relu-source-grid-integration-build.log';shards=TMP/'relu-source-grid-shard-tests.log'
jobs=re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())
require(jobs==['7341'],'build result differs from observed exit0')
require(re.findall(r'^Ran (\d+) tests? in ',shards.read_text(),re.M)==['25'] and shards.read_text().rstrip().endswith('OK'),'shard tests result')

# All final writes follow independent checks of actual results and preservation.
for name in ('receipt.json','value-metadata.json',*[p.name for p in GLOBAL.glob('*.log')]):
 source=GLOBAL/name
 if source.exists():
  dest=BATCH/'evidence/global-gates'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
actual={
 'integration-build':{'exit_code':0,'command':['/tmp/veritile-lean-toolchain/bin/lake','build','VeriTile','VeriTileMath','VeriTileFull','TritonBenchGVerified','TritonBenchGInProgress','TritonBenchGLegacy','TritonBenchSpecExamples','FPFrontendRegressions']},
 'catalog':{'exit_code':0,'result':'184 tasks;938 public;3549 task modules;3980 registered modules'},
 'shard-tests':{'exit_code':0,'tests':25},
 'numerical-check':{'exit_code':0,'command':['python3','-B','bench/deferred_numerical_experiments.py','--root','.','--check']},
 'exact-check':{'exit_code':0,'command':['python3','-B','bench/exact_numerical_bindings.py','--root','.','--output',str(TMP/'relu-source-grid-exact-current')]},
 'global':{'exit_code':1,'failed_gates':sorted(failures)}
}
for name,row in actual.items():
 source=TMP/('relu-source-grid-'+name+'.log')
 dest=BATCH/'evidence/integration'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(source,dest)
 row['log']={'path':dest.relative_to(BATCH).as_posix(),'sha256':hash_file(dest)}
for source,name in [(TMP/'rebind_relu_source_ledger.py','rebind-ledger.py'),(Path(__file__),'finalize.py')]:
 shutil.copyfile(source,BATCH/'evidence'/name)
shutil.copytree(TMP/'relu-source-grid-finalization-failure-01',BATCH/'evidence/finalization-attempts/01-ledger-json-encoding')
write_json(BATCH/'evidence/observed-exits.json',actual)
readme=BATCH/'README.md'
readme.write_text(readme.read_text()+'''\n## Observed integration validation\n\nThe registered eight-target build exited 0 with 7,341 jobs. Catalog generation,
all 25 sharding tests and both current numerical collector checks passed.
Catalog counts are 184 tasks, 938 public declarations, 3,549 task modules and
3,980 registered modules. The 238-test adapter result was reused only after
checking the exact tested source bytes and all test membership. No unnecessary
second run is claimed.\n\nThe actual global audit exited 1: scope_review fails because all 183 scopes
remain pending; comparator_preflight and library_replay fail because the
official comparator is unavailable to this host runner. Every other gate passed.
The historical pinned comparator replay does not validate these current sources.
The full-scope review inventory remains 66 confirmed gaps and 117 unreviewed.
''')
paths=set(state['new_source_sha256'])|{row['destination'] for row in manifest['production_files']}
paths|={'.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md',
        'lakefile.toml','lake-manifest.json','lean-toolchain'}
paths|={p.relative_to(ROOT).as_posix() for p in BATCH.rglob('*') if p.is_file()}
paths|={p.relative_to(ROOT).as_posix() for p in EXACT.rglob('*') if p.is_file()}
validation={
 'schema':1,'kind':'relu_source_grid_and_exact_numerical_adapter_integration',
 'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':state['base_commit'],
 'status':'validated_partial_progress_full_standard_incomplete',
 'inventory':inventory,'scope':{'requested':183,'accepted':0,'pending':183,'confirmed_gaps':66,'unreviewed':117},
 'partial_contract_inventory':{'execution':41,'protocol':4,'total':45,'tasks':10,'new_execution':2},
 'numerical_registration':queue['exact_numerical_summary'],
 'numerical_boundary':'All new rules unvalidated. Only three earlier statements registered; new ReLU source contracts require separate bindings.',
 'checked_input_sha256':{p:hash_file(ROOT/p) for p in sorted(paths)},
 'actual_runs':actual,'reused_python_tests':{'total':238,'existing':210,'new':28,'failures':0,'errors':0,
  'test_log_sha256':source_guard['all_python_tests_log_sha256'],'tested_source_guard_sha256':hash_file(EXACT/'evidence/validation/completed-test-source-guard.json'),
  'source_and_membership_rechecked':True,'rerun':False},
 'preservation':{'prior_public_statements_and_metadata':936,'prior_supplemental_declarations':43,
  'unrelated_scope_rows_unchanged':182,'prior_descriptive_numerical_rules':35,'legacy_numerical_rules':11,
  'retained_ledger_snapshot':'Semantically equal JSON with escaped non-ASCII text; original bytes independently bound to the base Git blob. Initial byte-identity guard failure retained.',
  'original_baseline_inputs':len(state['original_baseline_input_map']),
  'original_metadata_baseline_unchanged':state['metadata_baseline_sha256'],
  'predecessor_checkpoint_files_unchanged':len(state['predecessor_file_sha256'])},
 'global_gates':{k:v['status'] for k,v in receipt['gates'].items()},
 'native_execution':False,'current_comparator_success':False,
 'acceptance_boundary':'Complete declared source stages/dtypes/branches are still required. Numerical registration, proof compilation and partial source/byte composition do not establish whole-program acceptance.'
}
write_json(BATCH/'validation.json',validation)
print(json.dumps({'status':validation['status'],'sha256':hash_file(BATCH/'validation.json'),
                  'inventory':inventory,'scope':validation['scope'],'global_gates':validation['global_gates']},indent=2))
