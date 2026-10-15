"""Record actual validation and exact preservation for four new contracts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP=Path('/tmp/veritile-standardization-20261015')
BATCH=ROOT/'bench/review-checkpoints/2026-10-16/quantization-and-mixed-softmax'
OLD=ROOT/'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics'
BASE=ROOT/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
GLOBAL=TMP/'quant-soft-current-global'
sys.path.insert(0,str(ROOT/'bench'))
from adapter_common import canonical,hash_file,load,module_name,sha,write_json
from check_original_program_scope import task_bindings
from current_source_catalog import validate_supplement
from current_value_metadata import check_supplement_scope

def require(ok,message):
    if not ok:raise ValueError(message)
def hashes(path):
    return {p.relative_to(path).as_posix():hash_file(p) for p in sorted(path.rglob('*')) if p.is_file()}
def exact(path,h):require(hash_file(path)==h,'hash changed: '+str(path))
require(not (BATCH/'validation.json').exists(),'refuse final evidence overwrite')
state=load(BATCH/'evidence/preparation-state.json')
require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==state['base_commit'],'HEAD drift')
require(hashes(OLD)==state['predecessor_file_sha256'],'predecessor drift')
require(hashes(BASE)==state['metadata_baseline_sha256'],'original baseline drift')
for p,h in state['original_baseline_input_map'].items():exact(ROOT/p,h)
for p,h in state['new_source_sha256'].items():exact(ROOT/p,h)
catalog,prior=load(BATCH/'current-source-catalog.json'),load(OLD/'current-source-catalog.json')
supp,oldsupp=load(BATCH/'catalog-supplement.json'),load(OLD/'catalog-supplement.json')
validate_supplement(ROOT,ROOT/'bench/review-checkpoints',supp,set(catalog['expected_task_names']))
check_supplement_scope(supp,ROOT/'bench/review-checkpoints')
oldrows={t['task']:t for t in oldsupp['tasks']};newrows={t['task']:t for t in supp['tasks']}
names=set(state['changed_tasks'])
require(set(newrows)-set(oldrows)==names,'supplement membership')
require(all(newrows[name]==row for name,row in oldrows.items()),'old supplement changed')
all_decls=[d for t in supp['tasks'] for d in t['declarations']]
require(len(all_decls)==49 and sum(d['role']=='parametric_value_contract' for d in all_decls)==45
        and sum(d['role']=='protocol_contract' for d in all_decls)==4,'contract counts')
require(sum(bool(t['declarations']) for t in supp['tasks'])==14,'task count')
def claims(c):return {(t['task'],d['file'],d['declaration']):d for t in c['tasks'] for d in t['declarations']}
oldclaims,newclaims=claims(prior),claims(catalog)
require(len(oldclaims)==938 and all(newclaims.get(k)==v for k,v in oldclaims.items()),'old declarations changed')
added={(name,d['file'],d['declaration']) for name in names for d in newrows[name]['declarations']}
require(set(newclaims)-set(oldclaims)==added and len(added)==4,'new declarations')
require(set(catalog['registered_modules'])-set(prior['registered_modules'])=={module_name(p) for p in state['new_source_sha256']},'module delta')
require(all(catalog['registered_modules'].get(k)==v for k,v in prior['registered_modules'].items()),'old registration changed')
for field in ('build_input_sha256','python_input_sha256'):
    require(all(catalog[field].get(k)==v for k,v in prior[field].items()),'prior inputs changed')
    for p,h in catalog[field].items():exact(ROOT/p,h)
inventory={'tasks':catalog['summary']['tasks'],'public_declarations':catalog['summary']['public_claims'],
           'task_modules':catalog['summary']['task_modules'],'registered_modules':len(catalog['registered_modules'])}
require(inventory=={'tasks':184,'public_declarations':942,'task_modules':3560,'registered_modules':3991},'catalog inventory')
review=load(BATCH/'root-review.json')
for p,h in review['reviewed_input_sha256'].items():exact(ROOT/p,h)
for name in names:
    for p,h in newrows[name].get('shared_dependency_sha256',{}).items():
        require(review['reviewed_input_sha256'][p]==h,'unbound shared dependency');exact(ROOT/p,h)
index=load(BATCH/'evidence-index.json')
for p,h in index['retained_exact_artifacts'].items():exact(BATCH/p,h)
require(not list(BATCH.rglob('*.lean')) and not list(BATCH.rglob('*.olean')),'evidence contains live Lean modules')
exact(BATCH/'numerical-followup.json',hash_file(OLD/'numerical-followup.json'))
require(len(load(BATCH/'numerical-followup.json')['rules'])==35,'prior descriptive rules')
queuepath='bench/tritonbench_g/deferred_numerical_experiments.json'
queue=load(ROOT/queuepath)
require((ROOT/queuepath).read_bytes()==subprocess.check_output(['git','show',state['base_commit']+':'+queuepath],cwd=ROOT),'numeric queue changed')
require(len(queue['rules'])==29,'numeric count')
exact(TMP/'quant-soft-exact-current/deferred_numerical_experiments.json',hash_file(ROOT/queuepath))
exact(TMP/'quant-soft-exact-current/authority.json',hash_file(ROOT/'bench/review-checkpoints/2026-10-16/exact-numerical-bindings/evidence/validation/generated/authority.json'))
ledger,before=load(ROOT/'bench/original_program_scope.json'),load(BATCH/'evidence/ledger-before.json')
exact(BATCH/'evidence/ledger-before.json',state['scope_ledger_before_sha256'])
require((BATCH/'evidence/ledger-before.json').read_bytes()==subprocess.check_output(['git','show',state['base_commit']+':bench/original_program_scope.json'],cwd=ROOT),'ledger snapshot changed')
require(len(ledger['tasks'])==183 and all(t['status']=='pending' for t in ledger['tasks']),'scope status')
require(sum(t['initial_assessment']=='structural_gaps_confirmed' for t in ledger['tasks'])==85,'scope inventory')
require(ledger['catalog_sha256']==sha(canonical(catalog)),'catalog binding')
oldledger={t['task']:t for t in before['tasks']};catalogrows={t['task']:t for t in catalog['tasks']}
for row in ledger['tasks']:
    old=oldledger[row['task']]
    require(row['input_sha256']==task_bindings(ROOT,catalog,catalogrows[row['task']]),'task binding')
    if row['task'] not in names:
        require(row==old,'unrelated scope changed');continue
    allowed={'input_sha256','structural_gaps','initial_assessment','partial_progress'}
    require({k:v for k,v in row.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed},'historical task scope changed')
    require(row['partial_progress']['previous_structural_gaps']==old['structural_gaps'],'previous gaps lost')
    require(row['structural_gaps']==[{'kind':'remaining_original_program_scope','finding':item,'can_defer_as_numeric_assumption':False}
                                  for item in review['whole_stage_residuals'][row['task']]],'remaining scope changed')
allowed={'tasks','catalog_sha256','progress_checkpoint'}
require({k:v for k,v in ledger.items() if k not in allowed}=={k:v for k,v in before.items() if k not in allowed},'ledger metadata changed')
receipt=load(GLOBAL/'receipt.json')
require(receipt['status']=='failed' and receipt['catalog_sha256']==sha(canonical(catalog)),'global receipt binding')
failures={name for name,row in receipt['gates'].items() if row['status']!='passed'}
require(failures=={'scope_review','comparator_preflight','library_replay'},'unexpected gates: '+repr(failures))
for name,row in receipt['gates'].items():
    if 'log' in row:exact(GLOBAL/row['log'],row['log_sha256'])
    if 'exit_code' in row:require((row['exit_code']==0)==(row['status']=='passed'),'gate status')
scope=load(GLOBAL/'scope_review.log')
require(scope['errors']==['183 original-program scopes remain unaccepted'] and scope['counts']=={'not_accepted':183},'scope failure')
for name in ('comparator_preflight','library_replay'):
    require('Official comparator is required' in (GLOBAL/receipt['gates'][name]['log']).read_text(),'comparator failure')
build=TMP/'quant-soft-integration-build.log';shards=TMP/'quant-soft-shard-tests.log'
require(re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build.read_text())==['7352'],'build result')
require(re.findall(r'^Ran (\d+) tests? in ',shards.read_text(),re.M)==['25'] and shards.read_text().rstrip().endswith('OK'),'test result')
require(load(TMP/'quant-soft-exact-check.log')['status']=='passed','exact collector')

# Final writes occur only after actual results and all preservation checks.
for name in ('receipt.json','value-metadata.json',*[p.name for p in GLOBAL.glob('*.log')]):
    source=GLOBAL/name
    if source.exists():
        dest=BATCH/'evidence/global-gates'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
actual={
    'integration-build':{'exit_code':0,'jobs':7352,'targets':['VeriTile','VeriTileMath','VeriTileFull','TritonBenchGVerified','TritonBenchGInProgress','TritonBenchGLegacy','TritonBenchSpecExamples','FPFrontendRegressions']},
    'catalog':{'exit_code':0,'inventory':inventory},'shard-tests':{'exit_code':0,'tests':25},
    'numerical-check':{'exit_code':0},'exact-check':{'exit_code':0,'registered_rules':29},
    'rebind-ledger':{'exit_code':0,'changed_tasks':sorted(names)},
    'global':{'exit_code':1,'failed_gates':sorted(failures)}}
for name,row in actual.items():
    source=TMP/('quant-soft-'+name+'.log')
    dest=BATCH/'evidence/integration'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest)
    row['log']={'path':dest.relative_to(BATCH).as_posix(),'sha256':hash_file(dest)}
for source,name in [(TMP/'rebind_quant_soft_ledger.py','rebind-ledger.py'),(Path(__file__),'finalize.py')]:
    shutil.copyfile(source,BATCH/'evidence'/name)
write_json(BATCH/'evidence/observed-exits.json',actual)
readme=BATCH/'README.md'
readme.write_text(readme.read_text()+'''\n## Observed integration validation\n\nThe eight-target integration build passed with 7,352 jobs. All 25 sharding\ntests, catalog generation and both numerical collector checks passed. The\ncurrent catalog has 184 tasks, 942 public declarations, 3,560 task modules and\n3,991 registered modules. The previous 938 declaration records and 45\nsupplemental declarations are unchanged, as are the other 179 scope rows.\n\nThe actual global audit exited 1. Its only failing gates were scope_review\n(all 183 programs remain pending), comparator_preflight and library_replay\n(the official comparator is unavailable). All other global gates passed.\nNo current comparator success, native execution or new numerical registration\nis claimed. The original frozen metadata baseline remains byte-identical.\n\nTwo unsuccessful integration attempts are retained: initial cross-task source\nbindings rejected by the catalog, and a ledger rebind started before catalog\ngeneration finished. Neither altered existing proofs or accepted any task.\n''')
paths=set(state['new_source_sha256'])|{'.github/workflows/bench-audit.yml','bench/original_program_scope.json','bench/original_program_scope.md','lakefile.toml','lake-manifest.json','lean-toolchain',queuepath}
paths|={p.relative_to(ROOT).as_posix() for p in BATCH.rglob('*') if p.is_file()}
validation={'schema':1,'kind':'quantization_and_mixed_softmax_integration',
    'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'base_commit':state['base_commit'],
    'status':'validated_partial_progress_full_standard_incomplete','inventory':inventory,
    'scope':{'requested':183,'accepted':0,'pending':183,'confirmed_gaps':85,'unreviewed':98},
    'partial_contract_inventory':{'execution':45,'protocol':4,'total':49,'tasks':14,'new_execution':4},
    'checked_input_sha256':{p:hash_file(ROOT/p) for p in sorted(paths)},'actual_runs':actual,
    'preservation':{'prior_public_statements_and_metadata':938,'prior_supplemental_declarations':45,
        'unrelated_scope_rows_unchanged':179,'prior_descriptive_numerical_rules':35,'registered_numerical_rules':29,
        'original_metadata_baseline_unchanged':state['metadata_baseline_sha256'],
        'predecessor_checkpoint_files_unchanged':len(state['predecessor_file_sha256'])},
    'global_gates':{k:v['status'] for k,v in receipt['gates'].items()},
    'native_execution':False,'current_comparator_success':False,'new_exact_numeric_registrations':0,
    'acceptance_boundary':'All original stages, dtypes and branches must be covered. Compilation and partial execution contracts are not whole-program acceptance.'}
write_json(BATCH/'validation.json',validation)
print(json.dumps({'status':validation['status'],'sha256':hash_file(BATCH/'validation.json'),'inventory':inventory,'scope':validation['scope']},indent=2))
