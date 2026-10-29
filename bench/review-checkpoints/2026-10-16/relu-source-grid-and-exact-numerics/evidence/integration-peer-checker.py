"""Read-only integration review; does not import or execute production tests."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
B = ROOT / 'bench/review-checkpoints/2026-10-16/relu-source-grid-and-exact-numerics'
OLD = B.parent / 'fp64-admission-relu-access'
EXACT = B.parent / 'exact-numerical-bindings'
checks = []
hashes_checked = {}

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def load(p):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, 'duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(p.read_text(), object_pairs_hook=unique)

def exact(p, h):
    require(digest(p) == h, 'hash differs: ' + str(p))
    hashes_checked[str(p)] = h

def mapping(root, values):
    for p, h in values.items():
        exact(root / p, h)

def by_task(data):
    result = {t['task']: t for t in data['tasks']}
    require(len(result) == len(data['tasks']), 'duplicate task')
    return result

head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
require(head == 'ea6512dd1c663d24b2bcbdf0b0c29b89481c5a8a', 'HEAD drift')
supp, oldsupp = load(B/'catalog-supplement.json'), load(OLD/'catalog-supplement.json')
newrows, oldrows = by_task(supp), by_task(oldsupp)
require(set(newrows) == set(oldrows), 'supplement task membership changed')
newdecls = []
for task, old in oldrows.items():
    new = newrows[task]
    if task != 'relu_triton_kernel':
        require(new == old, 'unrelated supplement task changed: ' + task)
        continue
    require(new['declarations'][:len(old['declarations'])] == old['declarations'], 'old ReLU declarations changed')
    newdecls = new['declarations'][len(old['declarations']):]
    require(len(newdecls) == 2, 'new ReLU declaration count')
    require(new['required_value_stages'] == old['required_value_stages'] + [s for d in newdecls for s in d['covers_stages']], 'stages changed')
    require(all(new['source_sha256'].get(p) == h for p, h in old['source_sha256'].items()), 'prior ReLU source bindings changed')
    allowed = {'review', 'source_sha256', 'declarations', 'required_value_stages'}
    require({k:v for k,v in new.items() if k not in allowed} == {k:v for k,v in old.items() if k not in allowed}, 'old ReLU task metadata changed')
require(sum(len(t['declarations']) for t in oldrows.values()) == 43, 'old supplement count')
require(Counter(d['role'] for t in newrows.values() for d in t['declarations']) == {'parametric_value_contract':41, 'protocol_contract':4}, 'role counts')
checks.append('All 43 previous supplemental declarations and metadata preserved; exactly two ReLU source contracts added.')

catalog, prior = load(B/'current-source-catalog.json'), load(OLD/'current-source-catalog.json')
def claims(c):
    rows = [(t['task'], d) for t in c['tasks'] for d in t['declarations']]
    result = {(t,d['file'],d['declaration']):d for t,d in rows}
    require(len(result) == len(rows), 'duplicate declaration')
    return result
oldclaims, newclaims = claims(prior), claims(catalog)
require(len(oldclaims) == 936 and len(newclaims) == 938, 'public claim counts')
require(all(newclaims.get(k) == v for k,v in oldclaims.items()), 'previous catalog declaration changed')
require(set(newclaims)-set(oldclaims) == {('relu_triton_kernel',d['file'],d['declaration']) for d in newdecls}, 'public declaration delta')
state = load(B/'evidence/preparation-state.json')
newfiles = state['new_source_sha256']
require(len(newfiles) == 6, 'new source module count')
mapping(ROOT, newfiles)
require(set(catalog['registered_modules'])-set(prior['registered_modules']) == {p[:-5].replace('/','.') for p in newfiles}, 'registered module delta')
require(all(catalog['registered_modules'].get(k) == v for k,v in prior['registered_modules'].items()), 'old module registration changed')
require((catalog['summary']['tasks'], catalog['summary']['task_modules'], len(catalog['registered_modules'])) == (184,3549,3980), 'catalog counts')
require(not catalog['unregistered_task_modules'], 'unregistered task modules')
for field in ('build_input_sha256','python_input_sha256'):
    require(all(catalog[field].get(p) == h for p,h in prior[field].items()), 'old input changed: ' + field)
    mapping(ROOT, catalog[field])
for task in catalog['tasks']:
    for f in task['files']:
        exact(ROOT/f['path'], f['observed_sha256'])
sys.path.insert(0, str(ROOT/'bench'))
from adapter_common import declarations
for d in newdecls:
    parsed = declarations(ROOT/d['file'])[d['declaration']]
    require(parsed['kind'] == 'specification' and parsed['statement_sha256'] == d['statement_sha256'], 'new public statement binding')
    require(d['original_scope']['standard_accepted'] is False and d['native_binding_status'] == 'not_established', 'new declaration acceptance inflation')
checks.append('Catalog preserves all 936 earlier public claims; 938 public declarations, 3549 task modules, and 3980 registered modules match live bytes.')

mapping(OLD, state['predecessor_file_sha256'])
baseline_root = ROOT/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
mapping(baseline_root, state['metadata_baseline_sha256'])
mapping(ROOT, state['original_baseline_input_map'])
mapping(ROOT, load(B/'root-review.json')['reviewed_input_sha256'])
index = load(B/'evidence-index.json')
mapping(B, index['retained_exact_artifacts'])
require(index['new_source_sha256'] == newfiles, 'new source evidence index')
require(not list(B.rglob('*.lean')) and not list(B.rglob('*.olean')), 'live modules hidden in evidence')
checks.append('Predecessor checkpoint, original metadata baseline, original baseline inputs, root-reviewed sources, and retained evidence hashes match.')

queue = load(ROOT/'bench/tritonbench_g/deferred_numerical_experiments.json')
legacy = json.loads(subprocess.check_output(['git','show',head+':bench/tritonbench_g/deferred_numerical_experiments.json'],cwd=ROOT))
require(queue['rules'][:11] == legacy['rules'] and queue['input_sha256'] == legacy['input_sha256'], 'legacy queue drift')
require(len(queue['rules']) == 29 and all(r['status'] == 'pending_numeric_validation' for r in queue['rules'][11:]), 'new numeric count/status')
require(len(queue['exact_binding_provenance']) == 86, 'numeric provenance count')
require([len(r['cases']) for r in queue['exact_callback_requirements']] == [29,20,9], 'numerical cases')
registered = {(r['declaration']['file'],r['declaration']['declaration']) for r in queue['exact_callback_requirements']}
require(len(registered) == 3 and not registered.intersection((d['file'],d['declaration']) for d in newdecls), 'new ReLU falsely registered')
require(registered == {('bench/tritonbench_g/vector_addition/FP64GridCorrect.lean','vector_addition_fp64_defined_grid_correctness'),('bench/tritonbench_g/vector_addition/FP64GridCorrect.lean','vector_addition_fp64_floating_grid_correctness'),('bench/tritonbench_g/relu_triton_kernel/AccessGridCorrect.lean','relu_access_grid_correctness')}, 'numerical declaration membership')
require((B/'numerical-followup.json').read_bytes() == (OLD/'numerical-followup.json').read_bytes(), 'descriptive followup changed')
require(len(load(B/'numerical-followup.json')['rules']) == 35, 'descriptive count')
require((TMP/'relu-source-grid-exact-current/deferred_numerical_experiments.json').read_bytes() == (ROOT/'bench/tritonbench_g/deferred_numerical_experiments.json').read_bytes(), 'current numerical output mismatch')
checks.append('Queue retains all 11 legacy rules, adds 18 unvalidated families, binds exactly three old declarations over 58 cases, and excludes both new ReLU contracts; 35 descriptive followups unchanged.')

receipt, manifest = load(EXACT/'receipt.json'), load(EXACT/'integration-manifest.json')
for row in manifest['production_files']:
    exact(ROOT/row['destination'], receipt['artifacts'][row['source']])
for p,h in receipt['artifacts'].items():
    if p.startswith('bench/review-checkpoints/'):
        exact(ROOT/p,h)
guard = load(EXACT/'evidence/validation/completed-test-source-guard.json')
mapping(ROOT, guard['adapter_files'])
mapping(ROOT, receipt['existing_python_test_files_unchanged'])
replaced = {row['destination'] for row in manifest['production_files']}
for p,h in load(EXACT/'evidence/source-hashes.json')['baseline_repository_files'].items():
    if p not in replaced:
        exact(ROOT/p,h)
require({p.relative_to(ROOT).as_posix() for p in (ROOT/'bench/tests').glob('test_*.py')} == set(receipt['existing_python_test_files_unchanged'])|{'bench/tests/test_exact_numerical_bindings.py'}, 'Python test membership drift')
exact(EXACT/'evidence/validation/all-python-tests.log', guard['all_python_tests_log_sha256'])
exact(EXACT/'evidence/validation/test-results.json', guard['test_results_sha256'])
require(load(EXACT/'evidence/validation/test-results.json') == {'tests':238,'failures':0,'errors':0,'success':True}, 'reused test result')
checks.append('All seven integrated production files, six tested adapter/input files, five legacy test files, test membership, and retained 238-test results match the frozen source guards; no tests rerun.')

ledger, before = load(ROOT/'bench/original_program_scope.json'), load(B/'evidence/ledger-before.json')
blob = subprocess.check_output(['git','show',head+':bench/original_program_scope.json'],cwd=ROOT)
require(hashlib.sha256(blob).hexdigest() == state['scope_ledger_before_sha256'] and json.loads(blob) == before, 'ledger preimage binding')
oldledger = by_task(before)
require(len(ledger['tasks']) == 183 and all(t['status'] == 'pending' for t in ledger['tasks']), 'scope statuses')
require(sum(t['initial_assessment']=='structural_gaps_confirmed' for t in ledger['tasks']) == 66, 'confirmed scope gaps')
require(sum(t['initial_assessment']=='full_scope_not_yet_verified' for t in ledger['tasks']) == 117, 'unreviewed scope count')
require(ledger['catalog_sha256'] == hashlib.sha256(canon(catalog)).hexdigest(), 'ledger catalog binding')
exact(ROOT/ledger['deferred_queue']['path'], ledger['deferred_queue']['sha256'])
for row in ledger['tasks']:
    if row['task'] != 'relu_triton_kernel':
        require(row == oldledger[row['task']], 'unrelated ledger row changed: ' + row['task'])
    else:
        allowed = {'input_sha256','structural_gaps','initial_assessment','partial_progress'}
        require({k:v for k,v in row.items() if k not in allowed} == {k:v for k,v in oldledger[row['task']].items() if k not in allowed}, 'ReLU scope/coverage weakened')
        require(all(g['can_defer_as_numeric_assumption'] is False for g in row['structural_gaps']), 'structural gaps deferred')
checks.append('All 183 scopes remain pending: 66 confirmed gaps and 117 unreviewed; all 182 unrelated rows and existing ReLU coverage/status remain unchanged.')

validation = load(B/'validation.json')
mapping(ROOT, validation['checked_input_sha256'])
global_receipt = load(TMP/'relu-source-grid-current-global/receipt.json')
require(load(B/'evidence/global-gates/receipt.json') == global_receipt, 'retained global receipt mismatch')
require(global_receipt['catalog_sha256'] == ledger['catalog_sha256'], 'global catalog binding')
failures = {k for k,v in global_receipt['gates'].items() if v['status'] != 'passed'}
require(failures == {'scope_review','comparator_preflight','library_replay'}, 'unexpected global failure')
for gate, row in global_receipt['gates'].items():
    if 'log' in row:
        exact(B/'evidence/global-gates'/row['log'],row['log_sha256'])
scope = load(B/'evidence/global-gates/scope_review.log')
require(scope['errors'] == ['183 original-program scopes remain unaccepted'], 'unexpected scope error')
for gate in ('comparator_preflight','library_replay'):
    require('Official comparator is required' in (B/'evidence/global-gates'/global_receipt['gates'][gate]['log']).read_text(), 'unexpected comparator failure')
for row in validation['actual_runs'].values():
    exact(B/row['log']['path'],row['log']['sha256'])
build = (B/validation['actual_runs']['integration-build']['log']['path']).read_text()
require(re.findall(r'Build completed successfully \(([\d,]+) jobs\)\.',build) == ['7341'], 'build log result')
shards = (B/validation['actual_runs']['shard-tests']['log']['path']).read_text()
require(re.findall(r'^Ran (\d+) tests? in ',shards,re.M) == ['25'] and shards.rstrip().endswith('OK'), 'shard result')
require(validation['actual_runs']['integration-build']['exit_code'] == 0, 'build reported exit')
require(validation['actual_runs']['global']['exit_code'] == 1 and validation['current_comparator_success'] is False, 'global success inflation')
checks.append('Retained logs show successful 7341-job build and 25 shard tests; global failure is limited to 183 pending scopes and unavailable comparator/library replay, with no native/current comparator success claimed.')

workflow = (ROOT/'.github/workflows/bench-audit.yml').read_text()
require(workflow.count('relu-source-grid-and-exact-numerics/current-source-catalog.json') == 2 and workflow.count('relu-source-grid-and-exact-numerics/catalog-supplement.json') == 1 and 'fp64-admission-relu-access/current-source-catalog.json' not in workflow, 'workflow catalog wiring')
checks.append('Both workflow catalog consumers and the global supplement point to the new checkpoint.')

pinned = [B/'validation.json',B/'catalog-supplement.json',B/'current-source-catalog.json',B/'root-review.json',B/'evidence-index.json',B/'evidence/preparation-state.json',B/'evidence/ledger-before.json',ROOT/'bench/original_program_scope.json',ROOT/'bench/tritonbench_g/deferred_numerical_experiments.json',ROOT/'.github/workflows/bench-audit.yml',EXACT/'receipt.json',EXACT/'integration-manifest.json',EXACT/'evidence/validation/completed-test-source-guard.json',TMP/'relu-source-grid-current-global/receipt.json',Path(__file__)]
report = {'schema':1,'kind':'independent_current_integration_review','reviewer':'relu_current_review','status':'no_actionable_findings','base_commit':head,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'findings':[],'checks':checks,'pinned_sha256':{str(p):digest(p) for p in pinned},'verified_hash_entries':len(hashes_checked),'verified_hash_map_sha256':hashlib.sha256(canon(hashes_checked)).hexdigest(),'test_reruns':0,'repository_edits':0,'boundary':'Independent integration, preservation, metadata, registration and receipt review; no new native execution or comparator replay. Root semantic review and earlier independent source reviews are retained evidence, not replaced by this review.'}
out=TMP/'relu-current-integration-independent-review.json'
out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'report':str(out),'sha256':digest(out),'status':report['status'],'checks':len(checks),'verified_hash_entries':len(hashes_checked)}))
