#!/usr/bin/env python3
"""Read-only frozen-package audit and copied-root collector replay; no test rerun."""
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PACKAGE = BASE / 'numerical-exact-binding-repo-adapter-wip'
REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
CHECKPOINT = Path('bench/review-checkpoints/2026-10-16/exact-numerical-bindings')
COMMIT = '74cb67ae99cd5cf77096dc74868d8adcb243b46a'
RECEIPT_SHA = '0f8ee46effc304a9b9e70a900a5c95b8c300a4cd7f2d34fdf80bbd944e6c0b76'
MANIFEST_SHA = '3720edfdef74296b4aa42b0ecf6004ea68e4739f875e6565ad692f2c9545b540'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(path.read_bytes())


def read_json(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def run(label, command, expected=0):
    proc = subprocess.run(command, capture_output=True, text=True)
    (HERE / (label + '.stdout')).write_text(proc.stdout)
    (HERE / (label + '.stderr')).write_text(proc.stderr)
    row = {'command': command, 'exit_code': proc.returncode,
           'stdout_sha256': sha(proc.stdout.encode()),
           'stderr_sha256': sha(proc.stderr.encode())}
    (HERE / (label + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    require(proc.returncode == expected, label + ': ' + proc.stderr)
    return proc


def main():
    receipt = read_json(PACKAGE / 'receipt.json')
    require(digest(PACKAGE / 'receipt.json') == RECEIPT_SHA, 'receipt SHA')
    require((PACKAGE / 'receipt.json').read_bytes() == (PACKAGE / CHECKPOINT / 'receipt.json').read_bytes(), 'receipt copies')
    manifest = read_json(PACKAGE / CHECKPOINT / 'integration-manifest.json')
    require(digest(PACKAGE / CHECKPOINT / 'integration-manifest.json') == MANIFEST_SHA, 'manifest SHA')
    require(len(receipt['artifacts']) == 60, 'artifact count')
    for path, expected in receipt['artifacts'].items():
        candidate = (PACKAGE / path).resolve()
        require(candidate.is_relative_to(PACKAGE), 'artifact path escapes')
        require(digest(candidate) == expected, 'artifact SHA: ' + path)
    checkpoint_files = sorted(p for p in (PACKAGE / CHECKPOINT).rglob('*') if p.is_file())
    require(len(checkpoint_files) == 31, 'checkpoint file count')
    require(all(p.name == 'receipt.json' or str(p.relative_to(PACKAGE)) in receipt['artifacts'] for p in checkpoint_files), 'unrecorded checkpoint file')
    duplicates = 0
    for path in receipt['artifacts']:
        if path.startswith('output/'):
            twin = PACKAGE / CHECKPOINT / 'evidence/validation' / path.removeprefix('output/')
            require(twin.read_bytes() == (PACKAGE / path).read_bytes(), 'checkpoint validation copy: ' + path)
            duplicates += 1

    sources = read_json(PACKAGE / CHECKPOINT / 'evidence/source-hashes.json')
    baseline = sources['baseline_repository_files']
    require(len(baseline) == 292 and sources['tested_source_head'] == COMMIT, 'baseline inventory')
    requests = ''.join(COMMIT + ':' + path + '\n' for path in baseline).encode()
    proc = subprocess.run(['git', '-C', str(REPO), 'cat-file', '--batch'], input=requests, capture_output=True, check=True)
    blob_stream = proc.stdout
    blobs = {}
    position = 0
    for path, expected in baseline.items():
        end = blob_stream.index(b'\n', position)
        header = blob_stream[position:end].decode().split()
        require(len(header) == 3 and header[1] == 'blob', 'baseline git blob: ' + path)
        size = int(header[2]); data = blob_stream[end + 1:end + 1 + size]
        require(sha(data) == expected, 'tested commit baseline: ' + path)
        blobs[path] = data; position = end + 2 + size
    require(position == len(blob_stream), 'unexpected git blob tail')
    replacement = {row['destination']: row for row in manifest['production_files']}
    current_preimages = {}
    for path, expected in baseline.items():
        allowed = {expected}
        if path in replacement:
            allowed.add(digest(PACKAGE / replacement[path]['source']))
        current = digest(REPO / path)
        require(current in allowed, 'current baseline changed: ' + path)
        current_preimages[path] = current
    for row in manifest['production_files']:
        preimage = row['before_sha256']
        if preimage is not None:
            require(sha(subprocess.check_output(['git', '-C', str(REPO), 'show', COMMIT + ':' + row['destination']])) == preimage, 'production preimage: ' + row['destination'])
        else:
            status = subprocess.run(['git', '-C', str(REPO), 'cat-file', '-e', COMMIT + ':' + row['destination']], capture_output=True)
            require(status.returncode != 0, 'unexpected preexisting production file')

    test_files = receipt['existing_python_test_files_unchanged']
    require(test_files == sources['unchanged_existing_test_files'] and len(test_files) == 5, 'existing test source inventory')
    expected_tests = set()
    for path, expected in test_files.items():
        require(sha(blobs[path]) == expected, 'old test changed')
    test_sources = {path: blobs[path].decode() for path in test_files}
    test_sources['bench/tests/test_exact_numerical_bindings.py'] = (PACKAGE / 'bench/tests/test_exact_numerical_bindings.py').read_text()
    for path, source in test_sources.items():
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name.startswith('test_'):
                        expected_tests.add(Path(path).stem + '.' + node.name + '.' + method.name)
    log = (PACKAGE / 'output/all-python-tests.log').read_text()
    rows = [re.fullmatch(r'(test_\w+) \(([^)]+)\) \.\.\. ok', line) for line in log.splitlines()]
    observed = [row.group(2) for row in rows if row]
    require(len(observed) == 238 and len(set(observed)) == 238 and set(observed) == expected_tests, 'actual test names differ from AST')
    by_module = dict(sorted(collections.Counter(name.split('.')[0] for name in observed).items()))
    require(by_module['test_exact_numerical_bindings'] == 28 and sum(by_module.values()) == 238, 'test counts')
    require(re.search(r'\nRan 238 tests in [0-9.]+s\n\nOK\s*$', log) is not None, 'test log completion')
    results = read_json(PACKAGE / 'output/test-results.json')
    require(results == {'tests': 238, 'failures': 0, 'errors': 0, 'success': True}, 'test outcome')
    guard = read_json(PACKAGE / 'output/completed-test-source-guard.json')
    guarded = {row['source'] for row in manifest['production_files'] if row['source'].endswith('.py')}
    guarded.add(str(CHECKPOINT / 'evidence/fp64-source-routes/receipt.json'))
    require(set(guard['adapter_files']) == guarded and guard['source_head'] == COMMIT and guard['test_count'] == 238, 'test guard coverage')
    for path, expected in guard['adapter_files'].items():
        require(digest(PACKAGE / path) == expected, 'test guard source SHA')
    require(guard['test_results_sha256'] == digest(PACKAGE / 'output/test-results.json') and guard['all_python_tests_log_sha256'] == digest(PACKAGE / 'output/all-python-tests.log'), 'guard log hashes')
    for path in (PACKAGE / 'output').rglob('*.json'):
        obj = read_json(path)
        if not isinstance(obj, dict) or 'command' not in obj:
            continue
        stream_hashes = [obj['stdout_sha256'], obj['stderr_sha256']]
        require(sha(b'') in stream_hashes, 'cannot independently split combined log')
        require(digest(path.with_suffix('.log')) in stream_hashes, 'observed command log SHA: ' + path.name)
    require(read_json(PACKAGE / 'output/all-python-tests.json')['exit_code'] == 0, 'producer test exit')
    require(read_json(PACKAGE / 'output/cli-root-relocation.json')['exit_code'] == 0, 'producer relocation exit')
    require(read_json(PACKAGE / 'output/failed-attempts/01-cli-import-cache-guard/cli-root-relocation.json')['exit_code'] == 1, 'retained initial failure')

    path = 'bench/check_original_program_scope.py'
    old, new = functions(blobs[path].decode()), functions((PACKAGE / path).read_text())
    gate = copy.deepcopy(new['validate_task']); adapted_calls = 0
    for node in ast.walk(gate):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'validate_used_numerics':
            require(len(node.keywords) == 1 and node.keywords[0].arg == 'root' and ast.dump(node.keywords[0].value) == "Name(id='root', ctx=Load())", 'unexpected structural gate change')
            node.keywords = []; adapted_calls += 1
    require(adapted_calls == 1 and ast.dump(gate) == ast.dump(old['validate_task']), 'structural gate AST')
    unchanged = sorted(set(old) - {'validate_task', 'validate_used_numerics'})
    require(all(ast.dump(old[name]) == ast.dump(new[name]) for name in unchanged), 'other legacy scope AST')
    path = 'bench/deferred_numerical_experiments.py'
    old_collector = functions(blobs[path].decode())['collect']
    new_collector = functions((PACKAGE / path).read_text())['collect_legacy']; new_collector.name = 'collect'
    require(ast.dump(old_collector) == ast.dump(new_collector), 'legacy collector AST')

    # Build only reviewer-owned directories from independently verified commit blobs.
    first = HERE / 'first-root'; second = HERE / 'second-root'
    require(not first.exists() and not second.exists(), 'reviewer roots already exist; preserve prior run')
    for path, data in blobs.items():
        target = first / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    for row in manifest['production_files']:
        target = first / row['destination']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE / row['source'], target)
    for path in checkpoint_files:
        target = first / path.relative_to(PACKAGE); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    preflight = run('integration-preflight', [sys.executable, '-B', str(PACKAGE / CHECKPOINT / 'integrate.py'), '--package', str(PACKAGE), '--root', str(REPO)])
    require(json.loads(preflight.stdout)['baseline_files_verified'] == 292, 'preflight baseline count')
    run('collector-replay', [sys.executable, '-B', str(first / 'bench/deferred_numerical_experiments.py'), '--root', str(first), '--check'])
    generated = HERE / 'generated'
    run('authority-replay', [sys.executable, '-B', str(first / 'bench/exact_numerical_bindings.py'), '--root', str(first), '--output', str(generated)])
    for name in ('authority.json', 'deferred_numerical_experiments.json'):
        require((generated / name).read_bytes() == (PACKAGE / 'output/generated' / name).read_bytes(), 'reproduced bytes differ: ' + name)
    authority = read_json(generated / 'authority.json')
    require(len(authority['rules']) == 18 and len(authority['provenance']) == 86, 'new rule/binding count')
    require(all(r['status'] == 'pending_numeric_validation' and r['validation_evidence'] is None and r['native_whole_kernel_validated'] is False for r in authority['rules']), 'new rules unvalidated')
    require(all(p['status'] == 'explicitly_deferred_unvalidated' and p['validation_evidence'] is None for p in authority['provenance'].values()), 'all ports unvalidated')
    require([len(t['cases']) for t in authority['requirements']] == [29, 20, 9], 'case counts')
    require(sum(len(b['declarations']) for r in authority['rules'] for b in r['scope_bindings']) == 133, 'declaration port bindings')
    require(len(authority['unmigrated_descriptive_followups']['entries']) == 35, 'descriptive entries')
    require(len(sources['exact_input_bindings']) == 60 and len(sources['legacy_input_bindings']) == 99, 'bound input counts')
    require({k: v['sha256'] for k, v in sources['exact_input_bindings'].items()} == authority['input_sha256'], 'exact input SHA table')
    supplements = 0
    for key, row in sources['exact_input_bindings'].items():
        require(digest(first / row['repository_path']) == row['sha256'], 'exact input path SHA')
        supplements += int(row['new_supplement'])
    require(supplements == 1, 'supplement count')
    require(read_json(generated / 'deferred_numerical_experiments.json')['input_sha256'] == sources['legacy_input_bindings'], 'legacy input SHA table')
    shutil.copytree(first, second)
    relocation = '''
import importlib.util,runpy,sys
from pathlib import Path
first,second,repo,base=map(lambda s:Path(s).resolve(),sys.argv[1:])
allowed={str(p.resolve()) for folder in ('bench','scripts') for p in (first/folder).glob('*.py')}
allowed.update(str(p.resolve()) for p in (first/'bench/current_catalog_vendor').rglob('*.py'))
allowed.update(importlib.util.cache_from_source(p) for p in tuple(allowed))
sys.path[:0]=[str(first/'bench'),str(first/'scripts')]
attempts=[]
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
 if str(p) not in allowed and not p.is_relative_to(second) and (p.is_relative_to(repo) or p.is_relative_to(base)):
  attempts.append(str(p));raise RuntimeError('forbidden old data-root read: '+str(p))
sys.addaudithook(audit)
sys.argv=[str(first/'bench/deferred_numerical_experiments.py'),'--root',str(second),'--check']
runpy.run_path(sys.argv[0],run_name='__main__')
assert not attempts,attempts
'''
    command = [sys.executable, '-I', '-B', '-c', relocation, str(first), str(second), str(REPO), str(BASE)]
    run('reviewer-cli-root-relocation', command)
    (second / 'bench/tritonbench_g/vector_addition/vector_addition.py').unlink()
    negative = run('reviewer-cli-root-missing-input', command, expected=1)
    require('missing or escaping input: repo:bench/tritonbench_g/vector_addition/vector_addition.py' in negative.stderr and 'forbidden old data-root read' not in negative.stderr, 'missing source fallback')
    report = {'schema': 1, 'status': 'passed', 'reviewer_checker_sha256': digest(Path(__file__)),
              'frozen_receipt_sha256': RECEIPT_SHA, 'frozen_manifest_sha256': MANIFEST_SHA,
              'tested_source_commit': COMMIT, 'artifacts_verified': 60, 'checkpoint_files_verified': 31,
              'checkpoint_output_copies_verified': duplicates, 'git_baseline_blobs_verified': 292,
              'production_preimages_verified': 7, 'unchanged_existing_test_files': 5,
              'producer_test_log_verified': {'total': 238, 'existing': 210, 'added': 28, 'per_module': by_module, 'result': 'all 238 names match source AST and passed'},
              'reviewer_test_suite_reruns': 0, 'test_recovery_guard_files_verified': len(guarded),
              'structural_validate_task_change': 'only root=root argument added to one validate_used_numerics call',
              'other_unchanged_scope_functions': unchanged, 'legacy_collector_ast_unchanged_except_rename': True,
              'reviewer_collector_and_authority_reproduction': 'byte-identical',
              'reviewer_cli_root_relocation': 'passed with old data roots forbidden',
              'reviewer_missing_relocated_original_source': 'rejected without fallback',
              'exact_summary': authority['summary'], 'new_rules_all_unvalidated': True,
              'native_executions': 0, 'original_program_completion_increment': 0,
              'repository_modified_by_reviewer': False, 'frozen_package_modified_by_reviewer': False}
    (HERE / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
