"""Test the proposed repository files in a copied checkout, never the live tree."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE_REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
PROTOTYPE = HERE.parent / 'numerical-exact-binding-wip'
CHECKPOINT = 'bench/review-checkpoints/2026-10-16/'
PACKAGES = {
    'repo': '',
    'fp64_core': CHECKPOINT + 'fp64-admission-relu-access/evidence/fp64_core',
    'fp64_grid': CHECKPOINT + 'fp64-admission-relu-access/evidence/fp64_grid',
    'relu_access': CHECKPOINT + 'fp64-admission-relu-access/evidence/relu_access',
    'relu_original': CHECKPOINT + 'vector-integers-relu-program-order/evidence/relu',
    'relu_words': CHECKPOINT + 'vector-mixed-fp32-relu-words/evidence/words',
}
SUPPLEMENT = (CHECKPOINT +
              'exact-numerical-bindings/evidence/fp64-source-routes/receipt.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(root):
    authority = json.loads((PROTOTYPE / 'output/authority.json').read_text())
    legacy = json.loads((PROTOTYPE / 'legacy/deferred_numerical_experiments.json').read_text())
    paths = set(legacy['input_sha256'])
    for key in authority['input_sha256']:
        package, path = key.split(':', 1)
        if package == 'legacy':
            continue
        if package == 'reviews':
            package = 'fp64_grid' if path.startswith('vector-addition-fp64') else 'relu_access'
            path = 'independent-review.json'
        if path.startswith(('bench/', 'VeriTile/')):
            package = 'repo'
        if (package, path) == ('fp64_core', 'evidence/source-routes/receipt.json'):
            continue
        paths.add(str(Path(PACKAGES[package]) / path))
    paths.add(CHECKPOINT + 'fp64-admission-relu-access/numerical-followup.json')
    paths.update(str(p.relative_to(SOURCE_REPO)) for parent in ('bench', 'scripts')
                 for p in (SOURCE_REPO / parent).glob('*.py'))
    paths.update(str(p.relative_to(SOURCE_REPO)) for p in (SOURCE_REPO / 'bench/tests').glob('test_*.py'))
    paths.update(str(p.relative_to(SOURCE_REPO)) for p in (SOURCE_REPO / 'bench/current_catalog_vendor').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    baseline = SOURCE_REPO / 'bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/metadata-baseline'
    paths.update(str(p.relative_to(SOURCE_REPO)) for p in baseline.rglob('*') if p.is_file())
    for relative in paths:
        copy(SOURCE_REPO / relative, root / relative)
    for p in (HERE / 'bench').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            copy(p, root / p.relative_to(HERE))
    return len(paths)


def run_logged(label, command, output, *, cwd=None, env=None, success=True):
    run = subprocess.run(command, capture_output=True, text=True, cwd=cwd, env=env)
    (output / (label + '.log')).write_text(run.stdout + run.stderr)
    receipt = {'command': command, 'exit_code': run.returncode,
               'stdout_sha256': hashlib.sha256(run.stdout.encode()).hexdigest(),
               'stderr_sha256': hashlib.sha256(run.stderr.encode()).hexdigest()}
    (output / (label + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    if success and run.returncode:
        raise RuntimeError(label + ':\n' + run.stdout + run.stderr)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-after-tests', action='store_true')
    args = parser.parse_args()
    output = HERE / 'output'
    output.mkdir(exist_ok=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE_REPO, text=True).strip()
    if head != '74cb67ae99cd5cf77096dc74868d8adcb243b46a':
        raise ValueError('Unexpected source checkout: ' + head)
    with tempfile.TemporaryDirectory(prefix='veritile-numeric-adapter-') as tmp:
        root = Path(tmp) / 'first'
        root.mkdir()
        copied = prepare(root)
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTHONPATH': os.pathsep.join(map(str, (root / 'bench', root / 'scripts')))}
        run_logged('collector-generate', [sys.executable, '-B', str(root / 'bench/deferred_numerical_experiments.py'),
                                         '--root', str(root)], output, env=env)
        run_logged('collector-check', [sys.executable, '-B', str(root / 'bench/deferred_numerical_experiments.py'),
                                      '--root', str(root), '--check'], output, env=env)
        run_logged('exact-collector', [sys.executable, '-B', str(root / 'bench/exact_numerical_bindings.py'),
                                      '--root', str(root), '--output', str(output / 'generated')], output, env=env)
        run_logged('scope-help', [sys.executable, '-B', str(root / 'bench/check_original_program_scope.py'),
                                 '--help'], output, env=env)
        test_bootstrap = '''
import json,sys,unittest
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'bench'),str(root/'bench/tests'),str(root/'scripts')]
suite=unittest.defaultTestLoader.discover(str(root/'bench/tests'),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
Path(sys.argv[2]).write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),
 'errors':len(result.errors),'success':result.wasSuccessful()},indent=2)+'\\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
'''
        if args.resume_after_tests:
            guard = json.loads((output / 'completed-test-source-guard.json').read_text())
            if (guard['source_head'] != head or
                    any(digest(HERE / p) != h for p, h in guard['adapter_files'].items()) or
                    guard['test_results_sha256'] != digest(output / 'test-results.json') or
                    guard['all_python_tests_log_sha256'] != digest(output / 'all-python-tests.log')):
                raise ValueError('Completed-test source/evidence changed; cannot reuse')
        else:
            run_logged('all-python-tests', [sys.executable, '-I', '-B', '-c', test_bootstrap,
                                           str(root), str(output / 'test-results.json')], output, env=env)
        second = Path(tmp) / 'second'
        shutil.copytree(root, second)
        code_allow = [str(p.resolve()) for folder in ('bench', 'scripts') for p in (root / folder).glob('*.py')]
        code_allow += [str(p.resolve()) for p in (root / 'bench/current_catalog_vendor').rglob('*.py')]
        policy = Path(tmp) / 'policy.json'
        policy.write_text(json.dumps({'forbidden': [str(SOURCE_REPO), str(PROTOTYPE.parent), str(root)],
                                      'code_allow': code_allow}))
        relocation = '''
import importlib.util,json,runpy,sys
from pathlib import Path
first,second,policy=map(Path,sys.argv[1:4]);cfg=json.loads(policy.read_text())
sys.path[:0]=[str(first/'bench'),str(first/'scripts')]
allowed=set(cfg['code_allow'])
allowed.update(importlib.util.cache_from_source(p) for p in cfg['code_allow'])
forbidden=[Path(p) for p in cfg['forbidden']]
attempts=[]
def audit(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).absolute()
 if str(p) not in allowed and any(p.is_relative_to(r) for r in forbidden):
  attempts.append(str(p));raise RuntimeError('forbidden old data-root read: '+str(p))
sys.addaudithook(audit)
sys.argv=[str(first/'bench/deferred_numerical_experiments.py'),'--root',str(second),'--check']
runpy.run_path(sys.argv[0],run_name='__main__')
assert not attempts,attempts
'''
        run_logged('cli-root-relocation', [sys.executable, '-I', '-B', '-c', relocation,
                                          str(root), str(second), str(policy)], output)
        # Existing files in the first checkout may not fill missing second-root data.
        missing = second / 'bench/tritonbench_g/vector_addition/vector_addition.py'
        missing.unlink()
        negative = run_logged('cli-root-missing-input', [sys.executable, '-I', '-B', '-c', relocation,
                                                       str(root), str(second), str(policy)], output, success=False)
        if not negative.returncode or 'missing or escaping input: repo:bench/tritonbench_g/vector_addition/vector_addition.py' not in negative.stderr:
            raise ValueError('Missing relocated original source was not rejected correctly')
        for relative in ('bench/tritonbench_g/deferred_numerical_experiments.json',
                         'bench/tritonbench_g/deferred_numerical_experiments.md'):
            copy(root / relative, HERE / 'generated-repository-files' / relative)
    old = ast.parse((PROTOTYPE / 'legacy/check_original_program_scope.py').read_text())
    new = ast.parse((HERE / 'bench/check_original_program_scope.py').read_text())
    functions = [{n.name: n for n in t.body if isinstance(n, ast.FunctionDef)} for t in (old, new)]
    original_gate = functions[0]['validate_task']
    adapted_gate = functions[1]['validate_task']
    for call in ast.walk(adapted_gate):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'validate_used_numerics':
            if len(call.keywords) != 1 or call.keywords[0].arg != 'root' or not isinstance(call.keywords[0].value, ast.Name) or call.keywords[0].value.id != 'root':
                raise ValueError('Unexpected numeric root plumbing')
            call.keywords = []
    if ast.dump(original_gate) != ast.dump(adapted_gate):
        raise ValueError('Structural scope gate changed beyond passing the root')
    unchanged = [name for name in functions[0] if name not in {'validate_used_numerics', 'validate_task'}]
    if any(ast.dump(functions[0][name]) != ast.dump(functions[1][name]) for name in unchanged):
        raise ValueError('Other legacy scope functions changed')
    report = {'schema': 1, 'status': 'passed', 'source_head': head,
              'copied_repository_inputs_and_helpers': copied,
              'test_results': json.loads((output / 'test-results.json').read_text()),
              'actual_cli_root_relocation': 'passed_with_old_data_roots_forbidden',
              'missing_relocated_original_source': 'rejected_without_fallback',
              'structural_gate_ast_unchanged_except_explicit_root_keyword': True,
              'other_unchanged_legacy_scope_functions': unchanged,
              'supplemental_receipt_sha256': digest(HERE / SUPPLEMENT),
              'new_native_executions': 0, 'original_program_completion_increment': 0}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
