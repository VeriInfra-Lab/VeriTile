"""Read-only independent review of the tested production adapter package."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
PKG = TMP / 'numerical-exact-binding-repo-adapter-wip'
CP = PKG / 'bench/review-checkpoints/2026-10-16/exact-numerical-bindings'
OUT = TMP / 'exact-numerical-adapter-root-review.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def checked(p, h): assert sha(p) == h, str(p)
assert not OUT.exists()
checked(CP / 'receipt.json', '0f8ee46effc304a9b9e70a900a5c95b8c300a4cd7f2d34fdf80bbd944e6c0b76')
checked(CP / 'integration-manifest.json', '3720edfdef74296b4aa42b0ecf6004ea68e4739f875e6565ad692f2c9545b540')
receipt = read(CP / 'receipt.json')
for rel, h in receipt['artifacts'].items(): checked(PKG / rel, h)
source_hashes = read(CP / 'evidence/source-hashes.json')
for rel, h in source_hashes['baseline_repository_files'].items(): checked(ROOT / rel, h)
guard = read(CP / 'evidence/validation/completed-test-source-guard.json')
for rel, h in guard['adapter_files'].items(): checked(PKG / rel, h)
checked(CP / 'evidence/validation/all-python-tests.log', guard['all_python_tests_log_sha256'])
checked(CP / 'evidence/validation/test-results.json', guard['test_results_sha256'])
assert read(CP / 'evidence/validation/test-results.json') == {'tests': 238, 'failures': 0, 'errors': 0, 'success': True}
assert read(CP / 'evidence/validation/all-python-tests.json')['exit_code'] == 0
assert 'Ran 238 tests' in (CP / 'evidence/validation/all-python-tests.log').read_text()
for name, code in [('collector-generate', 0), ('collector-check', 0), ('exact-collector', 0),
                   ('scope-help', 0), ('cli-root-relocation', 0), ('cli-root-missing-input', 1)]:
    row = read(CP / ('evidence/validation/' + name + '.json'))
    assert row['exit_code'] == code, name
    log = CP / ('evidence/validation/' + name + '.log')
    assert sha(log) in {row['stdout_sha256'], row['stderr_sha256']}, name
assert 'missing or escaping input' in (CP / 'evidence/validation/cli-root-missing-input.log').read_text()
old_ast, new_ast = [ast.parse((base / 'bench/check_original_program_scope.py').read_text()) for base in (ROOT, PKG)]
old_funcs, new_funcs = [{n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)} for tree in (old_ast, new_ast)]
assert set(old_funcs) == set(new_funcs)
unchanged = []
for name, original in old_funcs.items():
    if name == 'validate_used_numerics': continue
    adapted = new_funcs[name]
    if name == 'validate_task':
        calls = [node for node in ast.walk(adapted) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'validate_used_numerics']
        assert len(calls) == 1 and len(calls[0].keywords) == 1
        kw = calls[0].keywords[0]
        assert kw.arg == 'root' and isinstance(kw.value, ast.Name) and kw.value.id == 'root'
        calls[0].keywords = []
    assert ast.dump(original) == ast.dump(adapted), name
    unchanged.append(name)
legacy_old = next(n for n in ast.parse((ROOT / 'bench/deferred_numerical_experiments.py').read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'collect')
legacy_new = next(n for n in ast.parse((PKG / 'bench/deferred_numerical_experiments.py').read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'collect_legacy')
legacy_new.name = 'collect'
assert ast.dump(legacy_old) == ast.dump(legacy_new)
manifest = read(CP / 'integration-manifest.json')
assert len(manifest['production_files']) == 7
for row in manifest['production_files']:
    assert row['source'] in receipt['artifacts']
    path = ROOT / row['destination']
    if row['before_sha256'] is None: assert not path.exists(), path
    else: checked(path, row['before_sha256'])
queue = read(PKG / 'generated-repository-files/bench/tritonbench_g/deferred_numerical_experiments.json')
legacy = read(ROOT / 'bench/tritonbench_g/deferred_numerical_experiments.json')
assert queue['rules'][:11] == legacy['rules'] and len(queue['rules']) == 29
assert queue['input_sha256'] == legacy['input_sha256']
assert len(queue['exact_binding_provenance']) == 86 and len(queue['exact_callback_requirements']) == 3
assert [len(r['cases']) for r in queue['exact_callback_requirements']] == [29,20,9]
assert all(r['status'] == 'pending_numeric_validation' and r['validation_evidence'] is None for r in queue['rules'][11:])
assert all('SourceGridCorrect' not in r['declaration']['file'] and 'SourceBlockCorrect' not in r['declaration']['file']
           for r in queue['exact_callback_requirements'])
review = {
    'schema': 1, 'kind': 'exact_numerical_adapter_independent_production_review',
    'reviewer': 'root', 'status': 'reviewed_exact_case_port_registration_without_scope_acceptance',
    'receipt_sha256': sha(CP / 'receipt.json'), 'manifest_sha256': sha(CP / 'integration-manifest.json'),
    'checker_path': str(Path(__file__).resolve()), 'checker_sha256': sha(__file__),
    'artifact_hashes_checked': len(receipt['artifacts']),
    'baseline_inputs_checked': len(source_hashes['baseline_repository_files']),
    'tested_adapter_hashes': guard['adapter_files'], 'retained_test_result': receipt['test_counts'],
    'independent_test_rerun': False,
    'scope_functions_unchanged_except_explicit_root': unchanged,
    'semantic_review': [
        'Root read the full prototype collector/gate/tests and its independently reproduced evidence, all production tool diffs, the complete production collector, gate and adapted 28-test suite, relocation validator and manifest copy helper.',
        'Data paths resolve from the explicit requested root. Live Lean declarations and callback sources, original Python and frozen source/configuration evidence are hash-bound. The one missing historical receipt is added under a new checkpoint; previous frozen evidence is preserved.',
        'The legacy collector body is AST-identical apart from its name. Scope structural functions are AST-identical except the explicit root keyword passed into numeric validation. The numerical consumer validates source-derived whole-queue authority, per-case callbacks and each required individual scalar-port reference.',
        'Unsupported dtype/configuration cases, missing side tables and missing callback ports fail closed. Signed ReLU branches separate literal positive zero from signed conversion; unsigned and bool zero use the selected fromNat route.',
        'Retained 238-test success is bound to unchanged source bytes. The actual collector CLI was loaded from one checkout and read data only through a second requested root; forbidden old-root reads were trapped. Missing relocated original Python failed without fallback. The initial import-cache harness failure is retained.',
        'Only the two earlier fp64 grid declarations and previous ReLU access-grid declaration are registered. ReLU binding is restricted to N1025/B1024/sm90. New source-block/grid contracts are not registered automatically. Every new rule remains unvalidated; no structural gap or whole-program completion is discharged.'
    ],
    'proof_or_gate_defects': [], 'original_program_completion_increment': 0,
    'structural_acceptance': False, 'native_executions': 0,
    'integration_boundary': 'Actual post-copy collector checks and current global audit still required.'
}
OUT.write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps({'review': str(OUT), 'sha256': sha(OUT), 'artifact_hashes_checked': review['artifact_hashes_checked'],
                  'baseline_inputs_checked': review['baseline_inputs_checked']}, indent=2))
