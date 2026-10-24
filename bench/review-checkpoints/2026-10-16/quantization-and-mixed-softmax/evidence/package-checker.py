"""Root source/statement/evidence review, without changing either frozen package."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
Q = TMP / 'quantize-integer-stage-wip'
S = TMP / 'token-softmax-mixed-output-wip'
OUT = TMP / 'quant-soft-packages-root-review.json'
assert not OUT.exists()
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import declarations, split_statement

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
checked = {}


def check(path, expected):
    path = Path(path)
    assert sha(path) == expected, str(path)
    checked[str(path)] = expected


check(Q / 'producer-receipt.json', '6de5e218b452196f28294bd90002cbd643ecb0eba677347be30ce8ba3402336b')
check(S / 'evidence/receipt.json', '5acdfee80d8f266c1e6c449f1ebc3863d6942d88b8c6f280625e09663ca2c17a')
qr, sr = read(Q / 'producer-receipt.json'), read(S / 'evidence/receipt.json')
for field in ('scope', 'dependencies', 'manifest', 'latest_lean_build', 'offline_original_admission'):
    check(qr[field]['path'], qr[field]['sha256'])
for item in qr['retained_build_attempts']:
    check(item['path'], item['sha256'])
qm = read(Q / 'artifact-manifest.json')
for rel, h in qm['files'].items():
    check(Q / rel, h)
for rel, h in qm['external_tooling_and_native_source_sha256'].items():
    check(rel, h)
qd = read(Q / 'dependency-bindings.json')
for item in qd['modules'].values():
    check(item['source'], item['source_sha256'])
    if item.get('olean'):
        check(item['olean'], item['olean_sha256'])
for rel, h in read(Q / 'scope.json')['original_source_sha256'].items():
    check(ROOT / rel, h)
qb = read(qr['latest_lean_build']['path'])
assert qb['status'] == 'passed' and len(qb['records']) == 4
q_axioms = 0
for item in qb['records']:
    assert item['exit_code'] == item['dependencies_exit_code'] == 0
    assert item['dependency_hashes_stable']
    rel = item['module'].replace('.', '/') + '.lean'
    check(Q / 'src' / rel, item['source_sha256'])
    assert qb['source_sha256'][rel] == item['source_sha256']
    check(item['log'], item['log_sha256'])
    log = Path(item['log']).read_text()
    assert 'error:' not in log and 'warning:' not in log
    q_axioms += log.count('axiom footprint ⊆ standard base ✓')
    for path, h in item['dependency_olean_sha256'].items():
        check(path, h)
assert q_axioms == 17
for public in qr['selected_public_declarations']:
    path = Q / 'src' / public['file']
    check(path, public['source_sha256'])
    parsed = declarations(path)[public['declaration']]
    assert parsed['kind'] == 'specification'
    assert parsed['statement_sha256'] == public['statement_sha256']
    assert split_statement(parsed['text'])[0] == public['statement']
qn = read(qr['offline_original_admission']['path'])
assert qn['status'] == 'passed' and len(qn['records']) == 2
quant_graphs = []
for item in qn['records']:
    check(item['original_source'], item['original_sha256'])
    check(item['copied_prefix'], item['copied_prefix_sha256'])
    for field in ('ttir', 'ttgir', 'llir', 'ptx'):
        check(item[field]['path'], item[field]['sha256'])
    name = '_quantize_global' if item['task'] == 'quantize_global' else '_quantize_global_transpose'
    node = lambda path: next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ast_text = ast.dump(node(item['original_source']))
    assert ast_text == ast.dump(node(item['copied_prefix']))
    assert hashlib.sha256(ast_text.encode()).hexdigest() == item['original_kernel_ast_sha256']
    ttir = Path(item['ttir']['path']).read_text()
    var = r'%[A-Za-z0-9_]+'
    defs = {}
    for line in ttir.splitlines():
        m = re.match(r'\s*(' + var + r') = ([A-Za-z0-9_.]+) (.*)', line)
        if m:
            defs[m[1]] = (m[2], re.findall(var, m[3]), m[3])
    rounds = [v for v, d in defs.items() if d[0] == 'tt.extern_elementwise' and '__nv_llrintf' in d[2]]
    assert len(rounds) == 1
    rounded = rounds[0]
    assert 'xf32>) -> tensor<' in defs[rounded][2] and 'xi64>' in defs[rounded][2]
    mul2 = defs[rounded][1][0]
    assert defs[mul2][0] == 'arith.mulf'
    prior = [v for v in defs[mul2][1] if defs[v][0] == 'arith.mulf']
    assert len(prior) == 1 and len([v for v, d in defs.items() if d[0] == 'arith.mulf']) == 2
    literals = [v for v in defs[mul2][1] if v != prior[0]]
    assert len(literals) == 1 and defs[literals[0]][0] == 'arith.constant'
    assert '1.270000e+02' in defs[literals[0]][2]
    trunc = [v for v, d in defs.items() if d[0] == 'arith.trunci' and d[1] == [rounded]]
    assert len(trunc) == 1 and 'xi64> to tensor<' in defs[trunc[0]][2] and 'xi8>' in defs[trunc[0]][2]
    stores = [line for line in ttir.splitlines() if line.strip().startswith('tt.store ')]
    assert len(stores) == 1 and re.findall(var, stores[0])[1] == trunc[0]
    quant_graphs.append({'task': item['task'], 'llrint_result': rounded, 'outer_multiply': mul2,
                         'inner_multiply': prior[0], 'stored_int8': trunc[0],
                         'native_equivalence': False, 'outer_operand_order_may_be_commuted': True})

for rel, h in sr['artifacts'].items():
    check(S / rel, h)
assert len(sr['modules']) == 7 and len(sr['public_declarations']) == 2
for public in sr['public_declarations']:
    parsed = declarations(S / public['file'])[public['declaration']]
    assert parsed['kind'] == 'specification' and parsed['statement_sha256'] == public['statement_sha256']
    assert len(public['selected_storage_pairs']) == 9
before = S / 'evidence/final-dependencies-before.json'
after = S / 'evidence/final-dependencies-after.json'
assert before.read_bytes() == after.read_bytes()
sd = read(before)
for rel, item in sd['files'].items():
    check(item['source_path'], item['source_sha256'])
    if item.get('olean_path'):
        check(item['olean_path'], item['olean_sha256'])
for rel, h in sd['preexisting_task_files_unchanged'].items():
    check(ROOT / rel, h)
for path, h in sd['toolchain'].items():
    check(path, h)
sv = read(S / 'evidence/verification-results.json')
assert len(sv) == 7
soft_axioms = 0
for item in sv:
    assert item['exit_code'] == 0
    assert item['source_sha256_before'] == item['source_sha256_after'] == sr['modules'][item['module']]
    check(S / item['module'], item['source_sha256_after'])
    check(S / item['source_snapshot'], item['snapshot_sha256'])
    check(S / item['log'], item['log_sha256'])
    log = (S / item['log']).read_text()
    assert 'error:' not in log and 'warning:' not in log
    soft_axioms += log.count('axiom footprint ⊆ standard base ✓')
assert soft_axioms == 9
sf = read(S / 'evidence/frontend/receipt.json')
assert len(sf['profiles']) == sf['frontend_successes'] == 18
for task, item in sf['sources'].items():
    check(ROOT / item['path'], item['sha256'])
    check(S / item['extracted_path'], item['extracted_sha256'])
    node = lambda path: next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel_token_softmax')
    ast_text = ast.dump(node(ROOT / item['path']), include_attributes=False)
    assert ast_text == ast.dump(node(S / item['extracted_path']), include_attributes=False)
    assert hashlib.sha256(ast_text.encode()).hexdigest() == item['ast_sha256']
# Replay the inspected graph checker while removing only its output/print statements.
script = S / 'audit_graph.py'
tree = ast.parse(script.read_text())
tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and (isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'write_text'
                   or isinstance(n.value.func, ast.Name) and n.value.func.id == 'print'))]
namespace = {'__file__': str(script)}
exec(compile(tree, str(script), 'exec'), namespace)
assert namespace['result'] == read(S / 'evidence/scalar-graph-audit.json')

report = {'schema': 1, 'kind': 'root_quantization_and_mixed_softmax_package_review',
          'reviewer': 'root', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'reviewed_partial_execution_extensions', 'proof_defects': [],
          'checked_file_sha256': dict(sorted(checked.items())),
          'quantization': {'source_modules_read': 4, 'fresh_compiles': 4, 'axiom_checks': q_axioms,
                           'dependency_modules': len(qd['modules']), 'connected_ttir_graphs': quant_graphs,
                           'semantic_review': 'The additive evaluator executes the exact llrint node using a total FP32-to-BitVec64 callback and the source-bound output register marker. Its signature map performs modulo-256 narrowing at actual named-region stores. Both memory formulas use initial-state data and ordered software FP32 multiplications, and execute before readback/frame proofs. Inactive values keep the original undefined-load behavior. Byte readback is an eight-bit value projection of tagged logical cells; it is not checked physical byte-memory execution. Global is the complete selected-signature kernel; transpose remains a supplied-coordinate tile stage, with original grouped mapping still missing.'},
          'mixed_softmax': {'source_modules_read': 7, 'fresh_compiles': 7, 'axiom_checks': soft_axioms,
                            'dependency_sources': len(sd['files']), 'connected_frontend_routes': 18,
                            'semantic_review': 'Nine independently selected fp16/bf16/fp32 input/output pairs retain the entire metadata load, cast/padded-row, maximum, subtraction, exp, sum, division and destination-cast graph. Execution is successful with a positive width, and a separate initial-state formula gives memory equality, active values and frame. The homogeneous route is definitionally the old source. Erased-surface equality binds both original translated kernels; exact Python AST extraction and connected frontend SSA are evidence, not native execution. Natural metadata and a nonzero output token stride remain the selected domain.'},
          'preservation': 'All old source/proof files and numerical queue entries remain unchanged at review time.',
          'remaining_scope': ['Host-stage and complete-grid composition for all four tasks',
                              'Unrepresented dtype/admission routes',
                              'Original grouped transpose pid mapping',
                              'Exact numerical registration for the four new public statements'],
          'whole_program_completion_increment': 0, 'numerical_validation_increment': 0,
          'new_builds_by_root': 0}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(OUT), 'sha256': sha(OUT), 'checked_file_count': len(checked),
                  'fresh_producer_compiles': 11, 'public_declarations': 4, 'acceptance_increment': 0}))
