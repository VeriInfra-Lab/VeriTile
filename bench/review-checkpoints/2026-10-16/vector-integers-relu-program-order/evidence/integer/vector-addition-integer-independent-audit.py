"""Read-only independent evidence and integer TTIR dataflow audit."""
from pathlib import Path
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re

repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
base = Path(__file__).resolve().parent
pkg = base / 'vector-addition-dtype-extension-wip'
public = base / 'vector-addition-integer-public-wip'
prior = base / 'vector-addition-wip'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
receipt = read(pkg / 'evidence/receipt.json')
scope = read(pkg / 'evidence/scope-review.json')
assert sha(pkg / 'evidence/receipt.json') == '85dc08106852c4a3f69516046182954693eb5afd0158064db358c708a70b9100'
assert sha(pkg / 'evidence/scope-review.json') == '99f8b349f2915998ef13338026a7e4f855cb2444089a57a931a0680d6de22934'
for rel, expected in receipt['artifacts'].items():
    assert sha(pkg / rel) == expected, rel
before = read(pkg / 'evidence/final-dependencies-before.json')
after = read(pkg / 'evidence/final-dependencies-after.json')
assert before == after
roots = {'repository': repo, 'package': pkg, 'frozen_vector_package': prior}
for rel, info in before['files'].items():
    assert sha(roots[info['source_kind']] / rel) == info['sha256'], rel
verification = read(pkg / 'evidence/verification-results.json')
logs = []
for row in verification:
    assert row['exit_code'] == 0
    actual = sha(pkg / row['module'])
    assert actual == row['source_sha256_before'] == row['source_sha256_after'] == receipt['modules'][row['module']]
    assert sha(pkg / row['log']) == row['log_sha256']
    logs.append((pkg / row['log']).read_text())
assert sum(s.count('axiom footprint') for s in logs) == 19
assert sum(s.count('non-circular ✓') for s in logs) == 3
assert all('warning:' not in s and 'error:' not in s for s in logs)
wrapper = read(public / 'compile-receipt.json')
assert wrapper['exit_code'] == 0
assert sha(public / wrapper['module']) == wrapper['source_sha256_before'] == wrapper['source_sha256_after']
assert wrapper['dependency_source_sha256'] == receipt['modules']
assert wrapper['dependency_receipt_sha256'] == sha(pkg / 'evidence/receipt.json')
assert sha(Path(wrapper['log'])) == wrapper['log_sha256']
wlog = Path(wrapper['log']).read_text()
assert wlog.count('axiom footprint') == 1 and 'warning:' not in wlog and 'error:' not in wlog

audit = pkg / 'evidence/extension-audit'
audit_receipt = read(audit / 'receipt.json')
for rel, expected in audit_receipt['artifacts'].items():
    assert sha(audit / rel) == expected
for rel, expected in audit_receipt['bindings'].items():
    assert sha(repo / rel) == expected
source = repo / 'bench/tritonbench_g/vector_addition/vector_addition.py'
assert sha(source) == audit_receipt['source_sha256']
assert sha(audit / 'extracted_add_kernel.py') == audit_receipt['extracted_kernel_sha256']
def kernel_ast(path):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'add_kernel')
    return ast.dump(node, include_attributes=False)
assert kernel_ast(source) == kernel_ast(audit / 'extracted_add_kernel.py')
front = read(audit / 'frontend-probes.json')
promotion = read(audit / 'promotion-matrix.json')
assert len(front) == len(promotion) == 289
front_by = {(r['x'], r['y']): r for r in front}
prom_by = {(r['x'], r['y']): r for r in promotion}
scope_by = {(r['x'], r['y']): r for r in scope['dtype_pairs']}
assert len(front_by) == len(prom_by) == len(scope_by) == 289
assert set(front_by) == set(prom_by) == set(scope_by)
assert Counter(r['status'] for r in front) == {'frontend_generated': 189, 'frontend_error': 100}
assert Counter(r['status'] for r in promotion) == {'promoted': 217, 'promotion_error': 72}
assert sorted(Counter(r['coverage'] for r in scope_by.values()).values()) == [11, 25, 28, 72, 72, 81]
for pair, row in front_by.items():
    if row['status'] == 'frontend_generated':
        assert sha(audit / row['ttir_path']) == row['ttir_sha256']
    assert scope_by[pair]['frontend_status'] == row['status']
    assert scope_by[pair]['status'] == prom_by[pair]['status']

integers = ['bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
width = lambda d: 1 if d == 'bool' else int(re.search(r'\d+', d).group())
signed = lambda d: d.startswith('int')
def promote(x, y):
    a, b = width(x), width(y)
    if signed(x) == signed(y): return (a, signed(x)) if a >= b else (b, signed(y))
    if signed(x): return (a, True) if a > b else (b, False)
    return (b, True) if b > a else (a, False)

# Independently follow each store operand's SSA graph. Ignore sanitizer
# computations only when they do not feed this graph; their native effects
# are intentionally still an open obligation.
graphs = []
for x in integers:
    for y in integers:
        row = front_by[x, y]
        assert row['status'] == 'frontend_generated'
        ir = (audit / row['ttir_path']).read_text()
        definitions = dict(re.findall(r'^\s*(%\w+) = (.*)$', ir, re.M))
        store = re.search(r'tt.store (%\w+), (%\w+), (%\w+) : tensor<1024x!tt.ptr<i(\d+)>>', ir)
        assert store
        ptr, value, mask, sw = store.groups()
        assert int(sw) == (8 if x == 'bool' else width(x))
        offsets_seen = set()
        masks_seen = {mask}
        def pointer(v):
            if v in ('%x_ptr', '%y_ptr', '%output_ptr'): return v
            rhs = definitions[v]
            op = rhs.split()[0]
            operands = re.findall(r'%\w+', rhs)
            assert op in ('tt.splat', 'tt.addptr', 'tt.bitcast'), rhs
            if op == 'tt.addptr': offsets_seen.add(operands[1])
            return pointer(operands[0])
        assert pointer(ptr) == '%output_ptr'
        def term(v):
            rhs = definitions[v]
            op = rhs.split()[0]
            operands = re.findall(r'%\w+', rhs)
            if op == 'tt.load':
                side = pointer(operands[0])
                assert side in ('%x_ptr', '%y_ptr')
                masks_seen.add(operands[1])
                w = int(re.search(r'tt.ptr<i(\d+)>', rhs).group(1))
                return ('load', side[1], w)
            if op == 'arith.constant':
                m = re.search(r'dense<(-?\d+)> : tensor<1024xi(\d+)>', rhs)
                assert m, rhs
                return ('const', int(m.group(2)), int(m.group(1)))
            if op in ('arith.extsi', 'arith.extui', 'arith.trunci'):
                widths = re.findall(r'tensor<1024xi(\d+)>', rhs)
                assert len(widths) == 2
                return (op.split('.')[1], int(widths[0]), int(widths[1]), term(operands[0]))
            if op == 'arith.bitcast':
                widths = re.findall(r'tensor<1024xi(\d+)>', rhs)
                assert len(widths) == 2 and widths[0] == widths[1], rhs
                return term(operands[0])
            if op == 'arith.addi':
                w = int(re.search(r'tensor<1024xi(\d+)>', rhs).group(1))
                return ('add', w, term(operands[0]), term(operands[1]))
            if op == 'arith.cmpi':
                assert rhs.startswith('arith.cmpi ne,'), rhs
                return ('ne', term(operands[0]), term(operands[1]))
            raise AssertionError(rhs)
        cw, cs = promote(x, y)
        observed = prom_by[x, y]['add_dtype']
        expected_promo = ('int' if cs or cw == 1 else 'uint') + str(cw)
        assert observed == expected_promo, (x, y, observed, expected_promo)
        def operand(side, d):
            w = width(d)
            node = ('load', side, 8 if d == 'bool' else w)
            if d == 'bool': node = ('ne', node, ('const', 8, 0))
            if w < cw: node = ('extsi' if signed(d) else 'extui', w, cw, node)
            return node
        expected = ('add', cw, operand('x', x), operand('y', y))
        dest = int(sw)
        if cw > dest: expected = ('trunci', cw, dest, expected)
        elif cw < dest: expected = ('extui', cw, dest, expected)
        actual = term(value)
        assert actual == expected, (x, y, actual, expected)
        assert len(offsets_seen) == len(masks_seen) == 1
        graphs.append({'x': x, 'y': y, 'promotion_width': cw, 'promotion_signed': cs,
                       'store_width': dest, 'store_dataflow': actual,
                       'same_logical_offset_and_mask': True})

result = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'read_only_audit': True,
    'package_receipt_sha256': sha(pkg / 'evidence/receipt.json'),
    'scope_sha256': sha(pkg / 'evidence/scope-review.json'),
    'artifact_hashes_verified': len(receipt['artifacts']),
    'dependency_hashes_verified': len(before['files']),
    'dependency_sources': dict(Counter(v['source_kind'] for v in before['files'].values())),
    'before_after_dependencies_equal': True,
    'compile_exit_codes': [r['exit_code'] for r in verification] + [wrapper['exit_code']],
    'producer_axiom_checks': 19, 'wrapper_axiom_checks': 1,
    'producer_noncircularity_checks': 3, 'warnings': 0,
    'wrapper_source_sha256': sha(public / wrapper['module']),
    'wrapper_receipt_sha256': sha(public / 'compile-receipt.json'),
    'wrapper_log_sha256': wrapper['log_sha256'],
    'ast_identical_frontend_source': True,
    'frontend_rows': len(front), 'ttir_hashes_verified': 189, 'frontend_errors': 100,
    'independently_traced_integer_store_graphs': graphs,
    'current_lakefile_sha256': sha(repo / 'lakefile.toml'),
    'lakefile_limit': 'The producer 158-file snapshot omits lakefile.toml. This review binds its current hash; root integration full build remains the environment verification.',
    'native_backend_recompiled': False,
    'gpu_executed': False,
    'script_sha256': sha(Path(__file__)),
}
out = base / 'vector-addition-integer-independent-audit.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'independently_traced_integer_store_graphs'}, indent=2))
print(json.dumps({'store_graph_count': len(graphs), 'output': str(out), 'sha256': sha(out)}))
