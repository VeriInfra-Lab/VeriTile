"""Check the actual two Python sources; no numerical or GPU experiment."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def check():
    pins = json.loads((HERE / 'source_pin.json').read_text())
    source = HERE / 'l2_norm_bwd_repaired.py'
    candidate = HERE / 'l2_norm_bwd_index_commuted.py'
    pair = pins['implementation_pair']
    for path, expected in ((source, pins['candidate_sha256']),
                           (HERE / 'Kernels.lean', pins['quotation_sha256']),
                           (candidate, pair['candidate_sha256'])):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    original = function(source, '_l2_norm_bwd_kernel_repaired')
    actual = function(candidate, '_l2_norm_bwd_kernel_index_commuted')
    expected = copy.deepcopy(original)
    expected.name = actual.name
    changes = []
    for stmt in expected.body:
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            pointer = stmt.target.id
            if pointer in ('X', 'DX', 'DY'):
                stride = {'X': 'stride_x_row', 'DX': 'stride_dx_row', 'DY': 'stride_dy_row'}[pointer]
                assert isinstance(stmt.op, ast.Add)
                assert ast.dump(stmt.value) == ast.dump(ast.parse(f'row * {stride}', mode='eval').body)
                stmt.value.left, stmt.value.right = stmt.value.right, stmt.value.left
                changes.append(pointer)
    assert changes == ['X', 'DX', 'DY']
    assert ast.dump(expected) == ast.dump(actual), 'candidate differs beyond the three row products'
    assert hashlib.sha256(ast.dump(original).encode()).hexdigest() == pins['kernel_ast_sha256']
    assert hashlib.sha256(ast.dump(actual).encode()).hexdigest() == pair['candidate_kernel_ast_sha256']
    wrapper = function(source, '_l2_norm_bwd')
    names = [n.id for n in ast.walk(wrapper) if isinstance(n, ast.Name)]
    assert '_l2_norm_bwd_kernel_repaired' in names
    assert '_l2_norm_bwd_kernel_index_commuted' not in names
    print('Pair AST check passed: identical annotations, JIT options and full body except X/DX/DY row products.')
    print('Production wrapper still dispatches the repaired source; no numerical or GPU experiment was run.')


if __name__ == '__main__':
    check()
