"""Translate the original five Python statements into the checked Lean syntax."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re

W = Path(__file__).resolve().parent
SOURCE = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/cosine_compute/cosine_compute.py')
LEAN = W / 'after/bench/tritonbench_g/cosine_compute/Proofs/NativeSource.lean'
tree = ast.parse(SOURCE.read_text())
kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cos_func')
assert [a.arg for a in kernel.args.args] == ['a', 'b', 'n_elements', 'BLOCK_SIZE']
assert [ast.unparse(a.annotation) for a in kernel.args.args] == ['tl.tensor', 'tl.tensor', 'tl.int32', 'tl.constexpr']
assert [ast.unparse(d) for d in kernel.decorator_list] == ['triton.jit']

def pointer(node):
    assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
    assert isinstance(node.left, ast.Name) and node.left.id in ['a', 'b']
    return node.left.id, expression(node.right)

def mask(node):
    assert len(node.keywords) == 1 and node.keywords[0].arg == 'mask'
    return expression(node.keywords[0].value)

def expression(node):
    if isinstance(node, ast.Name):
        if node.id == 'n_elements': return '(.literal n)'
        if node.id == 'BLOCK_SIZE': return '(.literal block)'
        assert node.id in ['offset', 'mask', 'a_value', 'b_value']
        return f'(.register "{node.id}")'
    if isinstance(node, ast.BinOp):
        operation = 'plus' if isinstance(node.op, ast.Add) else 'times' if isinstance(node.op, ast.Mult) else None
        assert operation is not None
        return f'(.{operation} {expression(node.left)} {expression(node.right)})'
    if isinstance(node, ast.Compare):
        assert len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt) and len(node.comparators) == 1
        return f'(.less {expression(node.left)} {expression(node.comparators[0])})'
    assert isinstance(node, ast.Call)
    function = ast.unparse(node.func)
    if function == 'tl.load':
        assert len(node.args) == 1
        region, offset = pointer(node.args[0])
        return f'(.load "{region}" {offset} {mask(node)} none)'
    assert not node.keywords
    if function == 'tl.program_id':
        assert len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0
        return '(.program 0)'
    if function == 'tl.arange':
        assert [ast.unparse(a) for a in node.args] == ['0', 'BLOCK_SIZE']
        return '(.laneRange 0 block)'
    if function == 'tl.cos':
        assert len(node.args) == 1
        return f'(.cosine {expression(node.args[0])})'
    assert isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
    assert [ast.unparse(a) for a in node.args] == ['tl.float32']
    return f'(.toFP32 {expression(node.func.value)})'

def translate(function):
    output = []
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            assert len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
            output.append(f'.assign "{statement.targets[0].id}" {expression(statement.value)}')
        else:
            assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
            call = statement.value
            assert ast.unparse(call.func) == 'tl.store' and len(call.args) == 2
            region, offset = pointer(call.args[0])
            output.append(f'.store "{region}" {offset} {expression(call.args[1])} {mask(call)}')
    return 'def sourceStatements (n block : Nat) : List SourceStmt :=\n  [' + ',\n   '.join(output) + ']'

text = LEAN.read_text()
observed = text[text.index('def sourceStatements'):text.index('\ninductive IndexExpr')].strip()
generated = translate(kernel)
normalize = lambda value: re.sub(r'\s+', '', value)
assert normalize(observed) == normalize(generated), (observed, generated)
mutations = {
    'wrong_scalar_precision': SOURCE.read_text().replace('a_value.to(tl.float32)', 'a_value.to(tl.float64)'),
    'wrong_trigonometric_op': SOURCE.read_text().replace('tl.cos(', 'tl.sin('),
    'wrong_mask': SOURCE.read_text().replace('offset < n_elements', 'offset <= n_elements'),
    'wrong_program_axis': SOURCE.read_text().replace('tl.program_id(0)', 'tl.program_id(1)'),
    'unmasked_load': SOURCE.read_text().replace('tl.load(a + offset, mask=mask)', 'tl.load(a + offset)'),
}
def rejected(candidate):
    changed = next(n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef) and n.name == 'cos_func')
    try:
        translated = translate(changed)
    except AssertionError:
        return True
    return normalize(translated) != normalize(observed)

assert not rejected(SOURCE.read_text()), 'unchanged-source control rejected'
for label, candidate in mutations.items():
    assert rejected(candidate), 'source mutation escaped translator: ' + label
output = W / 'evidence/python-to-lean-translation.json'
assert not output.exists()
sha = lambda data: hashlib.sha256(data).hexdigest()
result = {'status': 'passed', 'source_sha256': sha(SOURCE.read_bytes()),
          'lean_source_sha256': sha(LEAN.read_bytes()), 'source_statements': generated,
          'kernel_ast_sha256': sha(ast.dump(kernel).encode()), 'mutations_rejected': list(mutations),
          'unchanged_source_control': 'accepted', 'mutation_logic': 'translation error or unequal translation; equality rejects mutation evidence',
          'source_statement_count': len(kernel.body),
          'boundary': 'Exact task-local syntax translation audit. Lean proves lowering, finite execution and independent formulas; this script is not a verified Python parser or native numerical-conformance proof.'}
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'statements': len(kernel.body), 'mutations_rejected': len(mutations)}))
