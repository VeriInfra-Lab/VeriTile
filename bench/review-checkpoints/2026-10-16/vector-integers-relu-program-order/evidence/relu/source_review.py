import ast
import hashlib
import json
import re
import struct
from pathlib import Path

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
source = repo / 'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tree = ast.parse(source.read_text())
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
harness = functions['test_relu']
constructors = [n.value for n in harness.body if isinstance(n, ast.Assign) and
                isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'torch' and
                n.value.func.attr == 'tensor']
assert len(constructors) == 4
values = [ast.literal_eval(n.args[0]) for n in constructors]
for n in constructors:
    keywords = {k.arg: ast.unparse(k.value) for k in n.keywords}
    assert keywords == {'dtype': 'torch.float32', 'device': "'cuda'"}
word = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
input_words = [[word(x) for x in row] for row in values]
output_words = [[word(x if x >= 0 else 0.0) for x in row] for row in values]
software = root / 'bench/tritonbench_g/relu_triton_kernel/Proofs/SoftwareFP32.lean'
text = software.read_text()
def lean_literal_list(name):
    body = text.split('def ' + name + ' : List (List Nat) := ', 1)[1].split('\n\n', 1)[0]
    return ast.literal_eval(body)
assert lean_literal_list('harnessInputWords') == input_words
assert lean_literal_list('harnessExpectedWords') == output_words
assert not any(isinstance(n, ast.Assert) for n in ast.walk(functions['relu']))
result_assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'result_gold' for t in n.targets)]
assert len(result_assignments) == 1 and ast.unparse(result_assignments[0].value) == 'test_relu()'
report = {
    'source': source.relative_to(repo).as_posix(), 'source_sha256': sha(source),
    'function_ast_sha256': {name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                            for name, node in functions.items()},
    'wrapper_assertion_count': 0,
    'harness_input_values': values, 'harness_input_fp32_words': input_words,
    'harness_expected_fp32_words': output_words,
    'software_harness_lists_match_source_literals': True,
    'software_source_sha256': sha(software),
    'module_result_assignment': 'result_gold = test_relu()',
    'original_module_imported': False, 'gpu_executed': False,
    'limitations': [
        'CPU struct encoding checks only the twenty declared exactly representable harness literals, not general Torch constructor behavior.',
        'AST fingerprints bind manually reviewed host order; they are not a general Python-to-Lean refinement theorem.',
        'Ordered dictionary is modeled by an association list with the original four distinct literal keys; native allocation/dictionary/import errors remain unbound.',
    ],
}
(root / 'evidence/source-review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'source_review_sha256': sha(root / 'evidence/source-review.json'), 'literal_words': 20}))
