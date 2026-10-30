"""Exact selected JIT/DSL structure, ABI binding, and obvious original defects.
No GPU numerical/conversion/compiler conformance is inferred from these checks.
"""
import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = ast.parse((HERE / 'rmsnorm_fused_repaired.py').read_text())
node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'rms_norm_fwd_fused')
assert [a.arg for a in node.args.args] == ['X', 'Y', 'W', 'stride', 'N', 'eps', 'BLOCK_SIZE']
assert {a.arg for a in node.args.args if a.annotation is not None} == {'BLOCK_SIZE'}
# The checked wrapper binds stride=N. Typed regions and the removed epsilon
# declaration are DSL typing instrumentation, not inserted source operations.
lean = (HERE / 'Kernels.lean').read_text()
body = lean.split('def template ', 1)[1].split(':= triton {', 1)[1].split('\n}\n', 1)[0]
lines = [line.strip() for line in body.splitlines() if line.strip()]
assert lines.pop(0) == 'tl.region X = tl.float32, W = tl.float32, Y = tl.float32'
assert lines.pop(0) == 'eps = 0.0'
def norm(s):
    return re.sub(r'\s+', '', s.replace('$(N)', 'N').replace('$(B)', 'B').replace('BLOCK_SIZE', 'B').replace('stride', 'N'))
def statements(ns):
    out = []
    for n in ns:
        if isinstance(n, ast.For):
            assert not n.orelse
            out += [norm('for ' + ast.unparse(n.target) + ' in ' + ast.unparse(n.iter) + ' {')]
            out += statements(n.body) + ['}']
        else:
            out.append(norm(ast.unparse(n)))
    return out
assert list(map(norm, lines)) == statements(node.body)
assert '.mk [X,W] [Y] ((template X Y W N B).surfaceBody.drop 1)' in lean
runtime = (HERE/'Proofs/Runtime.lean').read_text()
assert 's.setReg "eps" (dtype := .real) (shape := []) (fun _ => eps)' in runtime
assert '(source X Y W N B) (bind (setup M r s) eps)' in runtime
for path in ['Correct.lean', 'FPEquiv.lean']:
    text = (HERE/path).read_text()
    assert 'specification kernel_' in text and 'Runtime.run' in text
    assert 'Host.invoke' not in text and 'scalar_equivalence' not in text
# Original int32 row*N can overflow while the repair admits the same geometry.
row, width = 2**17, 2**14
original_signed = ((row*width + 2**31) % 2**32) - 2**31
assert original_signed == -2**31 and row*width == 2**31
assert (row+1)*width <= (2**63-1)//4 and row+1 <= 2**31-1
assert 1*4+1 >= 2*2  # gapped input row stride=4 vs fresh packed 2x2 output
print('PASS: every selected JIT statement matches readable DSL; runtime scalar register is bound')
print('PASS: independent kernel-only public entries; original signed32 and gapped-output counterexamples')
