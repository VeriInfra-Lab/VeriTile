"""Exact decimal AST check of all eleven forward polynomials and 33 derivatives."""
import ast
import hashlib
import json
from pathlib import Path

import sympy as s
import numpy as np

HERE = Path(__file__).resolve().parent
original = HERE.parents[2] / "tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py"
repaired = HERE / "fifth_order_sph_harmonics_repaired.py"
x, y, z = s.symbols("x y z")
upstream = s.symbols("g0:11")

def scalar_graph(path, name, backward):
    text = path.read_text()
    kernel = next(n for n in ast.parse(text).body if getattr(n,"name",None)==name)
    env = dict(x=x,y=y,z=z)
    env.update({f"g_{i}":upstream[i] for i in range(11)})
    env.update(g_x=s.Integer(0),g_y=s.Integer(0),g_z=s.Integer(0))
    def evaluate(node):
        if isinstance(node,ast.Constant): return s.Rational(ast.get_source_segment(text,node))
        if isinstance(node,ast.Name): return env[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub): return -evaluate(node.operand)
        if isinstance(node,ast.BinOp):
            a,b=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Add): return a+b
            if isinstance(node.op,ast.Sub): return a-b
            if isinstance(node.op,ast.Mult): return a*b
        raise ValueError(ast.dump(node))
    for node in kernel.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key.startswith(("CONST","EXACT","VAR","Y")): env[key]=evaluate(node.value)
        if backward and isinstance(node,ast.AugAssign):
            env[node.target.id] += evaluate(node.value)
    return [env[key] for key in (["g_x","g_y","g_z"] if backward else [f"Y{i:02}" for i in range(11)])]

forward = scalar_graph(original,"fifth_order_fwd",False)
new_forward = scalar_graph(repaired,"fifth_order_fwd_repaired",False)
assert all(s.expand(a-b)==0 for a,b in zip(forward,new_forward))
backward = scalar_graph(original,"fifth_order_bwd",True)
new_backward = scalar_graph(repaired,"fifth_order_bwd_repaired",True)
derivatives = 0
old_mismatches = 0
for j,coord in enumerate([x,y,z]):
    for c in range(11):
        expected=s.diff(forward[c],coord)
        actual=s.expand(new_backward[j]).coeff(upstream[c])
        assert s.expand(actual-expected)==0
        old=s.expand(backward[j]).coeff(upstream[c])
        old_mismatches += s.expand(old-expected)!=0
        derivatives += 1
point={x:1,y:0,z:0,**{g:(1 if i==0 else 0) for i,g in enumerate(upstream)}}
old_value=backward[0].subs(point)
correct_value=s.diff(forward[0],x).subs(point)
assert old_value != correct_value
assert hashlib.sha256(original.read_bytes()).hexdigest()=="3879b6fb82f9f47763e3c0f059e17597f973dcf85ec1150227adbaaf1c756c0a"
old_kernel=next(n for n in ast.parse(original.read_text()).body if getattr(n,"name",None)=="fifth_order_bwd")
new_kernel=next(n for n in ast.parse(repaired.read_text()).body if getattr(n,"name",None)=="fifth_order_bwd_repaired")
def constants(node):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in node.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id.startswith(("CONST","EXACT"))}
a,b=constants(old_kernel),constants(new_kernel)
old_terms=[n for n in old_kernel.body if isinstance(n,ast.AugAssign)]
new_terms=[n for n in new_kernel.body if isinstance(n,ast.AugAssign)]
changed_occurrences=0
for old_term,new_term in zip(old_terms,new_terms):
    for old_node,new_node in zip(ast.walk(old_term),ast.walk(new_term)):
        if isinstance(new_node,ast.Name) and new_node.id.startswith('EXACT'):
            assert isinstance(old_node,ast.Name) and old_node.id.startswith('CONST')
            assert np.float32(a[old_node.id]).tobytes()==np.float32(b[new_node.id]).tobytes()
            changed_occurrences+=1
assert changed_occurrences==29
report=dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    forward_channels_unchanged=11,exact_partial_derivatives=derivatives,
    repaired_coefficient_occurrences=changed_occurrences,coefficient_occurrences_rounding_to_same_fp32=changed_occurrences,
    original_inconsistent_partial_derivatives=int(old_mismatches),
    exact_witness=dict(point=[1,0,0],upstream_channel=0,component="x",
                       original=str(old_value),derivative=str(correct_value),difference=str(old_value-correct_value)),
    mathematical_target="Original exact decimal degree-five polynomials and their true Jacobian transpose")
(HERE/"coefficient-validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(report)
