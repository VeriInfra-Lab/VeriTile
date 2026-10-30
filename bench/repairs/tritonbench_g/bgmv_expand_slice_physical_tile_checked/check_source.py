"""Bind the complete signed quotation to the unchanged selected JIT source.

Only declared primary specialization and explicit typed-DSL representations
are normalized. This is AST evidence, not a compiler or numerical theorem.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / '.veritile-cache/kernel-evidence/bgmv_expand_slice_physical_tile_checked'
SOURCE = HERE / 'bgmv_expand_slice_physical_tile_checked.py'
QUOTE = HERE / 'SurfaceSource.lean'

def expr(text):
    return ast.parse(text, mode='eval').body

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

bindings = dict(input_ptr='X', lora_ptr='W', out_ptr='Y', BLOCK_N='256', BLOCK_K='BK',
    xm_stride='K', xk_stride='1', l0_stride='N*K', lora_k_stride='K', lora_n_stride='1',
    cm_stride='columns', cn_stride='1', slice_offset='slice', SPLIT_N='N', CAST_TYPE='True')

class Canonical(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(expr(bindings[node.id]), node) if node.id in bindings else node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            assert not node.orelse
            return node.body
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Mult) and isinstance(node.right, ast.Constant) and node.right.value == 1:
            return node.left
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = ast.unparse(node.func)
        if name == 'tl.cast':
            assert len(node.args) == 2 and ast.unparse(node.args[1]) == 'tl.int64'
            return ast.Call(func=ast.Attribute(value=node.args[0], attr='to', ctx=ast.Load()),
                            args=[node.args[1]], keywords=[])
        if name == 'tl.cdiv':
            assert [ast.unparse(a) for a in node.args] == ['N.to(tl.int64)', 'N.to(tl.int64)']
            return expr('(N + N - 1) // N')
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
            assert len(node.args) == 1
            if ast.unparse(node.args[0]) == 'W.dtype.element_ty':
                node.args[0] = expr('tl.float16')
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and ast.unparse(node.targets[0]) == 'accumulator':
            # The shared typed DSL requires the source's implicit FP32 store
            # conversion to be explicit. No operation moves across this cast.
            value = node.value
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == 'to' and len(value.args) == 1
                    and ast.unparse(value.args[0]) == 'tl.float32'):
                node.value = ast.Call(func=ast.Attribute(value=value, attr='to', ctx=ast.Load()),
                                      args=[expr('tl.float32')], keywords=[])
        return node

def canonical(body):
    tree = Canonical().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[]))
    return ast.dump(tree, include_attributes=False)

text = QUOTE.read_text()
assert 'tl.region X = tl.float32, W = tl.float16, Y = tl.float32' in text
assert 'let lora_indices : Region .int' in text
assert 'let EVEN_K : Bool := h.even' in text and 'let ADD_INPUTS : Bool := h.add' in text
text = text.split('  triton {\n', 1)[1].rsplit('\n}', 1)[0]
for old, new in [('$((-1 : Int))', '-1'), ('$((h.N : Int))', 'N'),
                 ('$((1 : Int))', '1'), ('$((0 : Int))', '0'),
                 ('$(h.N*h.K)', 'N*K')]:
    text = text.replace(old, new)
text = re.sub(r'\$\(h\.(BK|K|columns|slice)\)', r'\1', text)
assert '$(' not in text
lines, depth = [], 0
for raw in text.splitlines():
    line = raw.strip()
    if not line or line.startswith('tl.region'):
        continue
    if line == 'if lora_index == -1 { return }':
        lines.extend(['    '*depth + 'if lora_index == -1:', '    '*(depth+1) + 'return'])
        continue
    if line.startswith('}'):
        depth -= 1
        line = line[1:].strip()
        if not line:
            continue
    opens = line.endswith('{')
    if opens:
        line = line[:-1].rstrip() + ':'
    lines.append('    '*depth + line)
    if opens:
        depth += 1
assert depth == 0
quote = ast.parse('\n'.join(lines)).body
module = ast.parse(SOURCE.read_text())
kernel = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_bgmv_expand_slice_kernel')
body = kernel.body
assert isinstance(body[0], ast.Expr) and isinstance(body[0].value.value, str)
source_ast, quote_ast = canonical(body[1:]), canonical(quote)
if source_ast != quote_ast:
    (OUT/'source-ast.txt').write_text(source_ast+'\n')
    (OUT/'quote-ast.txt').write_text(quote_ast+'\n')
    raise AssertionError('The specialized source and signed quotation differ')

wrapper = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_bgmv_expand_slice')
launches = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
            and ast.unparse(n.func.value) == '_bgmv_expand_slice_kernel']
assert len(launches) == 1
launch = launches[0]
assert ast.unparse(launch.func.slice) == '(N, B)'
assert [ast.unparse(a) for a in launch.args] == ['inputs','weights','output_tensor','N','K',
    'lora_indices_tensor','K','1','N * K','K','1','C','1','slice_offset']
assert {k.arg: ast.unparse(k.value) for k in launch.keywords} == {
    'BLOCK_N':'256','BLOCK_K':'BLOCK_K','SPLIT_N':'N','EVEN_K':'K % BLOCK_K == 0',
    'ADD_INPUTS':'add_inputs','CAST_TYPE':'True'}
report = {'status':'PASS','source_sha256':digest(SOURCE),'quote_sha256':digest(QUOTE),
    'checker_sha256':digest(Path(__file__)), 'all_specialized_statements_match':True,
    'source_statement_nodes':sum(isinstance(n,ast.stmt) for n in ast.walk(kernel))-2,
    'wrapper_launch_checked':True,'static_bindings':bindings,
    'representation_annotations':[
        'FP32 X/Y, FP16 W and signed metadata regions; CAST_TYPE=True follows the checked wrapper.',
        'Contiguous specialized strides substitute K, 1, N*K, K, 1, C, 1; multiplication by 1 is simplified.',
        'Signed tl.cdiv(N,N) is expanded as (N+N-1)//N; signed_length and signed_cdiv prove its nonempty value and arithmetic bound.',
        'tl.cast(0,tl.int64) uses equivalent method spelling; signed N/-1/0/1 constants use Int antiquotes.',
        'The implicit FP32 output-store conversion is explicit on each accumulator assignment.',
        'Actual signed pids, aranges, early return, signed range, branch structure, load/store masks and reduction axis are retained.'
    ],'GPU_execution':False,'native_numerical_conformance':False,'universal_compiler_profile_certification':False}
(OUT/'source-quote-controls.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: complete specialized JIT statement AST and wrapper launch match the signed quotation')
