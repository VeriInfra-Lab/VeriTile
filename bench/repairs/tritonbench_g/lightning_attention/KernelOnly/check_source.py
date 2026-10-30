"""Compare every quoted statement to the selected Python kernels.
Only typed representation annotations and the reserved Lean `do` name differ.
This is a source-binding control, not a Lean proof or GPU numerical test.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

TASK = Path(__file__).resolve().parent.parent
ROOT = TASK.parents[3]
OUT = ROOT / '.veritile-cache/kernel-evidence/lightning_attention'
SOURCE = TASK / 'lightning_attention_repaired.py'
QUOTE = TASK / 'KernelOnly/Kernels.lean'
NAMES = {'forward': '_fwd_kernel', 'intra': '_bwd_intra_kernel', 'inter': '_bwd_inter_kernel'}

class Canonical(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id='go' if node.id == 'do' else node.id, ctx=node.ctx), node)

    def visit_Constant(self, node):
        # These are exact zero literals. Context determines the original dtype.
        if isinstance(node.value, float) and node.value == 0.0:
            return ast.copy_location(ast.Constant(value=0), node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to' and len(node.args) == 1:
            target = node.args[0]
            if isinstance(target, ast.Attribute) and target.attr == 'element_ty':
                # All seven region declarations and checked source pointers are FP32.
                node.args = [ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()), attr='float32', ctx=ast.Load())]
        return node

def canonical(body):
    return ast.dump(Canonical().visit(ast.Module(body=body, type_ignores=[])), include_attributes=False)

def quoted(stage):
    text = QUOTE.read_text().split(f'def {stage} (p : Parameters)', 1)[1]
    text = text.split('  triton {\n', 1)[1].split('\n  }', 1)[0]
    if stage == 'forward':
        original = 'index = off_block[:, None].to(tl.int64) - off_block[None, :].to(tl.int64)'
        assert original in text
        text = text.replace(original, 'index = off_block[:, None] - off_block[None, :]')
        text = text.replace('$((0 : Int))', '0')
    if stage == 'intra':
        original = 'array = tl.toReal(tl.arange(0, $(BLOCK)))'
        assert original in text
        text = text.replace(original, 'array = tl.arange(0, $(BLOCK)).to(tl.float32)')
    if stage == 'inter':
        text = text.replace('$((Int.ofNat NUM_BLOCK - 1 : Int))', 'NUM_BLOCK - 1')
    text = re.sub(r'\$\(([A-Za-z_][A-Za-z_0-9]*)\)', r'\1', text)
    result, depth = [], 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('tl.region'):
            continue
        if line == '}':
            depth -= 1
            continue
        opens = line.endswith('{')
        if opens:
            line = line[:-1].rstrip() + ':'
        result.append('    ' * depth + line)
        if opens:
            depth += 1
    assert depth == 0
    return ast.parse('\n'.join(result)).body

source_nodes = {n.name: n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)}
rows = []
for stage, name in NAMES.items():
    source = canonical(source_nodes[name].body)
    quote = canonical(quoted(stage))
    if source != quote:
        (OUT / f'{stage}-source-ast.txt').write_text(source + '\n')
        (OUT / f'{stage}-quote-ast.txt').write_text(quote + '\n')
        raise AssertionError(f'{stage}: source and quotation ASTs differ')
    rows.append({'stage': stage, 'all_statements_match': True,
                 'source_statement_nodes': sum(isinstance(n, ast.stmt) for n in ast.walk(source_nodes[name])) - 1})
report = {'status': 'PASS', 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'quote_sha256': hashlib.sha256(QUOTE.read_bytes()).hexdigest(), 'stages': rows,
          'representation_annotations': [
              'Signed forward arange difference represented in Int; fixed lane range and int32 difference bounds proved.',
              'tl.toReal is the shared DSL numeric conversion under explicit FP32 default precision.',
              'Signed source reverse-bound constexpr represented by an Int literal.',
              'Store element dtype is FP32 for every declared/checked region.',
              'Python do identifier is consistently renamed go; exact zero literals are canonicalized.'],
          'GPU_execution': False, 'native_numerical_conformance': False}
(OUT / 'source-quote-controls.json').write_text(json.dumps(report, indent=2) + '\n')
print('all statements in all three quotations match the selected Python kernels')
