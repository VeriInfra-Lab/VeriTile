"""Compare every quoted arithmetic/store body with the pinned selected Python.

Only the documented source-adapter operations below are normalized. This is a
source binding check, not a compiler proof or a floating-point experiment.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

TASK = Path(__file__).resolve().parent
SOURCE = TASK / 'parallel_retention_attention_repaired.py'


def dump(node):
    return ast.dump(node, include_attributes=False)


def expr(text):
    return ast.parse(text, mode='eval').body


def integer_expression(node):
    names = set('B H T DK DV K V BK BV BTL BTS NV i i_bh i_c i_k i_v i_h i_kv j o_q o_k s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d'.split())
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)) and integer_expression(node.left) and integer_expression(node.right)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub)) and integer_expression(node.operand)
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and node.value.id in ('o_q', 'o_k')
    if isinstance(node, ast.Call):
        return ast.unparse(node.func) in ('tl.program_id', 'tl.arange', 'tl.cdiv') and not node.keywords and all(integer_expression(a) for a in node.args)
    return False


class Normalize(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'c':
            return self.visit(ast.Name(id=node.attr, ctx=ast.Load()))
        return node

    def visit_Name(self, node):
        node.id = {'K': 'DK', 'V': 'DV', 'do_': 'do'}.get(node.id, node.id)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, float) and node.value.is_integer():
            node.value = int(node.value)
        return node

    def visit_Return(self, node):
        assert node.value is None
        return None

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and dump(node.targets[0]).replace('Store()', 'Load()') == dump(node.value):
            return None  # Only a checked integer cast has become an identity.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            assert isinstance(node.value, ast.Tuple) and len(node.targets[0].elts) == len(node.value.elts)
            return [ast.Assign(targets=[left], value=right) for left, right in zip(node.targets[0].elts, node.value.elts)]
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.target.id in ('_', '_i'):
            assert not any(isinstance(n, ast.Name) and n.id == node.target.id for s in node.body for n in ast.walk(s))
            node.target.id = '_unused'
        if ast.unparse(node.iter) == 'range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS)':
            assert isinstance(node.target, ast.Name) and node.target.id == 'i'
            node.target.id = 'j'
            node.iter = expr('range(0, tl.cdiv(tl.cdiv(T, BTS) * BTS - (i_c + 1) * BTL, BTS), 1)')
            node.body.insert(0, ast.parse('i = tl.cdiv(T, BTS) * BTS - BTS - j * BTS').body[0])
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        name = ast.unparse(node.func)
        if name == 'tl.toReal':
            assert len(node.args) == 1 and not node.keywords
            assert integer_expression(node.args[0])
            return node.args[0]  # Source's implicit integer-to-FP32 promotion.
        if name == 'tl.full':
            assert dump(node.args[0]) == dump(expr('[]')) and len(node.args) == 2
            assert [(k.arg, ast.unparse(k.value)) for k in node.keywords] == [('dtype', 'tl.float32')]
            return node.args[1]  # Explicit source-default FP32 literal annotation.
        if name == 'tl.cast' and ast.unparse(node.args[1]) in ('tl.int32', 'tl.int64'):
            assert len(node.args) == 2 and not node.keywords
            assert integer_expression(node.args[0])
            return node.args[0]
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to' and len(node.args) == 1:
            if ast.unparse(node.args[0]) in ('tl.int32', 'tl.int64'):
                assert not node.keywords
                assert integer_expression(node.func.value)
                return node.func.value
        if name == 'tl.make_block_ptr':
            fields = ['base', 'shape', 'strides', 'offsets', 'block_shape', 'order']
            mapping = dict(zip(fields, node.args))
            assert len(mapping) == len(node.args)
            for keyword in node.keywords:
                assert keyword.arg in fields and keyword.arg not in mapping
                mapping[keyword.arg] = keyword.value
            assert set(mapping) == set(fields)
            node.args, node.keywords = [], [ast.keyword(arg=f, value=mapping[f]) for f in fields]
        if name == 'tl.advance' and isinstance(node.args[1], ast.List):
            node.args[1] = ast.Tuple(elts=node.args[1].elts, ctx=ast.Load())
        for keyword in node.keywords:
            if keyword.arg == 'boundary_check' and isinstance(keyword.value, ast.List):
                keyword.value = ast.Tuple(elts=keyword.value.elts, ctx=ast.Load())
        return node


def unquote(text):
    while '$(' in text:
        start = text.index('$(')
        depth, end = 1, start + 2
        while depth:
            depth += (text[end] == '(') - (text[end] == ')')
            end += 1
        inside = re.sub(r'\s*:\s*(Nat|ℝ)\b', '', text[start+2:end-1])
        text = text[:start] + '(' + inside + ')' + text[end:]
    assert '$' not in text
    return text


def quote(path, definition):
    whole = path.read_text()
    body = whole.split('def ' + definition, 1)[1].split('triton {', 1)[1].split('\n}', 1)[0]
    body = unquote(body)
    body = body.replace(': List Nat', '').replace('false', 'False').replace('true', 'True')
    lines = []
    for line in body.splitlines():
        if not line.strip() or line.strip().startswith('tl.region') or line.strip() == '}':
            continue
        if line.endswith('{'):
            line = line[:-1].rstrip() + ':'
        assert line.startswith('  ')
        lines.append(line[2:])
    return ast.parse('\n'.join(lines))


def main():
    functions = {f.name: f for f in ast.parse(SOURCE.read_text()).body if isinstance(f, ast.FunctionDef)}
    rows = []
    for py, filename, definition in [
        ('parallel_retention_fwd_kernel', 'SourceSyntax.lean', 'forwardSource'),
        ('_parallel_retention_bwd_dq', 'QuerySyntax.lean', 'source'),
        ('_parallel_retention_bwd_dkv', 'SourceSyntax.lean', 'backwardSource'),
        ('parallel_retention_fwd_fused_kernel', 'CandidateSyntax.lean', 'fusedForwardSource'),
        ('_parallel_retention_bwd_dq_fused', 'CandidateSyntax.lean', 'fusedSource'),
    ]:
        path = TASK / filename
        left = Normalize().visit(ast.Module(body=functions[py].body, type_ignores=[]))
        right = Normalize().visit(quote(path, definition))
        if dump(left) != dump(right):
            raise AssertionError(py + '\nPYTHON\n' + ast.unparse(ast.fix_missing_locations(left)) +
                                 '\nQUOTATION\n' + ast.unparse(ast.fix_missing_locations(right)))
        rows.append({'function': py, 'quotation': str(path.relative_to(TASK)),
                     'definition': definition, 'complete_body_equal': True,
                     'normalized_ast_sha256': hashlib.sha256(dump(left).encode()).hexdigest(),
                     'statements': sum(isinstance(n, ast.stmt) for n in ast.walk(left))})
    shell = Normalize().visit(ast.Module(body=functions['parallel_retention_bwd_kernel'].body, type_ignores=[]))
    prefix = ast.parse('''i_kv = tl.program_id(0)
i_c = tl.program_id(1)
i_bh = tl.program_id(2)
NV = tl.cdiv(DV, BV)
i_k = i_kv // NV
i_v = i_kv % NV
i_h = i_bh % H''')
    assert dump(ast.Module(body=shell.body[:7], type_ignores=[])) == dump(prefix)
    assert [ast.unparse(n.value.func) for n in shell.body[7:]] == [
        '_parallel_retention_bwd_dq', 'tl.debug_barrier', '_parallel_retention_bwd_dkv']
    calls = [shell.body[7].value, shell.body[9].value]
    expected = [
        'i_bh i_c i_k i_v i_h k v do dq s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T scale BTL BTS BK BV DK DV',
        'i_bh i_c i_k i_v i_h q k v do dk dv s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T scale BTL BTS BK BV DK DV']
    for call, parameters in zip(calls, expected):
        assert [ast.unparse(a) for a in call.args] + [ast.unparse(k.value) for k in call.keywords] == parameters.split()
        assert all(k.arg == ast.unparse(k.value) for k in call.keywords)
    report = {'status': 'PASS', 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'bodies': rows,
      'backward_shell': 'Exact seven index assignments and dQ/barrier/dKV calls; grid_decode binds helper indices.',
      'normalizations': ['Syntactic antiquotes, braces, name K/V/do_, tuple assignment split and block-pointer named arguments',
        'Explicit default FP32 constant annotations and implicit integer-to-FP32 promotions',
        'Checked integer widening and signed32 block-coordinate identity casts; source Domain proves range guards',
        'Unused loop binder spelling; descending i range reindexed by j with descendingCount/descendingStart'],
      'floating_arithmetic_reassociation': False, 'gpu_or_compiler_execution': False}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
