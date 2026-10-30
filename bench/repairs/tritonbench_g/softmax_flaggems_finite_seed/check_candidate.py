"""Whole-source and CPU boundary checks for the isolated finite-seed repair."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'softmax_flaggems'
SOURCE = HERE / 'softmax_flaggems_repaired.py'
EVIDENCE = ROOT / '.veritile-cache/softmax-flaggems-finite-seed-evidence'
FLOOR = -340282346638528859811704183484516925440.0
NAMES = ['softmax_kernel_non_inner', 'softmax_kernel_inner',
         'softmax_backward_kernel_non_inner', 'softmax_backward_kernel_inner']
spec = importlib.util.spec_from_file_location('finite_seed_helpers', OLD / 'check_contract.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(text, name):
    return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name).body


def unquote(text):
    while '$(' in text:
        start = text.index('$(')
        pos, depth = start + 2, 1
        while depth:
            depth += (text[pos] == '(') - (text[pos] == ')')
            pos += 1
        text = text[:start] + text[start+2:pos-1] + text[pos:]
    return text


def quote(index):
    text = (HERE / 'Kernels.lean').read_text().split(f'finiteSeedFlaggems{index}%[')[1]
    text = text.split('=> triton {', 1)[1].split('\n})', 1)[0]
    text = unquote(text)
    for typ in ('inputType', 'outputType'):
        text = text.replace(f'${typ}:tritonDType', 'tl.float32').replace(f'${typ}', 'tl.float32')
    text = text.replace('$programBase', ['pid_k * TILE_K', 'pid_m * N', 'pid_k * TILE_K', 'pid_m * TILE_M'][index])
    lines, depth = [], 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('tl.region'):
            continue
        if line.startswith('}'):
            depth -= 1
            line = line[1:].strip()
        if not line:
            continue
        opening = line.endswith('{')
        if opening:
            line = line[:-1].strip() + ':'
        lines.append('    ' * depth + line)
        depth += opening
    assert depth == 0
    return ast.parse('\n'.join(lines)).body


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'unused_start_n':
            node.id = '_'
        return node

    def visit_List(self, node):
        return ast.Tuple([self.visit(x) for x in node.elts], ast.Load())

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) == 'tl.load':
            node.keywords.sort(key=lambda item: item.arg)
        if isinstance(node.func, ast.Name) and node.func.id == 'prev_multiple_of':
            assert len(node.args) == 2 and not node.keywords
            a, b = node.args
            return ast.BinOp(ast.BinOp(ast.Call(ast.Attribute(ast.Name('tl', ast.Load()), 'cdiv', ast.Load()),
                [a, b], []), ast.Mult(), copy.deepcopy(b)), ast.Sub(), copy.deepcopy(b))
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to' and len(node.args) == 1:
            dtype = ast.unparse(node.args[0])
            # Pointer bounds separately justify the shared Nat projection.
            if dtype == 'tl.int64':
                return node.func.value
            if dtype == 'output_ptr.dtype.element_ty':
                node.args = [ast.Attribute(ast.Name('tl', ast.Load()), 'float32', ast.Load())]
        if ast.unparse(node.func) == 'tl.store':
            value = node.args[1]
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'to':
                assert len(value.args) == 1 and ast.unparse(value.args[0]) == 'tl.float32'
                node.args[1] = value.func.value
        return node


def canonical(statements):
    return ast.dump(Normalize().visit(ast.Module(copy.deepcopy(statements), [])), include_attributes=False)


def check_source():
    text = SOURCE.read_text()
    old = (OLD / SOURCE.name).read_text()
    seed = 'value=-340282346638528859811704183484516925440.0, dtype=tl.float32'
    assert text.count(seed) == 2
    restored = text.replace(seed, 'value=float("-inf"), dtype=tl.float32')
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(old)), 'unexpected source/domain change'
    helper_body = body(text, 'prev_multiple_of')
    assert len(helper_body) == 1 and ast.unparse(helper_body[0]) == 'return tl.cdiv(a, b) * b - b'
    expected = [canonical(quote(i)) for i in range(4)]
    for i, name in enumerate(NAMES):
        actual = canonical(body(text, name))
        if actual != expected[i]:
            (EVIDENCE / f'ast-mismatch-{i}.json').write_text(json.dumps({'source':actual,'quote':expected[i]},indent=2))
        assert actual == expected[i], name
    mutants = [
        ('finite-seed', seed, 'value=float("-inf"), dtype=tl.float32'),
        ('zero-seed', seed, 'value=0.0, dtype=tl.float32'),
        ('exp-sign', 'tl.exp(m - m_new)', 'tl.exp(m_new - m)'),
        ('reduction-axis', 'tl.max(inp, 0)', 'tl.max(inp, 1)'),
        ('mask-bound', 'n_offsets[:, None] < N', 'n_offsets[:, None] <= N'),
        ('loop-step', 'range(0, N, TILE_N)', 'range(0, N, 1)'),
        ('backward-padding', 'other=0)', 'other=1)'),
        ('integer-width', '.to(tl.int64)', '.to(tl.int32)'),
    ]
    for label, a, b in mutants:
        assert a in text, label
        changed = text.replace(a, b, 1)
        assert any(canonical(body(changed, name)) != expected[i] for i, name in enumerate(NAMES)), label
    assert torch.tensor(FLOOR, dtype=torch.float32).view(torch.int32).item() & 0xffffffff == 0xff7fffff
    return {'whole_module_only_two_seed_changes': True, 'four_full_quotes_match': True,
            'source_mutants_rejected': [x[0] for x in mutants], 'finite_seed_word': '0xff7fffff'}


def check_regression():
    # Reuse the existing regression against this candidate without changing its
    # historical file or weakening its finite/data/metadata assertions.
    original_here, original_binding = helper.HERE, helper.binding
    capture = io.StringIO()
    try:
        helper.HERE, helper.binding = HERE, check_source
        with contextlib.redirect_stdout(capture):
            helper.main()
    finally:
        helper.HERE, helper.binding = original_here, original_binding
    return capture.getvalue().splitlines()


def check_boundaries():
    records = []
    for inner in (True, False):
        for online in (False, True):
            n = (32769 if online else 32768) if inner else (8193 if online else 8192)
            tile = 4096 if inner and online else n
            shape = (1,n) if inner else (1,n,2)
            cases = ['first-inf','all-finite-zero','late-inf','all-finite-low','all-finite-floor',
                     'inf-prefix-low','inf-prefix-floor','all-inf','positive-inf','nan']
            for label in cases:
                value = torch.zeros(shape,dtype=torch.float32)
                column = value[0] if inner else value[0,:,0]
                if label == 'first-inf':column[0] = -float('inf')
                elif label == 'late-inf':column[-1] = -float('inf')
                elif label == 'all-finite-low':value.fill_(-1000)
                elif label == 'all-finite-floor':value.fill_(FLOOR)
                elif label.startswith('inf-prefix'):
                    column.fill_(-float('inf'))
                    column[-1] = FLOOR if label.endswith('floor') else -1000
                elif label == 'all-inf':column.fill_(-float('inf'))
                elif label == 'positive-inf':column[0] = float('inf')
                elif label == 'nan':column[0] = float('nan')
                ns,calls,_ = helper.load(SOURCE)
                x = helper.tensor(value)
                before = x.value.view(torch.int32).clone()
                y = ns['softmax'](x,dim=1,dtype=None)
                assert torch.equal(x.value.view(torch.int32),before)
                assert y is not x and y.ctx.saved_tensors == (y,) and y.value.dtype == torch.float32
                normalized = label not in ('all-inf','positive-inf','nan')
                if normalized:
                    # Independent shifted exponential ratio explicitly gives
                    # mass zero to -inf. No source recurrence is used here.
                    v = value.double()
                    shifted = v - v.amax(1,keepdim=True)
                    weights = torch.where(torch.isneginf(v),0.0,torch.exp(shifted))
                    expected = (weights/weights.sum(1,keepdim=True)).float()
                    assert torch.isfinite(expected).all()
                    torch.testing.assert_close(y.value,expected,rtol=3e-6,atol=3e-7)
                    torch.testing.assert_close(y.value,torch.softmax(value,1),rtol=3e-6,atol=3e-7)
                    g = helper.tensor(torch.linspace(-.5,.5,value.numel()).reshape(shape))
                    gradient,dimgrad,dtypegrad = ns['Softmax'].backward(y.ctx,g)
                    expected_gradient = expected*(g.value-(g.value*expected).sum(1,keepdim=True))
                    torch.testing.assert_close(gradient.value,expected_gradient,rtol=1e-4,atol=4e-7)
                    assert dimgrad is None and dtypegrad is None
                else:
                    actual_column = y.value[0] if inner else y.value[0,:,0]
                    assert torch.isnan(actual_column).all()
                assert calls[0][0] == NAMES[1 if inner else 0] and calls[0][4] == (not online)
                records.append({'inner':inner,'online':online,'shape':shape,'case':label,
                    'normalized_math_defined':normalized,'admitted':True,'input_preserved':True,
                    'nan_count':int(torch.isnan(y.value).sum()),'launches':calls})
    return records


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    report = {'source':check_source(),'regression':check_regression(),'boundaries':check_boundaries(),
              'source_sha256':digest(SOURCE),'quote_sha256':digest(HERE/'Kernels.lean'),
              'gpu_execution':False,'native_numerical_conformance_claimed':False}
    (EVIDENCE/'candidate-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source':report['source'],'regression':report['regression'],
        'boundary_cases':len(report['boundaries']),'gpu_execution':False}))


if __name__ == '__main__':main()
