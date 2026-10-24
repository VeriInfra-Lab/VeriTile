"""Bind the complete optimization candidate to exactly three discrete edits.

No payload arithmetic, compiler probe or GPU execution runs.
"""
import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, name
    return matches[0]


def assignment(fn, target):
    matches = [n for n in fn.body if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == target for t in n.targets)]
    assert len(matches) == 1, (fn.name, target)
    return matches[0]


def expression(text):
    return ast.parse(text, mode='eval').body


def verify_python(left, right):
    restored = copy.deepcopy(right)
    producer = function(restored, 'quantize_int8_perrow_kernel')
    row = assignment(producer, 'offs_am')
    assert dump(row.value) == dump(expression('pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)'))
    row.value = copy.deepcopy(assignment(function(left, producer.name), 'offs_am').value)
    gemm = function(restored, 'matmul_kernel')
    for target, pointer in [('a_scale', 'as_ptrs'), ('b_scale', 'bs_ptrs')]:
        load = assignment(gemm, target)
        assert dump(load.value) == dump(expression('tl.load(' + pointer + ')'))
        load.value = copy.deepcopy(assignment(function(left, gemm.name), target).value)
    assert dump(restored) == dump(left), 'unexpected whole Python/host/helper/config change'
    wrapper = function(right, 'quantize_int8_perrow')
    assert dump(assignment(wrapper, 'BLOCK_SIZE_M').value) == dump(expression('1'))
    assert dump(assignment(wrapper, 'grid').value) == dump(expression('(M // BLOCK_SIZE_M,)'))
    return restored


def quotation(source, name):
    start = source.index('def ' + name + ' (p : ')
    end = source.index('\n}\n', start) + 2
    result = source[start:end]
    assert result.count('triton {') == 1
    return result


def main():
    original = ast.parse((HERE / 'int8_matmul_quantization_i32_safe.py').read_text())
    candidate = ast.parse((HERE / 'int8_matmul_quantization_indices.py').read_text())
    verify_python(original, candidate)
    source = (HERE / 'Source.lean').read_text()
    left = quotation(source, 'quantize')
    right = quotation(source, 'unwrappedQuantize').replace('def unwrappedQuantize', 'def quantize', 1)
    old = 'offs_am = (pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows))) % $(p.rows)'
    new = 'offs_am = (pid_m * $(p.blockRows) + tl.arange(0, $(p.blockRows)))'
    assert right.count(new) == 1
    assert right.replace(new, old) == left, 'unexpected complete producer quotation change'
    for suffix in ['HalfHalf', 'HalfSingle', 'SingleHalf', 'SingleSingle']:
        left = quotation(source, 'gemm' + suffix)
        right = quotation(source, 'unmaskedGemm' + suffix).replace('def unmaskedGemm', 'def gemm', 1)
        for new, old in [
            ('a_scale = tl.load(as_ptrs)', 'a_scale = tl.load(as_ptrs, mask=offs_am < $(p.rows), other=0.0)'),
            ('b_scale = tl.load(bs_ptrs)', 'b_scale = tl.load(bs_ptrs, mask=offs_bn < $(p.columns), other=0.0)'),
        ]:
            assert right.count(new) == 1
            right = right.replace(new, old)
        assert right == left, 'unexpected complete GEMM quotation change: ' + suffix
    literal = next(line.split(' := ', 1)[1] for line in source.splitlines()
                   if line.startswith('def productionAST : String'))
    assert json.loads(literal) == dump(original), 'original production AST changed'
    print('PASS: complete Python AST and all five complete ordinary Lean quotation deltas; '
          'only producer row modulo and two GEMM scale-load masks removed; '
          'one-row producer launch, host/helpers/configurations/guards unchanged.')


if __name__ == '__main__':
    main()
