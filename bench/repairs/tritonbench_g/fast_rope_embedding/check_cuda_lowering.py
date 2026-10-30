"""Observe the selected FP32 compiler boundary without executing a GPU."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


# The rope ordered-graph checker, extended to ignore the i64 scalar stride parameter.
def check_graph(ptx, backward):
    """Check the complete stored-expression wiring, including operand order."""
    registers, pointers = {}, {}
    loaded = dict.fromkeys(('Q', 'C', 'S'), 0)
    stored = 0
    zero = ('bits', '00000000')

    def value(token):
        token = token.strip()
        if token.startswith('0f'):
            return ('bits', token[2:].lower())
        if token == '0':
            return zero
        return registers[token]

    def expected(side):
        a, b, c, s = ('Q', 0), ('Q', 1), ('C', 0), ('S', 0)
        s = ('sub', zero, s) if backward else s
        return ('fma', s, a, ('mul', c, b)) if side else ('fma', c, a, ('neg', ('mul', s, b)))

    for line in ptx.splitlines():
        param = re.search(r'ld\.param\.b64\s+(%\w+),\s*\[_rope_embedding_param_(\d+)\]', line)
        if param:
            dest, index = param.groups()
            # Q_row_stride may be i64; it is a scalar, never a data pointer.
            if int(index) in (0, 2, 4):
                pointers[dest] = {0: 'Q', 2: 'C', 4: 'S'}[int(index)]
        address = re.search(r'(?:mad\.wide\.s32|add\.s64)\s+(%\w+),\s*([^;]+);', line)
        if address:
            dest, args = address.groups()
            roles = [pointers[arg.strip()] for arg in args.split(',') if arg.strip() in pointers]
            if roles:
                assert len(set(roles)) == 1
                pointers[dest] = roles[0]
        load = re.search(r'ld\.global\.b32\s*\{\s*(%\w+)\s*\},\s*\[\s*(%\w+)', line)
        if load:
            dest, ptr = load.groups()
            role = pointers[ptr]
            registers[dest] = (role, loaded[role])
            loaded[role] += 1
        operation = re.search(r'\b(neg\.f32|sub\.f32|mul\.f32|fma\.rn\.f32)\s+([^;]+);', line)
        if operation:
            opcode, args = operation.groups()
            dest, *args = map(str.strip, args.split(','))
            registers[dest] = (opcode.split('.')[0], *(value(arg) for arg in args))
        move = re.search(r'mov\.b32\s+(%\w+),\s*(0f[0-9a-fA-F]+|0);', line)
        if move:
            registers[move[1]] = value(move[2])
        store = re.search(r'st\.global\.b32\s*\[\s*(%\w+)[^]]*\],\s*\{\s*(%\w+)\s*\}', line)
        if store:
            ptr, operand = store.groups()
            assert pointers[ptr] == 'Q'
            assert value(operand) == expected(stored), (stored, value(operand), expected(stored))
            stored += 1
    assert loaded == {'Q': 2, 'C': 1, 'S': 1} and stored == 2
    assert '.ftz.' not in ptx
    accesses = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+', ptx, re.M)
    assert len(accesses) == 6 and all(pred and op.endswith('.b32') for pred, op in accesses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source = Path(__file__).with_name('fast_rope_embedding_repaired.py')
    kernel = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_rope_embedding')
    kernel.decorator_list = [ast.parse('triton.jit', mode='eval').body]
    selected = output / 'selected.py'
    selected.write_text('import triton\nimport triton.language as tl\n' + ast.unparse(kernel) + '\n')
    spec = importlib.util.spec_from_file_location('fast_rope_embedding_selected', selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = dict.fromkeys(('Q', 'cos', 'sin'), '*fp32')
    signature.update(dict.fromkeys(('Q_row_stride', 'cos_row_stride', 'sin_row_stride', 'seqlen'), 'i32'))
    signature.update(dict.fromkeys(('head_dim', 'n_heads', 'BACKWARD_PASS', 'BLOCK_SIZE'), 'constexpr'))
    reports = []
    for sm, dimension, heads, stride_type in ((80, 35, 7, 'i32'), (90, 64, 5, 'i32'), (80, 35, 7, 'i64')):
        signature['Q_row_stride'] = stride_type
        for backward in (False, True):
            src = ASTSource(module._rope_embedding, signature=signature, constexprs={
                'head_dim': dimension, 'n_heads': heads, 'BACKWARD_PASS': backward,
                'BLOCK_SIZE': 32})
            compiled = triton.compile(src, target=GPUTarget('cuda', sm, 32),
                                      options={'num_warps': 4, 'enable_fp_fusion': True})
            ptx = compiled.asm['ptx']
            check_graph(ptx, backward)
            name = f'sm{sm}.d{dimension}.{stride_type}.backward{int(backward)}.ptx'
            (output / name).write_text(ptx)
            reports.append({'sm': sm, 'dimension': dimension, 'heads': heads,
                            'backward': backward, 'q_stride_type': stride_type, 'ordered_graph_checked': True,
                            'ptx': name, 'sha256': hashlib.sha256(ptx.encode()).hexdigest()})
            print(f'{name}: both complete ordered output graphs and six masked FP32 accesses checked')
    (output / 'report.json').write_text(json.dumps({
        'triton': triton.__version__, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'gpu_execution': False, 'numerical_replay': False,
        'scope': 'finite offline compiler observations; explicit lowering boundary, not compiler correctness proof',
        'cases': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
