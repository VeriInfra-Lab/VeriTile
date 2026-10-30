"""Observe the selected FP32 compiler boundary without executing a GPU."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


# Track both Q and K destinations and every ordered operand in the actual JIT.
def check_graph(ptx, backward):
    """Check the complete stored-expression wiring, including operand order."""
    registers, pointers = {}, {}
    loaded = dict.fromkeys(('Q', 'K', 'C', 'S'), 0)
    stored = {'Q': 0, 'K': 0}
    zero = ('bits', '00000000')

    def value(token):
        token = token.strip()
        if token.startswith('0f'):
            return ('bits', token[2:].lower())
        if token == '0':
            return zero
        return registers[token]

    def expected(role, ordinal):
        halves = loaded[role] // 2
        side, lane = divmod(ordinal, halves)
        assert side in (0, 1)
        a, b, c, s = (role, lane), (role, lane + halves), ('C', 0), ('S', 0)
        if backward:
            return ('fma', c, b, ('neg', ('mul', s, a))) if side else ('fma', c, a, ('mul', s, b))
        return ('fma', s, a, ('mul', c, b)) if side else ('fma', c, a, ('neg', ('mul', s, b)))

    for line in ptx.splitlines():
        param = re.search(r'ld\.param\.b64\s+(%\w+),\s*\[_triton_rope_param_(\d+)\]', line)
        if param:
            dest, index = param.groups()
            # Q_row_stride may be i64; it is a scalar, never a data pointer.
            if int(index) in (0, 2, 4, 6):
                pointers[dest] = {0: 'Q', 2: 'K', 4: 'C', 6: 'S'}[int(index)]
        address = re.search(r'(?:mad\.wide\.[su]32|add\.s64)\s+(%\w+),\s*([^;]+);', line)
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
            role = pointers[ptr]
            assert role in ('Q', 'K')
            assert value(operand) == expected(role, stored[role]), (role, stored[role], value(operand), expected(role, stored[role]))
            stored[role] += 1
    assert loaded['C'] == loaded['S'] == 1
    assert all(loaded[r] >= 2 and loaded[r] % 2 == 0 and stored[r] == loaded[r] for r in ('Q', 'K'))
    assert '.ftz.' not in ptx
    accesses = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+', ptx, re.M)
    assert len(accesses) == sum(loaded.values()) + sum(stored.values())
    assert all(pred and op.endswith('.b32') for pred, op in accesses)
    return {'loads': loaded, 'stores': stored, 'masked_fp32_accesses': len(accesses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source = Path(__file__).with_name('rope_transform_repaired.py')
    kernel = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_triton_rope')
    kernel.decorator_list = [ast.parse('triton.jit', mode='eval').body]
    selected = output / 'selected.py'
    selected.write_text('import triton\nimport triton.language as tl\n' + ast.unparse(kernel) + '\n')
    spec = importlib.util.spec_from_file_location('rope_transform_selected', selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = dict.fromkeys(('q_ptr', 'k_ptr', 'cos', 'sin'), '*fp32')
    signature.update(dict.fromkeys(('q_row_stride', 'k_row_stride', 'cos_row_stride', 'sin_row_stride', 'sl'), 'i32'))
    signature.update(dict.fromkeys(('bs', 'n_qh', 'n_kh', 'hd', 'pad_n_qh', 'pad_n_kh',
                                    'pad_hd', 'BLOCK_SIZE', 'BACKWARD_PASS'), 'constexpr'))
    reports = []
    for sm, dimension, qheads, kheads in ((80, 6, 3, 1), (90, 34, 5, 3)):
        for backward in (False, True):
            pad = triton.next_power_of_2
            src = ASTSource(module._triton_rope, signature=signature, constexprs={
                'bs': 2, 'n_qh': qheads, 'n_kh': kheads, 'hd': dimension,
                'pad_n_qh': pad(qheads), 'pad_n_kh': pad(kheads), 'pad_hd': pad(dimension),
                'BLOCK_SIZE': max(pad(qheads), pad(kheads)), 'BACKWARD_PASS': backward})
            compiled = triton.compile(src, target=GPUTarget('cuda', sm, 32),
                                      options={'num_warps': 4, 'enable_fp_fusion': True})
            ptx = compiled.asm['ptx']
            name = f'sm{sm}.d{dimension}.q{qheads}.k{kheads}.backward{int(backward)}.ptx'
            (output / name).write_text(ptx)
            wiring = check_graph(ptx, backward)
            reports.append({'sm': sm, 'dimension': dimension, 'q_heads': qheads, 'k_heads': kheads,
                            'backward': backward, 'ordered_graph_checked': True, 'wiring': wiring,
                            'ptx': name, 'sha256': hashlib.sha256(ptx.encode()).hexdigest()})
            print(f'{name}: all ordered Q/K output graphs and masked FP32 accesses checked')
    (output / 'report.json').write_text(json.dumps({
        'triton': triton.__version__, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'gpu_execution': False, 'numerical_replay': False,
        'scope': 'finite offline compiler observations; explicit lowering boundary, not compiler correctness proof',
        'cases': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
