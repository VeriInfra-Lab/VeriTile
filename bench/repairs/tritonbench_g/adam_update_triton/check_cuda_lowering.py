"""Offline lowering of the unchanged selected FP32 Lion kernel; no GPU execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def check_ordered_graph(ptx):
    """Compare active stored expressions with the independent native recurrence.

    Values are symbolic words; this checks instruction wiring and contraction,
    not any floating-point numerical identity or hardware conformance law.
    """
    registers, pointers = {}, {}
    loaded = dict.fromkeys(('P', 'G', 'M'), 0)
    stored = dict.fromkeys(('P', 'M'), 0)
    scalar_names = {3: 'lr', 4: 'wd', 5: 'beta1', 6: 'beta2'}
    zero, one = ('bits', '00000000'), ('bits', '3f800000')

    def value(token):
        token = token.strip()
        if token.startswith('0f'):
            return ('bits', token[2:].lower())
        if token in ('0', '0x0'):
            return zero
        return registers[token]

    def expected(output, lane):
        p, g, m = ((name, lane) for name in ('P', 'G', 'M'))
        lr, wd, b1, b2 = (('arg', name) for name in ('lr', 'wd', 'beta1', 'beta2'))
        difference = ('sub', m, g)
        update = ('fma', b1, difference, g)
        if output == 'M':
            return ('fma', b2, difference, g)
        decay = ('fma', ('neg', lr), wd, one)
        chosen = ('select', ('sub', zero, lr), lr, ('gt', update, zero))
        gate = ('select', one, zero, ('neu', update, zero))
        return ('fma', decay, p, ('mul', chosen, gate))

    for line in ptx.splitlines():
        param = re.search(r'ld\.param\.b(32|64)\s+(%\w+),\s*\[update_fn_kernel_param_(\d+)\]', line)
        if param:
            width, dest, index = param.groups()
            index = int(index)
            if width == '64' and index < 3:
                pointers[dest] = ('P', 'G', 'M')[index]
            elif index in scalar_names:
                registers[dest] = ('arg', scalar_names[index])
            continue
        address = re.search(r'add\.s64\s+(%\w+),\s*(%\w+),\s*([^;]+);', line)
        if address:
            dest, left, right = (token.strip() for token in address.groups())
            if left in pointers or right in pointers:
                pointers[dest] = pointers.get(left, pointers.get(right))
            continue
        load = re.search(r'ld\.global(?:\.v4)?\.b32\s*\{([^}]+)\},\s*\[\s*(%\w+)', line)
        if load:
            destinations, pointer = load.groups()
            region = pointers[pointer]
            for dest in destinations.split(','):
                registers[dest.strip()] = (region, loaded[region])
                loaded[region] += 1
            continue
        store = re.search(r'st\.global(?:\.v4)?\.b32\s*\[\s*(%\w+)[^]]*\],\s*\{([^}]+)\}', line)
        if store:
            pointer, operands = store.groups()
            region = pointers[pointer]
            assert region in stored, region
            for operand in operands.split(','):
                actual = value(operand)
                wanted = expected(region, stored[region])
                assert actual == wanted, (region, stored[region], actual, wanted)
                stored[region] += 1
            continue
        operation = re.search(r'\b(neg\.f32|sub\.f32|mul\.f32|fma\.rn\.f32|setp\.(?:neu|gt)\.f32|selp\.(?:f32|b32))\s+([^;]+);', line)
        if operation:
            opcode, operands = operation.groups()
            dest, *args = (token.strip() for token in operands.split(','))
            name = opcode.split('.')[1] if opcode.startswith('setp.') else opcode.split('.')[0]
            name = 'select' if name == 'selp' else name
            registers[dest] = (name, *(value(arg) for arg in args))
            continue
        move = re.search(r'mov\.b32\s+(%\w+),\s*([^;]+);', line)
        if move:
            dest, operand = move.groups()
            registers[dest] = value(operand)
    assert loaded['P'] == loaded['G'] == loaded['M'] == stored['P'] == stored['M'] > 0
    return stored['P']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = Path(__file__).with_name('adam_update_triton_repaired.py')
    kernel = next(n for n in ast.parse(source.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'update_fn_kernel')
    # Bypass host autotuner construction, retaining its exact JIT function body.
    kernel.decorator_list = [ast.parse('triton.jit', mode='eval').body]
    selected = output / 'selected_kernel.py'
    selected.write_text('import triton\nimport triton.language as tl\n' + ast.unparse(kernel) + '\n')
    spec = importlib.util.spec_from_file_location('adam_lion_lowering', selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for sm, block, warps, aligned_extent in ((80, 128, 4, False), (90, 128, 4, False),
                                            (90, 1024, 8, False), (90, 1024, 8, True)):
        signature = dict.fromkeys(('p_ptr', 'grad_ptr', 'exp_avg_ptr'), '*fp32')
        signature.update(dict.fromkeys(('lr', 'wd', 'beta1', 'beta2'), 'fp32'))
        signature.update(n_elements='i32', chunk_elements='i32', BLOCK_SIZE='constexpr')
        attrs = {(j,): [['tt.divisibility', 16]] for j in range(3)}
        if aligned_extent:
            attrs[(8,)] = [['tt.divisibility', 16]]
        profile = ASTSource(module.update_fn_kernel, signature=signature,
                            constexprs={'BLOCK_SIZE': block},
                            attrs=attrs)
        compiled = triton.compile(profile, target=GPUTarget('cuda', sm, 32),
                                  options={'num_warps': warps, 'enable_fp_fusion': True})
        ptx = compiled.asm['ptx']
        stem = f'lion.sm{sm}.b{block}.n16-{int(aligned_extent)}'
        (output / (stem + '.ptx')).write_text(ptx)
        (output / (stem + '.llir')).write_text(compiled.asm['llir'])
        for opcode in ('neg.f32', 'sub.f32', 'mul.f32', 'fma.rn.f32',
                       'setp.neu.f32', 'setp.gt.f32'):
            assert opcode in ptx, (stem, opcode)
        assert '.ftz.' not in ptx and 'add.f32' not in ptx
        assert ptx.count('fma.rn.f32') >= 4
        assert '0f3F800000' in ptx and '0f00000000' in ptx
        memory = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+', ptx, re.M)
        assert memory and all(pred for pred, _ in memory), memory
        assert all(op.endswith('.b32') for _, op in memory), memory
        lanes = check_ordered_graph(ptx)
        cases.append({'sm': sm, 'block': block, 'warps': warps,
                      'mask_extent_divisible_by_16': aligned_extent,
                      'runtime_scalar_abi': ['fp32'] * 4,
                      'pointer_alignment': 16, 'native_fma_preserved': True,
                      'ordered_native_graph_checked': True, 'symbolic_output_lanes_per_thread': lanes,
                      'explicit_ftz': False, 'masked_memory': memory,
                      'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest()})
        print(f'{stem}: runtime FP32 scalars, fused Lion graph, comparisons and masked memory retained')
    (output / 'report.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'triton': triton.__version__, 'gpu_execution': False,
        'scope': 'finite offline observations of the unchanged JIT body; no compiler proof',
        'cases': cases}, indent=2) + '\n')


if __name__ == '__main__':
    main()
