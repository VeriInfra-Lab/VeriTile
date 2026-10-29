"""Inspect both literal-add10 FP32 CUDA lowerings offline; does not execute the GPU."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source = Path(__file__).with_name('add_value_repaired.py')
    spec = importlib.util.spec_from_file_location('add_value_lowering', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for name in ('puzzle1_kernel', 'puzzle1_commuted_kernel'):
        for sm in (80, 90):
            ast_source = ASTSource(getattr(module, name), signature={
                'x_ptr': '*fp32', 'output_ptr': '*fp32', 'N': 'i32',
                'BLOCK_SIZE': 'constexpr'},
                constexprs={'BLOCK_SIZE': 1024}, attrs={})
            kernel = triton.compile(ast_source, target=GPUTarget('cuda', sm, 32))
            for stage in ('ttir', 'ttgir', 'llir', 'ptx'):
                (output / f'{name}.sm{sm}.{stage}').write_text(kernel.asm[stage])
            signature = next(line for line in kernel.asm['ttir'].splitlines()
                             if 'tt.func public' in line)
            assert re.findall(r'%(\w+):', signature) == ['x_ptr', 'output_ptr', 'N'], signature
            ptx = kernel.asm['ptx']
            memory = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+', ptx, re.M)
            assert memory and all(predicate for predicate, _ in memory), memory
            assert all(op in ('ld.global.b32', 'st.global.b32') for _, op in memory), memory
            assert re.search(r'setp\.lt\.s32', ptx)
            assert re.search(r'mul\.wide\.s32\s+%rd\d+,\s*%r\d+,\s*4;', ptx)
            assert re.search(r'add\.f32', ptx) and not re.search(r'add\.ftz\.f32', ptx)
            assert '.address_size 64' in ptx
            cases.append({'kernel': name, 'sm': sm, 'memory': memory,
                          'fp32_add': True, 'explicit_ftz': False})
            print(f'{name} SM{sm}: predicated scalar b32 loads/stores, signed offsets and FP32 addition passed')
    (output / 'report.json').write_text(json.dumps({
        'triton': triton.__version__, 'block_size': 1024, 'addend_literal': 10,
        'runtime_parameters': ['x_ptr', 'output_ptr', 'N'],
        'pointer_alignment_hints': False, 'gpu_execution': False, 'cases': cases}, indent=2) + '\n')


if __name__ == '__main__':
    main()
