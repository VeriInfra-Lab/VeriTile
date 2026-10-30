"""Offline selected FP32 CUDA compilation; does not establish GPU numerics."""
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
    source = Path(__file__).with_name('mul_exponent_compensator_repaired.py')
    spec = importlib.util.spec_from_file_location('mec_lowering', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for sm in (80, 90):
        for block in (1, 1024):
            ast_source = ASTSource(module.mul_kernel_repaired, signature={
                'src': '*fp32', 'dst': '*fp32', 'n_elements': 'i64', 'BLOCK_SIZE': 'constexpr'},
                constexprs={'BLOCK_SIZE': block}, attrs={})
            kernel = triton.compile(ast_source, target=GPUTarget('cuda', sm, 32))
            for stage in ('ttir', 'ttgir', 'llir', 'ptx'):
                (output / f'sm{sm}-b{block}.{stage}').write_text(kernel.asm[stage])
            ptx, ttir = kernel.asm['ptx'], kernel.asm['ttir']
            memory = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+', ptx, re.M)
            assert memory and all(predicate for predicate, _ in memory), memory
            assert all(op in ('ld.global.b32', 'st.global.b32') for _, op in memory), memory
            assert re.search(r'arith\.extsi.*i32 to i64', ttir)
            assert re.search(r'arith\.cmpi slt,.*(?:i64|tensor<.*xi64>)', ttir)
            assert 'mul.f32' in ptx and 'mul.ftz.f32' not in ptx
            assert '0f77800000' in ptx and '.address_size 64' in ptx
            cases.append({'sm': sm, 'block': block, 'memory': memory, 'i64_index': True, 'explicit_ftz': False})
            print(f'SM{sm}, B={block}: widened i64 indices/mask, predicated FP32 accesses, ordered multiply by 0x77800000 passed')
    (output / 'report.json').write_text(json.dumps({'triton': triton.__version__,
        'gpu_execution': False, 'cases': cases}, indent=2) + '\n')

if __name__ == '__main__':
    main()
