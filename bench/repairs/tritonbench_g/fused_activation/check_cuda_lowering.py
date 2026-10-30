"""Offline CUDA source lowering for the original and bias-commuted kernels."""
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
    source = Path(__file__).with_name('fused_activation_repaired.py')
    spec = importlib.util.spec_from_file_location('fused_activation_lowering', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for name in ('fused_add_mul_activation_kernel', 'fused_add_mul_activation_commuted_kernel'):
        for sm, n, block, activation in ((80, 3, 4, 'sigmoid'), (90, 3, 4, 'sigmoid'),
                                         (90, 3, 4, 'relu'), (90, 2049, 2048, 'sigmoid')):
            ast_source = ASTSource(getattr(module, name), signature={
                'x_ptr': '*fp32', 'bias_ptr': '*fp32', 'in_ptr': '*fp32',
                'num_weights': 'constexpr', 'xnumel': 'constexpr', 'multiplier': 'constexpr',
                'activation': 'constexpr', 'BLOCK_SIZE': 'constexpr'},
                constexprs={'num_weights': 2, 'xnumel': n, 'multiplier': 0.5,
                            'activation': activation, 'BLOCK_SIZE': block})
            kernel = triton.compile(ast_source, target=GPUTarget('cuda', sm, 32),
                                    options={'enable_fp_fusion': True})
            ptx = kernel.asm['ptx']
            stem = f'{name}.sm{sm}.n{n}.{activation}'
            (output / (stem + '.ptx')).write_text(ptx)
            (output / (stem + '.llir')).write_text(kernel.asm['llir'])
            fmas = re.findall(r'fma\.rn\.f32\s+([^;]+);', ptx)
            assert fmas and all('0f3F000000' in line for line in fmas), fmas
            assert 'add.f32' in ptx and '.ftz.' not in ptx
            if activation == 'sigmoid':
                for opcode in ('sub.f32', 'mul.f32', 'ex2.approx.f32', 'div.full.f32'):
                    assert opcode in ptx, opcode
                assert '0f3FB8AA3B' in ptx and '0f3F800000' in ptx
            else:
                assert 'max.f32' in ptx
            memory = re.findall(r'^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+', ptx, re.M)
            assert memory and all(pred for pred, _ in memory), memory
            assert all(op.endswith('.b32') for _, op in memory), memory
            cases.append({'kernel': name, 'sm': sm, 'n': n, 'block': block,
                          'activation': activation, 'fma_half_preserved': True,
                          'explicit_ftz': False, 'predicated_b32_memory': True})
            print(f'{stem}: FMA-half, FP32 activation graph and masked memory retained')
    (output / 'report.json').write_text(json.dumps({
        'triton': triton.__version__, 'gpu_execution': False, 'cases': cases}, indent=2) + '\n')


if __name__ == '__main__':
    main()
