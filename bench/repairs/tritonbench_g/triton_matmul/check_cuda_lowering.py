"""Offline instruction/index audit for the two unchanged source profiles."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import sys


def run(output):
    os.environ.pop('TRITON_INTERPRET', None)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    sys.dont_write_bytecode = True
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    path = Path(__file__).with_name('triton_matmul_repaired.py')
    spec = importlib.util.spec_from_file_location('triton_matmul_lowering', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module.matmul_kernel_repaired
    output.mkdir(parents=True, exist_ok=True)
    total = 0
    for dtype, bk, stages, mma_k in [('fp16', 64, 3, 16), ('fp8e4nv', 128, 4, 32)]:
        constants = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=bk, GROUP_SIZE_M=8)
        for scalar in ['i32', 'i64']:
            signature = {name: '*' + dtype if n < 3 else 'constexpr' if name in constants else scalar
                         for n, name in enumerate(kernel.arg_names)}
            compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants),
                                     target=GPUTarget('cuda', 90, 32),
                                     options={'num_warps': 8, 'num_stages': stages})
            ttir, ptx = compiled.asm['ttir'], compiled.asm['ptx']
            name = f'{dtype}_{scalar}'
            for extension in ['ttir', 'llir', 'ptx']:
                (output / f'{name}.{extension}').write_text(compiled.asm[extension])
            assert '.address_size 64' in ptx
            assert f'k{mma_k}.f32' in ptx
            assert 'wgmma.mma_async' in ptx or 'mma.sync' in ptx
            dots = [line for line in ttir.splitlines() if '= tt.dot ' in line]
            assert dots
            for dot in dots:
                seed = re.search(r'tt.dot %\w+, %\w+, (%\w+)', dot).group(1)
                assert not seed.startswith('%cst'), dot
                assert any(seed + ' = ' in loop for loop in ttir.splitlines() if 'scf.for' in loop)
            assert 'tt.fp_to_fp' in ttir if dtype == 'fp8e4nv' else 'arith.truncf' in ttir
            assert 'i64' in ttir and 'arith.divsi' in ttir and 'arith.remsi' in ttir
            total += 1
            print(name, f'fixed K={mma_k} FP32 live carry, explicit storage cast, signed64 index source passed', flush=True)
    print(f'{total} offline SM90 compilations passed. No GPU execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/veritile-triton-matmul-lowering-audit'))
    run(parser.parse_args().output_dir)
