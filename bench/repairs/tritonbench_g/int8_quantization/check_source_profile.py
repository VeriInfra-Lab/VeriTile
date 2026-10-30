"""Offline selected-source precision audit; no CUDA launch or numerical oracle.

Writes compiler artifacts to --output (default: /tmp/int8-quantization-profile).
The structural Lean theorem quantifies over ordered reduction plans; this audit
checks a representative actual plan and precision/cast instructions on sm80.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/int8-quantization-profile'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget

    source_path = Path(__file__).with_name('int8_quantization_repaired.py')
    spec = importlib.util.spec_from_file_location('int8_quantization_repaired', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for query in (False, True):
        source = ASTSource(module._per_block_int8_repaired,
            signature={'X': '*fp32', 'X_int8': '*i8', 'Scale': '*fp32'},
            constexprs={'L': 5, 'C': 3, 'BLK': 4, 'BLOCKS': 2, 'BASE': 0,
                        'WIDTH': 16, 'IS_Q': query})
        kernel = triton.compile(source, target=GPUTarget('cuda', 80, 32),
            options={'num_warps': 4, 'enable_fp_fusion': False})
        ttir, ptx = kernel.asm['ttir'], kernel.asm['ptx']
        assert 'arith.extsi' in ttir and 'i32 to i64' in ttir
        assert 'arith.divf' in ttir and 'arith.maxnumf' in ttir and 'arith.minnumf' in ttir
        assert 'arith.fptosi' in ttir and 'tensor<16xi8>' in ttir
        assert 'div.full.f32' in ptx and 'add.rn.f32' in ptx
        assert 'cvt.rzi.s16.f32' in ptx and 'st.global.b8' in ptx
        assert 'fma.rn.f32' not in ptx and 'fma.rn.ftz.f32' not in ptx
        if query:
            assert 'mul.rn.f32' in ptx and '0f3F553B94' in ptx
        artifacts = {}
        for fmt in ('ttir', 'ttgir', 'ptx'):
            path = output / (('query' if query else 'key') + '.' + fmt)
            data = kernel.asm[fmt].encode()
            path.write_bytes(data)
            artifacts[fmt] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
        records.append({'query': query, 'artifacts': artifacts})
    report = {
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'triton_version': triton.__version__, 'target': 'cuda-sm80',
        'specialization': {'length': 5, 'channels': 3, 'block': 4, 'width': 16, 'warps': 4},
        'fusion': False, 'compiled': records,
        'scope': 'Offline compilation only; no GPU execution or IEEE conformance claim.'
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
