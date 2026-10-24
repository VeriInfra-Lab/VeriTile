"""Actual selected wrapper capture and default JIT binding; offline SM80 only."""
from pathlib import Path
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'Whole001/attention_fwd_triton2_repaired.py'
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_CACHE_DIR'] = str(OUT / 'compiler-cache')
import torch
import triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('attention_fwd2_selected', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
kernel = module._attn_fwd


class CudaFacade(torch.Tensor):
    @property
    def device(self):
        return torch.device('cuda:0')


allocations = []
launches = []


def empty_like(q, dtype):
    tensor = torch.empty_like(q, dtype=dtype)
    allocations.append(tensor)
    return tensor


def empty(shape, device, dtype):
    assert device == torch.device('cuda:0')
    tensor = torch.empty(shape, dtype=dtype).as_subclass(CudaFacade)
    allocations.append(tensor)
    return tensor


class Capture:
    def __getitem__(self, grid):
        def launch(*args, **options):
            launches.append((grid, args, options))
        return launch


module.torch = SimpleNamespace(Tensor=CudaFacade, strided=torch.strided,
    float32=torch.float32, bfloat16=torch.bfloat16, empty=empty,
    empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
module._attn_fwd = Capture()
target = GPUTarget('cuda', 80, 32)
backend = CUDABackend(target)
binder = create_function_from_signature(kernel.signature, kernel.params, backend)
records = []
report = {'task': 'attention_fwd_triton2', 'source': str(SOURCE),
          'source_sha256': sha(SOURCE), 'script_sha256': sha(__file__),
          'triton': triton.__version__, 'target': dataclasses.asdict(target),
          'method': 'Actual selected wrapper on real CPU Tensor subclass with CUDA metadata/context facade and launcher capture; actual default JIT signature binder and packer; offline compilation',
          'actual_CUDA_launch': False, 'GPU_execution': False,
          'numerical_experiment': False, 'results': records}
for B, H, N in [(1, 1, 1), (1, 1, 64), (1, 1, 65), (1, 1, 129), (2, 3, 129),
                (0, 2, 64), (2, 0, 64), (2, 3, 0)]:
    name = f'B{B}_H{H}_N{N}'
    q, k, v = [torch.zeros((B, H, N, 96), dtype=torch.float32).as_subclass(CudaFacade) for _ in range(3)]
    qs = torch.ones((B, H, (N + 127) // 128), dtype=torch.float32).as_subclass(CudaFacade)
    ks = torch.ones((B, H, (N + 63) // 64), dtype=torch.float32).as_subclass(CudaFacade)
    inputs = [q, k, v, qs, ks]
    allocations.clear()
    launches.clear()
    output = module.forward(*inputs)
    assert output is allocations[0] and output.dtype == torch.bfloat16
    assert len(allocations) == 2 and allocations[1].dtype == torch.float32
    row = {'name': name, 'shape': [B, H, N, 96], 'wrapper_accepted': True,
           'output_shape': list(output.shape), 'output_dtype': str(output.dtype),
           'output_identity': output is allocations[0], 'allocations': len(allocations),
           'unused_M_shape': list(allocations[1].shape), 'launch_count': len(launches)}
    records.append(row)
    if not B * H * N:
        assert not launches
        row['status'] = 'empty_skip'
    else:
        assert len(launches) == 1
        grid, args, options = launches[0]
        assert all(args[i] is inputs[i] for i in range(5)) and args[5] is output
        bound, specialization, binder_options = binder(*args, **options)
        parsed, signature, constants, attrs = kernel._pack_args(backend, binder_options, bound, specialization, binder_options)
        compile_options = {k: v for k, v in options.items() if k in ['num_warps', 'num_stages']}
        row.update(grid=list(grid), launch_scalar_values=list(args[6:]),
                   launch_options=options, argument_names=kernel.arg_names,
                   signature=signature,
                   constants={kernel.arg_names[a[0]]: str(v) for a, v in constants.items()},
                   attrs={str(k): v for k, v in attrs.items()},
                   compiler_options=dataclasses.asdict(parsed))
        try:
            code = triton.compile(ASTSource(kernel, signature, constexprs=constants, attrs=attrs),
                                  target=target, options=compile_options)
            artifacts = {}
            for kind in ['ttir', 'ttgir', 'llir', 'ptx']:
                path = OUT / f'{name}.{kind}'
                path.write_text(code.asm[kind])
                artifacts[kind] = {'path': str(path), 'sha256': sha(path)}
            row.update(status='compiled', artifacts=artifacts, metadata=code.metadata._asdict())
        except Exception as error:
            path = OUT / f'{name}.error.txt'
            path.write_text(traceback.format_exc())
            row.update(status='compile_error', exception_type=type(error).__name__, exception=str(error),
                       error_artifact={'path': str(path), 'sha256': sha(path)})
    print(name, row['status'], flush=True)
    (OUT / 'binder-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
assert all(row['status'] in ['compiled', 'empty_skip'] for row in records)
print('PASS', len(records), 'wrapper profiles', flush=True)
