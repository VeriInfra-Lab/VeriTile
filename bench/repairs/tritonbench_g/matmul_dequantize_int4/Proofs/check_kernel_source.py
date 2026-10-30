"""Pin source and actual JIT binder specialization; compile without execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / 'matmul_dequantize_int4_repaired.py'
SHA = '3ffe3eae0fa2022409cf3d868ffc071ffec1950d5971a815a55b279162505a57'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert sha(SOURCE) == SHA
    os.environ['TRITON_CACHE_DIR'] = str(out / 'cache')
    os.environ.pop('TRITON_INTERPRET', None)
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    from triton.runtime.jit import create_function_from_signature
    spec = importlib.util.spec_from_file_location('int4_kernel_audit', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tuner = module.matmul4_kernel
    kernel = tuner.fn
    original = SOURCE.parents[3] / 'tritonbench_g/matmul_dequantize_int4/matmul_dequantize_int4.py'
    def function(path):
        return next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'matmul4_kernel')
    assert ast.dump(function(SOURCE)) == ast.dump(function(original))
    assert len(tuner.configs) == 17
    device = SimpleNamespace(type='cuda', index=0)
    class Tensor:
        def __init__(self, shape, dtype, base):
            self.shape, self.dtype, self.base = shape, dtype, base
            self.layout, self.ndim, self.device, self.is_cuda = torch.strided, 2, device, True
        def stride(self, dim=None):
            strides = (self.shape[1], 1)
            return strides if dim is None else strides[dim]
        def is_neg(self): return False
        def is_conj(self): return False
        def is_contiguous(self): return True
        def storage_offset(self): return 0
        def data_ptr(self): return self.base
        def element_size(self): return 4 if self.dtype == torch.int32 else 2
        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.base,
                                   nbytes=lambda: self.shape[0]*self.shape[1]*self.element_size())
    helpers = [node for node in ast.parse(SOURCE.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name in
               ['_tensor_view', '_disjoint', '_check_matmul']]
    namespace = {'torch': SimpleNamespace(Tensor=Tensor, float16=torch.float16,
                                          int32=torch.int32, strided=torch.strided),
                 '_I32': 1 << 31, '_U64': 1 << 64}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), '<actual-source-guards>', 'exec'), namespace)
    target = GPUTarget('cuda', 90, 32)
    backend = CUDABackend(target)
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    records = []
    for no_groups in [False, True]:
        M, K, N, group = 1, 256, 256, 256 if no_groups else 128
        shapes = [(M, K), (K//8, N), (K//group, N), (K//group, N//8), (M, N)]
        dtypes = [torch.float16, torch.int32, torch.float16, torch.int32, torch.float16]
        tensors = [Tensor(shape, dtype, (index+1)*(1 << 20))
                   for index, (shape, dtype) in enumerate(zip(shapes, dtypes))]
        a, q, s, z, c = tensors
        assert namespace['_check_matmul'](a, q, s, z, group, c) == (M, N, K)
        arguments = [a, q, c, s, z, M, N, K, *a.stride(), *q.stride(),
                     *c.stride(), *s.stride(), *z.stride(), group, no_groups]
        for index in [0, 15]:
            config = tuner.configs[index]
            kwargs = {**config.kwargs, 'num_warps': config.num_warps, 'num_stages': config.num_stages}
            bound, specialization, options = binder(*arguments, **kwargs)
            parsed, signature, constants, attributes = kernel._pack_args(
                backend, kwargs, bound, specialization, options)
            assert signature['M'] == 'constexpr'
            for name in ['stride_ak', 'stride_bn', 'stride_cn', 'stride_scales_n', 'stride_zeros_n']:
                assert signature[name] == 'constexpr'
            compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants,
                                               attrs=attributes), target=target,
                                      options=vars(parsed))
            artifacts = {}
            for extension in ['ttir', 'ttgir', 'llir', 'ptx']:
                path = out / f'config{index}-nogroups{int(no_groups)}.{extension}'
                path.write_text(compiled.asm[extension])
                artifacts[extension] = {'path': str(path), 'sha256': sha(path)}
            records.append({'config_index': index, 'shape_M_K_N': [M, K, N],
                            'group_size': group, 'NO_GROUPS': no_groups,
                            'actual_source_guards': 'accepted', 'signature': signature,
                            'specialization': {name: list(value) for name, value in
                                               zip(kernel.arg_names, specialization)},
                            'constants': {str(key): value for key, value in constants.items()},
                            'tuning': kwargs, 'artifacts': artifacts})
    report = {'source': {'path': str(SOURCE), 'sha256': sha(SOURCE)},
              'original_jit_ast_and_all_17_tunings_identical': True,
              'original': {'path': str(original), 'sha256': sha(original)},
              'triton_version': triton.__version__, 'target': 'sm90',
              'binding': 'Actual create_function_from_signature and JITFunction._pack_args; fake metadata-only Tensor pointers',
              'compiled_cases': records, 'tensor_allocations': 0, 'gpu_runs': 0,
              'numerical_experiments': 0, 'script_sha256': sha(__file__)}
    (out / 'source-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_JIT_compilations': len(records), 'scalar_one_and_unit_strides': 'constexpr',
                      'both_group_branches': True, 'GPU_runs': 0}))

if __name__ == '__main__':
    main()
