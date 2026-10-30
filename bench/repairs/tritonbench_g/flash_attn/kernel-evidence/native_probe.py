"""Unchanged selected wrapper, actual native binder, offline SM90 compile. No GPU."""
import ast
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'lean-toolchain').is_file())
SOURCE = HERE.parent / 'flash_attn_repaired.py'
os.environ.pop('TRITON_INTERPRET', None)
os.environ['TRITON_CACHE_DIR'] = str(HERE / 'native-cache')
os.environ['TMPDIR'] = str(HERE / 'tmp')
def pin(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def main():
    import torch
    import triton
    import triton.language as tl
    from triton._C.libtriton import native_specialize_impl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    torch.set_num_threads(1)
    (HERE / 'tmp').mkdir(exist_ok=True)
    output = HERE / 'native'
    output.mkdir(exist_ok=True)
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel')
    module = types.ModuleType('flash_attn_actual_probe')
    module.__file__ = str(SOURCE)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module([copy.deepcopy(node)], []), str(SOURCE), 'exec'), module.__dict__)
    kernel = module._fwd_kernel
    actual = ast.parse(kernel.src).body[0]
    assert ast.dump(actual.args) == ast.dump(node.args)
    assert [ast.dump(n) for n in actual.body] == [ast.dump(n) for n in node.body]
    class Tensor:
        def __init__(self, value):
            self.value = value
            self.device = torch.device('cuda:0')
            self.is_cuda = True
        def __getattr__(self, name):
            return getattr(self.value, name)
    target = GPUTarget('cuda', 90, 32)
    backend = CUDABackend(target)
    records = []
    for causal in (False, True):
        for width in (16, 32, 64, 128):
            length, batch, heads = 129, 2, 3
            def tensor(extra):
                backing = torch.empty((batch + 1, heads + extra, length + extra, width + extra))
                return Tensor(backing[1:, :heads, :length, :width])
            q, k, v = [tensor(extra) for extra in (1, 2, 3)]
            captured, events = {}, []
            @contextlib.contextmanager
            def device(owner):
                assert owner == torch.device('cuda:0')
                events.append('device-enter')
                try:
                    yield
                finally:
                    events.append('device-exit')
            class Launch:
                def __getitem__(self, grid):
                    def call(*args, **kwargs):
                        captured.update(grid=grid, args=args, kwargs=kwargs)
                        events.append('launch')
                    return call
            def empty_like(t):
                output = Tensor(torch.empty_like(t.value))
                captured['original_o'] = output
                return output
            facade = types.SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
                empty_like=empty_like, empty=lambda shape, **_: Tensor(torch.empty(shape)),
                cuda=types.SimpleNamespace(device=device))
            nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != node.name]
            ns = {'torch': facade, 'triton': triton, 'math': math, node.name: Launch()}
            exec(compile(ast.Module(nodes, []), str(SOURCE), 'exec'), ns)
            result = ns['flash_attn_triton'](q, k, v, causal=causal, sm_scale=0.375)
            assert result is captured['original_o']
            assert captured['args'][5] is result
            assert events == ['device-enter', 'launch', 'device-exit']
            assert captured['grid'] == (2, batch * heads, 1)
            values = [x.value if isinstance(x, Tensor) else x for x in captured['args']]
            kwargs = captured['kwargs']
            bound = dict(zip(kernel.arg_names, values))
            bound.update({k: v for k, v in kwargs.items() if k in kernel.arg_names})
            assert set(bound) == set(kernel.arg_names)
            bound = {name: bound[name] for name in kernel.arg_names}
            specialized = [('constexpr', bound[p.name]) if p.is_constexpr else
                native_specialize_impl(backend, bound[p.name], p.is_const,
                    not p.do_not_specialize, not p.do_not_specialize_on_alignment)
                for p in kernel.params]
            options = {name: kwargs[name] for name in ('num_warps', 'num_stages')}
            _, signature, constants, attrs = kernel._pack_args(backend, options, bound, specialized, None)
            assert all(signature[n] == '*fp32' for n in ('Q', 'K', 'V', 'O', 'L'))
            assert signature['sm_scale'] == 'fp32'
            compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                     target=target, options=options)
            label = f'causal{int(causal)}-d{width}'
            artifacts = []
            for ext in ('ttir', 'ttgir', 'llir', 'ptx'):
                path = output / f'{label}.{ext}'
                path.write_text(compiled.asm[ext])
                artifacts.append(pin(path))
            records.append({'label': label, 'width': width, 'causal': causal,
                'length': length, 'batch': batch, 'heads': heads, 'grid': captured['grid'], 'events': events,
                'original_output_returned_and_launched': True,
                'storage_offsets': [x.storage_offset() for x in (q, k, v, result)],
                'strides': [list(x.stride()) for x in (q, k, v, result)],
                'source_arguments': {name: value for name, value in bound.items() if not isinstance(value, torch.Tensor)},
                'signature': signature, 'constants': {kernel.arg_names[key[0]]: value for key, value in constants.items()},
                'attrs': {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
                'options': options, 'metadata': {**compiled.metadata._asdict(), 'target': str(target)},
                'artifacts': artifacts, 'status': 'offline_compile_pass'})
            print(label, 'PASS', flush=True)
    report = {'source': pin(SOURCE), 'probe': pin(Path(__file__)), 'target': 'CUDA SM90',
        'triton_version': triton.__version__, 'profiles': records,
        'scope': 'Unchanged actual wrapper AST and JIT body/signature; CPU storage with CUDA metadata/context/submission facade; actual native specialization and offline compilation.',
        'GPU_execution': False, 'full_launcher_execution': False}
    (HERE / 'native-probe.json').write_text(json.dumps(report, indent=2) + '\n')
if __name__ == '__main__':
    main()
