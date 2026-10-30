"""Actual checked source wrapper and native binder; offline SM90 artifacts only."""
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'bench/repairs/tritonbench_g/context_attn_fwd/context_attn_fwd_repaired.py'
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
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fwd_kernel_int8kv')
    module = types.ModuleType('context_fwd_actual_probe')
    module.__file__ = str(SOURCE)
    module.__dict__.update(torch=torch, triton=triton, tl=tl)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module([copy.deepcopy(node)], []), str(SOURCE), 'exec'), module.__dict__)
    kernel = module._fwd_kernel_int8kv
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
    for block in (64, 128):
        for width in (16, 32, 64, 128, 256):
            length, prompt, batch, heads, kv_heads = block + 1, 7, 2, 4, 2
            rows = batch * (length + 2)
            sk, sv = length + prompt + 2, length + prompt + 5
            q = torch.zeros(rows, heads, width)
            o = torch.zeros(rows + 2, heads, width)[1:-1]
            k = torch.zeros(batch, kv_heads + 1, sk, width)[:, :kv_heads]
            v = torch.zeros(batch, kv_heads + 2, sv, width)[:, :kv_heads]
            starts = torch.tensor([1, length + 3], dtype=torch.int32)
            sequences = torch.tensor([length + prompt, length + prompt], dtype=torch.int32)
            prompts = torch.tensor([prompt, prompt], dtype=torch.int32)
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

            facade = types.SimpleNamespace(Tensor=Tensor, strided=torch.strided,
                float32=torch.float32, int32=torch.int32,
                cuda=types.SimpleNamespace(device=device,
                    get_device_name=lambda _: 'Tesla profile' if block == 64 else 'SM90 profile'))
            nodes = []
            for n in tree.body:
                if isinstance(n, ast.Assign):
                    nodes.append(copy.deepcopy(n))
                elif isinstance(n, ast.FunctionDef) and n.name != node.name:
                    n = copy.deepcopy(n)
                    n.decorator_list = []
                    nodes.append(n)
            ns = {'torch': facade, 'triton': triton, node.name: Launch()}
            exec(compile(ast.Module(nodes, []), str(SOURCE), 'exec'), ns)
            assert ns['context_attention_fwd_ppl_int8kv'](
                *[Tensor(x) for x in (q, k, v, o, starts, sequences)], length + 3, Tensor(prompts)) is None
            assert events == ['device-enter', 'launch', 'device-exit']
            assert captured['grid'] == (2, batch * heads, 1)
            values = [x.value if isinstance(x, Tensor) else x for x in captured['args']]
            kwargs = {name: (value.value if isinstance(value, Tensor) else value)
                      for name, value in captured['kwargs'].items()}
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
            assert signature['sm_scale'] == 'fp32'
            assert signature['stride_qd'] == 'constexpr'
            compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs),
                                     target=target, options=options)
            label = f'b{block}-d{width}'
            artifacts = []
            for ext in ('ttir', 'ttgir', 'llir', 'ptx'):
                path = output / f'{label}.{ext}'
                path.write_text(compiled.asm[ext])
                artifacts.append(pin(path))
            records.append({'label': label, 'width': width, 'block': block,
                'uncached_length': length, 'prompt': prompt, 'batch': batch,
                'heads': heads, 'kv_heads': kv_heads, 'grid': captured['grid'], 'events': events,
                'output_storage_offset': o.storage_offset(), 'cache_shapes': [list(k.shape), list(v.shape)],
                'cache_strides': [list(k.stride()), list(v.stride())],
                'source_arguments': {name: value for name, value in bound.items()
                    if not isinstance(value, torch.Tensor)}, 'signature': signature,
                'constants': {kernel.arg_names[key[0]]: value for key, value in constants.items()},
                'attrs': {kernel.arg_names[key[0]]: value for key, value in attrs.items()},
                'options': options, 'metadata': {**compiled.metadata._asdict(), 'target': str(target)},
                'artifacts': artifacts, 'status': 'offline_compile_pass'})
            print(label, 'PASS', flush=True)
    report = {'source': pin(SOURCE), 'probe': pin(Path(__file__)), 'target': 'CUDA SM90',
        'triton_version': triton.__version__, 'profiles': records,
        'scope': 'Unchanged actual wrapper AST and JIT body/signature; CPU storage with CUDA metadata/context/submission facade; normal native specialization; offline compilation only.',
        'GPU_execution': False, 'full_launcher_execution': False}
    (HERE / 'native-probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
