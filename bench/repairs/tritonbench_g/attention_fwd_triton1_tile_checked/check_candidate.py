"""Exact candidate admission and actual JIT binding; no kernel execution."""
from pathlib import Path
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import struct
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / '.veritile-cache/kernel-evidence/attention_fwd_triton1_tile_checked'
SOURCE = HERE / 'attention_fwd_triton1_tile_checked.py'
ORIGINAL = ROOT / 'bench/repairs/tritonbench_g/attention_fwd_triton1/Whole001/attention_fwd_triton1_repaired.py'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
sys.dont_write_bytecode = True
os.environ.pop('TRITON_INTERPRET', None)
os.environ['TRITON_CACHE_DIR'] = str(OUT / 'triton-cache')
os.environ['TMPDIR'] = str(OUT)
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import triton
import triton._utils
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

torch.set_num_threads(1)
assert sha(ORIGINAL) == 'a87c37e5194ed1d3415ca59d17719be62e5b4a529f3307d6d07cabbaed7d6c54'
initial = json.loads((OUT / 'initial-source-hashes.json').read_text())
assert len(initial) == 42
assert all(sha(Path(path)) == expected for path, expected in initial.items())
assert triton._utils.TRITON_MAX_TENSOR_NUMEL == 1048576
before, after = ast.parse(ORIGINAL.read_text()), ast.parse(SOURCE.read_text())
# The exact source-pair check validates these separately. Keep the historical
# original-repair comparison scoped to the retained original definitions.
after.body = [node for node in after.body if not (isinstance(node, ast.FunctionDef)
              and node.name in {"attention_fwd_shifted_kernel", "_forward_shifted_with_state"})]
original_forward = next(node for node in before.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_forward_with_state')
candidate_forward = next(node for node in after.body if isinstance(node, ast.FunctionDef)
                         and node.name == '_forward_with_state')
guard = candidate_forward.body[len(original_forward.body)-3]
# Identify the sole added statement by its literal comparison, then remove it
# for a whole-module AST comparison. Comments and locations are irrelevant.
added = [node for node in candidate_forward.body if isinstance(node, ast.If)
         and ast.unparse(node.test) == 'dim * dim > 1048576']
assert len(added) == 1
guard = added[0]
index = candidate_forward.body.index(guard)
assert ast.unparse(candidate_forward.body[index-1].test) == 'batch == 0 or heads == 0 or length == 0'
candidate_forward.body.remove(guard)
assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)

spec = importlib.util.spec_from_file_location('attention_fwd1_tile_candidate', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
function = module.attention_fwd_kernel
target = GPUTarget('cuda', 80, 32)
backend = CUDABackend(target)
binder = create_function_from_signature(function.signature, function.params, backend)
rows = []
cases = [(f'live_d{dim}', (1,1,1,dim), False, False, False)
         for dim in (16,32,64,128,256,512,1024,2048,4096)]
cases += [(f'empty_{axis}', shape, False, False, False) for axis, shape in [
    ('batch', (0,1,1,2048)), ('head', (1,0,1,2048)), ('length', (1,1,0,2048))]]
cases += [(f'tail_store{int(store)}_ifcond{int(ifcond)}', (1,1,33,32), store, ifcond, True)
          for store, ifcond in itertools.product((False,True), repeat=2)]
cases += [('tiny_scalar_one', (1,1,1,16), False, False, True)]

for name, shape, store, ifcond, compile_control in cases:
    q, k, v = [torch.zeros(shape, dtype=torch.float32) for _ in range(3)]
    snapshots = [x.clone() for x in (q,k,v)]
    launches, allocations = [], []
    actual_empty = torch.empty

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((tuple(grid),args,kwargs))
            return launch

    def allocate(*args, **kwargs):
        result = actual_empty(*args, **kwargs)
        allocations.append(result)
        return result

    rejected = None
    try:
        with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)), \
                patch.object(torch.cuda, 'device', lambda _: contextlib.nullcontext()), \
                patch.object(torch, 'empty', allocate), \
                patch.object(module, 'attention_fwd_kernel', Capture()):
            returned = module.AttentionFunction.forward(None,q,k,v,store,ifcond)
    except ValueError as error:
        rejected = str(error)
    empty = 0 in shape[:3]
    admitted = empty or shape[-1] * shape[-1] <= 1048576
    assert len(allocations) == 2
    assert all(x.device == q.device and x.dtype == torch.float32 and x.is_contiguous()
               and not x.is_neg() and not x.is_conj() for x in allocations)
    assert all(torch.equal(x, old) for x, old in zip((q,k,v), snapshots))
    assert (rejected is None) == admitted
    assert len(launches) == int(admitted and not empty)
    row = {'case': name, 'shape': shape, 'store': store, 'ifcond': ifcond,
           'empty': empty, 'admitted': admitted, 'reject_reason': rejected,
           'actual_allocated_history_shape': list(allocations[0].shape),
           'actual_allocated_output_shape': list(allocations[1].shape),
           'input_snapshots_unchanged': True, 'source_allocation_order_preserved': True,
           'compile_control': compile_control, 'output_history_on_input_device': True,
           'output_history_ordinary_contiguous_fp32': True}
    if admitted:
        assert returned is allocations[1]
        row['public_return_is_actual_output'] = True
    else:
        assert rejected == 'carry tile exceeds Triton static tensor element limit'
    if launches:
        grid, args, kwargs = launches[0]
        assert args[0] is q and args[1] is k and args[2] is v
        assert args[3] is allocations[0] and args[4] is allocations[1]
        bound, specialization, binder_options = binder(*args, **kwargs)
        _, signature, constants, attrs = function._pack_args(
            backend, binder_options, bound, specialization, binder_options)
        assert kwargs['BD'] == shape[-1] == triton.next_power_of_2(shape[-1])
        assert kwargs['BD'] * kwargs['BD'] <= 1048576
        assert args[11] == shape[-1] ** -0.5
        actual_scale_word = struct.unpack('<I', struct.pack('<f', args[11]))[0]
        row.update(actual_host_scale={'python_value': args[11], 'fp32_word': actual_scale_word,
                                     'fp32_hex': f'{actual_scale_word:08x}'},
                   grid=grid, signature=signature,
                   constants={function.arg_names[key[0]]: str(value) for key,value in constants.items()},
                   attrs=str(attrs), actual_jit_binder=True, actual_tensor_identities=True,
                   launch_options=kwargs)
        if compile_control:
            options = {'num_warps': kwargs['num_warps'], 'num_stages': kwargs['num_stages'],
                       'enable_fp_fusion': True}
            code = triton.compile(ASTSource(function,signature,constexprs=constants,attrs=attrs),
                                  target=target,options=options)
            artifacts = {}
            for kind in ('ttir','ttgir','llir','ptx'):
                path = OUT / (name + '.' + kind)
                path.write_text(code.asm[kind])
                artifacts[kind] = {'path': str(path), 'sha256': sha(path)}
            assert 'tt.dot' in code.asm['ttir'] and 'tf32' not in code.asm['ttir'].lower()
            assert 'fma.rn.f32' in code.asm['ptx'] and '.tf32' not in code.asm['ptx']
            row.update(compiled=True, artifacts=artifacts, metadata=code.metadata._asdict())
    rows.append(row)
    print(name, 'accepted-empty' if empty else 'accepted-live' if admitted else 'rejected',
          'compiled' if row.get('compiled') else '', flush=True)

report = {'task': 'attention_fwd_triton1_tile_checked',
          'source': {'path': str(SOURCE), 'sha256': sha(SOURCE)},
          'original': {'path': str(ORIGINAL), 'sha256': sha(ORIGINAL)},
          'script': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
          'sole_ast_change': ast.unparse(guard),
          'all_42_old_files_unchanged': True, 'triton_version': triton.__version__,
          'target': dataclasses.asdict(target), 'results': rows,
          'boundary': 'Actual public forward, guards, allocation and JIT binder on ordinary CPU Torch '
                      'tensors; only CUDA classification/context and final launcher are replaced. '
                      'Five tiny controls compile offline. Boundary D1024 is admission/binder only. '
                      'Static tile admission does not promise sufficient resources on every GPU.',
          'gpu_execution': False, 'interpreter_execution': False, 'numerical_experiments': []}
(OUT / 'candidate-check.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
assert len(rows) == 17 and sum(bool(row.get('compiled')) for row in rows) == 5
print('PASS: sole live-tile guard, exact boundary, empty preservation, actual binder and five small compiler controls.')
