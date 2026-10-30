"""Exact candidate AST and actual runtime binder; offline SM90 compilation only."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'kernel-evidence'
(OUT / 'tmp').mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(OUT / 'tmp')
os.environ['TRITON_CACHE_DIR'] = str(OUT / 'triton-cache')
os.environ.pop('TRITON_INTERPRET', None)

import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import create_function_from_signature

OLD = HERE.with_name('int8_dequant_matmul')
SOURCE = HERE / 'int8_dequant_matmul_scalar_cast.py'
old_source = OLD / 'int8_dequant_matmul_repaired.py'
spec = importlib.util.spec_from_file_location('frozen_int8_diagnostic', OLD / 'check_counterexample.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

def pin(p):
    return {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

old_tree, tree = ast.parse(old_source.read_text()), ast.parse(SOURCE.read_text())
cast_names = ['M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn', 'stride_cm', 'stride_cn']
normalized = copy.deepcopy(tree)
fn = helper.function(normalized, helper.KERNEL)
changed = []
for statement in fn.body:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        name = getattr(statement.targets[0], 'id', None)
        if name in cast_names:
            expected = ast.parse(f'{name} = tl.cast({name}, tl.int64)').body[0]
            assert ast.dump(statement) == ast.dump(expected), name
            statement.value = ast.parse(f'{name}.to(tl.int64)', mode='eval').body
            changed.append(name)
assert changed == cast_names
assert ast.dump(normalized) == ast.dump(old_tree), 'Changes beyond the nine scalar casts'
kernel, configs, _ = helper.load_kernel(SOURCE, 'int8_dequant_scalar_candidate')
assert len(configs) == 418
target = GPUTarget('cuda', 90, 32)
backend = make_backend(target)
binder = create_function_from_signature(kernel.signature, kernel.params, backend)

# Original witnesses, both branch families, zero and unit K, tails, a larger
# block, and split-zero bias. These are compiler observations, not ISA numerics.
cases = [
    ('witness_1_cfg18', 1, 1, 32, 18, False),
    ('witness_1_cfg6', 1, 1, 32, 6, False),
    ('witness_2_cfg18', 2, 2, 32, 18, False),
    ('witness_2_cfg6', 2, 2, 32, 6, False),
    ('unit_all_bias', 1, 1, 1, 18, True),
    ('zero_k_split_bias', 1, 3, 0, 19, True),
    ('tail_split_bias', 17, 33, 65, 19, True),
    ('even_split', 17, 33, 64, 19, False),
    ('split16_tail', 1, 7, 33, 22, True),
    ('bk128_tail_bias', 65, 67, 129, 13, True),
]
reports = []
for name, m, n, k, index, bias_present in cases:
    config = configs[index]
    # Nonzero storage offsets use actual CPU tensor pointer observations.
    def backed(shape, dtype):
        count = 1
        for dimension in shape:
            count *= dimension
        return torch.empty(count + 8, dtype=dtype)[4:4+count].view(shape)
    a, b = backed((m, k), torch.int8), backed((k, n), torch.int8)
    c = torch.empty((m, n), dtype=torch.float16)
    x, w = backed((m,), torch.float32), backed((n,), torch.float32)
    bias = backed((n,), torch.float32) if bias_present else None
    args = (a, b, c, bias, x, w, m, n, k, 1.0 / (127.0 * 127.0),
            int(bias_present), k, 1, n, 1, n, 1)
    launch = dict(GROUP_M=8, ACC_TYPE=tl.float32, **config.kwargs,
                  EVEN_K=k % (config.kwargs['BLOCK_K'] * config.kwargs['SPLIT_K']) == 0,
                  num_warps=config.num_warps, num_stages=config.num_stages,
                  num_ctas=config.num_ctas,
                  debug=kernel.debug or triton.knobs.runtime.debug,
                  instrumentation_mode=triton.knobs.compilation.instrumentation_mode)
    bound, specialization, options = binder(*args, **launch)
    packed, signature, constants, attrs = kernel._pack_args(
        backend, launch, bound, specialization, options)
    assert signature['stride_ak'] == signature['stride_bn'] == signature['stride_cn'] == 'constexpr'
    if m == 1:
        assert signature['M'] == 'constexpr'
    record = dict(name=name, shape=[m, n, k], config_index=index,
                  config=config.kwargs, bias_present=bias_present,
                  EVEN_K=launch['EVEN_K'], signature=signature,
                  specialization=dict(zip(kernel.arg_names, specialization)),
                  constants={str(key): value for key, value in constants.items()},
                  attrs={str(key): value for key, value in attrs.items()},
                  options=packed.__dict__)
    try:
        code = triton.compile(ASTSource(kernel, signature, constexprs=constants, attrs=attrs),
                              target=target, options=packed.__dict__)
        artifacts = []
        for ext in ['ttir', 'ttgir', 'llir', 'ptx']:
            path = OUT / (name + '.' + ext)
            path.write_text(code.asm[ext])
            artifacts.append(pin(path))
        record.update(compilation='passed', artifacts=artifacts)
    except Exception as exc:
        record.update(compilation='failed', error_type=type(exc).__name__, error=str(exc))
    reports.append(record)
    print(json.dumps({key: record[key] for key in ['name', 'shape', 'config_index', 'bias_present', 'EVEN_K', 'compilation']}, default=str), flush=True)

result = dict(source=pin(SOURCE), baseline=pin(old_source), checker=pin(Path(__file__)),
              source_audit={'only_changes': cast_names, 'full_normalized_AST_equal': True,
                            'guard_and_wrapper_AST_unchanged': True, 'configurations': 418},
              binding='actual create_function_from_signature/native_specialize_impl and JITFunction._pack_args',
              pointer_observations='CPU tensor dtype, real pointer alignment and nonzero offsets; canonical wrapper scalar values',
              triton_version=triton.__version__, target='cuda sm90 warp32',
              no_specialization_disabled=True, actual_JIT_launch=False, GPU_execution=False,
              no_host_or_primitive_numerical_experiment=True, cases=reports)
(OUT / 'binder-validation.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
assert all(report['compilation'] == 'passed' for report in reports)
