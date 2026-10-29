"""Original binder and frontend at actual dynamic block/numel boundaries."""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys
import traceback

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
SOURCE = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/cosine_compute/cosine_compute.py')
work = PACKAGE / 'extent-frontend-attempts' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work.mkdir(parents=True)
os.environ['TRITON_CACHE_DIR'] = str(work / 'cache')
import torch
import triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir

original = SOURCE.read_text()
prefix = original.split('################################################################')[0]
nodes = lambda s: [ast.dump(n) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name in ['cos_func', 'cos']]
assert nodes(original) == nodes(prefix)
(work / 'source.py').write_text(prefix)
(work / 'probe_extent_frontend.py').write_bytes(Path(__file__).read_bytes())
spec = importlib.util.spec_from_file_location('cosine_extent_source', work / 'source.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
kernel = module.cos_func
target = GPUTarget('cuda', 90, 32)
backend = CUDABackend(target)
names = ['bool', 'uint8', 'int8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
         'float16', 'bfloat16', 'float32', 'float64', 'float8_e4m3fn', 'float8_e5m2',
         'float8_e4m3fnuz', 'float8_e5m2fnuz', 'complex32', 'complex64', 'complex128',
         'float8_e8m0fnu', 'float4_e2m1fn_x2']
extents = [1, 2**31, 2**40, 2**40+1, 2**52+1, 2**52+2, 2**63-1]
rows = []
for n in extents:
    block = triton.next_power_of_2(math.ceil(math.sqrt(n)))
    for name in names:
        row = {'N': n, 'BLOCK_SIZE': block, 'grid': triton.cdiv(n, block), 'dtype': name}
        stage = 'signature'
        try:
            dtype = getattr(torch, name)
            binder = jit.create_function_from_signature(kernel.signature, kernel.params, backend)
            params, specialization, options = binder(jit.MockTensor(dtype), jit.MockTensor(dtype), n, block)
            options, signature, constexprs, attrs = kernel._pack_args(backend, options, params, specialization, options)
            assert signature['n_elements'] == 'i32' and constexprs == {(3,): block}
            row.update(signature=signature, specialization=specialization,
                       constexprs={str(k): v for k, v in constexprs.items()}, options=vars(options))
            stage = 'frontend'
            src = ASTSource(kernel, signature, constexprs, attrs)
            context = ir.context()
            ir.load_dialects(context)
            backend.load_dialects(context)
            raw = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
            artifact = f'{n}-{name}.ttir'
            (work / artifact).write_text(str(raw))
            row.update(status='frontend_passed', artifact=artifact)
        except Exception:
            row.update(status='rejected', failure_stage=stage, exception=traceback.format_exc())
        rows.append(row)
    print(n, block, {status: sum(r['status'] == status for r in rows[-22:])
                     for status in ('frontend_passed', 'rejected')}, flush=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {'kind': 'cosine_original_extent_frontend_probe_not_acceptance',
           'source': str(SOURCE), 'source_sha256': sha(SOURCE), 'source_ast_unchanged': True,
           'torch_version': torch.__version__, 'triton_version': triton.__version__,
           'target': {'backend': 'cuda', 'arch': 90, 'warp_size': 32}, 'rows': rows,
           'gpu_launches': 0, 'large_tensor_allocations': 0, 'backend_compiles': 0,
           'artifacts': {str(p.relative_to(work)): sha(p) for p in work.iterdir() if p.is_file()}}
path = work / 'receipt.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': str(path), 'sha256': sha(path), 'profiles': len(rows)}))
