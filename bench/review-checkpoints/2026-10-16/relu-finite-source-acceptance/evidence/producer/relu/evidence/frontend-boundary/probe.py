import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch
import triton
import triton.language as tl
import triton.language.semantic as semantic
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = ['bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
         'float16', 'bfloat16', 'float32', 'float64', 'float8_e4m3fn', 'float8_e5m2',
         'float8_e4m3fnuz', 'float8_e5m2fnuz']
node = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'relu_kernel')
lines = SOURCE.read_text().splitlines(keepends=True)
text = 'import triton\nimport triton.language as tl\n' + ''.join(lines[min([node.lineno] + [n.lineno for n in node.decorator_list]) - 1:node.end_lineno])
extract = OUT / 'extracted_relu_kernel.py'
extract.write_text(text)
other = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
assert ast.dump(node, include_attributes=False) == ast.dump(other, include_attributes=False)
spec = importlib.util.spec_from_file_location('relu_original_dtype_audit', extract)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
target = GPUTarget('cuda', 90, 32)
backend = CUDABackend(target)
options = backend.parse_options({'num_warps': 4})
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)
results = []
for N, name in ((n, d) for n in [2**31, 2**31+1, 2**32-1, 2**32] for d in names):
    signature = {'x_ptr': '*' + jit.type_canonicalisation_dict[name], 'out_ptr': '*fp32'}
    row = {'N':N, 'input': name, 'output': 'float32', 'signature': signature}
    try:
        source = ASTSource(module.relu_kernel, signature, constexprs={'N': N, 'block_size': 1024})
        generated = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
        ttir = str(generated)
        path = OUT / (str(N) + '-' + name + '.raw.ttir')
        path.write_text(ttir)
        row.update(status='frontend_generated', ttir=path.name, ttir_sha256=sha(path),
                   semantic_lines=[l.strip() for l in ttir.splitlines() if any(s in l for s in ['cmpf', 'cmpi', 'select', 'extf', 'truncf', 'sitofp', 'uitofp', 'fp_to_fp', 'tt.store'])])
    except Exception as exc:
        row.update(status='frontend_error', error_type=type(exc).__name__, error=str(exc))
    results.append(row)
    print(N, name, row['status'], flush=True)
(OUT / 'frontend-profiles.json').write_text(json.dumps(results, indent=2) + '\n')
methods = ['computation_type_impl', 'greater_equal', 'where', 'cast', '_store_legacy']
(OUT / 'semantic-excerpts.txt').write_text('\n'.join(inspect.getsource(getattr(semantic.TritonSemantic, m)) for m in methods))
receipt = {
    'source': str(SOURCE), 'source_sha256': sha(SOURCE), 'extracted_kernel_sha256': sha(extract),
    'kernel_ast_identical': True,
    'environment': {'python': sys.version, 'torch': torch.__version__, 'triton': triton.__version__,
                    'semantic_path': semantic.__file__, 'semantic_sha256': sha(semantic.__file__),
                    'jit_path': jit.__file__, 'jit_sha256': sha(jit.__file__)},
    'target': {'backend': 'cuda', 'arch': 90, 'warp_size': 32}, 'options': vars(options),
    'constexprs': {'N_tested': [2**31,2**31+1,2**32-1,2**32], 'block_size': 1024},
    'canonical_profiles': len(names), 'profile_config_pairs':len(results), 'frontend_generated': sum(r['status'] == 'frontend_generated' for r in results),
    'original_module_imported': False, 'native_backend_compiled': False, 'gpu_executed': False,
    'dtype_canonicalization_inventory': {str(v): jit.type_canonicalisation_dict.get(str(v).split('.')[-1]) for v in vars(torch).values() if isinstance(v, torch.dtype)},
    'artifacts': {p.name: sha(p) for p in [Path(__file__), extract, OUT / 'frontend-profiles.json', OUT / 'semantic-excerpts.txt']},
}
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('receipt', sha(OUT / 'receipt.json'))
