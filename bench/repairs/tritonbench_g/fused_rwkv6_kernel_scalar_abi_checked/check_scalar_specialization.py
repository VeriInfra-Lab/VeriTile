"""Record the actual raw int-one specialization without rebuilding old controls."""
from pathlib import Path
import hashlib, json, os, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / '.veritile-cache/kernel-evidence/fused_rwkv6_kernel_scalar_abi_checked/scalar-specialization'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = str(OUT / 'cache')
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
backend = CUDABackend(GPUTarget('cuda', 80, 32))
profiles = []
for label, raw, expected in [('raw_int_one', 1, 'constexpr'), ('raw_float_one', 1.0, 'fp32')]:
    specialization = libtriton.native_specialize_impl(backend, raw, False, True, True)
    assert specialization[0] == expected, (label, specialization)
    profiles.append({'case': label, 'python_type': type(raw).__name__, 'specialization': specialization})
def pin(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
report = {'status': 'PASS', 'profiles': profiles, 'GPU_execution': False,
          'scope': 'Actual installed native scalar specializer; no tensor arithmetic or compiler rebuild.',
          'native_specializer': pin(libtriton.__file__), 'checker': pin(__file__)}
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
