"""Exact source preservation and metadata-only boundary controls; no GPU allocation."""
import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'bmm_optimized_grid_axes_checked.py'
ORIGINAL = HERE.parent / 'bmm_optimized/bmm_optimized_repaired.py'
old = ast.parse(ORIGINAL.read_text())
new = ast.parse(SOURCE.read_text())
old_validate = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_inputs')
validate = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_inputs')
assert len(validate.body) == len(old_validate.body) + 1
assert ast.dump(ast.Module(validate.body[:-1], [])) == ast.dump(ast.Module(old_validate.body, []))
assert ast.dump(ast.Module([n for n in new.body if n is not validate], [])) == ast.dump(ast.Module([n for n in old.body if n is not old_validate], []))

class Tensor:
    is_cuda = True
    ndim = 3
    dtype = 'float32'
    layout = 'strided'
    device = 'cuda:0'
    def __init__(self, shape): self.shape = shape
    def stride(self): return (self.shape[1] * self.shape[2], self.shape[2], 1)
    def numel(self): return math.prod(self.shape)
    def data_ptr(self): return 4096
    def is_neg(self): return False
    def is_conj(self): return False

ns = {'torch': SimpleNamespace(Tensor=Tensor, float32='float32', strided='strided'),
      'triton': SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d)}
exec(compile(ast.Module([validate], []), str(SOURCE), 'exec'), ns)
old_ns = dict(ns)
exec(compile(ast.Module([old_validate], []), str(ORIGINAL), 'exec'), old_ns)
checks = []
for name, dims, admitted in [
    ('z_last', (65535, 1, 1, 1), True),
    ('z_first_rejected', (65536, 1, 1, 1), False),
    ('y_last', (1, 1, 2097120, 1), True),
    ('y_first_rejected', (1, 1, 2097121, 1), False),
    ('frozen_y_witness', (1, 1, 8388481, 1), False),
    ('k_zero_live_z', (65536, 1, 1, 0), False),
    ('k_zero_live_y', (1, 1, 2097121, 0), False),
    ('k_zero_live_last', (1, 1, 2097120, 0), True),
    ('empty_batch_large_y', (0, 1, 8388481, 0), True),
    ('empty_m_large_z', (65536, 0, 1, 0), True),
    ('empty_n_large_z', (65536, 1, 0, 0), True),
]:
    batch, m, n, k = dims
    a, b = Tensor((batch, m, k)), Tensor((batch, k, n))
    old_ns['_validate_inputs'](a, b)
    try:
        ns['_validate_inputs'](a, b)
        got = True
    except ValueError as error:
        assert str(error) == 'live grid.y and grid.z must not exceed 65535'
        got = False
    assert got == admitted, (name, got)
    checks.append(name)
# The new empty exemption does not bypass any original signed-index checks.
for space in (old_ns, ns):
    try:
        space['_validate_inputs'](Tensor((0, 0, 0)), Tensor((0, 0, 2**31)))
    except ValueError as error:
        assert str(error) == 'formed padded source addresses exceed signed32 bounds'
    else:
        raise AssertionError('original empty-domain check bypassed')
print({'status': 'PASS', 'metadata_axis_cases': len(checks), 'kernel_and_all_configs_unchanged': True,
       'only_extra_source_node': 'live y/z validation', 'old_source_sha256': hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
       'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'GPU_execution': False})
