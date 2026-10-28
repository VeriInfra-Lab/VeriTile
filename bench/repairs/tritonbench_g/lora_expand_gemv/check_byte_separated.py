"""Candidate-only admission and actual-source controls; no GPU conformance claim."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import ast
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / '.veritile-cache/kernel-evidence/lora_expand_gemv_byte_separated'
OLD = HERE / 'lora_expand_gemv_repaired.py'
SOURCE = HERE / 'lora_expand_gemv_byte_separated.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD) == '8f36abb103e6dbe0b646c65a8d763a7d57afdd901a9888ea398987f219e70d81'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


adapter = module('lora_byte_separated_adapter', HERE / 'check_contract.py')
torch, triton = adapter.torch, adapter.triton
torch.set_num_threads(1)
adapter.source_binding()  # Frozen original production/quotation/numerical pins.
old_ast, new_ast = ast.parse(OLD.read_text()), ast.parse(SOURCE.read_text())
old_wrapper = next(n for n in old_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_bgmv_expand')
new_wrapper = next(n for n in new_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_bgmv_expand')
old_guard = next(i for i, n in enumerate(old_wrapper.body) if '_cdata' in ast.unparse(n))
start = next(i for i, n in enumerate(new_wrapper.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'output_storage')
assert old_guard == start
assert list(map(ast.dump, old_wrapper.body[:start])) == list(map(ast.dump, new_wrapper.body[:start]))
assert list(map(ast.dump, old_wrapper.body[start + 1:])) == list(map(ast.dump, new_wrapper.body[start + 3:]))
assert ast.dump(adapter.function(OLD, '_bgmv_expand_kernel')) == ast.dump(adapter.function(SOURCE, '_bgmv_expand_kernel'))
pins = json.loads((HERE / 'ByteSeparated/source_pin.json').read_text())
assert pins['candidate_sha256'] == sha(SOURCE)
assert pins['frozen_canonical_sha256'] == sha(OLD)
assert pins['kernel_ast_sha256'] == hashlib.sha256(ast.dump(adapter.function(SOURCE, '_bgmv_expand_kernel')).encode()).hexdigest()
quotation = (HERE / 'ByteSeparated/Kernels.lean').read_text()
assert json.dumps(ast.dump(new_ast)) in quotation
assert sha(SOURCE) in quotation
# Certify all actual original/candidate bodies and their two signed index ABIs.
# An old alias spelling cannot certify a copied Python function body.
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
body_checks = check_manifest(ROOT, HERE / 'ByteSeparated/jit-correspondence.json')
assert len(body_checks) == 4 and all(row['status'] == 'PASS' and
    row['mode'] == 'pasted_body' for row in body_checks)
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in {'clone', 'contiguous', 'copy_', 'empty', 'zeros', 'resolve_neg'}
               for n in ast.walk(new_wrapper))
adapter.SOURCE = SOURCE
adapter.host_checks()  # Same 25 complete calls, 38 rejections, failure frame.

candidate = module('lora_byte_separated_candidate', SOURCE)
legacy = module('lora_byte_separated_frozen', OLD)
kernel, pids = adapter.kernel_ast(SOURCE)
cuda_adapter = SimpleNamespace(Tensor=adapter.Tensor, strided=torch.strided,
    float16=torch.float16, float32=torch.float32, int32=torch.int32, int64=torch.int64,
    cuda=SimpleNamespace(device=lambda _: nullcontext()))
candidate.torch = legacy.torch = cuda_adapter


class Dispatch:
    def __init__(self):
        self.calls = 0

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls += 1
            values = [adapter.Pointer(t.data) if isinstance(t, adapter.Tensor) else t for t in args]
            for b in range(grid[1]):
                for n in range(grid[0]):
                    pids[:] = [n, b]
                    kernel(*values, **kwargs)
        return run


def description(t):
    storage = t.untyped_storage()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'storage_id': storage._cdata,
            'backing_start': storage.data_ptr(), 'backing_bytes': storage.nbytes(),
            'view_start': t.data_ptr(), 'storage_offset': t.storage_offset()}


records = []
for name in ['input', 'weight', 'index32', 'index64', 'partial_overlap',
             'same_backing_disjoint_views', 'adjacent_imports', 'empty_same_storage',
             'empty_view_nonempty_storage', 'fp32_backing_byte_tail', 'overlap_only_backing_tail']:
    x = torch.ones(1, 2)
    w = torch.ones(1, 2, 2, dtype=torch.float16)
    i = torch.tensor([0], dtype=torch.int32)
    if name == 'input':
        y = torch.from_dlpack(x)
    elif name == 'weight':
        w = torch.tensor([[[1., 2.]]], dtype=torch.float16)
        y = torch.from_dlpack(w).view(torch.float32).view(1, 1)
    elif name in {'index32', 'index64'}:
        x = torch.ones(1, 1)
        w = torch.ones(1, 1, 1, dtype=torch.float16)
        i = torch.tensor([0], dtype=torch.int64 if name == 'index64' else torch.int32)
        y = torch.from_dlpack(i).view(torch.float32).view(1, -1)
    elif name == 'partial_overlap':
        x = torch.ones(1, 16)
        w = torch.ones(1, 12, 16, dtype=torch.float16)
        y = torch.from_dlpack(x[:, 4:])
    elif name in {'same_backing_disjoint_views', 'adjacent_imports'}:
        backing = torch.ones(8)
        x, y = backing[:4].view(1, 4), backing[4:].view(1, 4)
        if name == 'adjacent_imports':
            x, y = torch.from_dlpack(x), torch.from_dlpack(y)
        w = torch.ones(1, 4, 4, dtype=torch.float16)
    elif name == 'fp32_backing_byte_tail':
        x = torch.zeros(17, dtype=torch.uint8)[:16].view(torch.float32).view(1, 4)
        x.fill_(1.)
        y = torch.zeros(21, dtype=torch.uint8)[:16].view(torch.float32).view(1, 4)
        w = torch.ones(1, 1, 4, dtype=torch.float16)
    elif name == 'overlap_only_backing_tail':
        backing = torch.zeros(32, dtype=torch.uint8)
        y = torch.from_dlpack(backing[:17])[:16].view(torch.float32).view(1, 4)
        x = torch.from_dlpack(backing[16:]).view(torch.float32).view(1, 4)
        w = torch.ones(1, 4, 4, dtype=torch.float16)
    else:
        backing = torch.empty(0 if name == 'empty_same_storage' else 16)
        x = backing[:0].view(0, 1)
        y = backing[:0].view(0, 1)
        w = torch.ones(1, 1, 1, dtype=torch.float16)
        i = torch.empty(0, dtype=torch.int32)
    values = [x, w, y, i]
    before = [adapter.raw(t).clone() for t in values]
    descriptors = [description(t) for t in values]
    dispatch = Dispatch()
    candidate._bgmv_expand_kernel = dispatch
    adapter.EVENTS.clear()
    try:
        result = candidate._bgmv_expand(*[adapter.Tensor(t) for t in values], False)
    except ValueError as error:
        assert 'backing bytes' in str(error)
        assert dispatch.calls == 0 and not adapter.EVENTS
        assert all(torch.equal(adapter.raw(t), old) for t, old in zip(values, before))
        admitted = False
    else:
        admitted = True
        assert result is None
        assert name in {'adjacent_imports', 'empty_same_storage', 'fp32_backing_byte_tail'}
        assert dispatch.calls == int(name != 'empty_same_storage')
        assert all(torch.equal(adapter.raw(values[j]), before[j]) for j in [0, 1, 3])
        if name == 'adjacent_imports':
            assert torch.equal(y, torch.full_like(y, 4.))
        elif name == 'fp32_backing_byte_tail':
            assert torch.equal(y, torch.tensor([[4., 0., 0., 0.]]))
            assert torch.equal(adapter.raw(y)[4:], before[2][4:])
    assert admitted == (name in {'adjacent_imports', 'empty_same_storage', 'fp32_backing_byte_tail'})
    records.append({'name': name, 'candidate_admitted': admitted,
                    'kernel_calls': dispatch.calls, 'tensors': descriptors})
    print(name, 'accepted' if admitted else 'rejected before metadata and dispatch')

preserved = json.loads((OUT / 'preserved-files-before.json').read_text())
assert all(sha(ROOT / p['path']) == p['sha256'] for p in preserved)
report = {'canonical_sha256': sha(OLD), 'candidate_sha256': sha(SOURCE),
          'kernel_AST_unchanged': True, 'only_wrapper_guard_changed': True,
          'ordinary_complete_cases': 25, 'ordinary_rejection_cases': 38,
          'partial_failure_frame': 'passed', 'additional_cases': records,
          'old_files_preserved': len(preserved), 'GPU_execution': False}
(OUT / 'candidate-controls.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: independent byte-separated candidate; existing controls plus eleven physical interval cases.')
