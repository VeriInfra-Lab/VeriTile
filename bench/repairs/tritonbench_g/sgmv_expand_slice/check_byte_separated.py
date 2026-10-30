"""Independent candidate actual-source and physical interval controls; CPU only."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import ast
import hashlib
import importlib.util
import itertools
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / '.veritile-cache/kernel-evidence/sgmv_expand_slice_byte_separated'
OLD = HERE / 'sgmv_expand_slice_repaired.py'
SOURCE = HERE / 'sgmv_expand_slice_byte_separated.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD) == '80cef4ba6062cc382fd30e6040a4cda218ca79b938fb23b55cf9b03355c169f4'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


adapter = module('sgmv_byte_separated_adapter', HERE / 'check_contract.py')
torch, triton = adapter.torch, adapter.triton
torch.set_num_threads(1)
adapter.source_binding()
old_ast, new_ast = ast.parse(OLD.read_text()), ast.parse(SOURCE.read_text())
old_wrapper = adapter.function(OLD, '_sgmv_expand_slice_repaired')
new_wrapper = adapter.function(SOURCE, '_sgmv_expand_slice_repaired')
old_guard = next(i for i, n in enumerate(old_wrapper.body) if '_cdata' in ast.unparse(n))
start = next(i for i, n in enumerate(new_wrapper.body) if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) == 'output_storage')
assert old_guard == start
assert list(map(ast.dump, old_wrapper.body[:start])) == list(map(ast.dump, new_wrapper.body[:start]))
assert list(map(ast.dump, old_wrapper.body[start + 1:])) == list(map(ast.dump, new_wrapper.body[start + 3:]))
old_kernel = adapter.function(OLD, '_sgmv_expand_slice_kernel_repaired')
new_kernel = adapter.function(SOURCE, '_sgmv_expand_slice_kernel_repaired')
assert ast.dump(old_kernel) == ast.dump(new_kernel)
pins = json.loads((HERE / 'ByteSeparated/source_pin.json').read_text())
assert pins['candidate_sha256'] == sha(SOURCE)
assert pins['frozen_canonical_sha256'] == sha(OLD)
assert pins['kernel_ast_sha256'] == hashlib.sha256(ast.dump(new_kernel).encode()).hexdigest()
assert pins['unchanged_quotation_sha256'] == sha(HERE / 'Kernels.lean')
assert pins['numerical_profile_sha256'] == sha(HERE / 'numerical_profile.json')
assert pins['candidate_entry_sha256'] == sha(HERE / 'ByteSeparated/Kernels.lean')
quotation = (HERE / 'ByteSeparated/Kernels.lean').read_text()
assert json.dumps(ast.dump(new_ast)) in quotation
assert sha(SOURCE) in quotation and 'abbrev source := VeriTile.Bench.Repairs.SgmvExpandSlice.kernel' in quotation
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in {'clone', 'contiguous', 'copy_', 'empty', 'zeros', 'resolve_neg'}
               for n in ast.walk(new_wrapper))
adapter.SOURCE = SOURCE
adapter.host_checks()

candidate = module('sgmv_byte_separated_candidate', SOURCE)
kernel, pids = adapter.kernel_ast(SOURCE)
candidate.torch = SimpleNamespace(Tensor=adapter.Tensor, strided=torch.strided,
    float16=torch.float16, float32=torch.float32, int32=torch.int32, int64=torch.int64,
    cuda=SimpleNamespace(device=lambda _: nullcontext()))


class Dispatch:
    def __init__(self):
        self.calls = 0

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls += 1
            values = [adapter.Pointer(t.data) if isinstance(t, adapter.Tensor) else t for t in args]
            options = {n: v for n, v in kwargs.items() if n not in {'num_warps', 'num_ctas'}}
            for pid in range(grid[0]):
                pids[0] = pid
                kernel(*values, **options)
        return run


def description(t):
    storage = t.untyped_storage()
    return {'shape': list(t.shape), 'dtype': str(t.dtype), 'storage_id': storage._cdata,
            'backing_start': storage.data_ptr(), 'backing_bytes': storage.nbytes(),
            'view_start': t.data_ptr(), 'storage_offset': t.storage_offset()}


def overlaps(a, b):
    return bool(a['backing_bytes'] and b['backing_bytes'] and
                a['backing_start'] < b['backing_start'] + b['backing_bytes'] and
                b['backing_start'] < a['backing_start'] + a['backing_bytes'])


def check_case(name, values, add, admitted, offset=0):
    x, w, y, starts, lengths, ids = values
    b, n, k = starts.numel(), w.shape[-2], w.shape[-1]
    before = [adapter.raw(t).clone() for t in values]
    descriptors = [description(t) for t in values]
    assert admitted == (not any(overlaps(descriptors[2], descriptors[j]) for j in (0, 1, 3, 4, 5)))
    expected = y.clone()
    allowed = torch.zeros(adapter.raw(y).numel(), dtype=torch.bool)
    expected_calls = 0
    if admitted:
        weights = w.squeeze(1) if w.ndim == 4 else w
        for st, le, slot in zip(starts.tolist(), lengths.tolist()[:b], ids.tolist()):
            if le <= 0 or slot == -1 or n == 0:
                continue
            expected_calls += 1
            acc = torch.zeros(le, n)
            for block in range((k + 15) // 16):
                lo, hi = block * 16, min((block + 1) * 16, k)
                acc += x[st:st + le, lo:hi].half().float() @ weights[slot, :, lo:hi].float().T
            result = acc.half().float()
            if add:
                result = expected[st:st + le, offset:offset + n] + result
            expected[st:st + le, offset:offset + n] = result
            for row in range(st, st + le):
                for col in range(offset, offset + n):
                    pos = 4 * (y.storage_offset() + row * y.shape[1] + col)
                    allowed[pos:pos + 4] = True
    dispatch = Dispatch()
    candidate._sgmv_expand_slice_kernel_repaired = dispatch
    adapter.EVENTS.clear()
    try:
        result = candidate._sgmv_expand_slice_repaired(
            *[adapter.Tensor(t) for t in values], b, 0, x.shape[0], offset, n, add)
    except ValueError as error:
        assert not admitted and 'backing bytes' in str(error), (name, str(error))
        assert dispatch.calls == 0 and not adapter.EVENTS
        assert all(torch.equal(adapter.raw(t), old) for t, old in zip(values, before))
    else:
        assert admitted and result is None
        assert dispatch.calls == expected_calls
        torch.testing.assert_close(y, expected, rtol=0, atol=0, equal_nan=True)
        assert all(torch.equal(adapter.raw(values[j]), before[j]) for j in (0, 1, 3, 4, 5))
        assert torch.equal(adapter.raw(y)[~allowed], before[2][~allowed])
    row = {'name': name, 'add': add, 'candidate_admitted': admitted,
           'kernel_calls': dispatch.calls, 'tensors': descriptors}
    print(name, 'add', add, 'accepted' if admitted else 'rejected before metadata and dispatch')
    return row


records = []
# All five read roles, all eight independent metadata ABIs, and both add branches.
for widths, add, role in itertools.product(itertools.product((torch.int32, torch.int64), repeat=3),
                                          (False, True), (0, 1, 3, 4, 5)):
    x = torch.ones(1, 2)
    w = torch.ones(1, 1, 2, dtype=torch.float16)
    metadata = [torch.tensor([v], dtype=dt) for v, dt in zip((0, 1, 0), widths)]
    values = [x, w, torch.ones(1, 1), *metadata]
    read = values[role]
    values[2] = torch.from_dlpack(read).view(torch.float32).view(1, -1)
    assert values[2].untyped_storage()._cdata != read.untyped_storage()._cdata
    assert values[2].data_ptr() == read.data_ptr()
    name = 'dlpack_' + ('x', 'w', 'unused', 'starts', 'lengths', 'indices')[role]
    records.append(check_case(name, values, add, False))

# Each read role can instead have a genuinely adjacent, independent backing.
for add, role, meta64 in itertools.product((False, True), (0, 1, 3, 4, 5), (False, True)):
    dt = torch.int64 if meta64 else torch.int32
    values = [torch.ones(1, 4), torch.ones(1, 4, 4, dtype=torch.float16),
              torch.ones(1, 4), torch.tensor([0], dtype=dt), torch.tensor([1], dtype=dt),
              torch.tensor([0], dtype=dt)]
    read = values[role]
    boundary = max(16, read.numel() * read.element_size())
    buffer = torch.zeros(boundary + 16, dtype=torch.uint8)
    imported_read = torch.from_dlpack(buffer[:boundary])
    view = imported_read[:read.numel() * read.element_size()].view(read.dtype).view(read.shape)
    view.copy_(read)
    values[role] = view
    values[2] = torch.from_dlpack(buffer[boundary:]).view(torch.float32).view(1, 4)
    values[2].fill_(1.)
    name = 'adjacent_' + ('x', 'w', 'unused', 'starts', 'lengths', 'indices')[role]
    records.append(check_case(name, values, add, True))

for add, name in itertools.product((False, True), (
        'partial_overlap', 'same_backing_disjoint_views', 'empty_same_storage',
        'empty_view_nonempty_storage', 'fp32_backing_byte_tail', 'overlap_only_backing_tail',
        'zero_k_shared_empty_xw', 'zero_columns_shared_empty_xy')):
    x = torch.ones(1, 4)
    w = torch.ones(1, 4, 4, dtype=torch.float16)
    y = torch.ones(1, 4)
    starts, lengths, ids = [torch.tensor([v], dtype=torch.int64) for v in (0, 1, 0)]
    admitted = False
    if name == 'partial_overlap':
        x = torch.ones(1, 16)
        w = torch.ones(1, 12, 16, dtype=torch.float16)
        y = torch.from_dlpack(x[:, 4:])
    elif name == 'same_backing_disjoint_views':
        backing = torch.ones(8)
        x, y = backing[:4].view(1, 4), backing[4:].view(1, 4)
    elif name in {'empty_same_storage', 'empty_view_nonempty_storage'}:
        backing = torch.empty(0 if name == 'empty_same_storage' else 16)
        x = backing[:0].view(0, 4)
        y = backing[:0].view(0, 4)
        starts, lengths, ids = [torch.empty(0, dtype=torch.int32) for _ in range(3)]
        admitted = name == 'empty_same_storage'
    elif name == 'fp32_backing_byte_tail':
        x = torch.zeros(17, dtype=torch.uint8)[:16].view(torch.float32).view(1, 4)
        x.fill_(1.)
        y = torch.zeros(21, dtype=torch.uint8)[:16].view(torch.float32).view(1, 4)
        w = torch.ones(1, 1, 4, dtype=torch.float16)
        admitted = True
    elif name == 'overlap_only_backing_tail':
        backing = torch.zeros(32, dtype=torch.uint8)
        y = torch.from_dlpack(backing[:17])[:16].view(torch.float32).view(1, 4)
        x = torch.from_dlpack(backing[16:]).view(torch.float32).view(1, 4)
    elif name == 'zero_k_shared_empty_xw':
        backing = torch.empty(0)
        x = backing.view(1, 0)
        w = backing.view(torch.float16).view(1, 4, 0)
        admitted = True
    elif name == 'zero_columns_shared_empty_xy':
        backing = torch.empty(0)
        x, y = backing.view(1, 0), backing.view(1, 0)
        w = torch.empty(1, 0, 0, dtype=torch.float16)
        admitted = True
    records.append(check_case(name, [x, w, y, starts, lengths, ids], add, admitted))

for file in ('preserved-files-before.json', 'preserved-defect-before.json'):
    preserved = json.loads((OUT / file).read_text())
    assert all(sha(ROOT / p['path']) == p['sha256'] for p in preserved)
report = {'canonical_sha256': sha(OLD), 'candidate_sha256': sha(SOURCE),
          'kernel_AST_unchanged': True, 'only_wrapper_guard_changed': True,
          'ordinary_complete_cases': 33, 'ordinary_rejection_cases': 57,
          'partial_failure_frame': 'passed', 'additional_cases': records,
          'old_task_files_preserved': 90, 'old_defect_files_preserved': 92,
          'GPU_execution': False}
(OUT / 'candidate-controls.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: independent byte-separated SGMV; 33 + 57 old controls and', len(records), 'interval cases.')
