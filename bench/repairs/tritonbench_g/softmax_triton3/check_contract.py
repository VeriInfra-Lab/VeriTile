"""Source, guard, logical Tensor, and complete masked-row checks. No GPU claim."""
import ast
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'softmax_triton3_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/softmax_triton3/softmax_triton3.py'
spec = importlib.util.spec_from_file_location('softmax_test_helpers', HERE.parent / 'softmax_triton1/check_contract.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
Tensor, Pointer, raw, function, next_power = base.Tensor, base.Pointer, base.raw, base.function, base.next_power


def source_checks():
    original, repaired = function(ORIGINAL, 'softmax_kernel'), function(SOURCE, 'softmax_kernel')
    # Integer widening is the only JIT change. The FP32 casts, optional mask
    # addition after input maximum subtraction, exp, sum and division remain.
    for statement in original.body:
        if isinstance(statement, ast.Assign) and statement.targets[0].id in ('row_idx', 'col_offsets'):
            statement.value = ast.Call(ast.Attribute(statement.value, 'to', ast.Load()),
                                       [ast.Attribute(ast.Name('tl', ast.Load()), 'int64', ast.Load())], [])
    for arg in original.args.args:
        if arg.arg in ('row_stride', 'n_cols'):
            arg.annotation = ast.Attribute(ast.Name('tl', ast.Load()), 'int64', ast.Load())
    assert ast.dump(original) == ast.dump(repaired)
    expected = ['not isinstance(x, torch.Tensor)', 'x.layout != torch.strided',
        'x.dtype != torch.float32', 'not x.is_cuda', 'x.ndim == 0',
        'not x.is_contiguous()', 'x.is_neg()', 'n and rows >= 2 ** 31',
        'n and x.data_ptr() % 4', 'n and block > 2 ** 20']
    guards = [n.test for n in function(SOURCE, '_check_tensor').body if isinstance(n, ast.If)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(s, mode='eval').body) for s in expected]
    for forbidden in ('.clone(', '.resolve_neg(', '.reshape(', '.view(', '.contiguous('):
        assert forbidden not in SOURCE.read_text()
    contract = (HERE / 'Contract.lean').read_text()
    for guard in ['x.shape == []', 'x.numel > 0 && rows x >= 2^31',
                  'x.numel > 0 && x.dataPtr.toNat % 4 != 0', 'blockSize (cols x) > 2^20',
                  'dim != -1 && dim != (x.shape.length : Int)-1', 'm.device != x.device',
                  'cols m != cols x', 'rows m != rows x']:
        assert guard in contract
    pending, visited = [HERE / 'Kernels.lean'], set()
    root = HERE.parents[3]
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        for line in source.read_text().splitlines():
            if line.startswith('import '):
                for module in line.split()[1:]:
                    assert '.Proofs.' not in module
                    if module.startswith('bench.'):
                        pending.append(root / (module.replace('.', '/') + '.lean'))


class Kernel:
    def __init__(self):
        self.calls, self.pid = [], 0
        def load(p, mask, other):
            result = torch.full_like(p.offset, other, dtype=torch.float32)
            result[mask] = raw(p.tensor)[p.tensor.storage_offset() + p.offset[mask]]
            return result
        def store(p, value, mask):
            raw(p.tensor)[p.tensor.storage_offset() + p.offset[mask]] = value[mask]
        tl = SimpleNamespace(program_id=lambda _: torch.tensor(self.pid, dtype=torch.int32),
            arange=lambda a, b: torch.arange(a, b, dtype=torch.int32), int64=torch.int64,
            float32=torch.float32, load=load, store=store, exp=torch.exp,
            max=lambda x, axis: x.max(dim=axis).values, sum=lambda x, axis: x.sum(dim=axis))
        node = function(SOURCE, 'softmax_kernel')
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        namespace = {'tl': tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(SOURCE), 'exec'), namespace)
        self.body = namespace['softmax_kernel']

    def __getitem__(self, grid):
        def launch(out, inp, stride, width, mask, *, BLOCK_SIZE, num_warps):
            self.calls.append((grid, out, inp, stride, width, mask, BLOCK_SIZE, num_warps))
            if inp.data is not None:
                for self.pid in range(grid[0]):
                    self.body(Pointer(out.data), Pointer(inp.data), stride, width,
                              Pointer(mask.data) if mask is not None else None, BLOCK_SIZE)
        return launch


def harness():
    kernel, allocations = Kernel(), []
    def empty_like(x, *, memory_format):
        assert memory_format is torch.contiguous_format
        y = Tensor(torch.full(x.shape, 73., dtype=x.dtype)) if x.data is not None else Tensor(shape=x.shape)
        allocations.append(y)
        return y
    namespace = {'torch': SimpleNamespace(Tensor=Tensor, float32=torch.float32, strided=torch.strided,
        contiguous_format=torch.contiguous_format, empty_like=empty_like,
        cuda=SimpleNamespace(device=lambda _: nullcontext())),
        'triton': SimpleNamespace(next_power_of_2=next_power), 'softmax_kernel': kernel}
    module = ast.Module([function(SOURCE, '_check_tensor'), function(SOURCE, 'softmax')], [])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return namespace['softmax'], kernel, allocations


def check_admission(inp, mask, dim):
    from math import prod
    def tensor_ok(x):
        return (isinstance(x, Tensor) and x.layout == torch.strided and x.dtype == torch.float32
            and x.is_cuda and x.ndim > 0 and x.is_contiguous() and not x.is_neg()
            and (not x.numel() or (prod(x.shape[:-1]) < 2**31 and x.data_ptr() % 4 == 0
                and max(next_power(x.shape[-1]), 2) <= 2**20)))
    return (tensor_ok(inp) and dim in (-1, inp.ndim - 1) and
        (mask is None or (tensor_ok(mask) and mask.device == inp.device
            and mask.shape[-1] == inp.shape[-1] and prod(mask.shape[:-1]) == prod(inp.shape[:-1]))))


def checks():
    softmax, kernel, allocations = harness()
    backing = torch.randn(300)
    cases = [(torch.randn(5), None, -1), (torch.randn(2, 3, 5), torch.randn(6, 5), 2),
        (torch.randn(6, 5), torch.randn(2, 3, 5), -1), (torch.randn(2, 1, 1, 3), None, 3),
        (backing[3:33].reshape(2, 3, 5), backing[7:37].reshape(6, 5), -1),
        (torch.as_strided(backing, (1, 5), (2**33, 1), 3), None, -1),
        (torch.as_strided(backing, (5, 1), (1, 2**34), 3), None, -1),
        (torch.empty(0, 3, 5), torch.empty(0, 5), -1), (torch.empty(2, 3, 0), torch.empty(6, 0), -1),
        (torch.empty(0), None, 0), (torch.as_strided(backing, (0, 5), (5, 1), 1000), None, -1),
        (torch.randn(1, 1025), torch.randn(1, 1025), -1),
        (torch.randn(1, 2049), None, -1), (torch.randn(1, 1), torch.randn(1), -1),
        (torch.tensor([[float('nan'), float('inf'), -float('inf'), -0., 1e-40]]), None, -1)]
    cases += [(base.unaligned(k, empty=True), None, -1) for k in (1, 2, 3)]
    for data, additive, dim in cases:
        x, mask = Tensor(data), None if additive is None else Tensor(additive)
        assert check_admission(x, mask, dim)
        inputs = [x] + ([] if mask is None else [mask])
        before = [bytes(t.data.untyped_storage()) for t in inputs]
        calls = len(kernel.calls)
        out = softmax(x, mask, dim)
        assert out is allocations[-1] and all(out is not t for t in inputs)
        assert out.shape == x.shape and out.dtype == torch.float32 and out.is_contiguous()
        assert before == [bytes(t.data.untyped_storage()) for t in inputs]
        logits = data if additive is None else data + additive.reshape(data.shape)
        torch.testing.assert_close(out.data, torch.softmax(logits, dim=-1), rtol=5e-6, atol=3e-7, equal_nan=True)
        if data.numel():
            grid, returned, actual, stride, width, actual_mask, block, warps = kernel.calls[-1]
            assert len(kernel.calls) == calls + 1
            assert returned is out and actual is x and actual_mask is mask
            assert stride == width == data.shape[-1] and grid == (data.numel() // width,)
            assert block == max(next_power(width), 2)
            assert warps == (16 if block >= 4096 else 8 if block >= 2048 else 4)
        else:
            assert len(kernel.calls) == calls
    x = Tensor(torch.randn(2, 3))
    other_device = Tensor(shape=(2, 3)); other_device.device = torch.device('cuda:1')
    rejects = [(object(), None, -1), (Tensor(shape=()), None, -1), (x, object(), -1),
        (x, Tensor(shape=()), -1), (x, None, 0), (x, None, -2),
        (x, Tensor(shape=(2, 4)), -1), (x, Tensor(shape=(3, 3)), -1), (x, other_device, -1),
        (Tensor(torch.randn(3, 2).t()), None, -1), (x, Tensor(torch.randn(3, 2).t()), -1),
        (Tensor(torch._neg_view(torch.ones(2, 3))), None, -1),
        (x, Tensor(torch._neg_view(torch.ones(2, 3))), -1),
        (Tensor(shape=(2**31, 1)), None, -1), (Tensor(shape=(1, 2**20+1)), None, -1)]
    for role in ('input', 'mask'):
        for kwargs in [{'dtype': torch.float16}, {'cuda': False}, {'layout': torch.sparse_coo}]:
            bad = Tensor(shape=(2, 3), **kwargs)
            rejects.append((bad, None, -1) if role == 'input' else (x, bad, -1))
        for k in (1, 2, 3):
            bad = Tensor(base.unaligned(k))
            rejects.append((bad, None, -1) if role == 'input' else (Tensor(torch.ones(2, 2)), bad, -1))
    for inp, mask, dim in rejects:
        assert not check_admission(inp, mask, dim)
        counts = len(kernel.calls), len(allocations)
        try:
            softmax(inp, mask, dim)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError('unexpected admission')
        assert counts == (len(kernel.calls), len(allocations))
    # Metadata-only checks cross the original broken branch without huge allocation.
    for shape in [(350001, 1), (2**31-1, 1), (2**20, 4096), (1, 2**20), (0, 2**40), (2**40, 0)]:
        inp = Tensor(shape=shape)
        out = softmax(inp)
        assert out.shape == shape
        if inp.numel():
            assert kernel.calls[-1][0] == (shape[0],)
    original = function(ORIGINAL, 'softmax')
    broken = [n for n in ast.walk(original) if isinstance(n, ast.Call)
              and any(k.arg == 'BLOCK_M' for k in n.keywords)]
    assert broken and any(k.arg == 'n_rows' for k in broken[0].keywords)
    # Distinguish the required subtract-then-mask ordering at large mask values.
    out = softmax(Tensor(torch.zeros(1, 2)), Tensor(torch.tensor([[1000., 1000.]]))).data
    assert torch.isnan(out).all()  # exp overflows; moving mask into max would give 0.5.
    return len(cases), len(rejects)


if __name__ == '__main__':
    source_checks()
    admitted, rejected = checks()
    base.index_checks()
    print(f'PASS: JIT/order binding; {admitted} CPU admissions; {rejected} rejections; '
          'optional masks, alias offsets, arbitrary ranks, empty axes and >350000 rows. GPU numerics untested.')
