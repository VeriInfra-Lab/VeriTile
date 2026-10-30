"""Targeted source, host, layout and finite-index checks. No GPU claim."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'softmax_triton1_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/softmax_triton1/softmax_triton1.py'


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_checks():
    import inspect, triton
    helper = ast.parse(inspect.getsource(triton.next_power_of_2.fn)).body[0]
    shifts = [node.value.right.value for node in helper.body
              if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitOr)]
    assert shifts == [1, 2, 4, 8, 16, 32]
    for exponent in range(21):
        for n in {0, max(0, 2**exponent-1), 2**exponent, 2**exponent+1}:
            assert triton.next_power_of_2(n) == next_power(n)
    original, repaired = function(ORIGINAL, 'softmax_kernel'), function(SOURCE, 'softmax_kernel')
    # Only widened integer indexing and explicit column strides changed.
    expected = ast.parse("""
row_idx = tl.program_id(0).to(tl.int64)
row_start_ptr = input_ptr + row_idx * input_row_stride
col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
input_ptrs = row_start_ptr + col_offsets * input_col_stride
""").body
    expected += original.body[4:-2]
    expected += ast.parse("output_ptrs = output_row_start_ptr + col_offsets * output_col_stride").body
    expected += original.body[-1:]
    assert [ast.dump(n) for n in repaired.body] == [ast.dump(n) for n in expected]
    assert [a.arg for a in repaired.args.args] == ['output_ptr', 'input_ptr', 'input_row_stride',
        'input_col_stride', 'output_row_stride', 'output_col_stride', 'n_cols', 'BLOCK_SIZE']
    assert all(ast.unparse(a.annotation) == 'tl.int64' for a in repaired.args.args[2:-1])
    expected = [
        'not isinstance(x, torch.Tensor)', 'x.layout != torch.strided',
        'x.dtype != torch.float32', 'not x.is_cuda', 'x.ndim != 2',
        'not x.is_contiguous()', 'x.is_neg()', 'n and n_rows >= 2 ** 31',
        'n and x.data_ptr() % 4', 'n == 0', 'BLOCK_SIZE > 2 ** 20',
        'BLOCK_SIZE >= 2048', 'BLOCK_SIZE >= 4096',
    ]
    guards = [n.test for n in function(SOURCE, 'softmax').body if isinstance(n, ast.If)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(s, mode='eval').body) for s in expected]
    contract = (HERE / 'Contract.lean').read_text()
    for text in ['x.layout != .strided', 'x.dtype != .floating .fp32', '!x.isCuda',
                 'x.shape.length != 2', '!x.isContiguous', 'x.isNeg', 'x.numel > 0 && rows x >= 2^31',
                 'x.numel > 0 && x.dataPtr.toNat % 4 != 0',
                 'x.numel > 0 && block (cols x) > 2^20']:
        assert text in contract
    root = HERE.parents[3]
    pending, visited = [HERE / 'Kernels.lean'], set()
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


def next_power(n):
    n -= 1
    for shift in (1, 2, 4, 8, 16, 32):
        n |= n >> shift
    return n + 1


class Tensor:
    def __init__(self, data=None, *, shape=None, cuda=True, dtype=torch.float32,
                 contiguous=True, negative=False, pointer=4096, layout=torch.strided):
        self.data = data
        self.shape = tuple(data.shape) if data is not None else tuple(shape)
        self.dtype = data.dtype if data is not None else dtype
        self.layout = data.layout if data is not None else layout
        self.is_cuda = cuda
        self.device = torch.device('cuda:0')
        self._contiguous, self._negative, self._pointer = contiguous, negative, pointer

    @property
    def ndim(self): return len(self.shape)
    def is_contiguous(self): return self.data.is_contiguous() if self.data is not None else self._contiguous
    def is_neg(self): return self.data.is_neg() if self.data is not None else self._negative
    def numel(self):
        from math import prod
        return prod(self.shape)
    def data_ptr(self):
        assert self.numel() > 0, 'empty inputs must not query data_ptr'
        return self.data.data_ptr() if self.data is not None else self._pointer


def raw(t):
    return torch.empty(0, dtype=torch.float32).set_(t.untyped_storage(), 0,
            (t.untyped_storage().nbytes() // 4,), (1,))


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, torch.as_tensor(offset, dtype=torch.int64)
    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + torch.as_tensor(offset, dtype=torch.int64))


class Kernel:
    def __init__(self):
        self.calls, self.pid = [], 0
        def load(p, mask, other):
            result = torch.full_like(p.offset, other, dtype=torch.float32)
            indices = p.tensor.storage_offset() + p.offset[mask]
            result[mask] = raw(p.tensor)[indices]
            return result
        def store(p, value, mask):
            indices = p.tensor.storage_offset() + p.offset[mask]
            raw(p.tensor)[indices] = value[mask]
        tl = SimpleNamespace(program_id=lambda _: torch.tensor(self.pid, dtype=torch.int32),
                             arange=lambda a, b: torch.arange(a, b, dtype=torch.int32),
                             load=load, store=store, int64=torch.int64,
                             max=lambda x, axis: x.max(dim=axis).values,
                             exp=torch.exp, sum=lambda x, axis: x.sum(dim=axis))
        node = function(SOURCE, 'softmax_kernel')
        node.decorator_list = []
        for a in node.args.args:
            a.annotation = None
        namespace = {'tl': tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(SOURCE), 'exec'), namespace)
        self.body = namespace['softmax_kernel']

    def __getitem__(self, grid):
        def launch(y, x, si, ci, so, co, n, *, num_warps, BLOCK_SIZE):
            self.calls.append((grid, y, x, si, ci, so, co, n, num_warps, BLOCK_SIZE))
            if x.data is not None:
                for self.pid in range(grid[0]):
                    self.body(Pointer(y.data), Pointer(x.data), si, ci, so, co, n, BLOCK_SIZE)
        return launch


def harness():
    kernel, allocations = Kernel(), []
    def empty_like(x, *, memory_format):
        assert memory_format is torch.contiguous_format
        if x.data is None:
            y = Tensor(shape=x.shape)
        else:
            y = Tensor(torch.full(x.shape, 73., dtype=x.dtype))
        allocations.append(y)
        return y
    module = ast.fix_missing_locations(ast.Module([function(SOURCE, 'softmax')], []))
    namespace = {'torch': SimpleNamespace(Tensor=Tensor, float32=torch.float32,
        strided=torch.strided, contiguous_format=torch.contiguous_format,
        empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: nullcontext())),
        'triton': SimpleNamespace(next_power_of_2=next_power), 'softmax_kernel': kernel}
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace['softmax'], kernel, allocations


def unaligned(offset, empty=False):
    backing = torch.arange(40, dtype=torch.uint8)
    storage = backing.untyped_storage()[offset:offset+32]
    return torch.empty(0, dtype=torch.float32).set_(storage, 0,
                (0, 2) if empty else (2, 2), (2, 1))


def host_checks():
    softmax, kernel, allocations = harness()
    backing = torch.arange(40, dtype=torch.float32)
    cases = [torch.randn(3, 5), torch.randn(2, 64), backing[3:15].reshape(3, 4),
             torch.as_strided(backing, (1, 4), (2**33, 1), 3),
             torch.as_strided(backing, (4, 1), (1, 2**33), 3),
             torch.as_strided(backing, (1, 1), (2**33, 2**34), 3),
             torch.empty(0, 5), torch.empty(3, 0), torch.empty(0, 0),
             torch.as_strided(backing, (0, 2), (2, 1), 1000),
             torch.tensor([[float('nan'), float('inf'), -float('inf'), -0., 1.e-40, 1.e30]]),
             torch.randn(2, 1025), torch.randn(2, 2049)]
    cases += [unaligned(k, empty=True) for k in (1, 2, 3)]
    for data in cases:
        x = Tensor(data)
        before = bytes(data.untyped_storage())
        calls = len(kernel.calls)
        y = softmax(x)
        assert y is allocations[-1] and y is not x
        assert y.shape == x.shape and y.dtype == torch.float32
        assert y.is_contiguous()
        assert bytes(data.untyped_storage()) == before
        torch.testing.assert_close(y.data, torch.softmax(data, dim=-1), rtol=3e-6, atol=2e-7, equal_nan=True)
        if data.numel():
            assert len(kernel.calls) == calls + 1
            grid, returned, actual, si, ci, so, co, n, warps, block = kernel.calls[-1]
            assert returned is y and actual is x
            assert grid == (data.shape[0],) and si == so == n == data.shape[1]
            assert ci == co == 1
            assert block == next_power(n)
            assert warps == (16 if block >= 4096 else 8 if block >= 2048 else 4)
        else:
            assert len(kernel.calls) == calls
    rejected = [object(), Tensor(shape=(2, 2), layout=torch.sparse_coo),
                Tensor(shape=(2, 2), dtype=torch.float16), Tensor(shape=(2, 2), cuda=False),
                Tensor(shape=()), Tensor(shape=(4,)), Tensor(shape=(2, 2, 1)),
                Tensor(torch.arange(6.).reshape(2, 3).t()),
                Tensor(torch._neg_view(torch.ones(2, 2))), Tensor(shape=(2**31, 1)),
                Tensor(shape=(1, 2**20+1))] + [Tensor(unaligned(k)) for k in (1, 2, 3)]
    for x in rejected:
        counts = len(kernel.calls), len(allocations)
        try:
            softmax(x)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(('unexpected admission', x))
        assert counts == (len(kernel.calls), len(allocations))
    for shape in [(2**31-1, 1), (2**20, 4096), (1, 2**20), (0, 2**40), (2**40, 0)]:
        x = Tensor(shape=shape)
        y = softmax(x)
        assert y.shape == shape
    # Original unit-column addressing leaves a transposed output cell unwritten.
    data = torch.tensor([[1., 2.], [3., 4.]]).t()
    output = torch.full_like(data, 73.)
    for kernel.pid in range(2):
        kernel.body(Pointer(output), Pointer(data), data.stride(0), 1, output.stride(0), 1, 2, 2)
    assert not torch.allclose(output, torch.softmax(data, dim=-1))
    assert raw(output)[3] == 73.
    # A stride-two input gives different logical values from unit offsets.
    data = torch.arange(12.).reshape(2, 6)[:, ::2]
    output = torch.full(data.shape, 73.)
    for kernel.pid in range(2):
        kernel.body(Pointer(output), Pointer(data), data.stride(0), 1, 3, 1, 3, 4)
    assert not torch.allclose(output, torch.softmax(data, dim=-1))
    return len(cases), len(rejected)


def signed32(x):
    x %= 2**32
    return x - 2**32 if x >= 2**31 else x


def index_checks():
    assert signed32(2**30 * 2) == -(2**31)
    for rows, cols in [(2**31-1, 1), (2**20, 4096), (7, 1025), (2, 2049)]:
        block = next_power(cols)
        for r in {0, rows-1}:
            for c in {0, cols-1, block-1}:
                # The repaired source widens each PID/lane before multiplication.
                assert signed32(r) * cols == r * cols
                assert (signed32(c) < cols) == (c < cols)
                if c < cols:
                    assert 4096+4*(signed32(r)*cols+signed32(c)) == 4096+4*(r*cols+c)
                    assert r*cols+c < rows*cols


if __name__ == '__main__':
    source_checks()
    admitted, rejected = host_checks()
    index_checks()
    print(f'PASS: source binding; {admitted} CPU Tensor admissions; {rejected} rejections; '
          'metadata limits; transpose and overflow counterexamples. GPU arithmetic untested.')
