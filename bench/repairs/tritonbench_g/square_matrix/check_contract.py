"""Both actual kernel variants, contracts, layouts and finite indices. No GPU claim."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'square_matrix_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/square_matrix/square_matrix.py'


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
    assert ast.dump(function(SOURCE, 'square_kernel')) == ast.dump(function(ORIGINAL, 'square_kernel'))
    expected_variant = function(ORIGINAL, 'square_kernel')
    expected_variant.name = 'square_index_commuted_kernel'
    product = expected_variant.body[1].value.right
    assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
    product.left, product.right = product.right, product.left
    assert ast.dump(function(SOURCE, expected_variant.name)) == ast.dump(expected_variant)
    expected = [
        'not isinstance(x, torch.Tensor)', 'x.layout != torch.strided',
        'x.dtype != torch.float32', 'not x.is_cuda', 'x.ndim != 2',
        'not x.is_contiguous()', 'x.is_neg()', 'n >= 2 ** 31',
        'n and x.data_ptr() % 4', 'n == 0', 'BLOCK_SIZE > 2 ** 20',
        'BLOCK_SIZE >= 2048', 'BLOCK_SIZE >= 4096',
    ]
    guards = [n.test for n in function(SOURCE, 'square').body if isinstance(n, ast.If)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(s, mode='eval').body) for s in expected]
    contract = (HERE / 'Contract.lean').read_text()
    for text in ['x.layout != .strided', 'x.dtype != .floating .fp32', '!x.isCuda',
                 'x.shape.length != 2', '!x.isContiguous', 'x.isNeg', 'x.numel >= 2^31',
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
    for entry in ('Correct.lean', 'FPEquiv.lean'):
        text = (HERE / entry).read_text()
        assert 'runWrapper' not in text and 'LocalFP32Square' not in text
        assert '.Proofs.Host' not in text and '.Proofs.SourceBinding' not in text


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
    def __init__(self, name='square_kernel'):
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
                             load=load, store=store)
        node = function(SOURCE, name)
        node.decorator_list = []
        for a in node.args.args:
            a.annotation = None
        namespace = {'tl': tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(SOURCE), 'exec'), namespace)
        self.body = namespace[name]

    def __getitem__(self, grid):
        def launch(y, x, si, so, n, *, num_warps, BLOCK_SIZE):
            self.calls.append((grid, y, x, si, so, n, num_warps, BLOCK_SIZE))
            if x.data is not None:
                for self.pid in range(grid[0]):
                    self.body(Pointer(y.data), Pointer(x.data), si, so, n, BLOCK_SIZE)
        return launch


def harness(name='square_kernel'):
    kernel, allocations = Kernel(name), []
    def empty_like(x, *, memory_format):
        assert memory_format is torch.contiguous_format
        if x.data is None:
            y = Tensor(shape=x.shape)
        else:
            y = Tensor(torch.full(x.shape, 73., dtype=x.dtype))
        allocations.append(y)
        return y
    module = ast.fix_missing_locations(ast.Module([function(SOURCE, 'square')], []))
    namespace = {'torch': SimpleNamespace(Tensor=Tensor, float32=torch.float32,
        strided=torch.strided, contiguous_format=torch.contiguous_format,
        empty_like=empty_like, cuda=SimpleNamespace(device=lambda _: nullcontext())),
        'triton': SimpleNamespace(next_power_of_2=next_power), 'square_kernel': kernel}
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace['square'], kernel, allocations


def unaligned(offset, empty=False):
    backing = torch.arange(40, dtype=torch.uint8)
    storage = backing.untyped_storage()[offset:offset+32]
    return torch.empty(0, dtype=torch.float32).set_(storage, 0,
                (0, 2) if empty else (2, 2), (2, 1))


def host_checks(name='square_kernel'):
    square, kernel, allocations = harness(name)
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
        y = square(x)
        assert y is allocations[-1] and y is not x
        assert y.shape == x.shape and y.dtype == torch.float32
        assert y.is_contiguous()
        assert bytes(data.untyped_storage()) == before
        torch.testing.assert_close(y.data, data * data, rtol=0, atol=0, equal_nan=True)
        if data.numel():
            assert len(kernel.calls) == calls + 1
            grid, returned, actual, si, so, n, warps, block = kernel.calls[-1]
            assert returned is y and actual is x
            assert grid == (data.shape[0],) and si == so == n == data.shape[1]
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
            square(x)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(('unexpected admission', x))
        assert counts == (len(kernel.calls), len(allocations))
    for shape in [(2**31-1, 1), (1, 2**20), (0, 2**40), (2**40, 0)]:
        x = Tensor(shape=shape)
        y = square(x)
        assert y.shape == shape
    # Concrete original transpose defect: output offset 3 is never written.
    data = torch.tensor([[1., 2.], [3., 4.]]).t()
    output = torch.full_like(data, 73.)
    for kernel.pid in range(2):
        kernel.body(Pointer(output), Pointer(data), data.stride(0), output.stride(0), 2, 2)
    assert not torch.equal(output, data * data)
    assert raw(output)[3] == 73.
    return len(cases), len(rejected)


def signed32(x):
    x %= 2**32
    return x - 2**32 if x >= 2**31 else x


def index_checks():
    assert signed32(2**30 * 2) == -(2**31)
    for rows, cols in [(2**31-1, 1), (2047, 2**20), (7, 1025), (2, 2049)]:
        assert rows * cols < 2**31
        block = next_power(cols)
        for r in {0, rows-1}:
            for c in {0, cols-1, block-1}:
                assert signed32(r * cols) == r * cols
                assert (signed32(c) < signed32(cols)) == (c < cols)
                if c < cols:
                    assert 4096+4*signed32(r*cols)+4*signed32(c) == 4096+4*(r*cols+c)
                    assert r*cols+c < rows*cols


if __name__ == '__main__':
    source_checks()
    admitted, rejected = host_checks()
    assert host_checks('square_index_commuted_kernel') == (admitted, rejected)
    index_checks()
    print(f'PASS: original and actual index-commuted source binding; each variant: '
          f'{admitted} CPU Tensor admissions; {rejected} rejections; '
          'metadata limits; transpose and overflow counterexamples. GPU arithmetic untested.')
