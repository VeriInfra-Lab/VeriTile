"""CPU source/host/stride checks; these do not establish GPU numerics."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'token_softmax_llama_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/token_softmax_llama/token_softmax_llama.py'


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def next_power(n):
    n -= 1
    for shift in (1, 2, 4, 8, 16, 32):
        n |= n >> shift
    return n + 1


def source_checks():
    original = function(ORIGINAL, '_fwd_kernel_token_softmax')
    repaired = function(SOURCE, '_fwd_kernel_token_softmax')
    # The numerical operations and active pointer formulas are preserved.
    assert [ast.dump(n) for n in repaired.body[5:]] == [ast.dump(n) for n in original.body[5:-1]]
    for a in repaired.args.args[4:-1]:
        assert ast.unparse(a.annotation) == 'tl.int64'
    assert '* stride_length' in ast.unparse(repaired.body[3])
    assert '* stride_start' in ast.unparse(repaired.body[4])
    body = ast.unparse(function(SOURCE, 'token_softmax_fwd'))
    assert all(word not in body for word in ('.clone(', '.contiguous(', '.cpu(', '.item(', '.reshape('))
    assert 'num_warps=num_warps' in body and 'torch.cuda.device(Logics.device)' in body
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()
    assert 'Correct' not in (HERE / 'FPEquiv.lean').read_text().split('namespace')[0]
    assert 'FPEquiv' not in (HERE / 'Correct.lean').read_text().split('namespace')[0]


class Tensor:
    def __init__(self, data=None, *, shape=None, strides=None, dtype=torch.float32,
                 layout=torch.strided, cuda=True, negative=False, pointer=4096):
        self.data = data
        self.shape = tuple(data.shape) if data is not None else tuple(shape)
        self.dtype = data.dtype if data is not None else dtype
        self.layout, self.is_cuda, self.device = layout, cuda, torch.device('cuda:0')
        self._strides = strides
        self._negative, self._pointer = negative, pointer
    def __getitem__(self, index): return self.data[index]
    @property
    def ndim(self): return len(self.shape)
    def stride(self, dim=None):
        value = self.data.stride() if self.data is not None else self._strides
        return value if dim is None else value[dim]
    def is_neg(self): return self.data.is_neg() if self.data is not None else self._negative
    def data_ptr(self):
        assert all(self.shape), 'no pointer query on empty tensors'
        return self.data.data_ptr() if self.data is not None else self._pointer
    def untyped_storage(self): return self.data.untyped_storage()


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // 4,), (1,))


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, torch.as_tensor(offset, dtype=torch.int64)
    def __add__(self, offset): return Pointer(self.tensor, self.offset + offset)


class Kernel:
    def __init__(self):
        self.calls, self.pids = [], (0, 0)
        def load(p, mask=None, other=None):
            if mask is None:
                return raw(p.tensor)[p.tensor.storage_offset() + p.offset]
            result = torch.full_like(p.offset, other, dtype=p.tensor.dtype)
            result[mask] = raw(p.tensor)[p.tensor.storage_offset() + p.offset[mask]]
            return result
        def store(p, value, mask):
            raw(p.tensor)[p.tensor.storage_offset() + p.offset[mask]] = value[mask]
        tl = SimpleNamespace(program_id=lambda dim: torch.tensor(self.pids[dim], dtype=torch.int32),
            arange=lambda a, b: torch.arange(a, b, dtype=torch.int32), int64=torch.int64,
            float32=torch.float32, load=load, store=store, exp=torch.exp,
            max=lambda x, axis: x.max(dim=axis).values, sum=lambda x, axis: x.sum(dim=axis))
        node = function(SOURCE, '_fwd_kernel_token_softmax')
        node.decorator_list = []
        for a in node.args.args: a.annotation = None
        namespace = {'tl': tl}
        exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(SOURCE), 'exec'), namespace)
        self.body = namespace[node.name]
    def __getitem__(self, grid):
        def launch(x, starts, lengths, out, ss, ls, rh, rc, oh, oc, *, BLOCK_SIZE, num_warps):
            self.calls.append((grid, x, starts, lengths, out, ss, ls, rh, rc, oh, oc, BLOCK_SIZE, num_warps))
            for b in range(grid[0]):
                for h in range(grid[1]):
                    self.pids = b, h
                    self.body(*map(lambda x: Pointer(x.data), (x, starts, lengths, out)),
                              ss, ls, rh, rc, oh, oc, BLOCK_SIZE)
        return launch


def harness():
    kernel = Kernel()
    nodes = [function(SOURCE, name) for name in ('_output_nonoverlap', '_check_contents', 'token_softmax_fwd')]
    for node in nodes: node.decorator_list = []
    namespace = {'torch': SimpleNamespace(Tensor=Tensor, float32=torch.float32, int32=torch.int32,
        strided=torch.strided, cuda=SimpleNamespace(device=lambda _: nullcontext())),
        'triton': SimpleNamespace(next_power_of_2=next_power), '_fwd_kernel_token_softmax': kernel}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), str(SOURCE), 'exec'), namespace)
    return namespace['token_softmax_fwd'], kernel


def checked_case(x, starts, lengths, out, maximum):
    call, kernel = harness()
    x0, s0, n0, o0 = [bytes(t.untyped_storage()) for t in (x, starts, lengths, out)]
    expected = out.clone()
    touched = set()
    for start, count in zip(starts.tolist(), lengths.tolist()):
        if count:
            expected[:, start:start+count] = torch.softmax(x[:, start:start+count], dim=1)
            for h in range(out.shape[0]):
                for j in range(start, start+count):
                    touched.add(out.storage_offset()+h*out.stride(0)+j*out.stride(1))
    tensors = list(map(Tensor, (x, starts, lengths, out)))
    assert call(*tensors, maximum) is None
    torch.testing.assert_close(out, expected, rtol=3e-6, atol=2e-7, equal_nan=True)
    for t, saved in zip((x, starts, lengths), (x0, s0, n0)):
        assert bytes(t.untyped_storage()) == saved
    final = bytes(out.untyped_storage())
    for i in range(len(final)//4):
        if i not in touched: assert final[4*i:4*i+4] == o0[4*i:4*i+4]
    if starts.numel() and x.numel() and maximum:
        event = kernel.calls[0]
        assert len(kernel.calls) == 1 and event[0] == (len(starts), x.shape[0])
        assert all(event[k+1] is tensors[k] for k in range(4))
        assert event[5:11] == (starts.stride(0), lengths.stride(0), *x.stride(), *out.stride())
        assert event[-1] == (16 if next_power(maximum)>=4096 else 8 if next_power(maximum)>=2048 else 4)
    else: assert not kernel.calls


def host_checks():
    admissions = 0
    for kind in range(6):
        base = torch.randn(7, 26)
        x = [base[:3, :10], base[:3, :20:2], base[:10, :3].T,
             base[0:1, :10].expand(3, 10), base[:3, 0:1].expand(3, 10), base[:3, :10]][kind]
        # Keep shape consistent for the transpose case.
        width = x.shape[1]
        outbase = torch.full((30, 30), 91.)
        out = outbase[2:2+x.shape[0], 3:3+2*width:2] if kind%2==0 else outbase[3:3+width, 2:2+x.shape[0]].T
        starts = torch.tensor([width-2, 999, 0, 999, 1, 999], dtype=torch.int32)[::2]
        lengths = torch.tensor([2, 999, 2, 999, 0, 999], dtype=torch.int32)[::2]
        checked_case(x, starts, lengths, out, 3)
        admissions += 1
    for width in (1, 1025, 2049, 4096):
        checked_case(torch.randn(2, width), torch.tensor([0], dtype=torch.int32),
                     torch.tensor([width], dtype=torch.int32), torch.full((2,width), 91.), width)
        admissions += 1
    for shape, batch, maximum in [((0,4),2,3),((2,0),2,3),((2,4),0,3),((2,4),2,0)]:
        checked_case(torch.empty(shape), torch.zeros(batch,dtype=torch.int32),
                     torch.zeros(batch,dtype=torch.int32), torch.full(shape,91.),maximum)
        admissions += 1
    call, kernel = harness()
    def args():
        return [Tensor(torch.randn(2,5)), Tensor(torch.tensor([0],dtype=torch.int32)),
                Tensor(torch.tensor([5],dtype=torch.int32)), Tensor(torch.empty(2,5)), 5]
    # A nonempty zero-capacity call validates metadata and rejects positive lengths.
    idle=args();idle[1]=Tensor(torch.tensor([0],dtype=torch.int32))
    idle[2]=Tensor(torch.tensor([1],dtype=torch.int32));idle[4]=0
    saved=bytes(idle[3].data.untyped_storage())
    try:call(*idle)
    except ValueError:pass
    else:raise AssertionError('positive length accepted at maximum zero')
    assert bytes(idle[3].data.untyped_storage())==saved
    controls = []
    for k in range(4): controls.append((k, None))
    controls += [(0, Tensor(torch.ones(2,5,dtype=torch.float16))),
                 (1, Tensor(torch.tensor([0],dtype=torch.int64))),
                 (0, Tensor(shape=(2,5),strides=(5,1),cuda=False)),
                 (0, Tensor(shape=(2,5),strides=(5,1),negative=True)),
                 (0, Tensor(shape=(10,),strides=(1,))),
                 (3, Tensor(torch.empty(2,4))),
                 (1, Tensor(torch.zeros(1,1,dtype=torch.int32))),
                 (2, Tensor(torch.zeros(2,dtype=torch.int32))),
                 (4, -1),(4,True),(4,1.5),(4,2**20+1),
                 (0,Tensor(shape=(2**16,5),strides=(5,1))),
                 (3,Tensor(torch.empty(2,1).expand(2,5))),
                 (0,Tensor(shape=(2,5),strides=(2**63,1))),
                 (0,Tensor(shape=(2,5),strides=(5,1),pointer=4097)),
                 (0,Tensor(shape=(2,5),strides=(2**62,1))),
                 (0,Tensor(shape=(2,5),strides=(5,1),pointer=2**64-4))]
    for pos, value in controls:
        a=args();a[pos]=value
        try: call(*a)
        except (TypeError,ValueError): pass
        else: raise AssertionError(('admitted negative control',pos,value))
    a=args();a[3]=a[0]
    try: call(*a)
    except ValueError: pass
    else: raise AssertionError('admitted aliased output')
    content_controls = [([-1],[1]),([0],[-1]),([0],[6]),([4],[2]),([6],[0]),
                        ([0,1],[3,2]),([2,0],[2,3]),([0,0],[1,1])]
    for starts, lengths in content_controls:
        a=args();a[1]=Tensor(torch.tensor(starts,dtype=torch.int32))
        a[2]=Tensor(torch.tensor(lengths,dtype=torch.int32))
        saved=bytes(a[3].data.untyped_storage())
        try: call(*a)
        except ValueError: pass
        else: raise AssertionError(('admitted invalid metadata',starts,lengths))
        assert bytes(a[3].data.untyped_storage())==saved
    assert not kernel.calls
    print(f'{admissions} direct-stride/empty admissions and {len(controls)+2+len(content_controls)} rejection controls passed; caller frames preserved')


if __name__ == '__main__':
    source_checks()
    host_checks()
