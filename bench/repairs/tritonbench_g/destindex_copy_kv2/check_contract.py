"""Actual original/repaired AST on CPU backing storage; CUDA launch is mocked."""
import ast
import contextlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'destindex_copy_kv2_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/destindex_copy_kv2/destindex_copy_kv2.py'
NAME = '_fwd_kernel_destindex_copy_kv'


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_binding():
    assert ast.dump(function(SOURCE, NAME), include_attributes=False) == ast.dump(
        function(ORIGINAL, NAME), include_attributes=False)
    wrapper = function(SOURCE, 'destindex_copy_kv_repaired')
    guards = [n for n in wrapper.body if isinstance(n, ast.If)
              and any(isinstance(s, ast.Raise) for s in n.body)]
    assert len(guards) == 16
    lean = (HERE.parent / 'destindex_copy_kv1/Contract.lean').read_text().split('def checkInputs', 1)[1].split('def checkArguments', 1)[0]
    assert re.findall(r'\.error \.(\w+)', lean) == [
        'layout', 'rank', 'dtype', 'cuda', 'device', 'contiguous', 'negative',
        'shapes', 'extent', 'tile', 'alignment', 'overlap', 'bounds', 'unique']
    assert 'Width m' in (HERE / 'Contract.lean').read_text()
    forbidden = {'clone', 'contiguous', 'resolve_neg', 'copy_', 'to'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in forbidden for n in ast.walk(wrapper))
    assert '.Proofs.' not in (HERE / 'Kernels.lean').read_text()
    assert not any(line.startswith('import ') and line.endswith('.FPEquiv')
                   for line in (HERE / 'Correct.lean').read_text().splitlines())
    assert not any(line.startswith('import ') and line.endswith('.Correct')
                   for line in (HERE / 'FPEquiv.lean').read_text().splitlines())


class Tensor:
    def __init__(self, data, device='cuda:0', cuda=True):
        self.data, self.device, self.is_cuda = data, torch.device(device), cuda
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self): return self.data.data_ptr()
    def cpu(self): return self.data.cpu()


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


def backing_bytes(t):
    return torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes(),), (1,))


class Pointer:
    def __init__(self, data, offset=0): self.data, self.offset = data, offset
    def __add__(self, offset): return Pointer(self.data, self.offset + offset)


def interpreter(path):
    pid = [0]
    def load(pointer, mask=None, other=0.0):
        storage = raw(pointer.data)
        offsets = pointer.offset + pointer.data.storage_offset()
        if isinstance(offsets, int):
            assert 0 <= offsets < len(storage)
            return int(storage[offsets])
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        mask = torch.broadcast_to(mask, offsets.shape)
        selected = offsets[mask].long()
        assert bool(((selected >= 0) & (selected < len(storage))).all()), 'source OOB'
        result = torch.full_like(offsets, other, dtype=pointer.data.dtype)
        result[mask] = storage[selected]
        return result
    def store(pointer, values, mask=None):
        storage = raw(pointer.data)
        if mask is None: mask = torch.ones_like(pointer.offset, dtype=torch.bool)
        mask = torch.broadcast_to(mask, pointer.offset.shape)
        indices = pointer.data.storage_offset() + pointer.offset[mask].long()
        assert bool(((indices >= 0) & (indices < len(storage))).all()), 'output OOB'
        storage[indices] = values[mask]
    node = function(path, NAME)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    ns = {'tl': SimpleNamespace(program_id=lambda _: pid[0],
        arange=lambda a, b: torch.arange(a, b, dtype=torch.int32), load=load, store=store)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), ns)
    return ns[NAME], pid


def main():
    source_binding()
    spec = importlib.util.spec_from_file_location('direct_kv2', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel, pid = interpreter(SOURCE)
    calls = []
    class Launch:
        def __getitem__(self, grid):
            def run(k, dest, out, *args, **kw):
                calls.append((grid, k, dest, out, args, kw))
                for p in reversed(range(grid[0])):
                    pid[0] = p
                    kernel(Pointer(k.data), Pointer(dest.data), Pointer(out.data), *args,
                           kw['BLOCK_DMODEL'], kw['BLOCK_HEAD'])
            return run
    module._fwd_kernel_destindex_copy_kv = Launch()
    module.torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
        int32=torch.int32, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    run = module.destindex_copy_kv_repaired
    cases = [(0,3,5), (2,0,3), (2,3,0), (1,1,1), (3,1,4), (3,3,8),
             (2,12,64), (2,12,128), (2,3,256), (1,1,512)]
    for count, heads, dims in cases:
        size = (count+1)*heads*dims
        bits = torch.arange(size+6, dtype=torch.int32)*16381
        bits[:6] = torch.tensor([0,-2147483648,2139095040,-8388608,2143289345,1], dtype=torch.int32)
        source = bits.view(torch.float32)[1:1+size].reshape(count+1,heads,dims)
        out_size = (count+3)*heads*dims
        output = torch.full((out_size+10,), 42., dtype=torch.float32)[3:3+out_size].reshape(count+3,heads,dims)
        destinations = torch.arange(count, 0, -1, dtype=torch.int32)
        args = [Tensor(source), Tensor(destinations), Tensor(output)]
        before = backing_bytes(output).clone()
        source_before, dest_before = backing_bytes(source).clone(), backing_bytes(destinations).clone()
        expected = output.view(torch.int32).clone()
        if heads and dims:
            for row, dest in enumerate(destinations.tolist()): expected[dest] = source.view(torch.int32)[row]
        old_calls = len(calls)
        assert run(*args) is None
        assert torch.equal(output.view(torch.int32), expected)
        after = backing_bytes(output)
        start, end = output.storage_offset()*4, (output.storage_offset()+output.numel())*4
        assert torch.equal(after[:start], before[:start]) and torch.equal(after[end:], before[end:])
        assert torch.equal(backing_bytes(source), source_before) and torch.equal(backing_bytes(destinations), dest_before)
        assert len(calls)-old_calls == int(count>0 and heads>0 and dims>0)
        if len(calls)>old_calls:
            assert all(a is b for a,b in zip(calls[-1][1:4], args))

    def fresh():
        return [Tensor(torch.arange(16., dtype=torch.float32).reshape(2,2,4)),
                Tensor(torch.tensor([2,0], dtype=torch.int32)), Tensor(torch.full((3,2,4),42.))]
    rejected = 0
    def reject(args):
        nonlocal rejected
        old_calls = len(calls)
        snapshots = [backing_bytes(t.data).clone() for t in args if isinstance(t, Tensor)]
        try: run(*args)
        except (TypeError,ValueError): pass
        else: raise AssertionError('unsupported request admitted')
        assert len(calls)==old_calls
        assert all(torch.equal(s,backing_bytes(t.data)) for s,t in zip(snapshots,[t for t in args if isinstance(t,Tensor)]))
        rejected += 1
    a=fresh();a[0]=object();reject(a)
    for i in range(3):
        a=fresh();a[i].is_cuda=False;reject(a)
        a=fresh();a[i].device=torch.device('cuda:1');reject(a)
        a=fresh();a[i]=Tensor(a[i].data.to(torch.float64));reject(a)
        a=fresh();a[i].data_ptr=lambda: 3;reject(a)
    a=fresh();a[0]=Tensor(a[0].data.reshape(-1));reject(a)
    a=fresh();a[1]=Tensor(torch.tensor([[2,0]],dtype=torch.int32));reject(a)
    a=fresh();a[2]=Tensor(torch.ones(3,2,5));reject(a)
    a=fresh();a[1]=Tensor(torch.tensor([0,1,2],dtype=torch.int32));reject(a)
    a=fresh();a[1]=Tensor(torch.tensor([0,1,2,0],dtype=torch.int32)[::2]);reject(a)
    a=fresh();a[0]=Tensor(torch.ones(2,2,8)[:,:,::2]);reject(a)
    a=fresh();a[2]=Tensor(torch.ones(3,2,8)[:,:,::2]);reject(a)
    a=fresh();a[0]=Tensor(torch._neg_view(a[0].data));reject(a)
    a=fresh();a[2]=Tensor(torch._neg_view(a[2].data));reject(a)
    for destinations in [[-1,0],[3,0],[1,1]]:
        a=fresh();a[1]=Tensor(torch.tensor(destinations,dtype=torch.int32));reject(a)
    a=fresh();a[2]=a[0];a[1]=Tensor(torch.tensor([0,1],dtype=torch.int32));reject(a)
    for i in range(3):
        a=fresh();a[i].numel=lambda: 1<<31;reject(a)
    a=fresh();a[0]=Tensor(torch.empty(0,1025,1025));a[1]=Tensor(torch.empty(0,dtype=torch.int32));a[2]=Tensor(torch.empty(0,1025,1025));reject(a)

    for dims in [3,5,96,129]:
        a=fresh();a[0]=Tensor(torch.ones(2,2,dims));a[2]=Tensor(torch.ones(3,2,dims));reject(a)

    # Original unstrided metadata pointer reads raw [0,1], not logical [0,2].
    original, opid = interpreter(ORIGINAL)
    source=torch.tensor([11.,22.]).reshape(2,1,1)
    dest=torch.tensor([0,1,2],dtype=torch.int32)[::2]
    output=torch.tensor([100.,200.,300.]).reshape(3,1,1)
    for p in range(2):
        opid[0]=p
        original(Pointer(source),Pointer(dest),Pointer(output),1,1,1,1,1,1,1,1,1)
    assert output.flatten().tolist()==[11.,22.,300.]
    assert output.flatten().tolist()!=[11.,200.,22.]
    print(f'Unchanged kernel AST, {len(cases)} admissions, {rejected} rejections, and original strided-metadata failure passed.')
    print('Raw FP32 bits and exterior bytes checked on CPU; CUDA context/launch mocked; no GPU claim.')


if __name__ == '__main__': main()
