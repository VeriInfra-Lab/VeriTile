"""Actual source AST replay over CPU storage; CUDA scheduling is mocked."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "spinning_lock_reduction_phased.py"
HISTORICAL = Path("/root/.codex/visualizations/2026/10/16/01a14539-f72c-7c83-aa71-3a0014d6e60a/veritile-work/bench/repairs/tritonbench_g/spinning_lock_reduction/spinning_lock_reduction_phased.py")


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


class Tensor:
    def __init__(self, data, device="cuda:0"):
        self.data, self.device = data, torch.device(device)
    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    def numel(self): return self.data.numel()
    def dim(self): return self.data.dim()
    def stride(self): return self.data.stride()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def is_conj(self): return self.data.is_conj()
    def data_ptr(self): return self.data.data_ptr()
    def detach(self): return self.data.detach()


class Pointer:
    def __init__(self, tensor, offset=0): self.tensor, self.offset = tensor, offset
    def __add__(self, offset): return Pointer(self.tensor, self.offset + offset)


class Casts(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        # Every exercised integer lies inside its admitted i64/u64 domain.
        # Torch arange already produces int64 vectors. Bitcasts are retained.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to" and not node.keywords:
            return node.func.value
        return node


def checks():
    if HISTORICAL.exists():
        assert ast.dump(function(SOURCE, "spinning_lock_kernel")) == ast.dump(function(HISTORICAL, "spinning_lock_kernel"))
    pending, seen = [HERE / "Kernels.lean"], set()
    while pending:
        source = pending.pop()
        if source in seen: continue
        seen.add(source)
        for line in source.read_text().splitlines():
            if line.startswith("import "):
                for name in line.split()[1:]:
                    assert ".Proofs." not in name
                    if name.startswith("bench."):
                        pending.append(HERE.parents[3] / (name.replace(".", "/") + ".lean"))
    spec = importlib.util.spec_from_file_location("spinning_candidate", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current_device, submission_devices = [0], []
    class DeviceContext:
        def __init__(self, device): self.device = torch.device(device)
        def __enter__(self):
            self.previous = current_device[0]; current_device[0] = self.device.index
        def __exit__(self, *exc): current_device[0] = self.previous
    module.torch = SimpleNamespace(Tensor=Tensor, float32=torch.float32, int32=torch.int32,
        strided=torch.strided, cuda=SimpleNamespace(device=DeviceContext))
    pid, events, submissions, queue = [0], [], [], []
    executions = [0]
    failed_phase = [None]

    def flat(ptr): return ptr.tensor.reshape(-1)
    def read(ptr):
        offsets = torch.as_tensor(ptr.offset)
        assert bool((offsets >= 0).all()) and bool((offsets < ptr.tensor.numel()).all())
        events.append(("load", pid[0]))
        return flat(ptr)[offsets].clone()
    def write(ptr, values, mask=None):
        offsets = torch.as_tensor(ptr.offset)
        mask = torch.ones_like(offsets, dtype=torch.bool) if mask is None else mask
        assert bool((offsets[mask] >= 0).all()) and bool((offsets[mask] < ptr.tensor.numel()).all())
        flat(ptr)[offsets[mask]] = values[mask]
        events.append(("store", pid[0]))
    def cas(ptr, expected, replacement):
        assert 0 <= ptr.offset < ptr.tensor.numel()
        old = int(flat(ptr)[ptr.offset])
        assert old == expected, "same-stream producer completion must make the poll ready"
        flat(ptr)[ptr.offset] = replacement
        events.append(("cas", pid[0]))
        return old
    def exchange(ptr, value):
        assert 0 <= ptr.offset < ptr.tensor.numel()
        assert events[-1] == ("barrier", pid[0])
        flat(ptr)[ptr.offset] = value
        events.append(("release", pid[0]))
    def static_assert(condition): assert condition
    namespace = {"tl": SimpleNamespace(
        constexpr=object, int64=torch.int64, uint64=torch.uint64, float32=torch.float32,
        uint32=torch.uint32, program_id=lambda axis: pid[0], static_assert=static_assert,
        cast=lambda value, dtype: value, zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        arange=lambda start, end: torch.arange(start, end, dtype=torch.int64),
        load=read, store=write, atomic_cas=cas, atomic_xchg=exchange,
        debug_barrier=lambda: events.append(("barrier", pid[0])))}
    node = Casts().visit(function(SOURCE, "spinning_lock_kernel"))
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)

    class Launch:
        def __getitem__(self, grid):
            def submit(*args, **kwargs):
                assert current_device[0] == args[0].device.index
                submission_devices.append((kwargs["PHASE"], current_device[0]))
                if kwargs["PHASE"] == failed_phase[0]: raise RuntimeError("mock submission error")
                submissions.append(kwargs["PHASE"])
                queue.append((grid, args, kwargs))
            return submit
    module.spinning_lock_kernel = Launch()

    def synchronize():
        while queue:
            grid, args, kwargs = queue.pop(0)
            for i in reversed(range(grid[0])):
                pid[0] = i
                converted = [Pointer(x.data) if isinstance(x, Tensor) else x for x in args]
                namespace["spinning_lock_kernel"](*converted, **kwargs)
                executions[0] += 1

    def make(sms=5, k=2, m=3, n=13, bm=2, bn=4):
        def tensor(shape, dtype, value):
            count = 1
            for dimension in shape: count *= dimension
            backing = torch.full((count+8,), value, dtype=dtype)
            result = Tensor(backing[4:4+count].reshape(shape))
            result.backing = backing
            return result
        p = tensor((sms*bm*bn+8,), torch.float32, 3.25)
        c = tensor((m,n), torch.float32, -17.5)
        l = tensor((sms,), torch.int32, 0)
        return [p,c,l,sms,k,m,n,*c.stride(),bm,bn]

    admitted = 0
    for sms,k,m,n,bm,bn in [(1,1,1,1,1,1),(2,2,3,9,2,4),(5,2,3,13,2,4),
        (5,3,5,21,4,8),(7,1,4,12,2,4),(7,2**64-1,3,12,2,4),
        (4,2**32+1,3,5,4,2),(2,2,0,7,2,4),(2,2,8,0,2,4),
        (3,2,7,2,2,1),(3,3,1,13,1,8),(5,2,3,20,4,4)]:
        args = make(sms,k,m,n,bm,bn)
        p,c,l = args[:3]
        if (sms,k)==(5,3):
            for tensor in (p,c,l): tensor.device = torch.device("cuda:1")
        snapshots = [t.backing.clone() for t in (p,c,l)]
        submissions.clear(); events.clear()
        assert module.spinning_lock(*args) is None
        assert submissions == [1,2] and len(queue)==2
        assert current_device[0] == 0
        assert all(torch.equal(t.backing, old) for t,old in zip((p,c,l),snapshots))
        synchronize()
        target = torch.full_like(c.data,-17.5)
        target[:min(m,bm),:min(n,sms*bn)] = 0
        assert torch.equal(c.data.view(torch.int32),target.view(torch.int32))
        for q in range(sms):
            expected = 3.25 if q%k==0 else 0.0
            assert bool((p.data[q*bm*bn:(q+1)*bm*bn] == expected).all())
            assert l.data[q] == (0 if q%k==0 else 1)
        assert bool((p.data[sms*bm*bn:]==3.25).all())
        for t,old in zip((p,c,l),snapshots):
            assert torch.equal(t.backing[:4],old[:4]) and torch.equal(t.backing[-4:],old[-4:])
        assert sum(e[0]=="release" for e in events)==9*sum(q%k!=0 for q in range(sms))
        expected_reads=9*sum(min(k-1,sms-q-1) for q in range(sms) if q%k==0)
        assert sum(e[0]=="cas" for e in events)==expected_reads
        assert sum(e[0]=="load" for e in events)==expected_reads
        # Reusing publications with a different group size needs no resets.
        args[4] = 3 if k!=3 else 2
        module.spinning_lock(*args); synchronize()
        assert torch.equal(c.data.view(torch.int32),target.view(torch.int32))
        assert current_device[0] == 0
        admitted += 2

    # Host exceptions retain only the phases already submitted.
    for phase in (1, 2):
        args = make(); failed_phase[0] = phase; submissions.clear()
        for tensor in args[:3]: tensor.device = torch.device("cuda:1")
        try: module.spinning_lock(*args)
        except RuntimeError: pass
        else: raise AssertionError("expected submission failure")
        assert submissions == ([] if phase == 1 else [1])
        assert current_device[0] == 0
        synchronize()
        if phase == 1: assert bool((args[1].data == -17.5).all())
        else:
            assert bool((args[1].data[:2,4:8] == 0).all())
            assert bool((args[1].data[:2,:4] == -17.5).all())
    failed_phase[0] = None
    rejected = 0
    def reject(edit):
        nonlocal rejected
        args=make(); edit(args)
        snapshot=[x.data.clone() for x in args[:3] if isinstance(x,Tensor)]
        submissions.clear()
        try: module.spinning_lock(*args)
        except (TypeError,ValueError): rejected+=1
        else: raise AssertionError("expected rejection")
        assert submissions==[] and queue==[]
        for x,old in zip([x for x in args[:3] if isinstance(x,Tensor)],snapshot):
            assert torch.equal(x.data.resolve_neg().view(torch.int32),old.view(torch.int32))
    for index,value in [(3,True),(3,0),(3,2**31),(4,0),(4,2**64),(5,-1),(6,2**63),
        (7,-1),(8,2**63),(9,0),(10,3),(9,2**21),(9,2048),(5,99),(7,1)]:
        reject(lambda a,i=index,v=value:a.__setitem__(i,v))
    reject(lambda a:a.__setitem__(0,object()))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(50,dtype=torch.float64))))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(50),"cpu")))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(50),"cuda:1")))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(100)[::2])))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(50)._neg_view())))
    reject(lambda a:a.__setitem__(0,Tensor(torch.ones(2))))
    reject(lambda a:a.__setitem__(2,Tensor(torch.zeros((1,5),dtype=torch.int32))))
    reject(lambda a:a.__setitem__(2,Tensor(torch.zeros(2,dtype=torch.int32))))
    reject(lambda a:setattr(a[0],"data_ptr",lambda:3))
    reject(lambda a:setattr(a[0],"data_ptr",lambda:2**64-4))
    reject(lambda a:setattr(a[0],"data_ptr",a[1].data_ptr))
    reject(lambda a:a[2].data.__setitem__(1,1))
    def negative_zero_ready(a):
        a[2].data[1]=1; a[0].data[8:16]= -0.0
    reject(negative_zero_ready)
    def beyond_payload_ready(a):
        a[2]=Tensor(torch.zeros(8,dtype=torch.int32)); a[2].data[7]=1
    reject(beyond_payload_ready)
    def masked_pointer_overflow(a):
        a[5]=1; a[6]=1; a[1]=Tensor(torch.ones((1,1)))
        a[7],a[8]=1,1
        a[1].data_ptr=lambda:2**64-4
    reject(masked_pointer_overflow)
    assert (1,1) in submission_devices and (2,1) in submission_devices and current_device[0] == 0
    print(f"spinning lock contract checks: {rejected} rejects, {admitted} admissions, {executions[0]} program executions")


if __name__=="__main__": checks()
