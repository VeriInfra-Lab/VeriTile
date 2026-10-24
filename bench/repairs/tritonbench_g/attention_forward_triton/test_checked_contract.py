from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
import hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
source = A / 'attention_forward_triton_repaired.py'
spec = importlib.util.spec_from_file_location('aft2_host_control', source)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

@dataclass(frozen=True)
class Device:
    type: str = 'cuda'
    index: int = 0

class Tensor:
    next_base = 2**32
    def __init__(self, shape, dtype='fp32', device=Device(), strides=None, neg=False, conj=False, base=None):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device
        self.ndim = len(shape)
        self.layout = 'strided'
        self.is_cuda = device.type == 'cuda'
        self.strides = tuple(strides) if strides is not None else tuple(prod(shape[i+1:]) for i in range(len(shape)))
        self.neg, self.conj = neg, conj
        self.base = Tensor.next_base if base is None else base
        Tensor.next_base += 2**24
    def stride(self, i=None): return self.strides if i is None else self.strides[i]
    def numel(self): return prod(self.shape)
    def is_neg(self): return self.neg
    def is_conj(self): return self.conj
    def data_ptr(self): return self.base
    def untyped_storage(self): return self
    def element_size(self): return 4 if self.dtype == 'fp32' else 2

allocations, launches = [], []
current_device, launch_devices, fail_launch = [0], [], [False]
class DeviceContext:
    def __init__(self, device): self.device=device
    def __enter__(self): self.previous=current_device[0];current_device[0]=self.device.index
    def __exit__(self,*exc): current_device[0]=self.previous
def empty(shape, device, dtype):
    t = Tensor(shape, dtype=dtype, device=device)
    allocations.append(t)
    return t
def empty_like(q, dtype): return empty(q.shape, q.device, dtype)
class LaunchSpy:
    def __getitem__(self, grid):
        def invoke(*args, **kw):
            assert current_device[0]==args[0].device.index
            launch_devices.append(current_device[0])
            if fail_launch[0]:raise RuntimeError('mock submission failure')
            launches.append((grid, args, kw))
        return invoke
m.torch = SimpleNamespace(Tensor=Tensor, strided='strided', float32='fp32', float16='fp16', cuda=SimpleNamespace(device=DeviceContext), empty=empty, empty_like=empty_like)
m._attn_fwd = LaunchSpy()

def inputs(z=1, h=1, n=65):
    return [Tensor((z,h,n,96)) for _ in range(3)] + [Tensor((z,h,(n+127)//128)),Tensor((z,h,(n+63)//64))]
records=[]
def success(name,args):
    allocations.clear();launches.clear()
    out=m.forward(*args)
    z,h,n,d=args[0].shape
    assert out is allocations[0] and out.dtype=='fp16' and out.shape==args[0].shape
    assert len(allocations)==2 and allocations[1].dtype=='fp32'
    assert len(launches)==int(bool(z*h*n))
    if launches:
        grid,actual,kw=launches[0]
        assert actual[:5]==tuple(args) and actual[5] is out
        assert grid==((n+127)//128,z*h,1)
        assert kw['num_stages']==2 and kw['num_warps']==8
    records.append(name)
def reject(name,thunk):
    launches.clear()
    try:thunk()
    except (ValueError,TypeError):assert not launches;records.append(name)
    else:raise AssertionError(name)
for n in [63,65,127,129]:success('tail_'+str(n),inputs(n=n))
success('aligned_full',inputs(2,3,128))
for z,h,n in [(0,2,64),(2,0,64),(2,3,0)]:success('empty',inputs(z,h,n))
for field,value in [('layout','sparse'),('is_cuda',False),('dtype','fp16'),('neg',True),('conj',True),('base',2**64-4)]:
    a=inputs();setattr(a[0],field,value);reject(field,lambda:m.forward(*a))
a=inputs();a[1]=a[0];reject('input_alias',lambda:m.forward(*a))
a=inputs();a[0].strides=(0,*a[0].strides[1:]);reject('zero_stride',lambda:m.forward(*a))
a=inputs();a[0].strides=(2**31,*a[0].strides[1:]);reject('overflow_stride',lambda:m.forward(*a))
a=inputs();a[4].strides=(4,4,2);reject('scale_stride',lambda:m.forward(*a))
reject('unsupported_specialization',lambda:m.forward(*inputs(n=64)))
reject('nontensor',lambda:m.forward(None,None,None,None,None))
for base in [1,-4,2**64-4]:
    reject('pointer',lambda base=base:m._check_pointers([Tensor((1,),base=base)],2))
for pointer_classes,integer_classes in m._SAVED_CLASSES:
    tensors=[Tensor((1,),base=4096 if c==16 else 4100) for c in pointer_classes]
    values=[1 if c==1 else 16 if c==16 else 3 for c in integer_classes]
    m._check_specialization(tensors,[values[i:i+4] for i in range(0,16,4)],values[16:])
    records.append('saved_class')
a=inputs(n=65)
for t in a:t.device=Device(index=1)
success('owning_device_mismatch',a)
assert current_device[0]==0 and launch_devices[-1]==1
fail_launch[0]=True
try:m.forward(*a)
except RuntimeError:pass
else:raise AssertionError('expected mock submission failure')
finally:fail_launch[0]=False
assert current_device[0]==0 and launch_devices[-1]==1
records.append('owning_device_exception_restoration')
print(json.dumps({'status':'PASS','controls':len(records),'GPU_execution':False}))
