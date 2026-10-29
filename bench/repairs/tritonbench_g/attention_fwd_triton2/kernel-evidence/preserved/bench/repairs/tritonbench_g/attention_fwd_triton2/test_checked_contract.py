from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from math import prod
import hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
source = A / 'Whole001/attention_fwd_triton2_repaired.py'
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
        self.strides = tuple(strides) if strides is not None else tuple(prod(shape[i+1:]) for i in range(len(shape)))
        self.neg, self.conj = neg, conj
        self.base = Tensor.next_base if base is None else base
        Tensor.next_base += 2**24
    def stride(self, i=None): return self.strides if i is None else self.strides[i]
    def numel(self): return prod(self.shape)
    def is_neg(self): return self.neg
    def is_conj(self): return self.conj
    def data_ptr(self): return self.base
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
m.torch = SimpleNamespace(Tensor=Tensor, strided='strided', float32='fp32', bfloat16='bf16', cuda=SimpleNamespace(device=DeviceContext), empty=empty, empty_like=empty_like)
m._attn_fwd = LaunchSpy()

def inputs(z=1, h=1, n=65):
    return [Tensor((z,h,n,96)) for _ in range(3)] + [Tensor((z,h,(n+127)//128)),Tensor((z,h,(n+63)//64))]
records = []
def success(name, args, empty_path=False):
    allocations.clear(); launches.clear()
    result = m.forward(*args)
    z,h,n,d = args[0].shape
    assert result is allocations[0] and result.dtype == 'bf16' and result.shape == args[0].shape
    assert len(allocations) == 2 and allocations[1].shape == (z,h,n) and allocations[1].dtype == 'fp32'
    assert len(launches) == (0 if empty_path else 1)
    if not empty_path:
        grid, actual, kw = launches[0]
        assert grid == ((n+127)//128,z*h,1)
        assert actual[:5] == tuple(args) and actual[5] is result
        assert actual[6:22] == tuple(s for t in (*args[:3],result) for s in t.stride())
        assert actual[22:] == (z,h)
        assert kw == dict(N_CTX=n,HEAD_DIM=96,BLOCK_M=128,BLOCK_N=64,STAGE=1,num_warps=8,num_stages=2)
    records.append({'name':name,'status':'PASS','path':'empty' if empty_path else 'launch','shape':list(args[0].shape)})
def rejected(name, args, typ, part):
    allocations.clear(); launches.clear()
    try: m.forward(*args)
    except typ as e:
        assert part in str(e),str(e)
        assert not launches
        records.append({'name':name,'status':'PASS','rejection':str(e)})
    else: raise AssertionError(name+' was accepted')
for n in [1,63,64,65,127,128,129]: success('canonical_N'+str(n),inputs(n=n))
success('multihead_Z2H3N129',inputs(2,3,129))
for z,h,n in [(0,2,64),(2,0,64),(2,3,0)]: success('empty_'+str((z,h,n)),inputs(z,h,n),True)
a=inputs(n=64);a[1]=a[0];a[2]=a[0];a[4]=a[3];success('readonly_input_alias',a)
a=inputs();a[0].dtype='fp16';rejected('reject_fp16_input',a,TypeError,'FP32')
a=inputs();a[3].dtype='bf16';rejected('reject_bf16_scale',a,TypeError,'FP32')
a=inputs();a[0].shape=(1,1,65,128);a[1].shape=a[0].shape;a[2].shape=a[0].shape;rejected('reject_D128',a,ValueError,'D must be 96')
a=inputs();a[2].shape=(1,1,64,96);rejected('reject_shape_mismatch',a,ValueError,'matching')
for i in range(3):
    a=inputs();a[i].strides=tuple(2*x for x in a[i].strides);rejected('reject_step2_'+str(i),a,ValueError,'canonical contiguous')
a=inputs();a[0].strides=(1,*a[0].strides[1:]);rejected('reject_noncanonical_singleton_stride',a,ValueError,'canonical contiguous')
a=inputs();a[0].neg=True;rejected('reject_lazy_neg',a,ValueError,'resolved')
a=inputs();a[1].conj=True;rejected('reject_lazy_conj',a,ValueError,'resolved')
a=inputs();a[3].shape=(1,1,65);rejected('reject_unpacked_scale_shape',a,ValueError,'packed scales')
a=inputs();a[4].strides=(4,4,2);rejected('reject_scale_stride',a,ValueError,'canonical packed')
a=inputs();a[2].device=Device(index=1);rejected('reject_other_device',a,ValueError,'one CUDA')
a=inputs();a[0].device=Device(type='cpu');rejected('reject_CPU',a,ValueError,'one CUDA')
a=inputs();a[0]=object();rejected('reject_non_tensor',a,TypeError,'ordinary')

def guard(name, f, reject=False):
    try: value=f()
    except ValueError as e:
        assert reject
        records.append({'name':name,'status':'PASS','rejection':str(e)})
    else:
        assert not reject
        records.append({'name':name,'status':'PASS','value':value})
limit=2**31-1
guard('signed_envelope_last',lambda:m._launch_envelope(1,1,0,[(limit-263,0,0,0)]))
guard('signed_envelope_first_rejected',lambda:m._launch_envelope(1,1,0,[(limit-262,0,0,0)]),True)
guard('gridY_65535',lambda:m._launch_envelope(65535,1,0,[]))
guard('gridY_65536_rejected',lambda:m._launch_envelope(65536,1,0,[]),True)
guard('pointer_last_aligned',lambda:m._check_pointer(Tensor((1,),base=2**64-44),10))
guard('pointer_extent_at_2pow64_rejected',lambda:m._check_pointer(Tensor((1,),base=2**64-40),10),True)
guard('pointer_unaligned_rejected',lambda:m._check_pointer(Tensor((1,),base=2**32+1),10),True)
guard('BF16_pointer_last_aligned',lambda:m._check_pointer(Tensor((1,),dtype='bf16',base=2**64-22),10))
a=inputs();a[1].layout='sparse';rejected('reject_nonstrided',a,ValueError,'strided')
for z,h,n in [(0,2,64),(2,0,64),(2,3,0)]:
    a=inputs(z,h,n)
    for t in a:t.strides=(17,9,5,2) if t.ndim==4 else (17,9,3)
    success('empty_noncanonical',a,True)
import ast
text=source.read_text();tree=ast.parse(text)
assert 'num_stages=2' in text and 'num_warps=8' in text
assert not any(isinstance(n,ast.Attribute) and n.attr in ('contiguous','resolve_conj','resolve_neg','clone') for n in ast.walk(tree))
for line in ['start_m = tl.program_id(0).to(tl.int64)','off_hz = tl.program_id(1).to(tl.int64)','offs_n = tl.arange(0, BLOCK_N).to(tl.int64)','offs_k = tl.arange(0, 128).to(tl.int64)','K_ptrs += BLOCK_N * stride_kn','V_ptrs += BLOCK_N * stride_vk']:
    assert line in text,line
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
records.append({'name':'owning_device_exception_restoration','status':'PASS'})
print(json.dumps({'status':'PASS','controls':len(records),'source_structure':'PASS','GPU_execution':False}))
