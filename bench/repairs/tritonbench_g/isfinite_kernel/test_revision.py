"""Actual JIT CPU structure tests, scalar callback adapter, and checked API controls."""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import torch
import triton
from triton.runtime.interpreter import TensorHandle

HERE = Path(__file__).resolve().parent
source = HERE / "isfinite_kernel_repaired.py"
spec = importlib.util.spec_from_file_location("isfinite_checked", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = HERE.parents[2] / "tritonbench_g/isfinite_kernel/isfinite_kernel.py"
oldtree, newtree = ast.parse(original.read_text()), ast.parse(source.read_text())
for name in ["StridedBuffer", "isfinite_func"]:
    old = next(n for n in oldtree.body if getattr(n,"name",None)==name)
    new = next(n for n in newtree.body if getattr(n,"name",None)==name)
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
assert hashlib.sha256(original.read_bytes()).hexdigest()=="9661c6a64cb20de85a75775c6daec1dc7e16397d1aa48b83296e9f37039a0ba6"

# The actual candidate differs only by commuting the block-index product.
import copy
left = copy.deepcopy(next(n for n in newtree.body if getattr(n, "name", None) == "isfinite_strided_chunk"))
right = copy.deepcopy(next(n for n in newtree.body if getattr(n, "name", None) == "isfinite_strided_chunk_index_commuted"))
right.name = left.name
product = right.body[1].value.left
assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
product.left, product.right = product.right, product.left
assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)

# Installed Triton CPU interpretation cannot call the CUDA libdevice externs.
# Adapt only those two scalar ports; all kernel source, masks, casts, pointer
# arithmetic, loads, and stores execute through the real JIT interpreter.
def classify(x):
    result = np.isfinite(x.handle.data)
    return module.tl.tensor(TensorHandle(result, module.tl.int1), module.tl.block_type(module.tl.int1, x.shape))
module._finitef = classify
module._isfinited = classify

source_cases = 0
for kernel in (module.isfinite_strided_chunk, module.isfinite_strided_chunk_index_commuted):
    for precision, width, exponent, fraction in [(torch.float32,4,255,23),(torch.float64,8,2047,52)]:
        high = exponent << fraction
        sign = 1 << (width*8-1)
        patterns = [0,sign,1,sign+1,1<<fraction,high-1,sign+high-1,
                    high,sign+high,high+1,high+(1<<(fraction-1)),sign+high+3]
        for n in [1,12,1023,1024,1025]:
            for sx, sy in [(1,1),(-1,-2),(0,3),(3,-1)]:
                stride_bytes = max(1,abs(sx))*width
                storage = torch.full((64+max(1,n)*stride_bytes,), 0x63, dtype=torch.uint8)
                xstart = 32 + ((n-1)*stride_bytes if sx<0 else 0)
                for i in range(n):
                    pattern = patterns[i%len(patterns)]
                    if sx==0: pattern=high
                    storage[xstart+i*sx*width:xstart+i*sx*width+width] = torch.tensor(
                        list(pattern.to_bytes(width,"little")),dtype=torch.uint8)
                before = storage.clone()
                out = torch.full((64+n*abs(sy),), 0xA5,dtype=torch.uint8)
                ystart = 16 + ((n-1)*abs(sy) if sy<0 else 0)
                x = storage[xstart:xstart+width].view(precision)
                y = out[ystart:ystart+1].view(torch.bool)
                kernel[(triton.cdiv(n,1024),)](x,y,sx,sy,n,BLOCK_SIZE=1024)
                expected = torch.full_like(out,0xA5)
                for i in range(n):
                    bits = int.from_bytes(before[xstart+i*sx*width:xstart+i*sx*width+width].tolist(),"little")
                    expected[ystart+i*sy] = (bits>>fraction)&exponent != exponent
                assert torch.equal(storage,before) and torch.equal(out,expected)
                source_cases += 1
        # Same-storage reinterpretation, interleaved disjoint accessed footprints.
        n, step = 12, 2*width
        storage = torch.full((64+n*step,),0xA5,dtype=torch.uint8)
        for i in range(n):
            storage[16+i*step:16+i*step+width]=torch.tensor(list(patterns[i].to_bytes(width,"little")),dtype=torch.uint8)
        before = storage.clone()
        kernel[(1,)](storage[16:16+width].view(precision),
            storage[16+width:17+width].view(torch.bool),step//width,step,n,BLOCK_SIZE=1024)
        expected = before.clone()
        for i in range(n): expected[16+width+i*step] = (patterns[i]>>fraction)&exponent != exponent
        assert torch.equal(storage,expected)
        source_cases += 1

actual_torch = torch
class DType:
    def __init__(self,name,itemsize): self.name,self.itemsize=name,itemsize
fp32,fp64,booltype,uint8 = DType("fp32",4),DType("fp64",8),DType("bool",1),DType("uint8",1)
device = SimpleNamespace(type="cuda",index=2)
calls,current = [],[0]
class Tensor:
    def __init__(self,n=1025,dtype=fp32,*,stride=1,pointer=4096,capacity=None,storage=4096,
                 negative=False,layout="strided",dev=device,shape=None):
        self.shape = (n,) if shape is None else shape
        self.dtype,self.layout,self.device = dtype,layout,dev
        self.pointer,self.storage,self.capacity = pointer,storage,capacity if capacity is not None else max(1,n)*dtype.itemsize*max(1,abs(stride))+128
        self._stride,self.negative = (stride,),negative
    def is_neg(self): return self.negative
    def stride(self): return self._stride
    def data_ptr(self): return self.pointer
    def element_size(self): return self.dtype.itemsize
    def untyped_storage(self): return SimpleNamespace(data_ptr=lambda:self.storage,nbytes=lambda:self.capacity)
    def numel(self): return math.prod(self.shape)
class Guard:
    def __init__(self,d): assert d==2
    def __enter__(self): self.saved=current[0];current[0]=2
    def __exit__(self,*args): current[0]=self.saved
class Kernel:
    def __getitem__(self,grid):
        def launch(x,y,sx,sy,n,**kwargs):
            assert current[0]==2 and grid==((n+1023)//1024,) and kwargs==dict(BLOCK_SIZE=1024)
            calls.append((x,y,sx,sy,n))
        return launch
module.torch = SimpleNamespace(Tensor=Tensor,strided="strided",float32=fp32,float64=fp64,bool=booltype,
                               cuda=SimpleNamespace(_DeviceGuard=Guard))
module.isfinite_strided_chunk = Kernel()
wrapper = module.isfinite_func_wrapper_rank_1_repaired
def pair(n=1025,dtype=fp32):
    return Tensor(n,dtype),Tensor(n,booltype,pointer=1<<50,storage=1<<50)
def reject(x,y):
    calls.clear()
    try: wrapper(x,out0=y)
    except (TypeError,ValueError): assert not calls and current[0]==0
    else: raise AssertionError("unsupported operands accepted")

rejections=0
x,y=pair()
for bad in [object(),Tensor(dtype=uint8),Tensor(layout="sparse"),Tensor(negative=True),
            Tensor(dev=SimpleNamespace(type="cpu",index=None)),Tensor(shape=(1,1025)),
            Tensor(pointer=4097),Tensor(capacity=4),Tensor(stride=1<<63),Tensor(n=-1),Tensor(n=True)]:
    reject(bad,y);rejections+=1
for bad in [Tensor(1024,booltype),Tensor(dtype=fp32),Tensor(dtype=booltype,stride=0),
            Tensor(dtype=booltype,pointer=4097),Tensor(dtype=booltype,negative=True),
            Tensor(dtype=booltype,dev=SimpleNamespace(type="cuda",index=3))]:
    reject(x,bad);rejections+=1

admissions=0
for dtype in [fp32,fp64]:
    for n in [0,1,1025,2**30+1,2**31,2**31+1]:
        for sx,sy in [(1,1),(-2,-3),(0,2)]:
            x,y=pair(n,dtype)
            x._stride,y._stride=(sx,),(sy,)
            x.capacity=max(1,n)*max(1,abs(sx))*dtype.itemsize+128
            y.capacity=max(1,n)*abs(sy)+128
            if sx<0:x.pointer+=(max(1,n)-1)*abs(sx)*dtype.itemsize
            if sy<0:y.pointer+=(max(1,n)-1)*abs(sy)
            calls.clear()
            assert wrapper(x,out0=y) is y and current[0]==0
            start=0
            for xx,yy,ssx,ssy,count in calls:
                assert (ssx,ssy)==(sx,sy) and count==min(module.CHUNK_SIZE,n-start)
                assert xx.unwrap() is x and yy.unwrap() is y
                assert xx.data_ptr()==x.data_ptr()+start*sx*dtype.itemsize
                assert yy.data_ptr()==y.data_ptr()+start*sy
                start+=count
            assert start==n
            admissions+=1

# Explicit raw StridedBuffer preserves arbitrary base dtype, byte pointer,
# metadata flags, signed/zero strides, and exact output identity.
for dtype in [fp32,fp64]:
    width=dtype.itemsize
    base=Tensor(256,uint8,negative=True)
    x=module.StridedBuffer(base,shape=(5,),strides=(-2,),dtype=dtype,offset=8)
    y=module.StridedBuffer(base,shape=(5,),strides=(-2*width,),dtype=booltype,offset=9*width)
    calls.clear()
    assert wrapper(x,out0=y) is y and len(calls)==1
    assert calls[0][0].unwrap() is base and calls[0][0].data_ptr()==x.data_ptr()
    admissions+=1

# Execute original empty wrapper AST, retaining the actual division-by-zero defect.
names={"heuristics_for_tile_size","heuristics_for_num_warps","isfinite_func_wrapper_rank_1"}
nodes=[n for n in oldtree.body if getattr(n,"name",None) in names]
env=dict(math=math,triton=triton,torch=module.torch,Union=__import__('typing').Union,StridedBuffer=module.StridedBuffer)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original),"exec"),env)
x,y=pair(0)
try:env["isfinite_func_wrapper_rank_1"](x,out0=y)
except ZeroDivisionError:pass
else:raise AssertionError("expected original empty launch defect")

report=dict(source_interpreter_cases=source_cases,guard_rejections=rejections,wrapper_admissions=admissions,
            original_empty_counterexample=True,large_extent_controls=[2**31,2**31+1],
            scalar_cpu_adapter="numpy.isfinite for CUDA libdevice externs only",gpu_execution=False)
(HERE/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(report)
