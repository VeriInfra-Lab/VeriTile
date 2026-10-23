"""Actual-source CPU interpreter replay with labeled CUDA host metadata facade."""
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cache_repair",HERE / "cache_transform_triton_repaired.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class Tensor:
    def __init__(self,value,**overrides):
        self.value=value
        self.__dict__.update(overrides)

    def __getattr__(self,name):
        if name == "is_cuda":
            return True
        if name == "device":
            return torch.device("cuda:0")
        return getattr(self.value,name)


class Launch:
    count=0

    def __getitem__(self,grid):
        def call(*args,**kwargs):
            Launch.count+=1
            assert kwargs["START"] == 0
            kernel[grid](*[x.value if isinstance(x,Tensor) else x for x in args],**kwargs)
        return call


allocations=[]

def allocate(shape,dtype,device):
    assert dtype==torch.float32
    count=shape[0]*shape[1]
    backing=torch.full((count+4,),0x5AA55AA5,dtype=torch.int32)
    allocations.append(backing)
    return Tensor(backing[2:2+count].view(torch.float32).reshape(shape))

def storage_bytes(value):
    storage=value.untyped_storage()
    return torch.empty(0,dtype=torch.uint8).set_(storage,0,(storage.nbytes(),),(1,))

accepted=0
rejected=0

kernel=repair.gather_cache_bytes
repair.gather_cache_bytes=Launch()
repair.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
    int32=torch.int32,int64=torch.int64,uint8=torch.uint8,
    empty=allocate,
    tensor=lambda rows,dtype,device: Tensor(torch.tensor(rows,dtype=dtype)),
    cuda=SimpleNamespace(device=lambda _: nullcontext()))


def equal(a,b):
    return torch.equal(a.view(torch.int32),b.view(torch.int32))


def case(values,width,prompt,alias=False,singleton=False,strided_lengths=False,negative_lengths=False):
    global accepted
    words=torch.tensor([0,-2147483648,1,2139095040,2143289345,-1],dtype=torch.int32)
    cos=words.repeat((9*width+5)//6)[:9*width].view(torch.float32).reshape(9,width)[1:8]
    sin=cos if alias else torch.arange(10*width,dtype=torch.float32).reshape(10,width)[1:9]
    if singleton and width == 1:
        cos=cos.as_strided(cos.shape,(1,19))
        sin=sin.as_strided(sin.shape,(1,23))
    before=(cos.clone(),sin.clone())
    storage_before=(storage_bytes(cos).clone(),storage_bytes(sin).clone())
    lengths=torch.tensor([99]+values+[99],dtype=torch.int32)[1:-1]
    if strided_lengths:
        lengths=torch.tensor([v for v in values for _ in range(2)],dtype=torch.int32)[::2]
    if negative_lengths:
        lengths=torch._neg_view(-lengths)
    length_before=storage_bytes(lengths).clone()
    args=(Tensor(lengths),Tensor(cos),Tensor(sin),prompt)
    launches=Launch.count
    result=repair.get_xine_cache(*args)
    rows=[r for v in values for r in range(v)] if prompt else [v-1 for v in values]
    assert isinstance(result,tuple) and len(result)==2
    for out,source in zip(result,(cos,sin)):
        assert out.shape==(len(rows),width)
        assert out.dtype==torch.float32 and out.is_contiguous() and not out.requires_grad
        assert equal(out.value,source[rows])
        if out.numel():
            assert out.untyped_storage().data_ptr()!=source.untyped_storage().data_ptr()
    assert Launch.count-launches==2*bool(rows and width)
    assert equal(cos,before[0]) and equal(sin,before[1])
    assert torch.equal(storage_bytes(cos),storage_before[0]) and torch.equal(storage_bytes(sin),storage_before[1])
    assert torch.equal(storage_bytes(lengths),length_before)
    for backing in allocations:
        assert torch.all(backing[:2]==0x5AA55AA5) and torch.all(backing[-2:]==0x5AA55AA5)
    accepted+=1
    return args


def reject(args,index,value):
    global rejected
    changed=list(args)
    changed[index]=value
    launches=Launch.count
    try:
        repair.get_xine_cache(*changed)
    except (ValueError,TypeError):
        assert Launch.count==launches
        rejected+=1
    else:
        raise AssertionError("invalid input admitted")


for width in (0,1,3,65):
    for prompt in (False,True):
        case([],width,prompt)
        case([1,3,7],width,prompt)
        case([1,3,7],width,prompt,alias=True,singleton=True)
        if prompt:
            case([0,2,0],width,True)
case([1,3,7],3,False,strided_lengths=True)
case([1,3,7],3,True,negative_lengths=True)
args=case([1,3,7],3,False)
reject(args,0,Tensor(torch.tensor([0,1,2],dtype=torch.int32)))
reject(args,0,Tensor(torch.tensor([-1,1,2],dtype=torch.int32)))
reject(args,0,Tensor(torch.tensor([8,1,2],dtype=torch.int32)))
reject(args,0,Tensor(args[0].value.to(torch.int64)))
reject(args,1,Tensor(args[1].value.to(torch.float16)))
reject(args,1,Tensor(torch._neg_view(args[1].value)))
reject(args,1,Tensor(args[1].value,is_cuda=False))
reject(args,2,Tensor(args[2].value,device=torch.device("cuda:1")))
reject(args,3,1)
print("Cache transform: actual source replay passed FP32 words, prompt/decode, tails, offsets, aliases, singleton strides, empty dimensions and guard rejection")

print(f"{accepted} admissions, {rejected} rejections, {Launch.count} actual source launches; output guard sentinels and complete input storage frames passed.")
