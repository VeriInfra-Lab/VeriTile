"""Actual-source CPU interpreter replay; CUDA metadata/context are mocked."""
import importlib.util
import ast
import copy
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"]="1"
import torch
import triton
import triton.language as tl

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("cat_repair",HERE/"index_select_cat_repaired.py")
repair=importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class Tensor:
    def __init__(self,value,**overrides):
        self.value=value
        self.__dict__.update(overrides)

    def __getattr__(self,name):
        if name=="is_cuda": return True
        if name=="device": return torch.device("cuda:0")
        return getattr(self.value,name)

    def __getitem__(self,key):
        return Tensor(self.value[key])


class Launch:
    count=0
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            Launch.count+=1
            kernel[grid](*[x.value if isinstance(x,Tensor) else x for x in args],**kwargs)
        return launch


kernel=repair.index_select_cat_fwd_kernel_repaired
repair.index_select_cat_fwd_kernel_repaired=Launch()
repair.torch=SimpleNamespace(Tensor=Tensor,float32=torch.float32,int64=torch.int64,
    strided=torch.strided,cuda=SimpleNamespace(device=lambda _:nullcontext()))


def case(ids,width,rows=3,singleton=False):
    case.count+=1
    case.empty+=not bool(ids and width)
    special=torch.tensor([0,-2147483648,1,2139095040,2143289345,-1],dtype=torch.int32)
    source=special.repeat(((rows+2)*width+5)//6)[:(rows+2)*width].view(torch.float32).reshape(rows+2,width)[1:-1]
    backing=torch.full((len(ids)+2,width),37.,dtype=torch.float32)
    output=backing[1:-1]
    if singleton and width==1:
        source=source.as_strided(source.shape,(1,17))
        output=output.as_strided(output.shape,(1,29))
    index=torch.tensor([99]+ids+[99],dtype=torch.int64)[1:-1]
    source_before=source.clone().view(torch.int32)
    args=(Tensor(output),Tensor(source),Tensor(index))
    launches=Launch.count
    assert repair.index_select_cat_fwd_repaired(*args) is args[0]
    assert Launch.count-launches==bool(ids and width)
    expected=source[ids] if width and ids else torch.empty((len(ids),width))
    assert torch.equal(output.view(torch.int32),expected.view(torch.int32))
    assert torch.equal(source.view(torch.int32),source_before)
    assert torch.equal(index,torch.tensor(ids,dtype=torch.int64))
    assert torch.all(backing[[0,-1]]==37.)
    return args


def reject(args,index,value):
    changed=list(args)
    changed[index]=value
    before=Launch.count
    saved=args[0].value.clone().view(torch.int32)
    try: repair.index_select_cat_fwd_repaired(*changed)
    except (ValueError,TypeError,IndexError):
        reject.count+=1
        assert Launch.count==before
        assert torch.equal(args[0].value.view(torch.int32),saved)
    else: raise AssertionError("unsupported input admitted")


case.count=0
case.empty=0
reject.count=0
kernels=(kernel,repair.index_select_cat_fwd_kernel_index_commuted)
for kernel in kernels:
    for width in (0,1,3,512,513):
        case([],width)
        case([2,0,2,1,0,1,2],width,singleton=True)
    case([123,-456],0,rows=0)
    args=case([0,2],3)
    # Read roles may share physical storage even with their distinct dtypes.
    raw=torch.tensor([1,0,1,0],dtype=torch.int64)
    source=raw.view(torch.float32).reshape(4,2)
    index=raw[:3]
    out=torch.empty(3,2)
    before=raw.clone()
    wrapper=Tensor(out)
    assert repair.index_select_cat_fwd_repaired(wrapper,Tensor(source),Tensor(index)) is wrapper
    assert torch.equal(out.view(torch.int32),source[index].view(torch.int32))
    assert torch.equal(raw,before)
reject(args,2,Tensor(torch.tensor([-1,2],dtype=torch.int64)))
reject(args,2,Tensor(torch.tensor([3,2],dtype=torch.int64)))
reject(args,2,Tensor(torch.tensor([0,2],dtype=torch.int32)))
reject(args,1,Tensor(args[1].value.to(torch.float16)))
reject(args,0,Tensor(torch.empty(3,3)))
reject(args,0,Tensor(args[1].value[:2]))
reject(args,1,Tensor(torch._neg_view(args[1].value)))
reject(args,2,Tensor(args[2].value,device=torch.device("cuda:1")))
reject(args,1,Tensor(args[1].value,is_cuda=False))
reject(args,1,None)
reject(args,2,Tensor(torch.tensor([0,99,2,99],dtype=torch.int64)[::2]))
for remainder in (1,2,3):
    raw=torch.empty(args[1].numel()*4+4,dtype=torch.uint8)
    value=torch.empty(0,dtype=torch.float32).set_(raw.untyped_storage()[remainder:],0,args[1].shape,args[1].stride())
    reject(args,1,Tensor(value))
raw=torch.empty(20,dtype=torch.uint8)
index=torch.empty(0,dtype=torch.int64).set_(raw.untyped_storage()[4:],0,(2,),(1,))
reject(args,2,Tensor(index))
raw=torch.zeros(40,dtype=torch.uint8)
source=torch.empty(0,dtype=torch.float32).set_(raw.untyped_storage(),0,(3,3),(3,1))
out=torch.empty(0,dtype=torch.float32).set_(raw.untyped_storage()[4:],0,(2,3),(3,1))
reject((Tensor(out),Tensor(source),args[2]),0,Tensor(out))

# The displayed candidate differs only by commuting the existing integer product.
tree=ast.parse((HERE/'index_select_cat_repaired.py').read_text())
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
original=functions['index_select_cat_fwd_kernel_repaired']
candidate=copy.deepcopy(functions['index_select_cat_fwd_kernel_index_commuted'])
assert ast.unparse(original.body[2])=='indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX).to(tl.int64)'
assert ast.unparse(candidate.body[2])=='indices = BLOCK_SIZE_INDEX * pid0 + tl.arange(0, BLOCK_SIZE_INDEX).to(tl.int64)'
candidate.name=original.name
candidate.body[2]=copy.deepcopy(original.body[2])
assert ast.dump(candidate,include_attributes=False)==ast.dump(original,include_attributes=False)
source_text=(HERE/'Kernels.lean').read_text()
assert 'import bench.' not in source_text and 'def target' not in source_text
assert 'source_ptr = tl.float32, index_ptr = tl.int64' in source_text
assert 'other=$((0 : Int))' in source_text

# Execute the original wrapper/JIT source on physically allocated CPU backing.
# CUDA metadata/context are facades; no test benchmark's automatic GPU call runs.
original_path=HERE.parents[2]/'tritonbench_g/index_select_cat/index_select_cat.py'
tree=ast.parse(original_path.read_text())
tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
env={'torch':repair.torch,'triton':triton,'tl':tl}
exec(compile(tree,str(original_path),'exec'),env)
original_kernel=env['index_select_cat_fwd_kernel']
class OriginalLaunch:
    count=0
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            OriginalLaunch.count+=1
            original_kernel[grid](*[x.value if isinstance(x,Tensor) else x for x in args],**kwargs)
        return launch
env['index_select_cat_fwd_kernel']=OriginalLaunch()
original_wrapper=env['index_select_cat_fwd']
out=torch.full((3,1),37.)
original_wrapper(Tensor(out),Tensor(torch.tensor([[1.],[2.]])),Tensor(torch.tensor([0,1,0],dtype=torch.int64)))
assert out[:,0].tolist()==[1.,2.,37.]
source=torch.arange(8,dtype=torch.float32).reshape(4,2)[::2]
backing=torch.full((3,2),37.)
original_wrapper(Tensor(backing[:2]),Tensor(source),Tensor(torch.tensor([0,1],dtype=torch.int64)))
assert backing.tolist()==[[0.,1.],[37.,37.],[4.,5.]]
alias=torch.tensor([[10.],[20.]])
original_wrapper(Tensor(alias),Tensor(alias),Tensor(torch.tensor([1,0],dtype=torch.int64)))
assert alias[:,0].tolist()==[20.,20.]
backing=torch.tensor([[999.],[1.],[2.]])
out=torch.empty(1,1)
original_wrapper(Tensor(out),Tensor(backing[1:]),Tensor(torch.tensor([-1],dtype=torch.int64)))
assert out.item()==999.

print(f"Index select: {case.count} ordinary cases ({case.empty} empty), 2 shared-read cases, "
      f"{Launch.count} repaired-kernel CPU Triton launches, {reject.count} rejections; both source variants passed.")
print(f"Original source counterexamples reproduced with {OriginalLaunch.count} CPU Triton launches: "
      "N>M truncation, output stride/frame mismatch, input/output alias corruption, unchecked negative index.")
print("Exact candidate AST, FP32 payload bits, masks, tails and frames passed; CUDA metadata/context are facades, no GPU execution.")
