"""Actual source Triton CPU interpreter controls; native CUDA numerics are not tested."""
import os
os.environ['TRITON_INTERPRET']='1'
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('checked_matmul_source',HERE/'matmul_kernel_repaired.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
current_device, launch_devices, fail_launch = [torch.device('cuda',0)], [], [False]

class DeviceContext:
    def __init__(self,device):self.device=device
    def __enter__(self):self.previous=current_device[0];current_device[0]=self.device
    def __exit__(self,*exc):current_device[0]=self.previous

class LogicalTensor:
    """CUDA host metadata backed by the same CPU storage used by the interpreter."""
    def __init__(self,value):self.value=value;self.device=torch.device('cuda',1)
    def __getattr__(self,name):return getattr(self.value,name)

class TorchHost:
    Tensor=(torch.Tensor,LogicalTensor)
    cuda=SimpleNamespace(device=DeviceContext)
    def __getattr__(self,name):return getattr(torch,name)

class Launch:
    def __init__(self,kernel):self.kernel=kernel
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            assert current_device[0]==args[1].device
            launch_devices.append(current_device[0])
            if fail_launch[0]:raise RuntimeError('mock submission failure')
            actual=tuple(value.value if isinstance(value,LogicalTensor) else value for value in args)
            return self.kernel[grid](*actual,**kwargs)
        return call

module.torch=TorchHost()
module.matmul_kernel=Launch(module.matmul_kernel)

def raw(t):
    return torch.as_strided(t,(t.untyped_storage().nbytes()//t.element_size(),),(1,),storage_offset=0)

def matrix(rows,cols,kind):
    base=torch.full((max(1,rows*cols*4+rows*9+cols*9+31),),-77.,dtype=torch.float32)
    if kind=='transpose': strides=(1,max(rows,1)+3)
    elif kind=='step2': strides=(max(cols,1)*2+3,2)
    else: strides=(max(cols,1)+3,1)
    view=torch.as_strided(base,(rows,cols),strides,storage_offset=5)
    if rows*cols:
        view.copy_(((torch.arange(rows*cols).reshape(rows,cols)%5)-2)/8)
    return view

def run(m,n,k,bm=16,bn=16,bk=16,kind='padded',alias=False,cast=False):
    a,b,c=matrix(m,k,kind),matrix(k,n,kind),matrix(m,n,kind)
    if alias:
        assert m==n==k
        b=a
    if cast: a.fill_(2051);b.fill_(1)
    before_a,before_b,before_c=raw(a).clone(),raw(b).clone(),raw(c).clone()
    expected=(a.double()@b.double()).half().float()
    ret=module.matmul(c,a,b,m,n,k,bm,bn,bk)
    assert ret is None and torch.equal(c,expected),(m,n,k,kind)
    assert torch.equal(raw(a),before_a) and torch.equal(raw(b),before_b)
    touched=set(c.storage_offset()+i*c.stride(0)+j*c.stride(1) for i in range(m) for j in range(n))
    outside=torch.tensor([i for i in range(raw(c).numel()) if i not in touched],dtype=torch.int64)
    assert torch.equal(raw(c)[outside],before_c[outside])

def main():
    tree=ast.parse((HERE/'matmul_kernel_repaired.py').read_text())
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_kernel')
    assert [a.arg for a in kernel.args.args][3:12]==['M','N','K','stride_am','stride_ak','stride_bk','stride_bn','stride_cm','stride_cn']
    calls=[n for n in ast.walk(kernel) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    loads=[n for n in calls if n.func.attr=='load']
    assert len(loads)==2 and all({k.arg for k in n.keywords}=={'mask','other'} for n in loads)
    assert any(n.func.attr=='dot' and len(n.args)==3 and not n.keywords for n in calls)
    assert any(n.func.attr=='cast' and ast.unparse(n.args[1])=='tl.float16' for n in calls)
    original_is_cuda=torch.Tensor.is_cuda
    torch.Tensor.is_cuda=property(lambda self:True)
    try:
        count=0
        for kind in ['padded','transpose','step2']:
            for m,n,k in [(1,1,1),(17,19,33),(35,18,3),(2,3,0),(0,3,5),(2,0,5),(0,0,0)]:
                run(m,n,k,kind=kind);count+=1
        for bm,bn,bk in [(32,16,32),(16,32,64),(64,32,16)]:
            run(33,35,67,bm,bn,bk);count+=1
        run(3,3,3,alias=True);count+=1
        run(1,1,1,cast=True);count+=1
        rejected=0
        for mutate in ['alias','negative','shape','dtype','unaligned','block','booldim','stride','bound']:
            a,b,c=matrix(3,5,'padded'),matrix(5,4,'padded'),matrix(3,4,'padded')
            dims=[3,4,5];blocks=[16,16,16]
            if mutate=='alias':c=torch.as_strided(a,(3,4),a.stride())
            if mutate=='negative':a=torch._neg_view(a)
            if mutate=='shape':dims[0]=2
            if mutate=='dtype':a=a.half()
            if mutate=='unaligned':a=torch.frombuffer(bytearray(61),dtype=torch.float32,count=15,offset=1).reshape(3,5)
            if mutate=='block':blocks[0]=24
            if mutate=='booldim':dims[0]=True
            if mutate=='stride':a=torch.as_strided(a,(3,5),(0,1))
            if mutate=='bound':
                dims=[1,4,5];a=torch.as_strided(a,(1,5),(2**31,1));c=matrix(1,4,'padded')
            try:module.matmul(c,a,b,*dims,*blocks)
            except ValueError:rejected+=1
            else:raise AssertionError(mutate)
        a,b,c=[LogicalTensor(t) for t in (matrix(3,5,'padded'),matrix(5,4,'padded'),matrix(3,4,'padded'))]
        expected=(a.value.double()@b.value.double()).half().float()
        assert module.matmul(c,a,b,3,4,5,16,16,16) is None
        assert torch.equal(c.value,expected)
        assert current_device[0]==torch.device('cuda',0) and launch_devices[-1]==torch.device('cuda',1)
        fail_launch[0]=True
        try:module.matmul(c,a,b,3,4,5,16,16,16)
        except RuntimeError as error:assert str(error)=='mock submission failure'
        else:raise AssertionError('expected mock submission failure')
        finally:fail_launch[0]=False
        assert current_device[0]==torch.device('cuda',0) and launch_devices[-1]==torch.device('cuda',1)
        print(f'PASS: {count} actual source CPU interpreter calls; offsets, A/B aliases, all tails, output holes, empty/K0, exact None and 2051→2052 cast; {rejected} guard rejections. No GPU execution.')
        print('PASS: additional cuda:1 host admission executes the CPU kernel and restores ambient cuda:0 on success and submission exception.')
    finally:torch.Tensor.is_cuda=original_is_cuda

if __name__=='__main__':main()
