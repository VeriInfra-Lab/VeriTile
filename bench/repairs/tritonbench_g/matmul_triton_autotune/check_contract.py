"""Execute widened candidate source on CPU; native instruction numerics remain deferred."""
import os
os.environ['TRITON_INTERPRET']='1'
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
import triton.language as tl

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'matmul_triton_autotune_view_safe.py'
class LogicalTensor:
    """Logical CUDA metadata with CPU storage for the existing interpreter."""
    def __init__(self,value,device):self.value,self.device=value,device
    def __getattr__(self,name):return getattr(self.value,name)

class TorchHost:
    Tensor=(torch.Tensor,LogicalTensor)
    def __init__(self):
        self.current_device=torch.device('cuda',0)
        self.cuda=SimpleNamespace(device=self.device)
    def __getattr__(self,name):return getattr(torch,name)
    @contextmanager
    def device(self,device):
        previous=self.current_device;self.current_device=device
        try:yield
        finally:self.current_device=previous
    def empty(self,shape,device,dtype):
        if device.type=='cuda':return LogicalTensor(torch.empty(shape,dtype=dtype),device)
        return torch.empty(shape,device=device,dtype=dtype)

class Kernel:
    def __init__(self,body,config,host):
        self.body,self.config,self.calls,self.host=body,config,[],host
        self.launch_devices,self.fail_launch=[],False
    def __getitem__(self,grid):
        def launch(a,b,c,*args,**kwargs):
            assert self.host.current_device==a.device==b.device==c.device
            self.launch_devices.append(self.host.current_device)
            if self.fail_launch:raise RuntimeError('mock submission failure')
            self.calls.append((a,b,c))
            assert len(args)==9 and args[3:]==(*a.stride(),*b.stride(),*c.stride())
            actual=tuple(value.value if isinstance(value,LogicalTensor) else value for value in (a,b,c))
            self.body[grid(self.config)](*actual,*args,**self.config,**kwargs)
        return launch

def instantiate(config):
    nodes=ast.parse(SOURCE.read_text()).body
    selected=[]
    for node in nodes:
        if isinstance(node,ast.FunctionDef) and node.name in ['matmul_kernel','leaky_relu','matmul_view_safe']:
            if node.name in ['matmul_kernel','leaky_relu']:
                node.decorator_list=[ast.Attribute(ast.Name('triton',ast.Load()),'jit',ast.Load())]
                ast.fix_missing_locations(node)
            selected.append(node)
        elif isinstance(node,ast.Assign):selected.append(node)
    host=TorchHost()
    ns={'torch':host,'triton':triton,'tl':tl}
    exec(compile(ast.Module(selected,[]),str(SOURCE),'exec'),ns)
    kernel=Kernel(ns['matmul_kernel'],config,host);ns['matmul_kernel']=kernel
    return ns['matmul_view_safe'],kernel

def main():
    original=ast.parse((HERE.parents[2]/'tritonbench_g/matmul_triton_autotune/matmul_triton_autotune.py').read_text().split('################################################################')[0])
    repaired=ast.parse(SOURCE.read_text())
    names=['is_cuda','get_cuda_autotune_config','get_hip_autotune_config','get_autotune_config','leaky_relu']
    for name in names:
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in repaired.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    source=ast.unparse(repaired)
    assert '.resolve_' not in source and '.contiguous(' not in source
    assert 'tl.dot(a, b, accumulator)' in source and 'c = accumulator.to(tl.float16)' in source
    assert 'safe_k = tl.where' in source and 'safe_cm = tl.where' in source
    configs=[dict(BLOCK_SIZE_M=m,BLOCK_SIZE_N=256,BLOCK_SIZE_K=k,GROUP_SIZE_M=g) for m,k,g in [(128,64,8),(64,32,8),(128,16,1)]]
    old_cuda=torch.Tensor.is_cuda
    torch.Tensor.is_cuda=property(lambda self:True)
    try:
        count=0
        for config in configs:
            call,kernel=instantiate(config)
            for m,n,k in [(1,1,1),(129,257,65),(1025,3,3),(2,3,0),(0,3,4),(2,0,4)]:
                for activation in ['', 'leaky_relu', 'identity']:
                    a=(((torch.arange(m*k)%5)-2)/8).reshape(m,k)
                    b=torch.ones((k,n),dtype=torch.float32)/8
                    before_a,before_b=a.clone(),b.clone();kernel.calls.clear()
                    out=call(a,b,activation)
                    expected=a@b
                    if activation=='leaky_relu':expected=torch.where(expected>=0,expected,0.01*expected)
                    assert out.dtype==torch.float16 and torch.equal(out,expected.half())
                    assert torch.equal(a,before_a) and torch.equal(b,before_b)
                    if m*n:assert kernel.calls[0][0] is a and kernel.calls[0][1] is b and kernel.calls[0][2] is out
                    else:assert not kernel.calls
                    count+=1
            for kind in ['offset','transposeB','broadcastB','alias','singletonStride']:
                a=torch.ones((3,5));b=torch.ones((5,4))
                if kind=='offset':a=torch.ones(20)[3:18].reshape(3,5);b=torch.ones((8,9))[2:7,1:5]
                if kind=='transposeB':b=torch.ones((4,5)).t()
                if kind=='broadcastB':b=torch.ones((1,4)).expand(5,4)
                if kind=='alias':a=torch.ones((3,3));b=a
                if kind=='singletonStride':a=torch.as_strided(torch.ones(5),(1,5),(2**40,1))
                out=call(a,b)
                assert torch.equal(out,(a@b).half());count+=1
        rejects=0
        call,kernel=instantiate(configs[0])
        for kind in ['dtype','rank','shape','noncontiguousA','negativeA','negativeB','unalignedA','unalignedB','activation']:
            a=torch.ones((3,5));b=torch.ones((5,4));activation=''
            if kind=='dtype':a=a.half()
            if kind=='rank':a=a[0]
            if kind=='shape':b=b[:4]
            if kind=='noncontiguousA':a=torch.ones((5,3)).t()
            if kind=='negativeA':a=torch._neg_view(a)
            if kind=='negativeB':b=torch._neg_view(b)
            if kind=='unalignedA':a=torch.frombuffer(bytearray(61),dtype=torch.float32,count=15,offset=1).reshape(3,5)
            if kind=='unalignedB':b=torch.frombuffer(bytearray(81),dtype=torch.float32,count=20,offset=1).reshape(5,4)
            if kind=='activation':activation=None
            kernel.calls.clear()
            try:call(a,b,activation)
            except (TypeError,ValueError):rejects+=1
            else:raise AssertionError(kind)
            assert not kernel.calls
        a,b=[LogicalTensor(value,torch.device('cuda',1)) for value in (torch.ones((3,5)),torch.ones((5,4)))]
        out=call(a,b)
        assert out is kernel.calls[-1][2] and out.device==a.device
        assert torch.equal(out.value,(a.value@b.value).half())
        assert kernel.host.current_device==torch.device('cuda',0) and kernel.launch_devices[-1]==torch.device('cuda',1)
        kernel.fail_launch=True
        try:call(a,b)
        except RuntimeError as error:assert str(error)=='mock submission failure'
        else:raise AssertionError('expected mock submission failure')
        finally:kernel.fail_launch=False
        assert kernel.host.current_device==torch.device('cuda',0) and kernel.launch_devices[-1]==torch.device('cuda',1)
        print(f'PASS: {count} actual source CPU interpreter calls across all3 configs, tails/partial groups, K0/empty, offsets/aliases/broadcast/stridedB and 2**40 singleton stride, original objects forwarded; {rejects} rejection classes. No GPU execution.')
        print('PASS: additional cuda:1 host admission executes the CPU kernel and restores ambient cuda:0 on success and submission exception.')
    finally:torch.Tensor.is_cuda=old_cuda

if __name__=='__main__':main()
