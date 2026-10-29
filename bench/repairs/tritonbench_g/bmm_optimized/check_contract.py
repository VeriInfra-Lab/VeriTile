"""Run each preserved BMM configuration through the actual source CPU interpreter."""
import os
os.environ['TRITON_INTERPRET']='1'
import ast
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import torch
import triton
import triton.language as tl
import logging

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'bmm_optimized_repaired.py'

class Kernel:
    def __init__(self,body,config):self.body,self.config,self.calls=body,config,[]
    def __getitem__(self,grid):
        def launch(a,b,c,m,n,k):
            meta={'M':m,'N':n,'K':k,**self.config}
            assert a.is_contiguous() and b.is_contiguous()
            self.calls.append((a,b,c))
            self.body[grid(meta)](a,b,c,m,n,k,**self.config,
                DIVISIBLE_M=m%self.config['TILE_M']==0,
                DIVISIBLE_N=n%self.config['TILE_N']==0,DIVISIBLE_K=k%32==0)
        return launch

def instantiate(config):
    nodes=ast.parse(SOURCE.read_text()).body
    kernel=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='bmm_kernel')
    kernel.decorator_list=[ast.Attribute(ast.Name('triton',ast.Load()),'jit',ast.Load())]
    ast.fix_missing_locations(kernel)
    ns={'torch':torch,'triton':triton,'tl':tl,'logging':logging}
    exec(compile(ast.Module([kernel]+[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in ['bmm','_validate_inputs']],[]),str(SOURCE),'exec'),ns)
    body=ns['bmm_kernel']
    ns['bmm_kernel']=Kernel(body,config)
    return ns['bmm'],ns['bmm_kernel']

def main():
    original=ast.parse((HERE.parents[2]/'tritonbench_g/bmm_optimized/bmm_optimized.py').read_text().split('################################################################')[0])
    repaired=ast.parse(SOURCE.read_text())
    old=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='bmm_kernel')
    new=next(n for n in repaired.body if isinstance(n,ast.FunctionDef) and n.name=='bmm_kernel')
    assert [ast.dump(n) for n in old.decorator_list]==[ast.dump(n) for n in new.decorator_list]
    calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert any(n.func.attr=='dot' and ast.unparse(n.keywords[0].value)=='False' for n in calls)
    assert len([n for n in calls if n.func.attr=='load' and any(k.arg=='other' and k.value.value==0.0 for k in n.keywords)])==2
    assert '.resolve_' not in ast.unparse(repaired)
    configs=[dict(TILE_M=m,TILE_N=n,TILE_K=32,GROUP_M=1 if m==n==32 else 2) for m,n in [(32,32),(64,32),(64,64),(128,32),(128,64),(128,128)]]
    old_cuda,old_device=torch.Tensor.is_cuda,torch.cuda.device
    old_contiguous=torch.Tensor.contiguous
    events=[]
    def contiguous(t,*a,**k):events.append('contiguous');return old_contiguous(t,*a,**k)
    torch.Tensor.is_cuda=property(lambda self:True)
    torch.cuda.device=lambda device:nullcontext()
    torch.Tensor.contiguous=contiguous
    try:
        count=0
        for stages in [2,3]:
            for config in configs:
                for m,n,k in [(3,5,33),(65,35,32),(2,3,0)]:
                    call,kernel=instantiate(config)
                    a=torch.ones((2,m,k),dtype=torch.float32)
                    b=torch.ones((2,k,n),dtype=torch.float32)
                    before=(a.clone(),b.clone());events.clear()
                    out=call(a,b)
                    assert out.shape==(2,m,n) and torch.equal(out,torch.full_like(out,k))
                    assert torch.equal(a,before[0]) and torch.equal(b,before[1])
                    assert events[:2]==['contiguous','contiguous'] and len(kernel.calls)==1
                    count+=1
        call,kernel=instantiate(configs[0])
        for kind in ['offset','transpose','broadcast','alias','emptyBatch','emptyM','emptyN']:
            a=torch.arange(24,dtype=torch.float32).reshape(2,3,4)/8
            b=torch.ones((2,4,3),dtype=torch.float32)
            if kind=='offset': a=torch.arange(50,dtype=torch.float32)[5:29].reshape(2,3,4)/8
            if kind=='transpose':a=a.transpose(1,2);b=torch.ones((2,3,3))
            if kind=='broadcast':a=a[:1].expand(2,-1,-1);b=b[:1].expand(2,-1,-1)
            if kind=='alias':a=torch.ones((2,3,3));b=a
            if kind=='emptyBatch':a=a[:0];b=b[:0]
            if kind=='emptyM':a=a[:,:0]
            if kind=='emptyN':b=b[:,:,:0]
            expected=torch.bmm(a,b);events.clear();kernel.calls.clear()
            out=call(a,b)
            assert torch.equal(out,expected) and events[:2]==['contiguous','contiguous']
            assert bool(kernel.calls)==bool(out.numel());count+=1
        rejects=0
        for kind in ['rank','dtype','negative','shape','unaligned']:
            a=torch.ones((1,3,4));b=torch.ones((1,4,3))
            if kind=='rank':a=a[0]
            if kind=='dtype':a=a.half()
            if kind=='negative':a=torch._neg_view(a)
            if kind=='shape':b=b[:,:3]
            if kind=='unaligned':a=torch.frombuffer(bytearray(49),dtype=torch.float32,count=12,offset=1).reshape(1,3,4)
            events.clear()
            try:call(a,b)
            except ValueError:rejects+=1
            else:raise AssertionError(kind)
            assert events==[]
        print(f'PASS: {count} actual source CPU interpreter executions across12 preserved config/stage pairs, K tails and zero/empty cases; offsets/transposes/broadcast/aliases, preserved contiguous calls; {rejects} rejection classes. No GPU execution.')
    finally:
        torch.Tensor.is_cuda=old_cuda;torch.cuda.device=old_device;torch.Tensor.contiguous=old_contiguous

if __name__=='__main__':main()
