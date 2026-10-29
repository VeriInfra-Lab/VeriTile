"""Actual repaired source AST on CPU; native arithmetic/library validation deferred."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'dequantize_matmul_repaired.py'
spec=importlib.util.spec_from_file_location('add_helpers',HERE.parent/'add_example/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
class Tensor(helpers.TensorFacade):
    @property
    def ndim(self):return self.data.ndim
    def stride(self,axis):return self.data.stride(axis)

def function(name):return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

def main():
    assert hashlib.sha256(ast.dump(function('dequantize_kernel')).encode()).hexdigest()=='4ff75876d411a4501ccb00beb039fab92b5fb612edf312448f398019001f82e6'
    configs=[]
    deco=function('dequantize_kernel').decorator_list[0]
    for conf in next(kw.value for kw in deco.keywords if kw.arg=='configs').elts:
        configs.append(ast.literal_eval(conf.args[0]))
    assert len(configs)==17
    assert min(c['BLOCK_SIZE_K'] for c in configs)==32 and max(c['BLOCK_SIZE_K'] for c in configs)==256
    assert min(c['BLOCK_SIZE_N'] for c in configs)==32 and max(c['BLOCK_SIZE_N'] for c in configs)==256
    pids=[0,0];calls=[];selection=[0]
    class Pointer:
        def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
        def __add__(self,offset):return Pointer(self.tensor,offset)
    def physical(t):
        return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes()//t.element_size(),),(1,))
    def load(ptr,mask,other):
        result=torch.full_like(ptr.offset,other,dtype=ptr.tensor.dtype)
        result[mask]=physical(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offset[mask].long()]
        return result
    def store(ptr,value,mask):
        physical(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offset[mask].long()]=value[mask].to(ptr.tensor.dtype)
    tl=SimpleNamespace(int64=torch.int64,program_id=lambda axis:torch.tensor(pids[axis],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),load=load,store=store)
    node=function('dequantize_kernel');node.decorator_list=[]
    for arg in node.args.args:arg.annotation=None
    kns={'tl':tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),kns)
    class Launch:
        def __getitem__(self,grid):
            def run(b,s,fpb,*args):
                cfg=configs[selection[0]];gx,gy=grid(cfg);calls.append('launch')
                for pk in reversed(range(gx)):
                    for pn in reversed(range(gy)):
                        pids[:]=[pk,pn]
                        kns['dequantize_kernel'](Pointer(b.data),Pointer(s.data),Pointer(fpb.data),*args,**cfg)
            return run
    def empty(shape,device,dtype):calls.append('allocation');return Tensor(torch.empty(shape,dtype=dtype),str(device))
    def mm(a,b,out):calls.append('mm');torch.mm(a.data,b.data,out=out.data)
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int8=torch.int8,empty=empty,mm=mm,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()),
        _C=SimpleNamespace(_overlaps=lambda a,b:torch._C._overlaps(a.data,b.data)))
    ns={'torch':fake,'triton':SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d),'dequantize_kernel':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function('_check'),function('matmul_dequantize_int8')],type_ignores=[])),str(SOURCE),'exec'),ns)
    wrapper=function('matmul_dequantize_int8')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','resolve_neg') for n in ast.walk(wrapper))
    source=(HERE/'Kernels.lean').read_text();assert '.Proofs.' not in source
    contract=(HERE/'Contract.lean').read_text()
    for field in ('rank','layout','dtype','device','contiguous','noNeg','shape','inputAlias','extent','launch','alignment','output'):
        assert f'  {field} :' in contract
    kernel=function('dequantize_kernel')
    assignments={n.targets[0].id:ast.unparse(n.value) for n in kernel.body if isinstance(n,ast.Assign)}
    assert assignments['bs_offs']=='(n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_scale'
    assert assignments['k_block_idx']=='tl.program_id(axis=0).to(tl.int64)'
    assert assignments['n_block_idx']=='tl.program_id(axis=1).to(tl.int64)'
    accepted=0
    for cfg in range(len(configs)):
        selection[0]=cfg
        shapes=[(2,33,35)] if cfg else [(0,3,4),(2,0,3),(2,3,0),(2,1,1),(3,7,5),(1,33,35)]
        for M,K,N in shapes:
            for provided in (False,True):
                aa=torch.arange(M*K+6,dtype=torch.float32)/7-2;a=aa[2:2+M*K].reshape(M,K)
                bb=torch.arange(K*N+6,dtype=torch.int32).to(torch.int8);b=bb[2:2+K*N].reshape(K,N)
                ss=torch.arange(N+4,dtype=torch.float32)/31;s=ss[1:1+N]
                if M==1:a=torch.as_strided(a,(M,K),(2**40,1))
                if K==1:b=torch.as_strided(b,(K,N),(2**40,1))
                if N==1:s=torch.as_strided(s,(N,),(2**62,))
                cc=torch.full((M*N+8,),123.0);out=Tensor(cc[3:3+M*N].reshape(M,N)) if provided else None
                old=[v.clone() for v in (aa,bb,ss,cc)];calls.clear()
                got=ns['matmul_dequantize_int8'](Tensor(a),Tensor(b),Tensor(s),out)
                expected=torch.mm(a,b.float()*s)
                torch.testing.assert_close(got.data,expected,rtol=0,atol=0,equal_nan=True)
                assert got.shape==(M,N) and got.dtype==torch.float32 and got.device==torch.device('cuda:0')
                if provided:assert got is out
                for original,prior in zip((aa,bb,ss),old):torch.testing.assert_close(original,prior,rtol=0,atol=0)
                torch.testing.assert_close(cc[:3],old[3][:3],rtol=0,atol=0)
                torch.testing.assert_close(cc[3+M*N:],old[3][3+M*N:],rtol=0,atol=0)
                assert calls==['allocation']*(1 if provided else 2)+(['launch'] if K and N else [])+['mm']
                accepted+=1
    a=Tensor(torch.ones(2,3));b=Tensor(torch.ones(3,4,dtype=torch.int8));s=Tensor(torch.ones(4))
    bad=[([],b,s,None),(Tensor(torch.ones(3)),b,s,None),(Tensor(a.data.to_sparse()),b,s,None),
        (Tensor(a.data.double()),b,s,None),(a,Tensor(b.data.float()),s,None),(a,b,Tensor(s.data.double()),None),
        (Tensor(a.data,'cpu',False),b,s,None),(a,Tensor(b.data,'cuda:1'),s,None),
        (Tensor(torch.ones(3,2).T),b,s,None),(Tensor(torch._neg_view(a.data)),b,s,None),
        (a,Tensor(torch.ones(2,4,dtype=torch.int8)),s,None),(a,b,Tensor(torch.ones(5)),None),
        (a,b,s,[]),(a,b,s,Tensor(torch.ones(2,5))),(a,b,s,Tensor(torch.ones(2,4,dtype=torch.float64))),
        (a,b,s,Tensor(torch.ones(4,2).T)),(a,b,s,Tensor(torch._neg_view(torch.ones(2,4)))),
        (Tensor(torch.empty(0,2**31,device='meta')),Tensor(torch.empty(2**31,1,dtype=torch.int8,device='meta')),Tensor(torch.empty(1,device='meta')),None),
        (Tensor(torch.empty(0,1,device='meta')),Tensor(torch.empty(1,65535*32+1,dtype=torch.int8,device='meta')),Tensor(torch.empty(65535*32+1,device='meta')),None),
        (Tensor(torch.empty(2**61,0,device='meta')),Tensor(torch.empty(0,1,dtype=torch.int8)),Tensor(torch.ones(1)),None)]
    alias=torch.ones(16);bad.append((Tensor(alias[:6].reshape(2,3)),b,s,Tensor(alias[8:16].reshape(2,4))))
    mixed=torch.ones(2,4)
    bad.append((Tensor(mixed),Tensor(mixed.view(torch.int8).reshape(4,8)),Tensor(torch.ones(8)),None))
    bad.extend((a,b,Tensor(helpers.unaligned_tensor(i)),None) for i in (1,2,3))
    for args in bad:
        calls.clear()
        try:ns['matmul_dequantize_int8'](*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid inputs accepted')
        assert not calls
    # Read-only FP32 A/scales can share storage.
    shared=torch.arange(10,dtype=torch.float32)
    aa=Tensor(shared[:8].reshape(2,4));bb=Tensor(torch.arange(16,dtype=torch.int8).reshape(4,4));ss=Tensor(shared[2:6])
    got=ns['matmul_dequantize_int8'](aa,bb,ss)
    torch.testing.assert_close(got.data,aa.data@(bb.data.float()*ss.data),rtol=0,atol=0)
    accepted+=1
    print(f'PASS: {accepted} complete actual AST output/metadata/frame cases over all 17 configs; {len(bad)} guard rejections; canonical singleton strides and input alias.')
    print('CUDA is mocked. CPU torch.mm is exercised; native selected library plans and FP primitive conformance remain deferred.')
if __name__=='__main__':main()
