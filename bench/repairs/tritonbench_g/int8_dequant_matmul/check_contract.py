"""Execute the complete repaired AST on CPU storage; native FP validation is separate."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'int8_dequant_matmul_repaired.py'
spec=importlib.util.spec_from_file_location('add_helpers',HERE.parent/'add_example/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
class Tensor(helpers.TensorFacade):
    @property
    def ndim(self):return self.data.ndim
    def storage_offset(self):return self.data.storage_offset()
    def zero_(self):self.data.zero_();return self

def function(name):return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
def main():
    torch.set_num_threads(1)
    source=SOURCE.read_text()
    ast_literal=next(l.split(' := ',1)[1] for l in (HERE/'Proofs/Source.lean').read_text().splitlines() if l.startswith('def productionAST :'))
    assert json.loads(ast_literal)==ast.dump(ast.parse(source))
    kernel=function('_int8_matmul_rowwise_dequantize')
    assert hashlib.sha256(ast.dump(kernel).encode()).hexdigest()=='6f143759eb16c660d5ff874831400cce929c147e2b103030b045472fc2c8db47'
    wrapper=function('int8_matmul_rowwise_dequantize')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','resolve_neg') for n in ast.walk(wrapper))
    class Config:
        def __init__(self,kwargs,**kw):self.kwargs=kwargs;self.__dict__.update(kw)
    triton=SimpleNamespace(Config=Config,cdiv=lambda n,d:(n+d-1)//d)
    confns={'triton':triton}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function('get_configs_io_bound')],type_ignores=[])),str(SOURCE),'exec'),confns)
    configs=eval(compile(ast.Expression(next(k.value for k in kernel.decorator_list[0].keywords if k.arg=='configs')),str(SOURCE),'eval'),confns)
    assert len(configs)==418
    for c in configs:
        m,n,k,s=(c.kwargs[x] for x in ('BLOCK_M','BLOCK_N','BLOCK_K','SPLIT_K'))
        assert 16<=m<=256 and 32<=n<=256 and k*s<=2048 and 1<=s<=16
        assert hasattr(c,'pre_hook')==(s!=1)
    pids=[0,0];calls=[];selection=[0];trials=[[]]
    class Pointer:
        def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
        def __add__(self,offset):return Pointer(self.tensor,self.offset+offset)
        @property
        def dtype(self):return SimpleNamespace(element_ty=self.tensor.dtype)
    def physical(t):return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes()//t.element_size(),),(1,))
    def load(ptr,mask=None,other=0):
        offsets=torch.as_tensor(ptr.offset);mask=torch.ones_like(offsets,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,offsets.shape)
        value=torch.full(offsets.shape,other,dtype=ptr.tensor.dtype)
        value[mask]=physical(ptr.tensor)[ptr.tensor.storage_offset()+offsets[mask].long()]
        return value
    def store(ptr,value,mask,atomic=False):
        offsets=torch.as_tensor(ptr.offset);mask=torch.broadcast_to(mask,offsets.shape)
        loc=ptr.tensor.storage_offset()+offsets[mask].long();values=torch.broadcast_to(value,offsets.shape)[mask]
        flat=physical(ptr.tensor)
        flat[loc]=(flat[loc]+values).to(ptr.tensor.dtype) if atomic else values.to(ptr.tensor.dtype)
    tl=SimpleNamespace(int64=torch.int64,int32=torch.int32,float32=torch.float32,
        program_id=lambda axis:torch.tensor(pids[axis],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),
        cdiv=triton.cdiv,max_contiguous=lambda v,_:v,multiple_of=lambda v,_:v,
        load=load,store=store,atomic_add=lambda p,v,mask:store(p,v,mask,True),
        zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),dot=lambda a,b:a.int()@b.int())
    kernel.decorator_list=[]
    for a in kernel.args.args:a.annotation=None
    kns={'tl':tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel],type_ignores=[])),str(SOURCE),'exec'),kns)
    class Launch:
        def __getitem__(self,grid):
            def run(a,b,c,bias,x,w,M,N,K,div,has_bias,*strides,**kwargs):
                assert strides==(K,1,N,1,N,1)
                for ci in trials[0]+[selection[0]]:
                    cfg=configs[ci];gx,gz=grid(cfg.kwargs)
                    if hasattr(cfg,'pre_hook'):calls.append('zero');cfg.pre_hook({'C':c})
                    calls.append('launch')
                    for z in reversed(range(gz)):
                        for pid in reversed(range(gx)):
                            pids[:]=[pid,z]
                            kns['_int8_matmul_rowwise_dequantize'](Pointer(a.data),Pointer(b.data),Pointer(c.data),
                                None if bias is None else Pointer(bias.data),Pointer(x.data),Pointer(w.data),
                                *[torch.tensor(v,dtype=torch.int64) for v in (M,N,K)],div,has_bias,
                                *[torch.tensor(v,dtype=torch.int64) for v in strides],**cfg.kwargs,
                                EVEN_K=(K%(cfg.kwargs['BLOCK_K']*cfg.kwargs['SPLIT_K'])==0),**kwargs)
            return run
    def empty(shape,device,dtype):calls.append('allocation');return Tensor(torch.full(shape,float('nan'),dtype=dtype),str(device))
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,int8=torch.int8,float32=torch.float32,float16=torch.float16,
        empty=empty,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()),
        _C=SimpleNamespace(_overlaps=lambda a,b:torch._C._overlaps(a.data,b.data)))
    ns={'torch':fake,'triton':triton,'tl':tl,'_int8_matmul_rowwise_dequantize':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function('_check'),wrapper],type_ignores=[])),str(SOURCE),'exec'),ns)
    invoke=ns['int8_matmul_rowwise_dequantize']
    def expected(a,b,x,w,bias,cfg):
        M,K=a.shape;N=b.shape[1];bk=cfg['BLOCK_K'];split=cfg['SPLIT_K'];result=torch.zeros(M,N,dtype=torch.float16)
        for z in reversed(range(split)):
            acc=torch.zeros(M,N,dtype=torch.int32)
            for start in range(0,K,bk*split):
                lo=start+z*bk;hi=min(lo+bk,K)
                if lo<hi:acc+=a[:,lo:hi].int()@b[lo:hi].int()
            v=(w[None,:]*(x[:,None]*(acc.float()*torch.tensor(1/(127*127),dtype=torch.float32)))).half()
            if bias is not None and z==0:v=v+bias.half()[None,:]
            result=v if split==1 else result+v
        return result
    accepted=0
    unique={tuple(c.kwargs.values()):i for i,c in enumerate(configs)}
    for ci in unique.values():
        selection[0]=ci;trials[0]=[]
        M,K,N=2,35,3
        aa=torch.arange(M*K+8).to(torch.int8);bb=(torch.arange(K*N+8)*3).to(torch.int8)
        xx=torch.linspace(-2,3,M+4);ww=torch.linspace(-3,2,N+4);bi=torch.linspace(-1,1,N+4)
        args=[Tensor(aa[3:3+M*K].reshape(M,K)),Tensor(bb[2:2+K*N].reshape(K,N)),Tensor(xx[1:1+M]),Tensor(ww[1:1+N]),Tensor(bi[1:1+N])]
        old=[v.clone() for v in (aa,bb,xx,ww,bi)];calls.clear();got=invoke(*args)
        want=expected(*[a.data for a in args],configs[ci].kwargs)
        torch.testing.assert_close(got.data,want,rtol=0,atol=0,equal_nan=True)
        assert got.shape==(M,N) and got.dtype==torch.float16 and got.device==args[0].device
        assert got.data.stride()==(max(N,1),1)
        for v,o in zip((aa,bb,xx,ww,bi),old):torch.testing.assert_close(v,o,rtol=0,atol=0)
        assert calls==['allocation']+(['zero'] if configs[ci].kwargs['SPLIT_K']!=1 else [])+['launch']
        accepted+=1
    # Empty/K=0, tails, singleton inert strides, aliases, and invalid FP payloads.
    for M,K,N,bias in [(0,3,4,True),(3,0,4,True),(3,4,0,False),(0,4,0,False),(0,4,3,False),(1,1,1,True),(17,33,35,False),(3,64,5,True)]:
        selection[0]=19;trials[0]=[0,19,20]
        a=Tensor(torch.ones(M,K,dtype=torch.int8));b=Tensor(torch.ones(K,N,dtype=torch.int8))
        x=Tensor(torch.full((M,),float('inf')));w=Tensor(torch.zeros(N));bi=Tensor(torch.ones(N)) if bias else None
        if M==1:a.data=torch.as_strided(a.data,(M,K),(2**62,2**62))
        calls.clear();got=invoke(a,b,x,w,bi)
        assert got.shape==(M,N) and got.data.stride()==(max(N,1),1)
        assert got.dtype==torch.float16 and got.device==a.device
        torch.testing.assert_close(got.data,expected(a.data,b.data,x.data,w.data,None if bi is None else bi.data,configs[selection[0]].kwargs),rtol=0,atol=0,equal_nan=True)
        if M==0 or N==0:assert calls==['allocation']
        accepted+=1
    selection[0]=18;trials[0]=[]
    ints=torch.arange(16,dtype=torch.int8);floats=torch.arange(8,dtype=torch.float32)
    a=Tensor(ints[:6].reshape(2,3));b=Tensor(ints[1:7].reshape(3,2));x=Tensor(floats[:2]);w=Tensor(floats[1:3]);bi=Tensor(floats[2:4])
    got=invoke(a,b,x,w,bi);torch.testing.assert_close(got.data,expected(a.data,b.data,x.data,w.data,bi.data,configs[18].kwargs),rtol=0,atol=0);accepted+=1
    a=Tensor(torch.ones(2,3,dtype=torch.int8));b=Tensor(torch.ones(3,4,dtype=torch.int8));x=Tensor(torch.ones(2));w=Tensor(torch.ones(4));bi=Tensor(torch.ones(4))
    bad=[([],b,x,w,bi),(a,b,x,w,[]),(Tensor(torch.ones(3,dtype=torch.int8)),b,x,w,bi),
        (Tensor(a.data.to_sparse()),b,x,w,bi),(Tensor(a.data.float()),b,x,w,bi),(a,b,Tensor(x.data.half()),w,bi),
        (a,b,x,w,Tensor(bi.data.half())),(Tensor(a.data,'cpu',False),b,x,w,bi),(a,Tensor(b.data,'cuda:1'),x,w,bi),
        (Tensor(torch.ones(3,2,dtype=torch.int8).T),b,x,w,bi),(Tensor(torch._neg_view(a.data)),b,x,w,bi),
        (a,Tensor(torch.ones(2,4,dtype=torch.int8)),x,w,bi),(a,b,Tensor(torch.ones(3)),w,bi),(a,b,x,w,Tensor(torch.ones(5))),
        (a,b,x,w,Tensor(torch.ones(8)[::2])),(a,b,x,w,Tensor(torch._neg_view(bi.data)))]
    shared=torch.ones(16);bad.append((Tensor(shared.view(torch.int8)[:6].reshape(2,3)),b,Tensor(shared[:2]),w,bi))
    bad.extend((a,b,x,Tensor(helpers.unaligned_tensor(i)),bi) for i in (1,2,3))
    class Metadata(Tensor):
        def __init__(self,shape,dtype,offset=0):
            super().__init__(torch.empty(1,dtype=dtype));self._shape=shape;self._offset=offset
        @property
        def shape(self):return self._shape
        @property
        def ndim(self):return len(self._shape)
        def numel(self):
            import math
            return math.prod(self.shape)
        def storage_offset(self):return self._offset
    def metadata_case(M,K,N,offsets=(0,0,0,0,0)):
        return (Metadata((M,K),torch.int8,offsets[0]),Metadata((K,N),torch.int8,offsets[1]),
                Metadata((M,),torch.float32,offsets[2]),Metadata((N,),torch.float32,offsets[3]),Metadata((N,),torch.float32,offsets[4]))
    bad.extend([metadata_case(2**61,0,1),metadata_case(16*(2**31-1)+1,0,1)])
    for axis in range(5):
        offsets=[0]*5;offsets[axis]=2**63
        bad.append(metadata_case(2,3,4,offsets))
    for args in bad:
        calls.clear()
        try:invoke(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('bad input admitted')
        assert not calls
    ns['_check'](*metadata_case(0,2**62,3))
    ns['_check'](*metadata_case(3,2**62,0))
    print(f'PASS: {accepted} complete AST output/metadata/frame cases, {len(bad)} guard rejections; all 418 config bounds/hooks, {len(unique)} distinct tile/split configurations.')
    print('CPU emulation uses INT32 wrap and FP16 operations; selected native MMA/WGMMA and floating primitive conformance remain deferred.')
if __name__=='__main__':main()
