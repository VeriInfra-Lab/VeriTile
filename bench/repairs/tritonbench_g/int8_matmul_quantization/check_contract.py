"""Whole actual-source CPU execution and checked-domain regression; CUDA is mocked."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'int8_matmul_quantization_repaired.py'
spec=importlib.util.spec_from_file_location('add_helpers',HERE.parent/'add_example/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
class Tensor(helpers.TensorFacade):
    @property
    def ndim(self):return self.data.ndim
    def element_size(self):return self.data.element_size()
    def storage_offset(self):return self.data.storage_offset()
    def stride(self,dim):return self.data.stride(dim)
    def t(self):return Tensor(self.data.t(),str(self.device))
    def contiguous(self):return Tensor(self.data.contiguous(),str(self.device))
    def abs(self):return Tensor(self.data.abs(),str(self.device))
    def amax(self,*args,**kw):return Tensor(self.data.amax(*args,**kw),str(self.device))
    def squeeze(self,*args):return Tensor(self.data.squeeze(*args),str(self.device))
    def to(self,dtype):return Tensor(self.data.to(dtype),str(self.device))
    def __truediv__(self,other):return Tensor(self.data/(other.data if isinstance(other,Tensor) else other),str(self.device))
    def zero_(self):self.data.zero_();return self
    def fill_(self,v):self.data.fill_(v);return self

def function(name):return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)

def main():
    torch.set_num_threads(1)
    ast_literal=next(l.split(' := ',1)[1] for l in (HERE/'Kernels.lean').read_text().splitlines() if l.startswith('def productionAST :'))
    assert json.loads(ast_literal)==ast.dump(ast.parse(SOURCE.read_text()))
    for name,digest in KERNEL_HASHES.items():assert hashlib.sha256(ast.dump(function(name)).encode()).hexdigest()==digest
    pids=[0,0];calls=[];chosen=[0];trials=[[]]
    def configs(name):
        return [ast.literal_eval(c.args[0]) for c in next(k.value for k in function(name).decorator_list[0].keywords if k.arg=='configs').elts]
    gemmconfigs=configs('matmul_kernel');assert len(gemmconfigs)==34
    assert len(configs('quantize_int8_perrow_kernel'))==4
    class Pointer:
        def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset
        def __add__(self,offset):return Pointer(self.tensor,self.offset+offset)
    def physical(t):return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes()//t.element_size(),),(1,))
    def load(p,mask=None,other=0):
        offsets=torch.as_tensor(p.offset);mask=torch.ones_like(offsets,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,offsets.shape)
        value=torch.full(offsets.shape,other,dtype=p.tensor.dtype)
        value[mask]=physical(p.tensor)[p.tensor.storage_offset()+offsets[mask].long()]
        return value
    def store(p,v,mask=None,atomic=False):
        offsets=torch.as_tensor(p.offset);mask=torch.ones_like(offsets,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,offsets.shape)
        loc=p.tensor.storage_offset()+offsets[mask].long();v=torch.broadcast_to(v,offsets.shape)[mask];flat=physical(p.tensor)
        flat[loc]=(flat[loc]+v).to(p.tensor.dtype) if atomic else v.to(p.tensor.dtype)
    triton=SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d,next_power_of_2=lambda n:1<<(n-1).bit_length() if n else 0)
    tl=SimpleNamespace(int64=torch.int64,int32=torch.int32,int8=torch.int8,float32=torch.float32,float16=torch.float16,
        program_id=lambda axis:torch.tensor(pids[axis],dtype=torch.int32),arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),
        cdiv=triton.cdiv,load=load,store=store,atomic_add=lambda p,v,mask:store(p,v,mask,True),
        zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),dot=lambda a,b:a.int()@b.int(),
        abs=torch.abs,max=lambda a,axis:torch.amax(a,dim=axis),maximum=torch.maximum)
    kns={'tl':tl}
    for name in KERNEL_HASHES:
        node=function(name);node.decorator_list=[]
        for a in node.args.args:a.annotation=None
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),kns)
    class QuantLaunch:
        def __getitem__(self,grid):
            def launch(fpa,a,scales,M,K,*strides):
                calls.append('quantize')
                assert strides[:5]==(max(K,1),1,max(K,1),1,1)
                for pid in reversed(range(grid[0])):
                    pids[:]=[pid,0]
                    kns['quantize_int8_perrow_kernel'](Pointer(fpa.data),Pointer(a.data),Pointer(scales.data),
                        torch.tensor(M,dtype=torch.int64),torch.tensor(K,dtype=torch.int64),
                        *[torch.tensor(x,dtype=torch.int64) for x in strides[:5]],*strides[5:])
            return launch
    class GemmLaunch:
        def __getitem__(self,grid):
            def launch(a,sa,b,sb,c,M,N,K,*strides):
                assert strides==(max(K,1),1,1,max(N,1) if b.is_contiguous() else 1,1 if b.is_contiguous() else max(K,1),1,max(N,1),1)
                selected=trials[0]+[chosen[0]]
                for pos,ci in enumerate(selected):
                    if pos<len(trials[0]) or trials[0]:c.zero_();calls.append('reset')
                    cfg=gemmconfigs[ci];gx,gz=grid(cfg);calls.append('gemm')
                    for z in reversed(range(gz)):
                        for pid in reversed(range(gx)):
                            pids[:]=[pid,z]
                            kns['matmul_kernel'](Pointer(a.data),Pointer(sa.data),Pointer(b.data),Pointer(sb.data),Pointer(c.data),
                                *[torch.tensor(v,dtype=torch.int64) for v in (M,N,K,*strides)],**cfg)
            return launch
    def allocate(shape,device,dtype,zero):
        calls.append('allocation');shape=(shape,) if isinstance(shape,int) else shape
        return Tensor(torch.zeros(shape,dtype=dtype) if zero else torch.empty(shape,dtype=dtype),str(device))
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,int8=torch.int8,float16=torch.float16,float32=torch.float32,
        empty=lambda shape,device,dtype:allocate(shape,device,dtype,False),zeros=lambda shape,device,dtype:allocate(shape,device,dtype,True),
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()),_C=SimpleNamespace(_overlaps=lambda a,b:torch._C._overlaps(a.data,b.data)))
    tree=ast.parse(SOURCE.read_text());names=['_tensor','_check_row','_check_gemm','_check_helper','quantize_int8_perrow','matmul_quantize_int8','matmul_int8','quantize_int8']
    body=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'torch':fake,'triton':triton,'quantize_int8_perrow_kernel':QuantLaunch(),'matmul_kernel':GemmLaunch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(SOURCE),'exec'),ns)
    assert ns['_GEMM_TILES']==[tuple(c[k] for k in ('BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K','SPLIT_K','GROUP_SIZE_M')) for c in gemmconfigs]
    def expected(a,sa,b,sb,cfg,dtype=torch.float16):
        M,K=a.shape;N=b.shape[1];bk=cfg['BLOCK_SIZE_K'];split=cfg['SPLIT_K'];result=torch.zeros(M,N,dtype=dtype)
        for z in reversed(range(split)):
            acc=torch.zeros(M,N,dtype=torch.int32)
            for start in range(0,K,bk*split):
                lo=start+z*bk;hi=min(lo+bk,K)
                if lo<hi:acc+=a[:,lo:hi].int()@b[lo:hi].int()
            value=(acc.float()*sa.float()[:,None]*sb[None,:]).half()
            result=value.to(dtype) if split==1 else result+value.to(dtype)
        return result
    accepted=0
    for ci in range(34):
        chosen[0]=ci;trials[0]=[0,17] if ci==0 else []
        for column in (False,True):
            for supplied in (None,torch.float16,torch.float32):
                M,K,N=2,35,3
                aa=torch.arange(M*K+6).to(torch.int8);bb=(torch.arange(K*N+6)*3).to(torch.int8)
                a=Tensor(aa[2:2+M*K].reshape(M,K));b=Tensor(bb[1:1+K*N].reshape(N,K).T if column else bb[1:1+K*N].reshape(K,N))
                sa=Tensor(torch.tensor([-0.13,0.11],dtype=torch.float16 if column else torch.float32));sb=Tensor(torch.tensor([0.21,-0.17,0.05]))
                backing=torch.full((M*N+8,),9.,dtype=supplied or torch.float16);out=Tensor(backing[3:3+M*N].reshape(M,N)) if supplied else None
                old=[v.clone() for v in (aa,bb,sa.data,sb.data,backing)];got=ns['matmul_int8'](a,sa,b,sb,out)
                torch.testing.assert_close(got.data,expected(a.data,sa.data,b.data,sb.data,gemmconfigs[ci],supplied or torch.float16),rtol=0,atol=0,equal_nan=True)
                assert got.dtype==(supplied or torch.float16) and got.shape==(M,N) and got.device==a.device
                if supplied:assert got is out
                for v,oldv in zip((aa,bb,sa.data,sb.data),old):torch.testing.assert_close(v,oldv,rtol=0,atol=0)
                torch.testing.assert_close(backing[:3],old[-1][:3],rtol=0,atol=0);torch.testing.assert_close(backing[3+M*N:],old[-1][3+M*N:],rtol=0,atol=0)
                accepted+=1
    for M,K in [(0,0),(0,7),(1,1),(3,5),(5,33)]:
        backing=torch.linspace(-2,2,M*K+6);fpa=Tensor(backing[2:2+M*K].reshape(M,K))
        if M==1:fpa.data=torch.as_strided(fpa.data,(M,K),(2**62,2**62 if K==1 else 1))
        old=backing.clone();calls.clear();a,sa=ns['quantize_int8_perrow'](fpa)
        if M:
            scale=fpa.data.abs().amax(1)/127
            torch.testing.assert_close(a.data,(fpa.data/scale[:,None]).to(torch.int8),rtol=0,atol=0)
            torch.testing.assert_close(sa.data,scale.half(),rtol=0,atol=0)
        else:assert calls==['allocation','allocation']
        assert a.shape==(M,K) and a.dtype==torch.int8 and sa.shape==(M,) and sa.dtype==torch.float16
        torch.testing.assert_close(backing,old,rtol=0,atol=0);accepted+=1
    chosen[0]=17;trials[0]=[0,17]
    for M,K,N in [(0,0,3),(2,5,0),(2,33,5),(1,1,1)]:
        fpa=Tensor(torch.arange(M*K).float().reshape(M,K)/17);b=Tensor(torch.arange(K*N).to(torch.int8).reshape(N,K).T);sb=Tensor(torch.ones(N)*0.21)
        got=ns['matmul_quantize_int8'](fpa,b,sb)
        if M:
            scales=fpa.data.abs().amax(1)/127;a=(fpa.data/scales[:,None]).to(torch.int8)
            torch.testing.assert_close(got.data,expected(a,scales.half(),b.data,sb.data,gemmconfigs[17]),rtol=0,atol=0,equal_nan=True)
        assert got.shape==(M,N);accepted+=1
    # Torch helper: original input offset, all literal axis branches, and tuple metadata.
    for axis in (-2,-1,0,1):
        weight=Tensor(torch.linspace(-2,3,20)[2:17].reshape(3,5));prior=weight.data.clone()
        q,s,third=ns['quantize_int8'](weight,axis)
        want_s=weight.data.abs().amax(axis,keepdim=True)/127
        torch.testing.assert_close(q.data,(weight.data/want_s).to(torch.int8),rtol=0,atol=0)
        torch.testing.assert_close(s.data,want_s.squeeze(axis),rtol=0,atol=0)
        assert third is None and s.dtype==torch.float32
        assert (q.data.stride()==(1,3)) if axis==0 else q.data.is_contiguous()
        torch.testing.assert_close(weight.data,prior,rtol=0,atol=0);accepted+=1
    # Axis0 .contiguous() is a no-op for degenerate transpose views.
    for M,N in [(1,3),(3,1),(1,1),(0,3),(3,0)]:
        for axis in (-2,-1,0,1):
            if (M,N)[axis]==0:continue
            data=torch.arange(M*N).float().reshape(M,N)
            if M==1 or N==1:data=torch.as_strided(data,(M,N),(2**40 if M==1 else max(N,1),2**40 if N==1 else 1))
            weight=Tensor(data);scale=data.abs().amax(axis,keepdim=True)/127
            expected_q=(data/scale).to(torch.int8)
            if axis==0:expected_q=expected_q.t().contiguous().t()
            got,s,third=ns['quantize_int8'](weight,axis)
            assert got.data.stride()==expected_q.stride() and got.shape==(M,N) and third is None
            torch.testing.assert_close(got.data,expected_q,rtol=0,atol=0)
            torch.testing.assert_close(s.data,scale.squeeze(axis),rtol=0,atol=0,equal_nan=True)
            accepted+=1
    # Standalone K=0 and empty matrices preserve the same API and supplied object.
    for M,K,N in [(0,0,3),(2,0,3),(3,4,0)]:
        a=Tensor(torch.empty(M,K,dtype=torch.int8));b=Tensor(torch.empty(K,N,dtype=torch.int8));sa=Tensor(torch.ones(M));sb=Tensor(torch.ones(N));out=Tensor(torch.full((M,N),3.,dtype=torch.float16))
        got=ns['matmul_int8'](a,sa,b,sb,out);assert got is out
        torch.testing.assert_close(got.data,expected(a.data,sa.data,b.data,sb.data,gemmconfigs[17]),rtol=0,atol=0);accepted+=1
    # Same-dtype read aliases and arbitrary inactive supplied-output strides.
    ints=torch.arange(16,dtype=torch.int8);floats=torch.linspace(-1,1,8)
    a=Tensor(ints[:6].reshape(2,3));b=Tensor(ints[1:7].reshape(3,2));sa=Tensor(floats[:2]);sb=Tensor(floats[1:3])
    got=ns['matmul_int8'](a,sa,b,sb)
    torch.testing.assert_close(got.data,expected(a.data,sa.data,b.data,sb.data,gemmconfigs[17]),rtol=0,atol=0,equal_nan=True);accepted+=1
    for M,N in [(3,0),(0,0),(0,3)]:
        a=Tensor(torch.empty(M,4,dtype=torch.int8));b=Tensor(torch.empty(4,N,dtype=torch.int8));sa=Tensor(torch.ones(M));sb=Tensor(torch.ones(N))
        out=Tensor(torch.as_strided(torch.empty(0,dtype=torch.float32),(M,N),(2**40,2**41)))
        got=ns['matmul_int8'](a,sa,b,sb,out)
        assert got is out and got.data.stride()==(2**40,2**41);accepted+=1
    for payload in [[0.,0.,0.],[float('inf'),1.,-1.],[float('nan'),2.,-2.],[3.4e38,-3.4e38,0.]]:
        fpa=Tensor(torch.tensor([payload],dtype=torch.float32));q,scale=ns['quantize_int8_perrow'](fpa)
        expected_scale=torch.maximum(torch.tensor(0.),fpa.data.abs().amax(1))/127
        torch.testing.assert_close(q.data,(fpa.data/expected_scale[:,None]).to(torch.int8),rtol=0,atol=0)
        torch.testing.assert_close(scale.data,expected_scale.half(),rtol=0,atol=0,equal_nan=True);accepted+=1
    fpa=Tensor(torch.ones(2,3));a=Tensor(torch.ones(2,3,dtype=torch.int8));b=Tensor(torch.ones(3,4,dtype=torch.int8));sa=Tensor(torch.ones(2));sb=Tensor(torch.ones(4));out=Tensor(torch.ones(2,4,dtype=torch.float16))
    bad=[('quantize_int8_perrow',([],)),('quantize_int8_perrow',(Tensor(torch.ones(3)),)),('quantize_int8_perrow',(Tensor(fpa.data.half()),)),
        ('quantize_int8_perrow',(Tensor(torch._neg_view(fpa.data)),)),('quantize_int8_perrow',(Tensor(torch.ones(3,2).T),)),
        ('quantize_int8_perrow',(Tensor(torch.empty(2,0)),)),('quantize_int8',(fpa,True)),('quantize_int8',(fpa,2)),('quantize_int8',(Tensor(torch.empty(0,2)),0)),
        ('matmul_int8',(a,sa,b,sb,Tensor(out.data.double()))),('matmul_int8',(a,sa,Tensor(b.data.float()),sb,None)),
        ('matmul_int8',(a,Tensor(sa.data.double()),b,sb,None)),('matmul_int8',(a,sa,b,Tensor(sb.data.half()),None)),
        ('matmul_int8',(a,sa,b,Tensor(sb.data,'cuda:1'),None)),('matmul_int8',(a,sa,Tensor(torch.ones(6,4,dtype=torch.int8)[::2,::1]),sb,None)),
        ('matmul_quantize_int8',(fpa,b,Tensor(torch.ones(5)),None)),('matmul_quantize_int8',(fpa,b,sb,[]))]
    for offset in (1,2,3):bad.append(('matmul_int8',(a,sa,b,Tensor(helpers.unaligned_tensor(offset)),None)))
    shared=torch.ones(32)
    bad.extend([('matmul_int8',(Tensor(shared.view(torch.int8)[:6].reshape(2,3)),Tensor(shared[:2]),b,sb,None)),
        ('matmul_int8',(a,Tensor(shared[:2]),b,sb,Tensor(shared[:8].reshape(2,4)))),
        ('matmul_quantize_int8',(fpa,Tensor(torch.ones(3,3,dtype=torch.int8)),Tensor(torch.ones(3)),fpa))])
    class Metadata(Tensor):
        def __init__(self,shape,dtype,offset=0):
            super().__init__(torch.empty(1,dtype=dtype));self._shape=shape;self._offset=offset
        @property
        def shape(self):return self._shape
        @property
        def ndim(self):return len(self._shape)
        def numel(self):
            import math
            return math.prod(self._shape)
        def storage_offset(self):return self._offset
        def is_contiguous(self):return True
    for shape,offset in [((2,2**20+1),0),((2**31,1),0),((2,3),2**61)]:
        bad.append(('quantize_int8_perrow',(Metadata(shape,torch.float32,offset),)))
    for M,K,N,ao,bso,oo in [(2**63,3,4,0,0,0),(2**40,3,2**20,0,0,0),(2,3,4,2**63,0,0),(2,3,4,0,2**61,0),(2,3,4,0,0,2**61)]:
        bad.append(('matmul_int8',(Metadata((M,K),torch.int8,ao),Metadata((M,),torch.float32),Metadata((K,N),torch.int8),Metadata((N,),torch.float32,bso),Metadata((M,N),torch.float32,oo))))
    for name,args in bad:
        calls.clear()
        try:ns[name](*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError((name,'unsupported input admitted'))
        assert not calls
    print(f'PASS: {accepted} full actual-AST API cases, {len(bad)} guard rejections; 34 GEMM configs, row/column-major B, FP16/FP32 row scales, offset supplied outputs and all helper axes.')
    print('CUDA and allocation/device routing are mocked; native ISA/floating primitive conformance remains deferred.')

KERNEL_HASHES = {'quantize_int8_perrow_kernel': 'e580b9c28f2b64c70210defd35d11615de206f6fc96fcb6f021c20548ae339db', 'matmul_kernel': '4c328c543b1a4bb9f5102528f3f2ced7d1ae067d123839931ffa7abd1c3889ca'}
if __name__=='__main__':main()
