"""Actual-source CPU controls. CUDA allocation/launch are mocked; ISA numerics are not validated."""
import ast
import contextlib
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'int4_matmul_repaired.py'

def unwrap(x):
    if isinstance(x,Tensor):return x.data
    if isinstance(x,(list,tuple)):return type(x)(unwrap(v) for v in x)
    return x
class Tensor:
    def __init__(self,data,device='cuda:0',**overrides):self.data=data;self.device=torch.device(device);self.overrides=overrides
    @property
    def dtype(self):return self.overrides.get('dtype',self.data.dtype)
    @property
    def layout(self):return self.overrides.get('layout',self.data.layout)
    @property
    def shape(self):return self.overrides.get('shape',tuple(self.data.shape))
    @property
    def ndim(self):return self.overrides.get('ndim',len(self.shape))
    @property
    def is_cuda(self):return self.device.type=='cuda'
    @property
    def requires_grad(self):return self.overrides.get('requires_grad',self.data.requires_grad)
    def numel(self):return math.prod(self.shape)
    def data_ptr(self):return self.overrides.get('pointer',self.data.data_ptr())
    def element_size(self):return self.data.element_size()
    def storage_offset(self):return self.overrides.get('offset',self.data.storage_offset())
    def stride(self,dim=None):
        strides=self.overrides.get('strides',self.data.stride());return strides if dim is None else strides[dim]
    def untyped_storage(self):return self.data.untyped_storage()
    def is_neg(self):return self.overrides.get('neg',self.data.is_neg())
    def is_conj(self):return self.overrides.get('conj',self.data.is_conj())
    def to(self,arg):
        if isinstance(arg,torch.dtype):return Tensor(self.data.to(arg),self.device)
        return Tensor(self.data,torch.device(arg))
    def cuda(self,rank):
        assert self.device.index==rank
        return self
    def __getattr__(self,name):
        fn=getattr(self.data,name)
        if not callable(fn):return fn
        def call(*args,**kw):
            result=fn(*(unwrap(a) for a in args),**{k:unwrap(v) for k,v in kw.items()})
            if result is self.data:return self
            return Tensor(result,self.device) if isinstance(result,torch.Tensor) else result
        return call
    def __bool__(self):return bool(self.data)
    def __getitem__(self,item):return Tensor(self.data[item],self.device)
    def __setitem__(self,item,value):self.data[item]=unwrap(value)
    def __neg__(self):return Tensor(-self.data,self.device)
    def __add__(self,x):return Tensor(self.data+unwrap(x),self.device)
    def __sub__(self,x):return Tensor(self.data-unwrap(x),self.device)
    def __mul__(self,x):return Tensor(self.data*unwrap(x),self.device)
    def __truediv__(self,x):return Tensor(self.data/unwrap(x),self.device)
    def __lt__(self,x):return Tensor(self.data<unwrap(x),self.device)
    def __gt__(self,x):return Tensor(self.data>unwrap(x),self.device)
    def __lshift__(self,x):return Tensor(self.data<<unwrap(x),self.device)
    def __rshift__(self,x):return Tensor(self.data>>unwrap(x),self.device)
    def __and__(self,x):return Tensor(self.data&unwrap(x),self.device)
    def __iadd__(self,x):self.data+=unwrap(x);return self


def main():
    torch.set_num_threads(1)
    tree=ast.parse(SOURCE.read_text())
    literal=next(line.split(' := ',1)[1] for line in (HERE/'Kernels.lean').read_text().splitlines() if line.startswith('def productionAST :'))
    assert json.loads(literal)==ast.dump(tree)
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_kernel')
    assert hashlib.sha256(ast.dump(kernel).encode()).hexdigest()==KERNEL_HASH
    configs=[ast.literal_eval(c.args[0]) for c in next(k.value for k in kernel.decorator_list[0].keywords if k.arg=='configs').elts]
    assert len(configs)==36
    pids=[0,0];selected=[0];trials=[[]];launches=[];currentDevice=[3]
    @contextlib.contextmanager
    def device(owner):
        previous=currentDevice[0];currentDevice[0]=torch.device(owner).index
        try:yield
        finally:currentDevice[0]=previous
    def physical(t):return torch.empty(0,dtype=t.dtype).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes()//t.element_size(),),(1,))
    class Pointer:
        def __init__(self,tensor,offset=0):self.tensor,self.offset=tensor,offset;self.dtype=SimpleNamespace(element_ty=tensor.dtype)
        def __add__(self,offset):return Pointer(self.tensor,self.offset+offset)
    def load(p,mask=None,other=0):
        offsets=torch.as_tensor(p.offset);mask=torch.ones_like(offsets,dtype=torch.bool) if mask is None else torch.broadcast_to(mask,offsets.shape)
        value=torch.full(offsets.shape,other,dtype=p.tensor.dtype);value[mask]=physical(p.tensor)[p.tensor.storage_offset()+offsets[mask].long()];return value
    def store(p,v,mask,atomic=False):
        offsets=torch.as_tensor(p.offset);mask=torch.broadcast_to(mask,offsets.shape)
        ids=p.tensor.storage_offset()+offsets[mask].long();values=torch.broadcast_to(v,offsets.shape)[mask];flat=physical(p.tensor)
        flat[ids]=flat[ids]+values if atomic else values
    triton=SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d)
    tl=SimpleNamespace(int32=torch.int32,float32=torch.float32,program_id=lambda axis:torch.tensor(pids[axis],dtype=torch.int32),
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),cdiv=triton.cdiv,load=load,store=store,
        atomic_add=lambda p,v,mask:store(p,v,mask,True),zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype),dot=lambda a,b:a@b)
    kernel.decorator_list=[]
    for arg in kernel.args.args:arg.annotation=None
    kernel.returns=None;kns={'tl':tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel],type_ignores=[])),str(SOURCE),'exec'),kns)
    class Launch:
        def __getitem__(self,grid):
            def launch(a,w,c,s,z,M,N,K,*strides):
                assert currentDevice[0]==a.device.index
                assert strides==(*a.stride(),*w.stride(),*c.stride(),*s.stride(),*z.stride(),K//s.shape[0])
                for pos,cfgid in enumerate(trials[0]+[selected[0]]):
                    if pos<len(trials[0]) or trials[0]:c.zero_()
                    cfg=configs[cfgid];gx,gz=grid(cfg);launches.append(cfgid)
                    for sp in reversed(range(gz)):
                        for pid in reversed(range(gx)):
                            pids[:]=[pid,sp]
                            kns['matmul_kernel'](*(Pointer(t.data) for t in (a,w,c,s,z)),M,N,K,*strides,**cfg)
            return launch
    def zeros(*shape,device=None,dtype=None):
        dims=shape[0] if len(shape)==1 and isinstance(shape[0],(tuple,list)) else shape
        return Tensor(torch.zeros(dims,dtype=dtype),device or 'cuda:0')
    def where(condition,a,b):return Tensor(torch.where(unwrap(condition),unwrap(a),unwrap(b)),condition.device)
    fake=SimpleNamespace(Tensor=Tensor,FloatTensor=Tensor,IntTensor=Tensor,strided=torch.strided,float32=torch.float32,int32=torch.int32,
        zeros=zeros,where=where,isfinite=lambda t:Tensor(torch.isfinite(t.data),t.device),cuda=SimpleNamespace(device=device))
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='matmul_kernel']
    ns={'torch':fake,'triton':triton,'matmul_kernel':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(SOURCE),'exec'),ns)
    def pack(values,dim):
        if dim==0:
            result=torch.zeros(values.shape[0]//8,values.shape[1],dtype=torch.int64)
            for i in range(8):result|=values[i::8,:].long()<<(4*i)
        else:
            result=torch.zeros(values.shape[0],values.shape[1]//8,dtype=torch.int64)
            for i in range(8):result|=values[:,i::8].long()<<(4*i)
        return result.int()
    def unpack(w,s,z):
        K,N=w.shape[0]*8,w.shape[1];group=K//s.shape[0]
        k=torch.arange(K);n=torch.arange(N)
        q=(w[k//8,:].long()>>(4*(k%8))[:,None])&15
        zp=(z[:,n//8].long()>>(4*(n%8))[None,:])&15
        return (q.float()-zp[k//group,:].float())*s[k//group,:]
    count=0
    for ci in range(36):
        selected[0]=ci;trials[0]=[0,9] if ci==0 else []
        for provided in (False,True):
            M,K,N,G=3,40,8,8
            a=Tensor((torch.arange(M*K)%5-2).float().reshape(M,K));w=Tensor(pack((torch.arange(K*N)%16).reshape(K,N),0))
            s=Tensor((torch.arange(K//G*N)%3+1).float().reshape(K//G,N));z=Tensor(pack((torch.arange(K//G*N)%16).reshape(K//G,N),1))
            out=Tensor(torch.full((M,N),9.)) if provided else None
            before=[v.data.clone() for v in (a,w,s,z)];got=ns['matmul_dequantize_int4_s2'](a,w,s,z,G,out)
            torch.testing.assert_close(got.data,a.data@unpack(w.data,s.data,z.data),rtol=0,atol=0)
            assert got.dtype==torch.float32 and got.shape==(M,N) and got.stride()==(N,1) and got.device==a.device
            if provided:assert got is out
            for t,old in zip((a,w,s,z),before):torch.testing.assert_close(t.data,old,rtol=0,atol=0)
            assert currentDevice[0]==3;count+=1
    # Original input words remain unrestricted for GEMM/unpack.
    selected[0]=0;trials[0]=[]
    for payload in (float('nan'),float('inf'),-float('inf'),0.0,torch.finfo(torch.float32).tiny/4):
        a=Tensor(torch.ones(1,8));w=Tensor(pack(torch.ones(8,8,dtype=torch.int32)*2,0))
        sc=Tensor(torch.full((1,8),payload));zp=Tensor(torch.zeros(1,1,dtype=torch.int32))
        got=ns['matmul_dequantize_int4_s2'](a,w,sc,zp,8)
        torch.testing.assert_close(got.data,a.data@unpack(w.data,sc.data,zp.data),rtol=0,atol=0,equal_nan=True)
        decoded=ns['unpack_int4'](w,sc,zp)
        torch.testing.assert_close(decoded.data,unpack(w.data,sc.data,zp.data).half(),rtol=0,atol=0,equal_nan=True)
        count+=1
    for default in (torch.float32,torch.float64):
        torch.set_default_dtype(default)
        for K,N,G in [(8,8,8),(8,16,8),(16,8,16),(16,16,8),(40,24,8)]:
            weight=Tensor(torch.linspace(-3,5,K*N,dtype=torch.float32).reshape(K,N));old=weight.data.clone()
            qw,sc,qz,group=ns['quantize_int4'](weight,G,0)
            values=weight.data.T.contiguous().reshape(N,K//G,G)
            hi=values.amax(-1).clamp_min(0);lo=values.amin(-1).clamp_max(0);scale=(hi-lo)/15
            zp=(-lo/scale).round().clamp(0,15).int();q=(values/scale[:,:,None]+zp[:,:,None]).round().clamp(0,15).int().reshape(N,K).T
            wantw=pack(q,0);wantz=pack(zp.T,1)
            torch.testing.assert_close(qw.data,wantw,rtol=0,atol=0);torch.testing.assert_close(qz.data,wantz,rtol=0,atol=0)
            torch.testing.assert_close(sc.data,scale.T.contiguous(),rtol=0,atol=0)
            for t,want in ((qw,wantw),(sc,scale.T.contiguous()),(qz,wantz)):
                rows,cols=t.shape;expected=(1,rows) if 1 in t.shape else (cols,1)
                assert t.stride()==expected and t.device==weight.device
            assert group==G and len({t.untyped_storage().data_ptr() for t in (weight,qw,sc,qz)})==4
            got=ns['unpack_int4'](qw,sc,qz)
            torch.testing.assert_close(got.data,unpack(qw.data,sc.data,qz.data).half(),rtol=0,atol=0)
            assert got.dtype==torch.float16 and got.shape==(K,N) and got.stride()==(1,K)
            torch.testing.assert_close(weight.data,old,rtol=0,atol=0)
            # Every actual quantizer-returned descriptor is accepted by GEMM.
            a=Tensor(torch.ones(1,K,dtype=torch.float32));ns['_validate_matmul'](a,qw,sc,qz,G,None)
            count+=1
    torch.set_default_dtype(torch.float32)
    a=Tensor(torch.ones(2,8));w=Tensor(torch.zeros(1,8,dtype=torch.int32));s=Tensor(torch.ones(1,8));z=Tensor(torch.zeros(1,1,dtype=torch.int32))
    reject=0
    def rejected(fn):
        nonlocal reject
        try:fn()
        except (ValueError,TypeError):reject+=1
        else:raise AssertionError('guard accepted invalid request')
    for bad in [Tensor(a.data,device='cpu'),Tensor(a.data,dtype=torch.float16),Tensor(a.data,layout=torch.sparse_coo),
                Tensor(a.data,ndim=3),Tensor(a.data,shape=(0,8)),Tensor(a.data,offset=1),Tensor(a.data,neg=True),
                Tensor(a.data,conj=True),Tensor(a.data,requires_grad=True),Tensor(a.data,pointer=a.data_ptr()+1),Tensor(a.data,strides=(16,1))]:
        rejected(lambda bad=bad:ns['_validate_matmul'](bad,w,s,z,8,None))
    for group in [False,0,-1,4,8.0]:rejected(lambda group=group:ns['_validate_matmul'](a,w,s,z,group,None))
    rejected(lambda:ns['_validate_matmul'](a,w,s,z,8,a))
    rejected(lambda:ns['_validate_matmul'](a,w,Tensor(s.data,device='cuda:1'),z,8,None))
    rejected(lambda:ns['quantize_int4'](Tensor(torch.zeros(8,8)),8,0))
    rejected(lambda:ns['quantize_int4'](Tensor(torch.full((8,8),float('nan'))),8,0))
    rejected(lambda:ns['quantize_int4'](Tensor(torch.full((8,8),float('inf'))),8,0))
    maximum=torch.finfo(torch.float32).max
    overflowing=torch.full((8,8),maximum);overflowing[:4]=-maximum
    rejected(lambda:ns['quantize_int4'](Tensor(overflowing),8,0))
    rejected(lambda:ns['quantize_int4'](Tensor(torch.full((8,8),torch.nextafter(torch.tensor(0.),torch.tensor(1.)).item())),8,0))
    rejected(lambda:ns['quantize_int4'](Tensor(torch.ones(8,8)),8,1))
    rejected(lambda:ns['quantize_int4'](Tensor(torch.ones(8,8)),8,False))
    # Actual pre-division packed stride overflow from the retained successor review.
    hugeN=4194304
    rejected(lambda:ns['_validate_matmul'](Tensor(torch.ones(1,8)),Tensor(w.data,shape=(1,hugeN),strides=(hugeN,1)),
        Tensor(s.data,shape=(1,hugeN),strides=(hugeN,1)),Tensor(z.data,shape=(1,hugeN//8),strides=(hugeN//8,1)),8,None))
    print(f'PASS: {count} complete actual-AST API cases, {reject} rejections; all 36 GEMM configs and both helper default dtypes.')
    print('CUDA is mocked; native floating/TF32/atomic primitive conformance remains deferred.')

KERNEL_HASH='02f08fa8532ed593ba0a2ae2668ca438ca50137f6a261831c036ede09feb732d'
if __name__=='__main__':main()
