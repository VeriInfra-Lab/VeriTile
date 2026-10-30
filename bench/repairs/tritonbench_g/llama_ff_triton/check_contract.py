"""Checked wrapper and actual source-AST regression tests. CPU execution only."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'llama_ff_triton_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/llama_ff_triton/llama_ff_triton.py'
spec=importlib.util.spec_from_file_location('rms_helpers',HERE.parent/'fast_rms_layernorm/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,raw,Pointer=helpers.function,helpers.raw,helpers.Pointer
EVENTS=[]

class Tensor(helpers.Tensor):
    def is_conj(self):return self.data.is_conj()
    def storage_offset(self):return self.data.storage_offset()
    def view(self,*shape):
        t=Tensor(self.data.view(*shape),self.is_cuda,self.device.index or 0);EVENTS.append(('view',self,t));return t
    def t(self):
        t=Tensor(self.data.t(),self.is_cuda,self.device.index or 0);EVENTS.append(('transpose',self,t));return t

def typedraw(t):
    s=t.untyped_storage()
    return torch.empty(0,dtype=t.dtype).set_(s,0,(s.nbytes()//t.element_size(),),(1,))

def kernel_ast(path):
    node=function(path,'ff_llama');node.decorator_list=[]
    for a in node.args.args:a.annotation=None
    class Convert(ast.NodeTransformer):
        def visit_Call(self,n):
            n=self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='to':
                return ast.copy_location(ast.Call(func=ast.Name(id='convert',ctx=ast.Load()),args=[n.func.value,*n.args],keywords=n.keywords),n)
            return n
    node=Convert().visit(node)
    pid=[0]
    def convert(t,dtype,bitcast=False):
        return t.view(dtype) if bitcast else t.to(dtype)
    def offsets(p,mask):
        off=torch.as_tensor(p.offsets,dtype=torch.int64)
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.broadcast_to(torch.as_tensor(mask,dtype=torch.bool),off.shape)
        idx=p.tensor.storage_offset()+off[mask]
        assert torch.all(idx>=0) and torch.all(idx<typedraw(p.tensor).numel()),'active source access outside backing'
        return off,mask,idx
    def load(p,mask=None,other=0):
        off,mask,idx=offsets(p,mask);out=torch.full_like(off,other,dtype=p.tensor.dtype);out[mask]=typedraw(p.tensor)[idx];return out
    def store(p,v,mask=None):
        off,mask,idx=offsets(p,mask);typedraw(p.tensor)[idx]=torch.broadcast_to(v,off.shape)[mask].to(p.tensor.dtype)
    tl=SimpleNamespace(program_id=lambda axis:torch.tensor(pid[0]),arange=torch.arange,load=load,store=store,
      float32=torch.float32,float16=torch.float16,float8e5=torch.float8_e5m2,int64=torch.int64,
      cdiv=lambda a,b:(a+b-1)//b,zeros=torch.zeros,sum=lambda a,axis:a.sum(dim=axis),dot=lambda a,b:a.float()@b.float(),
      sigmoid=torch.sigmoid,math=SimpleNamespace(rsqrt=torch.rsqrt),extra=SimpleNamespace(cuda=SimpleNamespace(libdevice=SimpleNamespace(pow=torch.pow))))
    scope={'tl':tl,'convert':convert};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope['ff_llama'],pid

def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    for key,path in [('original_sha256',ORIGINAL),('candidate_sha256',SOURCE),('quotation_sha256',HERE/'Kernels.lean')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[key]
    assert hashlib.sha256(ast.dump(function(SOURCE,'ff_llama')).encode()).hexdigest()==pins['historical_kernel_ast_sha256']
    src=(HERE/'Proofs/Source.lean').read_text()
    assert json.dumps(ast.dump(ast.parse(SOURCE.read_text()))) in src
    assert pins['candidate_sha256'] in src
    expected=[
      'not all(isinstance(t, torch.Tensor) for t in tensors)',
      'any(t.layout != torch.strided for t in tensors)',
      'x.dtype != torch.float16 or w1.dtype not in (torch.float16, torch.int8) or not w1.dtype == w3.dtype == rms_w.dtype',
      'x.ndim != 3 or w1.ndim != 2 or w3.ndim != 2 or rms_w.ndim != 1',
      'not all(t.is_cuda and t.device == x.device for t in tensors)',
      'any(not t.is_contiguous() or t.storage_offset() != 0 or t.is_neg() or t.is_conj() for t in tensors)',
      'K == 0 or N == 0 or w1.shape != (N, K) or w3.shape != (N, K) or rms_w.shape != (K,)',
      'max(M, N, K) >= 2**31','tiles_m * tiles_n > 2**31 - 1',
      '2 * (M * K + formed_k) >= 2**63 or weight_bytes * (N * K + formed_k) >= 2**63 or 2 * (tiles_m * 16 * N + tiles_n * 16) >= 2**63',
      'w1.dtype == torch.int8 and any(x.untyped_storage()._cdata == t.untyped_storage()._cdata for t in (w1, w3, rms_w))',
      'M and any(t.data_ptr() % 16 for t in tensors)','M > 0']
    wrapper=function(SOURCE,'kernel_ff')
    checks=[n.test for n in wrapper.body if isinstance(n,ast.If)]
    assert list(map(ast.dump,checks))==[ast.dump(ast.parse(s,mode='eval').body) for s in expected]
    calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'clone','contiguous','reshape','detach','resolve_neg'}
    assert calls.count('view')==2 and calls.count('t')==2

def host_checks():
    torch.manual_seed(17)
    node=function(SOURCE,'kernel_ff');scope={'torch':torch,'triton':SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),scope)
    active=[2];k,pid=kernel_ast(SOURCE)
    @contextlib.contextmanager
    def device(d):
        old=active[0];active[0]=d.index;EVENTS.append(('enter',d.index))
        try:yield
        finally:active[0]=old;EVENTS.append(('exit',old))
    def empty(shape,dtype,device):
        t=Tensor(torch.empty(shape,dtype=dtype),device=device.index);EVENTS.append(('allocate',t));return t
    class Launch:
        fail=False
        def __getitem__(self,grid):
            def run(*args,**kw):
                meta=dict(kw,M=args[5],N=args[6]);actual_grid=grid(meta)
                EVENTS.append(('launch',args,kw,actual_grid,active[0]))
                assert active[0]==args[0].device.index
                if self.fail:args[3].data.reshape(-1)[:1].fill_(99);raise RuntimeError('injected launch')
                values=[Pointer(t.data) for t in args[:5]]+[torch.tensor(v) for v in args[5:]]
                options={name:kw[name] for name in ('USE_FP8','EPS','BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K')}
                for p in range(actual_grid[0]):pid[0]=p;k(*values,**options)
            return run
    launch=Launch();scope['ff_llama']=launch
    scope['torch']=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float16=torch.float16,int8=torch.int8,
      empty=empty,cuda=SimpleNamespace(device=device))
    fn=scope['kernel_ff']
    def inputs(b,s,n,k,byte=False):
        x=torch.randn(b,s,k,dtype=torch.float16)*.2
        weights=[torch.randn(n,k)*.2,torch.randn(n,k)*.2,torch.randn(k)*.2]
        return [Tensor(x)]+[Tensor(t.to(torch.float8_e5m2).view(torch.int8) if byte else t.half()) for t in weights]
    good=[inputs(b,s,n,k,byte) for b,s,n,k in ((1,1,1,1),(1,17,19,63),(2,9,17,64),(1,3,5,65),(1,2,3,129),(0,3,5,65),(2,0,3,1)) for byte in (False,True)]
    # Same-width read aliases and arbitrary singleton strides are preserved.
    shared=torch.randn(256,dtype=torch.float16)
    good += [[Tensor(shared[:64].view(1,8,8)),Tensor(shared[:64].view(8,8)),Tensor(shared[:64].view(8,8)),Tensor(shared[:8])]]
    singleton=inputs(1,1,1,1)
    singleton[0]=Tensor(torch.as_strided(torch.randn(8,dtype=torch.float16),(1,1,1),(81,55,33)))
    singleton[1]=Tensor(torch.as_strided(torch.randn(8,dtype=torch.float16),(1,1),(44,17)))
    good.append(singleton)
    special=inputs(1,2,3,65,True);special[0].data.reshape(-1)[:4]=torch.tensor([float('nan'),float('inf'),-float('inf'),-0.],dtype=torch.float16)
    special[1].data.reshape(-1)[:4]=torch.tensor([124,125,-4,-3],dtype=torch.int8);good.append(special)
    for ts in good:
        before=[raw(t.data).clone() for t in ts];EVENTS.clear();out=fn(*ts)
        x,w,v,r=[t.data for t in ts]
        if w.dtype==torch.int8:w,v,r=[t.view(torch.float8_e5m2).half() for t in (w,v,r)]
        z=(x*r).float();rho=torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        u=(z@w.float().T)*rho;vv=(z@v.float().T)*rho;expected=(u*torch.sigmoid(u)*vv).half()
        torch.testing.assert_close(out.data,expected,rtol=5e-3,atol=5e-3,equal_nan=True)
        assert out.shape==(ts[0].shape[0],ts[0].shape[1],ts[1].shape[0]) and out.dtype==torch.float16
        allocated=next(e[1] for e in EVENTS if e[0]=='allocate')
        assert out.untyped_storage()._cdata==allocated.untyped_storage()._cdata
        assert out is EVENTS[-1][2] and EVENTS[-1][0]=='view'
        for t,saved in zip(ts,before):assert torch.equal(raw(t.data),saved)
        assert active[0]==2
        launches=[e for e in EVENTS if e[0]=='launch']
        if x.numel():
            assert len(launches)==1
            _,args,kw,grid,dev=launches[0];m,n,kdim=args[5:8]
            assert args[8:]==(kdim,1,1,kdim,1,kdim,n,1,1)
            assert kw==dict(USE_FP8=ts[1].dtype==torch.int8,EPS=1e-6,BLOCK_SIZE_M=16,BLOCK_SIZE_N=16,BLOCK_SIZE_K=64,num_stages=2,num_warps=4)
            for actual,original in ((args[0],ts[0]),(args[1],ts[1]),(args[2],ts[2]),(args[4],ts[3])):
                assert actual.data_ptr()==original.data_ptr() and actual.untyped_storage()._cdata==original.untyped_storage()._cdata
        else:assert not launches and not any(e[0]=='enter' for e in EVENTS)
    bad=[]
    for side in range(4):
        template=inputs(1,2,4,4)
        base=template[side].data
        vals=[None,Tensor(base.float()),Tensor(base,cuda=False),Tensor(base,device=1),Tensor(torch._neg_view(base)),Tensor(torch.cat([base.reshape(-1),base.reshape(-1)])[1:1+base.numel()].view(base.shape))]
        if base.ndim>1:vals.append(Tensor(base.transpose(-1,-2)))
        for value in vals:
            ts=inputs(1,2,4,4);ts[side]=value;bad.append(ts)
    bad += [inputs(1,2,0,4),inputs(1,2,4,0)]
    for side in (1,2,3):
        ts=inputs(1,2,4,4);ts[side]=Tensor(torch.ones((3,4) if side<3 else (3,),dtype=torch.float16));bad.append(ts)
    # Mixed-width alias is rejected before views/allocation.
    ts=inputs(1,2,4,4,True);ts[1]=Tensor(ts[0].data.view(torch.int8).reshape(4,4));bad.append(ts)
    class Misaligned(Tensor):
        def data_ptr(self):return super().data_ptr()+1
    for side in range(4):
        ts=inputs(1,2,4,4);ts[side]=Misaligned(ts[side].data);bad.append(ts)
    class Metadata(Tensor):
        def __init__(self,data,dims):super().__init__(data);self.dims=torch.Size(dims)
        @property
        def shape(self):return self.dims
        @property
        def ndim(self):return len(self.dims)
    for dims,n in [((1,2**31,1),1),((1,2**31-1,1),512)]:
        ts=inputs(1,1,n,1);ts[0]=Metadata(ts[0].data,dims);bad.append(ts)
    for ts in bad:
        EVENTS.clear()
        try:fn(*ts)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported input admitted')
        assert not EVENTS
    ts=inputs(1,2,4,4);before=[raw(t.data).clone() for t in ts];launch.fail=True;EVENTS.clear()
    try:fn(*ts)
    except RuntimeError as e:assert str(e)=='injected launch'
    else:raise AssertionError('launch failure swallowed')
    assert active[0]==2 and EVENTS[-1]==('exit',2)
    for t,saved in zip(ts,before):assert torch.equal(raw(t.data),saved)
    old,oldpid=kernel_ast(ORIGINAL);ts=inputs(1,1,1,1);out=Tensor(torch.empty(1,1,dtype=torch.float16))
    try:old(*[Pointer(t.data) for t in (ts[0],ts[1],ts[2],out,ts[3])],*[torch.tensor(v) for v in (1,1,1,1,1,1,1,1,1,1,1,1)],USE_FP8=False,EPS=1e-6,BLOCK_SIZE_M=16,BLOCK_SIZE_N=16,BLOCK_SIZE_K=64)
    except AssertionError as e:assert 'outside backing' in str(e)
    else:raise AssertionError('original K-tail access escape not reproduced')
    print(f'{len(good)} complete FP16/int8 cases and {len(bad)} rejection cases passed; exact aliases, canonical launch, caller bytes and failure context checked.')
    print('Original unmasked K-tail escape reproduced. CPU AST execution is not GPU/native FP conformance.')

if __name__=='__main__':source_binding();host_checks()
