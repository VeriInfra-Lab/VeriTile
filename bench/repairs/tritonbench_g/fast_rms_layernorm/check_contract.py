"""Source/host regression checks. CPU arithmetic is not GPU conformance evidence."""
import ast
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'fast_rms_layernorm_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/fast_rms_layernorm/fast_rms_layernorm.py'
KERNELS=('_rms_layernorm_forward','_gemma_rms_layernorm_forward','_rms_layernorm_backward','_rms_layernorm_weight_backward')
EVENTS=[]

def function(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)

def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    for key,path in [('original_sha256',ORIGINAL),('candidate_sha256',SOURCE),('quotation_sha256',HERE/'Kernels.lean')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[key]
    assert pins['original_sha256']=='4a3553e0f4dee342c393edf0b8aaf82504153d1bd39b724072e566612fb9c1e8'
    for name in KERNELS:
        assert hashlib.sha256(ast.dump(function(SOURCE,name)).encode()).hexdigest()==pins['historical_kernel_asts'][name]
    expected={
      '_check_tensor':['not isinstance(tensor, torch.Tensor)','tensor.layout != torch.strided','tensor.dtype != torch.float32','not tensor.is_cuda','torch.cuda.current_device() != tensor.device.index','not tensor.is_contiguous()','tensor.is_neg()'],
      '_check_forward':['X.device != W.device','X.ndim == 0 or X.shape[-1] == 0','X.shape[-1] > MAX_FUSED_SIZE','W.numel() != X.shape[-1]','X.numel() // X.shape[-1] > 2147483647','X.untyped_storage()._cdata == W.untyped_storage()._cdata','(X.numel() and X.data_ptr() % 16) or W.data_ptr() % 16','type(eps) is not float','not math.isfinite(eps)','type(gemma) is not bool']}
    for name,tests in expected.items():
        actual=[n.test for n in function(SOURCE,name).body if isinstance(n,ast.If)]
        assert list(map(ast.dump,actual))==[ast.dump(ast.parse(t,mode='eval').body) for t in tests]
    contract=(HERE/'Contract.lean').read_text()
    for clause in ['t.tensor,t.strided,t.fp32,t.cuda,t.current,t.contiguous,!t.negative','decide (r.inputDevice=r.weightDevice)','ranked r.shape','decide (0<cols r.shape)','decide (cols r.shape≤65536)','decide (r.weightShape.prod=cols r.shape)','decide (rows r.shape≤2147483647)','decide (m.input.storageId≠m.weight.storageId)','decide (r.shape.dims.prod=0 ∨ m.input.address%16=0)','decide (m.weight.address%16=0)','m.epsilonFloat,m.epsilonFinite,m.gemmaBool','g.shape.prod=ctx.call.cells']:
        assert clause in contract,clause
    native=(HERE/'Proofs/NativeGraph.lean').read_text()
    assert json.dumps(ast.dump(ast.parse(SOURCE.read_text()))) in native
    assert pins['candidate_sha256'] in native
    forward=function(SOURCE,'forward');backward=function(SOURCE,'backward')
    for f in (forward,backward):
        calls=[n.func.attr for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        assert not set(calls)&{'resolve_neg','reshape','contiguous','detach'}
    assert not any(isinstance(n,ast.Name) and n.id=='_snapshot' for n in ast.walk(forward))
    assert sum(isinstance(n,ast.Name) and n.id=='_snapshot' for n in ast.walk(backward))==1
    print('Original/hash binding and all four historical kernel ASTs preserved; Python guards match the Lean contract.')


def raw(t):
    s=t.untyped_storage()
    return torch.empty(0,dtype=torch.uint8).set_(s,0,(s.nbytes(),),(1,))
def fpraw(t):
    s=t.untyped_storage()
    return torch.empty(0,dtype=torch.float32).set_(s,0,(s.nbytes()//4,),(1,))

class Tensor:
    def __init__(self,data,cuda=True,device=0):
        self.data=data;self.is_cuda=cuda;self.device=torch.device('cuda',device) if cuda else torch.device('cpu')
    @property
    def layout(self):return self.data.layout
    @property
    def dtype(self):return self.data.dtype
    @property
    def ndim(self):return self.data.ndim
    @property
    def shape(self):return self.data.shape
    def numel(self):return self.data.numel()
    def is_contiguous(self):return self.data.is_contiguous()
    def is_neg(self):return self.data.is_neg()
    def data_ptr(self):
        assert self.numel()>0,'empty pointer queried'
        return self.data.data_ptr()
    def untyped_storage(self):return self.data.untyped_storage()
    def stride(self,*a):return self.data.stride(*a)
    def view(self,*shape):
        v=Tensor(self.data.view(*shape),self.is_cuda,self.device.index or 0);EVENTS.append(('view',self,v));return v
    def clone(self,**kw):
        v=Tensor(self.data.clone(**kw),self.is_cuda,self.device.index or 0);EVENTS.append(('clone',self,v));return v
    def zero_(self):self.data.zero_();EVENTS.append(('zero',self));return self

class Pointer:
    def __init__(self,t,off=0):self.tensor,self.offsets=t,off
    @property
    def dtype(self):return SimpleNamespace(element_ty=self.tensor.dtype)
    def __add__(self,off):return Pointer(self.tensor,self.offsets+off)

def interpreter(path,name):
    pid=[0]
    def offsets(p,mask):
        off=torch.as_tensor(p.offsets,dtype=torch.int64)
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.as_tensor(mask,dtype=torch.bool)
        indices=p.tensor.storage_offset()+off[mask]
        assert torch.all(indices>=0) and torch.all(indices<fpraw(p.tensor).numel()),'active access out of bounds'
        return off,mask,indices
    def load(p,mask=None,other=0):
        off,mask,idx=offsets(p,mask)
        out=torch.full_like(off,other,dtype=torch.float32);out[mask]=fpraw(p.tensor)[idx];return out
    def store(p,v,mask=None):
        off,mask,idx=offsets(p,mask)
        values=torch.broadcast_to(torch.as_tensor(v),off.shape)
        fpraw(p.tensor)[idx]=values[mask]
    node=function(path,name);node.decorator_list=[]
    for a in node.args.args:a.annotation=None
    tl=SimpleNamespace(program_id=lambda _:torch.tensor(pid[0]),arange=torch.arange,load=load,store=store,
        float32=torch.float32,int64=torch.int64,math=SimpleNamespace(rsqrt=torch.rsqrt),
        sum=lambda v,axis:v.sum(dim=axis),full=lambda shape,v,dtype:torch.full(shape,v,dtype=dtype),
        cast=lambda v,dtype:torch.as_tensor(v,dtype=dtype))
    scope={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope[name],pid

def module(path):
    scope={'__name__':'checked_fast_rms'}
    exec(compile(path.read_text().split('##################################################################################################################################################')[0],str(path),'exec'),scope)
    def empty(shape,dtype,device):
        dev=torch.device(device);t=Tensor(torch.empty(shape,dtype=dtype),True,dev.index or 0);EVENTS.append(('allocate',t));return t
    def empty_like(t):return empty(t.shape,t.dtype,t.device)
    scope['torch']=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        contiguous_format=torch.contiguous_format,empty=empty,empty_like=empty_like,
        cuda=SimpleNamespace(current_device=lambda:0))
    for name in KERNELS:
        if name not in scope:continue
        kernel,pid=interpreter(path,name)
        class Launch:
            def __init__(self,k,p,n):self.kernel,self.pid,self.name=k,p,n
            def __getitem__(self,grid):
                def run(*args,**kw):
                    EVENTS.append(('launch',self.name,grid,args,kw))
                    vals=[Pointer(a.data) if isinstance(a,Tensor) else a for a in args]
                    bound={k:v for k,v in kw.items() if k in ('GEMMA','BLOCK_SIZE')}
                    for row in range(grid[0]):self.pid[0]=row;self.kernel(*vals,**bound)
                return run
        scope[name]=Launch(kernel,pid,name)
    return SimpleNamespace(**scope)

class Context:
    def save_for_backward(self,*values):self._saved=values;self._versions=[t.data._version for t in values]
    @property
    def saved_tensors(self):
        if [t.data._version for t in self._saved]!=self._versions:raise RuntimeError('saved tensor version changed')
        return self._saved

def reference(x,w,g,gemma,eps=.7):
    xx=x.double().clone().requires_grad_();ww=w.double().clone().requires_grad_()
    y=xx*torch.rsqrt(xx.square().mean(-1,keepdim=True)+eps)*(ww.reshape(-1)+int(gemma))
    dx,dw=torch.autograd.grad(y,(xx,ww),g.double())
    return y.detach(),dx,dw

def host_checks():
    mod=module(SOURCE);F=mod.Fast_RMS_Layernorm
    good=[((2,n),(n,)) for n in (1,3,8,257,2048,8192,32768,65536)]
    good += [((2,3,7),(1,7)),((7,),(7,)),((2,0,7),(7,)),((1,1),())]
    passed=0
    for shape,ws in good:
      for gemma in (False,True):
        x=Tensor(torch.randn(shape));w=Tensor(torch.randn(ws));g=Tensor(torch.randn(shape))
        before=[raw(t.data).clone() for t in (x,w,g)];expect=reference(x.data,w.data,g.data,gemma)
        EVENTS.clear();ctx=Context();y=F.forward(ctx,x,w,.7,gemma);dx,dw,a,b=F.backward(ctx,g)
        assert a is None and b is None
        for actual,expected in zip((y,dx,dw),expect):torch.testing.assert_close(actual.data.double(),expected,rtol=4e-5,atol=4e-5)
        assert (y.shape,dx.shape,dw.shape)==(x.shape,x.shape,w.shape)
        for t,old in zip((x,w,g),before):assert torch.equal(raw(t.data),old)
        sx,sw,sr=ctx.saved_tensors
        for saved,original in ((sx,x),(sw,w)):
            assert saved.untyped_storage()._cdata==original.untyped_storage()._cdata
            if original.numel():assert saved.data_ptr()==original.data_ptr()
        clones=[e for e in EVENTS if e[0]=='clone'];assert len(clones)==1 and clones[0][1].untyped_storage()._cdata==g.untyped_storage()._cdata
        assert dx.untyped_storage()._cdata==clones[0][2].untyped_storage()._cdata
        for out in (y,dx,dw):assert all(out.untyped_storage()._cdata!=t.untyped_storage()._cdata for t in (x,w,g))
        launches=[e for e in EVENTS if e[0]=='launch']
        assert [e[1] for e in launches]==([KERNELS[int(gemma)],KERNELS[3],KERNELS[2]] if x.numel() else [])
        if not x.numel():assert torch.equal(raw(dw.data),torch.zeros_like(raw(dw.data)))
        else:
            assert launches[0][3][1:8:2]==(shape[-1],shape[-1],1,1)
            assert launches[2][3][1:10:2]==(shape[-1],shape[-1],1,1,1)
        passed+=1
    # Both Python zero signs are admitted and reach the runtime scalar slot intact.
    for eps in (0.0,-0.0):
      for gemma in (False,True):
        EVENTS.clear();ctx=Context()
        F.forward(ctx,Tensor(torch.ones(1,4)),Tensor(torch.ones(4)),eps,gemma)
        launch=next(e for e in EVENTS if e[0]=='launch')
        delivered=launch[3][9]
        assert struct.pack('<d',delivered)==struct.pack('<d',eps)
        packed=struct.unpack('<I',struct.pack('<f',delivered))[0]
        assert packed==(0x80000000 if math.copysign(1.0,eps)<0 else 0)
    print('Both modes preserve admitted +0.0/-0.0 in the runtime epsilon slot; host FP32 packing keeps the sign.')
    bad=[]
    for side in (0,1):
      vals=[None,Tensor(torch.eye(2).to_sparse()),Tensor(torch.ones(2,4,dtype=torch.float64)),Tensor(torch.ones(2,4),False),Tensor(torch.ones(4,4).T),Tensor(torch._neg_view(torch.ones(2,4)))]
      vals += [Tensor(torch.ones(32)[off:off+8].view(2,4)) for off in (1,2,3)]
      for v in vals:
        pair=[Tensor(torch.ones(2,4)),Tensor(torch.ones(4))];pair[side]=v;bad.append((*pair,.7,False))
    base=torch.ones(16)
    huge=Tensor(torch.empty(0));huge.data=torch.empty(0)
    bad += [(Tensor(torch.ones(2,4)),Tensor(torch.ones(4),device=1),.7,False),
      (Tensor(torch.tensor(1.)),Tensor(torch.ones(1)),.7,False),
      (Tensor(torch.empty(2,0)),Tensor(torch.empty(0)),.7,False),
      (Tensor(torch.empty(0,65537)),Tensor(torch.ones(65537)),.7,False),
      (Tensor(torch.ones(2,4)),Tensor(torch.ones(5)),.7,False),
      (Tensor(base.view(4,4)),Tensor(base[:4]),.7,False)]
    for eps,gemma in [(1,False),(True,False),(.7,1),(float('nan'),False),(float('inf'),False)]:
        bad.append((Tensor(torch.ones(2,4)),Tensor(torch.ones(4)),eps,gemma))
    for args in bad:
        EVENTS.clear()
        try:F.forward(Context(),*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported call admitted')
        assert EVENTS==[]
    # Caller aliases in G remain supported: G is copied before either backward source.
    for role in ('x','w','r'):
        shape=(4,1) if role=='r' else (1,4)
        x=Tensor(torch.randn(shape));w=Tensor(torch.randn(shape[-1]));ctx=Context();F.forward(ctx,x,w,.7)
        g={'x':x,'w':w,'r':ctx.saved_tensors[2]}[role]
        expect=reference(x.data,w.data,g.data.view(shape),False)
        old=raw(g.data).clone();dx,dw,_,_=F.backward(ctx,g)
        torch.testing.assert_close(dx.data.double(),expect[1],rtol=4e-5,atol=4e-5);assert torch.equal(raw(g.data),old)
    x=Tensor(torch.randn(2,4));w=Tensor(torch.randn(4));ctx=Context();F.forward(ctx,x,w,.7)
    for item in [None,Tensor(torch.ones(8),False),Tensor(torch.ones(4,2).T),Tensor(torch.ones(8,dtype=torch.float64)),Tensor(torch._neg_view(torch.ones(8))),Tensor(torch.ones(7)),Tensor(torch.ones(8),device=1)]:
        EVENTS.clear()
        try:F.backward(ctx,item)
        except (TypeError,ValueError):pass
        else:raise AssertionError('unsupported backward call admitted')
        assert not EVENTS
    x.data.add_(1);EVENTS.clear()
    try:F.backward(ctx,Tensor(torch.ones(2,4)))
    except RuntimeError as e:assert 'version' in str(e)
    else:raise AssertionError('changed saved version accepted')
    assert not EVENTS
    # Real source layouts with singleton strides, aligned offset windows, and unaligned G.
    backing=torch.randn(96)
    extra=[(torch.as_strided(backing,(1,4),(99,1),4),torch.randn(4)),
           (torch.as_strided(backing,(2,1),(1,77)),torch.as_strided(torch.randn(8),(1,),(66,))),
           (torch.as_strided(backing,(1,1),(99,77)),torch.randn(1)),
           (backing[4:36].view(4,8),torch.randn(8))]
    for xd,wd in extra:
        x=Tensor(xd);w=Tensor(wd);g=Tensor(torch.randn(xd.numel()+1)[1:].view(xd.shape))
        ctx=Context();F.forward(ctx,x,w,.7);expected=reference(xd,wd,g.data,False)
        before=[raw(t.data).clone() for t in (x,w,g)]
        dx,dw,_,_=F.backward(ctx,g)
        for actual,ref in ((dx,expected[1]),(dw,expected[2])):torch.testing.assert_close(actual.data.double(),ref,rtol=4e-5,atol=4e-5)
        for t,snapshot in zip((x,w,g),before):assert torch.equal(raw(t.data),snapshot)
    # Different Tensor storage handles can expose the same read-only bytes.
    # The existing guard admits this, so the kernel placement cannot require
    # physical disjointness for X/W merely from distinct storage identities.
    shared=torch.ones(4)
    x=Tensor(shared.view(1,4));w=Tensor(torch.from_numpy(shared.numpy()));g=Tensor(torch.randn(1,4))
    assert x.untyped_storage()._cdata!=w.untyped_storage()._cdata and x.data_ptr()==w.data_ptr()
    expected=reference(x.data,w.data,g.data,False);old=raw(x.data).clone()
    ctx=Context();y=F.forward(ctx,x,w,.7);dx,dw,_,_=F.backward(ctx,g)
    for actual,ref in zip((y,dx,dw),expected):torch.testing.assert_close(actual.data.double(),ref,rtol=4e-5,atol=4e-5)
    assert torch.equal(raw(x.data),old) and torch.equal(raw(w.data),old)
    # Pointer metadata is not queried for an empty X, even on unaligned backing.
    backing=torch.arange(80,dtype=torch.uint8).untyped_storage()[1:65]
    x=Tensor(torch.empty(0,dtype=torch.float32).set_(backing,0,(0,4),(4,1)))
    ctx=Context();F.forward(ctx,x,Tensor(torch.ones(4)),.7);F.backward(ctx,x)
    class HugeRows(Tensor):
        @property
        def shape(self):return torch.Size((2147483648,1))
        def numel(self):return 2147483648
    EVENTS.clear()
    try:F.forward(Context(),HugeRows(torch.empty(0)),Tensor(torch.ones(1)),.7)
    except ValueError as e:assert 'grid' in str(e)
    else:raise AssertionError('oversized grid admitted')
    assert not EVENTS
    # Inject partial private writes at each source launch; all caller bytes survive.
    class FailedLaunch:
        def __getitem__(self,grid):
            def fail(*args,**kw):args[0].data.reshape(-1)[:1].fill_(99);raise RuntimeError('injected launch')
            return fail
    for name in (KERNELS[0],KERNELS[3],KERNELS[2]):
        x=Tensor(torch.randn(2,4));w=Tensor(torch.randn(4));g=Tensor(torch.randn(2,4));ctx=Context()
        if name!=KERNELS[0]:F.forward(ctx,x,w,.7)
        before=[raw(t.data).clone() for t in (x,w,g)]
        scope=F.forward.__globals__;saved=scope[name];scope[name]=FailedLaunch()
        try:
            try:
                if name==KERNELS[0]:F.forward(ctx,x,w,.7)
                else:F.backward(ctx,g)
            except RuntimeError as e:assert str(e)=='injected launch'
            else:raise AssertionError('launch failure swallowed')
        finally:scope[name]=saved
        for t,snapshot in zip((x,w,g),before):assert torch.equal(raw(t.data),snapshot)
    # A raw saved-input mutation can bypass PyTorch's version counter. Backward
    # then uses current X/W with the saved r; this is not the original adjoint.
    x=Tensor(torch.randn(2,4));w=Tensor(torch.randn(4));g=Tensor(torch.randn(2,4));ctx=Context();F.forward(ctx,x,w,.7)
    saved_r=ctx.saved_tensors[2].data.clone().view(2,1)
    fpraw(x.data).add_(.125)
    normalized=x.data*saved_r;weighted=g.data*w.data
    expected_dx=saved_r/4*(4*weighted-normalized*(weighted*normalized).sum(-1,keepdim=True))
    expected_dw=(g.data*normalized).sum(0)
    dx,dw,_,_=F.backward(ctx,g)
    torch.testing.assert_close(dx.data,expected_dx,rtol=4e-5,atol=4e-5)
    torch.testing.assert_close(dw.data,expected_dw,rtol=4e-5,atol=4e-5)
    # Original source overwrites caller upstream and returns no weight gradient.
    old=module(ORIGINAL);x=Tensor(torch.randn(2,4));w=Tensor(torch.randn(4));g=Tensor(torch.randn(2,4));ctx=Context()
    old.Fast_RMS_Layernorm.forward(ctx,x,w,.7);before=raw(g.data).clone()
    _,dw,_,_=old.Fast_RMS_Layernorm.backward(ctx,g)
    assert dw is None and not torch.equal(raw(g.data),before)
    print(f'{len(bad)+1} forward rejection cases, 7 backward rejection cases and {passed+len(extra)+2} complete forward/backward cases passed.')
    print('Direct saved X/W identity, private G→dX identity, dW-before-dX order, full caller backing bytes, empty zero fill and saved-version failure checked.')
    print('Original upstream mutation and missing dW reproduced. CPU AST execution and mocked runtime/version gates do not establish GPU numerical conformance.')

if __name__=='__main__':source_binding();host_checks()
