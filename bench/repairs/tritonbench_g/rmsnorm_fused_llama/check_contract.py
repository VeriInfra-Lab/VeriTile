"""Actual wrapper/kernel-AST checks; no GPU numerical-conformance claim."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'rmsnorm_fused_llama_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rmsnorm_fused_llama/rmsnorm_fused_llama.py'
spec=importlib.util.spec_from_file_location('fast_rms_helpers',HERE.parent/'fast_rms_layernorm/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,Pointer,raw,fpraw=helpers.function,helpers.Pointer,helpers.raw,helpers.fpraw

def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    for key,path in [('original_sha256',ORIGINAL),('candidate_sha256',SOURCE),('quotation_sha256',HERE/'Kernels.lean')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[key]
    kernel=function(SOURCE,'_rms_norm_fwd_fused')
    assert ast.dump(kernel)==ast.dump(function(ORIGINAL,'_rms_norm_fwd_fused'))
    assert hashlib.sha256(ast.dump(kernel).encode()).hexdigest()==pins['kernel_ast_sha256']
    wrapper=function(SOURCE,'rmsnorm_forward')
    checks=[n.test for n in wrapper.body if isinstance(n,ast.If)]
    expected=['not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor)',
      'x.layout != torch.strided or weight.layout != torch.strided',
      'x.dtype != torch.float32 or weight.dtype != torch.float32','x.ndim < 1 or weight.ndim != 1',
      'not x.is_contiguous() or not weight.is_contiguous()',
      'x.is_neg() or x.is_conj() or weight.is_neg() or weight.is_conj()',
      'x.device != weight.device or not x.is_cuda','N <= 0 or N > 16384 or weight.numel() != N',
      'x.numel() + 16383 >= 2**31','type(eps) is not float or not 0 < eps < float("inf")',
      'x.numel() and (x.data_ptr() % 16 or weight.data_ptr() % 16)','x.numel() == 0']
    assert list(map(ast.dump,checks))==[ast.dump(ast.parse(s,mode='eval').body) for s in expected]
    contract=(HERE/'Contract.lean').read_text()
    for clause in ['x.layout=.strided ∧ w.layout=.strided','x.dtype=.floating .fp32 ∧ w.dtype=.floating .fp32',
      '1≤x.shape.length ∧ w.shape.length=1','x.isContiguous && w.isContiguous','!(x.isNeg || x.isConj || w.isNeg || w.isConj)',
      'decide (x.device=w.device) && x.isCuda','0<width x ∧ width x≤16384 ∧ w.numel=width x','x.numel+16383<2^31',
      'eps.pythonFloat && eps.finite && decide (0<eps.value)','x.numel=0 ∨ (x.dataPtr.toNat%16=0 ∧ w.dataPtr.toNat%16=0)']:
        assert clause in contract
    calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'clone','contiguous','reshape','resolve_neg','detach'}
    assert calls.count('view')==1
    print('Original kernel/decorator AST unchanged; candidate/hash binding and ordered admission checks match Lean.')

class Tensor(helpers.Tensor):
    def is_conj(self):return self.data.is_conj()
    def view(self,*shape):
        out=Tensor(self.data.view(*shape),self.is_cuda,self.device.index or 0);helpers.EVENTS.append(('view',self,out));return out

def kernel_ast(path):
    k,pid=helpers.interpreter(path,'_rms_norm_fwd_fused')
    tl=k.__globals__['tl'];tl.zeros=lambda shape,dtype:torch.zeros(shape,dtype=dtype);tl.sqrt=torch.sqrt;tl.float16=torch.float16
    def store(p,v,mask=None):
        off=torch.as_tensor(p.offsets,dtype=torch.int64)
        mask=torch.ones_like(off,dtype=torch.bool) if mask is None else torch.as_tensor(mask,dtype=torch.bool)
        indices=p.tensor.storage_offset()+off[mask]
        assert torch.all(indices>=0) and torch.all(indices<fpraw(p.tensor).numel()),'source store out of bounds'
        fpraw(p.tensor)[indices]=torch.broadcast_to(torch.as_tensor(v),off.shape)[mask].float()
    tl.store=store
    return k,pid

def host_checks():
    scope={'__name__':'rmsnorm_llama_candidate'};exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),scope)
    events=[];active=[2];kernel,pid=kernel_ast(SOURCE)
    @contextlib.contextmanager
    def device(d):
        saved=active[0];active[0]=d.index;events.append(('enter',d.index))
        try:yield
        finally:active[0]=saved;events.append(('exit',saved))
    def empty_like(x,memory_format):
        out=Tensor(torch.empty_like(x.data,memory_format=memory_format),device=x.device.index);events.append(('allocate',out));return out
    class Launch:
        fail=False
        def __getitem__(self,grid):
            def run(x,y,w,stride,n,eps,**kw):
                events.append(('launch',grid,x,y,w,stride,n,eps,kw,active[0]))
                assert active[0]==x.device.index
                if self.fail:y.data.reshape(-1)[:1].fill_(99);raise RuntimeError('injected launch')
                for row in range(grid[0]):pid[0]=row;kernel(Pointer(x.data),Pointer(y.data),Pointer(w.data),stride,n,eps,kw['BLOCK_SIZE'])
            return run
    launch=Launch();scope['_rms_norm_fwd_fused']=launch
    scope['torch']=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
      contiguous_format=torch.contiguous_format,empty_like=empty_like,cuda=SimpleNamespace(device=device))
    fn=scope['rmsnorm_forward']
    good=[(torch.randn(2,n),torch.randn(n)) for n in (1,3,16,128,257,8192,16384)]
    base=torch.randn(128)
    good += [(torch.randn(2,3,7),torch.randn(7)),(torch.randn(7),torch.randn(7)),
      (torch.empty(0,8),torch.randn(8)),(base[4:36].view(4,8),base[8:16]),
      (torch.as_strided(base,(1,8),(99,1),4),torch.randn(8)),
      (torch.as_strided(base,(2,1),(1,77)),torch.as_strided(torch.randn(8),(1,),(66,)))]
    storage=torch.arange(80,dtype=torch.uint8).untyped_storage()[1:65]
    good += [(torch.empty(0,dtype=torch.float32).set_(storage,0,(0,8),(8,1)),torch.randn(12)[1:9])]
    good += [(torch.tensor([[float('nan'),-0.,float('inf'),-float('inf')]]),torch.ones(4))]
    for xd,wd in good:
        events.clear();helpers.EVENTS.clear();before=[raw(t).clone() for t in (xd,wd)]
        x,w=Tensor(xd),Tensor(wd);out=fn(x,w,1e-5)
        expected=(xd*(1/torch.sqrt(xd.square().mean(-1,keepdim=True)+1e-5))*wd).half().float()
        torch.testing.assert_close(out.data,expected,rtol=2e-3,atol=2e-3,equal_nan=True)
        assert out.shape==x.shape and out.dtype==torch.float32
        assert out is events[0][1] and all(out.untyped_storage()._cdata!=t.untyped_storage()._cdata for t in (x,w))
        for t,saved in zip((xd,wd),before):assert torch.equal(raw(t),saved)
        assert active[0]==2
        if x.numel():
            assert len(events)==4
            _,grid,kx,ky,kw,stride,n,eps,options,current=events[2]
            assert ky is out and kw is w and stride==n==xd.shape[-1]
            assert grid==(xd.numel()//n,) and options=={'BLOCK_SIZE':16384,'num_warps':8}
            assert kx.untyped_storage()._cdata==x.untyped_storage()._cdata and kx.data_ptr()==x.data_ptr()
        else:assert len(events)==1 and not helpers.EVENTS
    bad=[]
    for side in (0,1):
        values=[None,Tensor(torch.eye(2).to_sparse()),Tensor(torch.ones(4,dtype=torch.float16)),Tensor(torch.ones(4),False),Tensor(torch.ones(4,4).T),Tensor(torch._neg_view(torch.ones(4)))]
        values += [Tensor(torch.randn(32)[off:off+4]) for off in (1,2,3)]
        for v in values:
            pair=[Tensor(torch.randn(2,4)),Tensor(torch.randn(4))];pair[side]=v;bad.append((*pair,1e-5))
    bad += [(Tensor(torch.tensor(1.)),Tensor(torch.ones(1)),1e-5),
      (Tensor(torch.ones(2,4)),Tensor(torch.ones(1,4)),1e-5),
      (Tensor(torch.empty(2,0)),Tensor(torch.empty(0)),1e-5),
      (Tensor(torch.empty(0,16385)),Tensor(torch.ones(16385)),1e-5),
      (Tensor(torch.ones(2,4)),Tensor(torch.ones(5)),1e-5),
      (Tensor(torch.ones(2,4)),Tensor(torch.ones(4),device=1),1e-5)]
    bad += [(Tensor(torch.randn(2,4)),Tensor(torch.randn(4)),e) for e in (1,True,0.,-1.,float('inf'),float('nan'))]
    for x,w,eps in bad:
        events.clear();helpers.EVENTS.clear()
        try:fn(x,w,eps)
        except (ValueError,TypeError):pass
        else:raise AssertionError('unsupported input admitted')
        assert not events and not helpers.EVENTS
    class PaddedOverflow(Tensor):
        @property
        def shape(self):return torch.Size((2147483648-16383,1))
        def numel(self):return 2147483648-16383
    try:fn(PaddedOverflow(torch.empty(0)),Tensor(torch.ones(1)),1e-5)
    except ValueError as e:assert '32-bit' in str(e)
    else:raise AssertionError('padded overflow admitted')
    x,w=Tensor(torch.randn(2,4)),Tensor(torch.randn(4));before=[raw(t.data).clone() for t in (x,w)]
    launch.fail=True;events.clear()
    try:fn(x,w,1e-5)
    except RuntimeError as e:assert str(e)=='injected launch'
    else:raise AssertionError('launch failure swallowed')
    assert active[0]==2 and events[-1]==('exit',2)
    for t,saved in zip((x,w),before):assert torch.equal(raw(t.data),saved)
    old,old_pid=kernel_ast(ORIGINAL);xd=torch.randn(4,8)[::2];yd=torch.empty_like(xd);wd=torch.randn(8);old_pid[0]=1
    try:old(Pointer(xd),Pointer(yd),Pointer(wd),xd.stride(0),8,1e-5,16384)
    except AssertionError:pass
    else:raise AssertionError('original output row-stride escape not reproduced')
    print(f'{len(bad)+1} admission rejection cases and {len(good)} direct-view cases passed; empty skips, exact return identity and full caller bytes checked.')
    print('Original row-stride store escape reproduced; partial launch failure restores device context. CPU AST checks are not native FP/GPU conformance.')

if __name__=='__main__':source_binding();host_checks()
