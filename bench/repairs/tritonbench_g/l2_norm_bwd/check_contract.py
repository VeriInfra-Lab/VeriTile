"""Actual source/host regression checks, not native floating-point evidence."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'l2_norm_bwd_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/l2_norm_bwd/l2_norm_bwd.py'
spec = importlib.util.spec_from_file_location('l21_helpers', HERE.parent / 'l2_norm_triton1/check_contract.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
function, raw_storage, Pointer = helpers.function, helpers.raw_storage, helpers.Pointer
CHECKS = [
    ('x.layout != torch.strided or dy.layout != torch.strided', 'x.layout != .strided || dy.layout != .strided', 'layout : x.layout=.strided ∧ dy.layout=.strided'),
    ('x.dtype != torch.float32 or dy.dtype != torch.float32', 'x.dtype != .floating .fp32 || dy.dtype != .floating .fp32', 'dtype : x.dtype=.floating .fp32 ∧ dy.dtype=.floating .fp32'),
    ('not x.is_cuda or not dy.is_cuda', '!(x.isCuda && dy.isCuda)', 'cuda : x.device.kind="cuda" ∧ dy.device.kind="cuda"'),
    ('x.device != dy.device', 'x.device != dy.device', 'sameDevice : x.device=dy.device'),
    ('dy.ndim == 0', 'dy.shape=[]', 'rank : dy.shape≠[]'),
    ('N == 0 or N > 16384', 'dy.shape.getLastD 0=0 || dy.shape.getLastD 0>16384', 'width : 0<dy.shape.getLastD 0 ∧ dy.shape.getLastD 0≤16384'),
    ('x.numel() % N != 0', 'x.numel%dy.shape.getLastD 0 != 0', 'divisible : x.numel%dy.shape.getLastD 0=0'),
    ('dy.numel() < x.numel()', 'dy.numel<x.numel', 'enough : x.numel≤dy.numel'),
    ('not x.is_contiguous() or not dy.is_contiguous()', '!(x.isContiguous && dy.isContiguous)', 'contiguous : x.isContiguous=Bool.true ∧ dy.isContiguous=Bool.true'),
    ('x.is_neg() or dy.is_neg()', 'x.isNeg || dy.isNeg', 'noNeg : x.isNeg=Bool.false ∧ dy.isNeg=Bool.false'),
    ('x.numel() > 0 and (x.data_ptr() % 4 != 0 or dy.data_ptr() % 4 != 0)', 'x.numel>0 && (x.dataPtr.toNat%4 != 0 || dy.dataPtr.toNat%4 != 0)', 'alignment : 0<x.numel → x.dataPtr.toNat%4=0 ∧ dy.dataPtr.toNat%4=0'),
]


def source_binding():
    pins=json.loads((HERE/'source_pin.json').read_text())
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==pins['original_sha256']=='5292032417de1cc1b43af26b5ed7846094dd07baf0154f8e06f933532552ab37'
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==pins['candidate_sha256']
    assert hashlib.sha256((HERE/'Kernels.lean').read_bytes()).hexdigest()==pins['quotation_sha256']
    kernel=function(SOURCE,'_l2_norm_bwd_kernel_repaired')
    assert hashlib.sha256(ast.dump(kernel).encode()).hexdigest()==pins['kernel_ast_sha256']=='5f37bb0fb3b81b4af25055aa95a9b506e9d90ebf67848c2f42cc2fbe8ec794a3'
    wrapper=function(SOURCE,'_l2_norm_bwd')
    guards=[n.test for n in wrapper.body if isinstance(n,ast.If)]
    expected=['not isinstance(x, torch.Tensor) or not isinstance(dy, torch.Tensor)',*(r[0] for r in CHECKS)]
    expected+=['N > BLOCK_N','dy.numel() < x.numel()','x.device != dy.device']
    assert list(map(ast.dump,guards))==[ast.dump(ast.parse(v,mode='eval').body) for v in expected]
    lean=(HERE/'Contract.lean').read_text()
    checks=lean.split('def checkTensorInputs',1)[1].split('def checkInputs',1)[0]
    assert re.findall(r'\bif (.*?) then',checks)==[r[1] for r in CHECKS]
    for _,_,field in CHECKS: assert field in lean.split('inductive InputError')[0]
    calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert calls.count('view')==3
    assert not set(calls)&{'clone','contiguous','resolve_neg','reshape','detach'}
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    print('Original/hash binding preserved; historical kernel/JIT AST unchanged; 12 admission guards match Lean; three input/output alias views.')


class Tensor(helpers.TensorFacade):
    def stride(self,*args): return self.data.stride(*args)
    def element_size(self): return self.data.element_size()
    def view(self,*shape): return Tensor(self.data.view(*shape),self.is_cuda)


def interpreter(path,name):
    kernel,pid=helpers.interpreter(path,name)
    kernel.__globals__['tl'].sum=lambda x,axis=0:x.sum(dim=axis)
    return kernel,pid


def host_checks():
    spec=importlib.util.spec_from_file_location('l2bwd_source',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    kernel,pid=interpreter(SOURCE,'_l2_norm_bwd_kernel_repaired')
    def empty_like(x):
        out=Tensor(torch.empty_like(x.data));calls.append(('allocation',out));return out
    class Launch:
        def __getitem__(self,grid):
            def run(x,dy,dx,*args):
                calls.append((grid,x,dy,dx,args))
                for row in range(grid[0]):
                    pid[0]=row;kernel(Pointer(x.data),Pointer(dy.data),Pointer(dx.data),*args)
            return run
    module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    module._l2_norm_bwd_kernel_repaired=Launch()
    base=torch.arange(256,dtype=torch.float32)
    one=Tensor(torch.ones(8))
    bad=[]
    for side in (0,1):
        unsupported=[(None,TypeError),(Tensor(torch.eye(2).to_sparse()),TypeError),
            (Tensor(torch.ones(8,dtype=torch.float64)),TypeError),(Tensor(torch.ones(8),cuda=False),ValueError),
            (Tensor(torch.ones(4,2).T),ValueError),(Tensor(torch._neg_view(torch.ones(8))),ValueError)]
        for misalignment in (1,2,3):
            storage=torch.arange(80,dtype=torch.uint8).untyped_storage()[misalignment:misalignment+64]
            data=torch.empty(0,dtype=torch.float32).set_(storage,0,(8,),(1,))
            assert data.data_ptr()%4==misalignment
            unsupported.append((Tensor(data),ValueError))
        for item,error in unsupported:
            pair=[one,one];pair[side]=item;bad.append((*pair,error))
    different=Tensor(torch.ones(8));different.device=torch.device('cuda:1')
    bad += [(one,different,ValueError),(Tensor(torch.tensor(1.)),Tensor(torch.tensor(1.)),ValueError),
        (Tensor(torch.empty(0)),Tensor(torch.empty(0,0)),ValueError),
        (Tensor(torch.empty(0)),Tensor(torch.empty(0,16385)),ValueError),
        (Tensor(torch.ones(7)),one,ValueError),(Tensor(torch.ones(16)),one,ValueError)]
    for x,dy,error in bad:
        calls.clear()
        try:module._l2_norm_bwd(x,dy)
        except error:pass
        else:raise AssertionError('unsupported arguments admitted')
        assert not calls
    storage=torch.arange(80,dtype=torch.uint8).untyped_storage()[1:65]
    empty_unaligned=torch.empty(0,dtype=torch.float32).set_(storage,0,(0,8),(8,1))
    good=[(torch.randn(2,n),torch.randn(3,n),1e-5) for n in (1,3,16,32,128,129,16384)]
    good += [(torch.randn(2,6),torch.randn(4,4),1e-5),(torch.tensor(2.),torch.randn(1),1e-5),
        (torch.empty(0,3,5),torch.empty(0,8),1e-5),(empty_unaligned,empty_unaligned,1e-5),
        (torch.empty(0,8),torch.randn(3,8),1e-5),(base[1:33].view(4,8),base[2:50].view(3,16),1e-5),
        (torch.as_strided(base,(1,8),(99,1),3),torch.randn(8),1e-5),(base[:16],base[:16],1e-5),
        (torch.tensor([[float('nan'),-0.,float('inf'),-float('inf')]]),torch.ones(2,4),1e-5)]
    good += [(torch.as_strided(base,(2,1),(1,77)),torch.as_strided(base,(3,1),(1,66)),1e-5),
        (torch.as_strided(base,(1,1),(99,77)),torch.as_strided(base,(1,1),(88,66)),1e-5),
        (torch.as_strided(base,(0,1),(99,77)),torch.as_strided(base,(0,1),(88,66)),1e-5)]
    good += [(torch.randn(2,4),torch.randn(3,4),eps) for eps in (-0.,-1.,float('nan'),float('inf'))]
    for xd,dd,eps in good:
        calls.clear();before=[raw_storage(v).view(torch.int32).clone() for v in (xd,dd)]
        x,dy=Tensor(xd),Tensor(dd);out=module._l2_norm_bwd(x,dy,eps)
        n=dd.shape[-1];xx=xd.view(-1,n);gg=dd.view(-1,n)[:xx.shape[0]]
        var=(xx*xx).sum(-1,keepdim=True);rstd=1/torch.sqrt(var+eps)
        expected=(gg*rstd-(gg*xx).sum(-1,keepdim=True)*(1/(var+eps))*rstd*xx).view(xd.shape)
        torch.testing.assert_close(out.data,expected,rtol=5e-6,atol=2e-6,equal_nan=True)
        assert out.shape==x.shape
        for v,snapshot in zip((xd,dd),before):assert torch.equal(raw_storage(v).view(torch.int32),snapshot)
        allocation=calls[0][1];grid,kx,kd,ky,args=calls[1]
        assert len(calls)==2 and grid==(xd.numel()//n,)
        for k,v in ((kx,xd),(kd,dd)):
            assert k.data.untyped_storage()._cdata==v.untyped_storage()._cdata
            if v.numel():assert k.data.data_ptr()==v.data_ptr()
        assert ky is allocation and out.data.untyped_storage()._cdata==allocation.data.untyped_storage()._cdata
        assert all(out.data.untyped_storage()._cdata!=v.untyped_storage()._cdata for v in (xd,dd))
        assert args[:6]==(*kx.stride(),*kd.stride(),*ky.stride())
        assert args[6]==n and args[8]==1<<(n-1).bit_length()
    old,old_pid=interpreter(ORIGINAL,'_l2_norm_bwd_kernel')
    x=torch.randn(2,8);dy=torch.randn(4,8)[::2];dx=torch.empty_like(x)
    old_pid[0]=1;old(Pointer(x),Pointer(dy),Pointer(dx),x.stride(0),8,1e-5,8)
    xx=x[1];gg=dy[1];q=(xx*xx).sum()+1e-5;r=1/torch.sqrt(q)
    correct=gg*r-(gg*xx).sum()*(1/q)*r*xx
    assert not torch.allclose(dx[1],correct,rtol=1e-5,atol=1e-6)
    x=torch.ones(4,8)[::2];dy=torch.ones(2,8);dx=torch.empty_like(x)
    try:old(Pointer(x),Pointer(dy),Pointer(dx),x.stride(0),8,1e-5,8)
    except (AssertionError,IndexError):pass
    else:raise AssertionError('original stride escape was not reproduced')
    print(f'{len(bad)} rejection and {len(good)} direct-view cases passed, including extra DY prefix, scalar X, aliases, offsets 4/8/12 and empty unaligned storage.')
    print('Original wrong-DY-row and out-of-bounds stride defects reproduced. CPU graph checks do not establish native numerical conformance.')


if __name__=='__main__':
    source_binding();host_checks()
