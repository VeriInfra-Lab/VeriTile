"""CPU execution of the actual JIT source and host; no GPU conformance claim."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('kl_support', D.parent/'cross_entropy1/Proofs/probe.py')
support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)
torch = support.torch; torch.set_num_threads(1)
SOURCE = D/'kldiv_ops_repaired.py'
ORIGINAL = D.parents[2]/'tritonbench_g/kldiv_ops/kldiv_ops.py'
for path, digest in json.loads((D/'source_pin.json').read_text())['files'].items():
    assert hashlib.sha256((D.parents[3]/path).read_bytes()).hexdigest() == digest

def kernels(path):
    return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name.startswith('_kldiv_kernel_')]
assert kernels(SOURCE) == kernels(D/'kldiv_ops_conditional.py')
for n in ast.walk(ast.parse(SOURCE.read_text())):
    if isinstance(n, ast.Attribute):
        assert n.attr not in ('resolve_neg', 'contiguous', 'clone', 'copy_')

class TorchHost(support.TorchHost):
    def zeros(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        out = torch.zeros(*args, **kwargs)
        self.runner.allocations.append(out)
        return out
    def tensor(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return torch.tensor(*args, **kwargs)

class Kernel:
    def __init__(self, run, kernel): self.run, self.kernel = run, kernel
    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            self.run.launches.append((grid, args, kwargs))
            return self.kernel[grid](*args, **kwargs)
        return invoke

class Runner:
    def __init__(self, name, path=SOURCE, original=False):
        self.allocations, self.launches = [], []
        spec = importlib.util.spec_from_file_location('kl_ops_'+name, path)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        self.module.torch = TorchHost(self)
        suffix = ''
        for direction in ('forward', 'backward'):
            key = '_kldiv_kernel_'+direction+suffix
            setattr(self.module, key, Kernel(self, getattr(self.module, key)))

def fwd(run, x, y, flag, mode, eps=1e-6):
    return run.module.kldiv_forward_triton(x,y,flag,mode,eps)
def bwd(run, dst, target, grad, flag):
    return run.module.kldiv_backward_triton(target,grad,dst,flag)

rejected, cases = [], []
x0 = lambda: torch.tensor([[.1,.2,.3],[.4,.5,.6]])
y0 = lambda: torch.tensor([[.2,.3,.4],[.5,.6,.7]])
g0 = lambda: torch.tensor(1.)
r = Runner('guards'); m = r.module

def reject(name, f, kind=(TypeError, ValueError)):
    try: f()
    except kind: rejected.append(name)
    else: raise AssertionError('admitted '+name)

reject('CPU', lambda: fwd(r,x0(), y0(), False, 'none'))
with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)):
    for name, x, y in [
        ('nonTensor', [], y0()), ('fp16', x0().half(), y0()), ('fp64', x0(), y0().double()),
        ('integer', x0().int(), y0()), ('sparse', x0().to_sparse(), y0()),
        ('device', torch.empty((2,3),device='meta'), y0()), ('rank', x0().flatten(), y0().flatten()),
        ('shape', x0(), y0().T.contiguous()), ('prediction stride', x0().T, y0().T),
        ('target stride', x0(), torch.ones(2,6)[:,::2]),
        ('prediction negative', torch._neg_view(x0()), y0()),
        ('target negative', x0(), torch._neg_view(y0())),
        ('prediction alignment', torch.frombuffer(bytearray(25),dtype=torch.float32,offset=1,count=6).view(2,3),y0()),
        ('target alignment', x0(), torch.frombuffer(bytearray(26),dtype=torch.float32,offset=2,count=6).view(2,3)),
    ]:
        reject(name, lambda x=x,y=y: fwd(r,x,y,False,'none'))
    for flag in (0, 1, None, 'false'):
        reject('flag '+str(flag), lambda flag=flag: fwd(r,x0(),y0(),flag,'none'))
    for mode in ('bad', None, 0, []):
        reject('mode '+str(mode), lambda mode=mode: fwd(r,x0(),y0(),False,mode))
    for name, g in [('gradient rank1', torch.ones(3)), ('gradient rank3', torch.ones(1,2,3)),
                    ('gradient fp16', g0().half()), ('gradient negative',torch._neg_view(g0())),
                    ('gradient stride',torch.ones(2,6)[:,::2]),
                    ('gradient alignment',torch.frombuffer(bytearray(7),dtype=torch.float32,offset=3,count=1).view(()))]:
        reject(name, lambda g=g: bwd(r,x0(),y0(),g,False))
    x=x0();reject('destination target alias',lambda:bwd(r,x,x,g0(),False))
    x=x0();reject('destination gradient alias',lambda:bwd(r,x,y0(),x,False))
    z=torch.ones(12);reject('disjoint views in same allocation',lambda:bwd(r,z[:6].view(2,3),z[6:].view(2,3),g0(),False))
    reject('backward nonbool',lambda:bwd(r,x0(),y0(),g0(),0))
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    for epsilon in (None,0,True,torch.tensor(1e-6),'1e-6'):
        reject('epsilon '+str(epsilon),lambda epsilon=epsilon:fwd(r,x0(),y0(),False,'sum',epsilon))
assert not r.allocations and not r.launches


def snapshot(*tensors):
    return {t.untyped_storage()._cdata: (t, support.storage_bytes(t)) for t in tensors}

def check_kept(before):
    for t, old in before.values(): assert torch.equal(old, support.storage_bytes(t))

def forward(name, x, y, flag, mode, eps=1e-6):
    run = Runner(name)
    kept = snapshot(x,y)
    yd, xd = y.double(), x.double()
    expected = torch.exp(yd)*(yd-xd) if flag else yd*(torch.log(torch.maximum(yd,torch.tensor(eps,dtype=torch.float32).double()))-xd)
    if mode != 'none':
        expected = expected.sum()
        if mode == 'mean': expected = expected / x.numel()
        if mode == 'batchmean': expected = expected / x.shape[0]
    with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)), \
         patch.object(torch.Tensor,'contiguous',side_effect=AssertionError('packing')), \
         patch.object(torch.Tensor,'resolve_neg',side_effect=AssertionError('normalization')):
        out = fwd(run,x,y,flag,mode,eps)
    assert len(run.allocations)==1 and len(run.launches)==1
    grid,args,kw = run.launches[0]
    assert args[0] is x and args[2] is y and args[4] is run.allocations[0]
    block=kw['BLOCK_SIZE']; chunks=(x.shape[1]+block-1)//block
    assert block > 0 and grid == (x.shape[0],)
    assert run.allocations[0].shape == (x.shape if mode=='none' else (x.shape[0],))
    assert out.dtype==torch.float32 and out.device==x.device
    if mode=='none': assert out is run.allocations[0]
    else: assert out.ndim==0
    torch.testing.assert_close(out.double(),expected,atol=4e-5,rtol=4e-5,equal_nan=True)
    check_kept(kept)
    cases.append(name)


def backward(name, dst, target, grad, flag):
    run = Runner(name)
    before = support.storage_bytes(dst)
    kept = snapshot(target,grad)
    td=target.double(); expected = -torch.exp(td) if flag else -td
    expected_out=expected*grad.double()
    scalar_one = torch.equal(grad,torch.tensor(1.,dtype=grad.dtype))
    with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)), \
         patch.object(torch.Tensor,'contiguous',side_effect=AssertionError('packing')), \
         patch.object(torch.Tensor,'resolve_neg',side_effect=AssertionError('normalization')):
        out = bwd(run,dst,target,grad,flag)
    assert not run.allocations and len(run.launches)==1
    args=run.launches[0][1]; assert args[0] is target and args[2] is dst
    torch.testing.assert_close(dst.double(),expected,atol=4e-5,rtol=4e-5,equal_nan=True)
    if scalar_one: assert out is dst
    else:
        assert out is not dst and out.shape==dst.shape
        torch.testing.assert_close(out.double(),expected_out,atol=4e-5,rtol=4e-5,equal_nan=True)
    changed=support.storage_bytes(dst); active=support.footprint(dst); base=dst.untyped_storage().data_ptr()
    for i in range(len(before)):
        if base+i not in active: assert changed[i]==before[i]
    check_kept(kept)
    cases.append(name)

for shape in ((0,4),(3,0),(0,0),(1,1),(2,3),(2,1025),(2,16385),(1,32769)):
    for flag in (False,True):
        for mode in ('none','sum','mean','batchmean'):
            n=shape[0]*shape[1]
            x=torch.linspace(-.5,.5,n).reshape(shape)
            y=torch.linspace(.1,.8,n).reshape(shape)
            if not flag and n: y.flatten()[0]=0.
            forward('forward_'+str((shape,flag,mode)),x,y,flag,mode)
        for g in (torch.tensor(1.),torch.tensor(.25),torch.full(shape,.5)):
            backward('backward_'+str((shape,flag,g.ndim)),torch.full(shape,42.),y,g,flag)
for shape,stride in [((1,3),(77,1)),((2,1),(1,7)),((0,4),(19,7))]:
    for flag in (False,True):
        x=torch.empty_strided(shape,stride).fill_(.1); y=torch.empty_strided(shape,stride).fill_(.3)
        forward('degenerate_forward_'+str((shape,flag)),x,y,flag,'sum')
        backward('degenerate_backward_'+str((shape,flag)),x,y,g0(),flag)
x=torch.arange(20,dtype=torch.float32)/10
forward('overlapping_reads',x[2:8].view(2,3),x[4:10].view(2,3),False,'none')
backward('offset_destination',torch.full((20,),42.)[4:10].view(2,3),y0(),g0(),False)
for tail in (1,2,3):
    dst=support.partial_float_storage(tail)
    backward('partial_tail_'+str(tail),dst,y0(),g0(),False)
x=torch.empty(0,3)
backward('empty_alias',x,x,torch.tensor(1.),False)
x=torch.tensor([[0.,-0.,-1.,float('nan'),float('inf'),float('-inf')]])
y=torch.tensor([[0.,0.,-.2,.3,float('inf'),float('-inf')]])
for flag in (False,True):
    forward('exceptional_forward_'+str(flag),x,y,flag,'none')
    backward('exceptional_backward_'+str(flag),torch.zeros_like(x),y,g0(),flag)

for eps in (0.,-.1,1e-40,float('inf'),float('-inf'),float('nan')):
    for flag in (False,True):
        for mode in ('none','sum','mean','batchmean'):
            forward('epsilon_'+str((eps,flag,mode)),x0(),y0(),flag,mode,eps)

# Ordinary failure prefixes: prelaunch leaves destination unchanged; errors
# after a completed launch retain derivative writes and their byte complement.
failures=[]
class FailedKernel:
    def __getitem__(self, grid):
        def fail(*args, **kwargs): raise RuntimeError("injected setup failure")
        return fail
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    run=Runner('forward_setup_failure'); x=x0(); y=y0(); kept=snapshot(x,y)
    run.module._kldiv_kernel_forward=FailedKernel()
    reject('forward setup failure',lambda:fwd(run,x,y,False,'sum'),RuntimeError)
    check_kept(kept);failures.append('forward setup')
    run=Runner('forward_sum_failure'); kept=snapshot(x,y)
    with patch.object(torch.Tensor,'sum',side_effect=RuntimeError('injected sum failure')):
        reject('forward sum failure',lambda:fwd(run,x,y,False,'sum'),RuntimeError)
    check_kept(kept);failures.append('forward sum')
    run=Runner('backward_setup_failure'); dst=x0(); kept=snapshot(dst,y)
    run.module._kldiv_kernel_backward=FailedKernel()
    reject('backward setup failure',lambda:bwd(run,dst,y,g0(),False),RuntimeError)
    check_kept(kept);failures.append('backward setup')
    for op in ('tensor','equal'):
        run=Runner('backward_'+op+'_failure'); dst=torch.full((10,),42.)[2:8].view(2,3)
        old=support.storage_bytes(dst); kept=snapshot(y)
        setattr(run.module.torch,op,lambda *a,**k:(_ for _ in ()).throw(RuntimeError('injected finish failure')))
        reject('backward '+op+' failure',lambda:bwd(run,dst,y,g0(),False),RuntimeError)
        torch.testing.assert_close(dst,-y);check_kept(kept)
        now=support.storage_bytes(dst); assert torch.equal(now[:8],old[:8]) and torch.equal(now[32:],old[32:])
        failures.append('backward '+op)

# Separate original-program failures, with CPU routing only.
with tempfile.TemporaryDirectory(prefix='veritile-kl-ops-original-') as tmp:
    path=Path(tmp)/'original.py';path.write_text(ORIGINAL.read_text().split('def test_kldiv',1)[0])
    old=Runner('original',path,True)
    out=fwd(old,torch.zeros(1,3),torch.ones(1,3),False,'sum',0.)
    assert torch.isnan(out)
    y=torch.tensor([1.,1.,2.,2.,3.,3.])[::2].view(1,3)
    out=fwd(old,torch.zeros(1,3),y,False,'none')
    assert not torch.allclose(out,y*torch.log(y))
    t=torch._neg_view(torch.ones(1,3)); dst=torch.zeros(1,3)
    bwd(old,dst,t,torch.tensor(1.),False)
    assert torch.equal(dst,-torch.ones(1,3)) and not torch.equal(dst,-t)
print('3 original counterexamples reproduced')
print(f'{len(rejected)-len(failures)} rejected; {len(failures)} failure prefixes; {len(cases)} actual source/reference/frame cases passed')
