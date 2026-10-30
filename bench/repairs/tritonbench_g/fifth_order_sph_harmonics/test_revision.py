"""Real Torch autograd and real Triton CPU source tests; no launcher adapter."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
os.environ['TRITON_INTERPRET']='1'
import torch
import triton
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fifth_order_checked', HERE/'fifth_order_sph_harmonics_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tree=ast.parse((HERE/'fifth_order_sph_harmonics_repaired.py').read_text())
kernel=next(n for n in tree.body if getattr(n,'name',None)=='fifth_order_fwd_repaired')
# The independent reference uses Torch autograd on the forward polynomial AST.
assignments=[n for n in kernel.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
             and n.targets[0].id.startswith(('CONST','VAR','Y'))]
code=compile(ast.Module(body=assignments,type_ignores=[]),'<forward polynomial>','exec')
def polynomial(x):
    env=dict(x=x[:,0],y=x[:,1],z=x[:,2]);exec(code,env)
    return torch.stack([env[f'Y{i:02}'] for i in range(11)],1)

torch.manual_seed(918)
source_cases=0
for n in [1,7,63,64,65,127,129]:
  for block in [1,64,128]:
    x=torch.randn(n+2,3)[1:n+1].clone()
    before=x.clone();width,col=17,3
    y=torch.full((n+2,width),321.0)[1:n+1]
    m.fifth_order_fwd_repaired[(triton.cdiv(n,block),)](x,y,block,n*3,n*width,col,width,enable_fp_fusion=False)
    assert torch.equal(y[:,:col],torch.full_like(y[:,:col],321.0))
    assert torch.equal(y[:,col+11:],torch.full_like(y[:,col+11:],321.0))
    assert torch.equal(x,before)
    expected=polynomial(x)
    torch.testing.assert_close(y[:,col:col+11],expected,atol=0,rtol=0)
    for layout in ['dense','transpose','sliced','zero']:
      if layout=='dense':up=torch.randn(n,width)
      elif layout=='transpose':up=torch.randn(width,n).t()
      elif layout=='sliced':up=torch.randn(n*2,width*3)[::2,::3]
      else:up=torch.randn(1,1).expand(n,width)
      gx=torch.zeros_like(x)
      m.fifth_order_bwd_repaired[(triton.cdiv(n,block),)](x,gx,up,block,n*3,n*width,col,width,*up.stride(),enable_fp_fusion=False)
      refx=x.double().requires_grad_()
      gy=(polynomial(refx)*up[:,col:col+11].double()).sum()
      expectedg=torch.autograd.grad(gy,refx)[0].float()
      torch.testing.assert_close(gx,expectedg,atol=3e-3,rtol=2e-4)
      source_cases+=1

# CPU adaptation only replaces the explicit CUDA residency check/device guard.
# Actual Tensor metadata, allocations, slices, storage, .apply, saved tensors,
# once_differentiable, backward arity and kernel launch remain unchanged.
def cpu_tensor(t):
    if not isinstance(t,torch.Tensor):raise TypeError('expected Tensor')
    if t.layout!=torch.strided or t.dtype!=torch.float32 or t.is_neg() or t.is_conj() or t.ndim!=2:
        raise ValueError('unsupported metadata')
m._tensor=cpu_tensor
m.torch.cuda._DeviceGuard=lambda index:contextlib.nullcontext()
autograd_cases=0
for n in [0,1,65]:
  for arity in range(1,6):
    x=torch.randn(n,3,requires_grad=True)
    output=None if arity==1 else torch.full((n,11),99.0)
    args=[x,output,None,32,0][:arity]
    result=m.FifthOrderSphericalHarmonicRepaired.apply(*args)
    if output is not None:assert result is output
    result.sum().backward()
    ref=x.detach().double().requires_grad_()
    expected=torch.autograd.grad(polynomial(ref).sum(),ref)[0].float()
    torch.testing.assert_close(x.grad,expected,atol=3e-3,rtol=2e-4)
    autograd_cases+=1
# Caller padding and source block/column values are retained by backward.
x=torch.randn(65,3,requires_grad=True);out=torch.full((65,19),55.0)
result=m.FifthOrderSphericalHarmonicRepaired.apply(x,out,None,128,4)
assert result is out and torch.all(out[:,:4]==55) and torch.all(out[:,15:]==55)
up=torch.randn(19,65).t();result.backward(up)
ref=x.detach().double().requires_grad_()
expected=torch.autograd.grad((polynomial(ref)*up[:,4:15].double()).sum(),ref)[0].float()
torch.testing.assert_close(x.grad,expected,atol=3e-3,rtol=2e-4)
autograd_cases+=1
# Ordinary saved-Tensor version check remains in force.
x=torch.randn(1,3,requires_grad=True)
y=m.FifthOrderSphericalHarmonicRepaired.apply(x)
with torch.no_grad():x.add_(1)
try:y.sum().backward()
except RuntimeError as e:assert 'modified by an inplace operation' in str(e)
else:raise AssertionError('saved-coordinate mutation was not detected')

rejections=0
def reject(*args):
 global rejections
 try:m._forward_contract(*args)
 except (TypeError,ValueError):rejections+=1
 else:raise AssertionError('bad forward accepted')
x=torch.randn(3,3);out=torch.empty(3,11)
for bad in [0,3,2048,True,1.5]:reject(x,out,bad,0)
for bad in [-1,True,1.5]:reject(x,out,64,bad)
reject(x,None,64,1)
reject(x.t(),out,64,0)
reject(x,out.requires_grad_(),64,0)
reject(x,x,64,0)
reject(x,torch.empty(4,11),64,0)
reject(x,torch.empty(3,12)[:,1:],64,0)
reject(x,torch.empty(3,11),64,1)
# These controls exercise source-supported bounds without allocating large tensors.
large_admissions=[(2**31-1,11,1,0),((2**31-1)*1024,11,1024,0),(0,11,64,0)]
for values in large_admissions:m._launch_contract(*values)
for values in [(2**31,11,1,0),(2**63//11,11,1024,0),(2**32+1,2**31-1,1024,0)]:
    try:m._launch_contract(*values)
    except ValueError:rejections+=1
    else:raise AssertionError('out-of-range source launch admitted')
report=dict(source_interpreter_cases=source_cases,autograd_cases=autograd_cases,
 guard_rejections=rejections,saved_coordinate_version_check=True,
 upstream_layouts=['dense','transpose','sliced','zero'],gpu_execution=False,
 large_launch_metadata_admissions=large_admissions,
 cpu_adapter='CUDA residency and device context only; no kernel/launcher changes')
(HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
