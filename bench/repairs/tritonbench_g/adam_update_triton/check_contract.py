"""CPU checks of actual guards, aliasing host flow, chunks and restored trials."""
import ast
import contextlib
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'adam_update_triton_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/adam_update_triton/adam_update_triton.py'
spec=importlib.util.spec_from_file_location('l21_helpers',HERE.parent/'l2_norm_triton1/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
function,Pointer,raw_storage=helpers.function,helpers.Pointer,helpers.raw_storage
KERNEL_HASH='684889e60713c2eb4552878c1fbe017b183718d664cb0d53601cbb37550bc6a5'
CHUNK=65535*128
GUARDS=[
 'not all(isinstance(t, torch.Tensor) for t in tensors)',
 'any(t.layout != torch.strided for t in tensors)',
 'any(t.dtype != torch.float32 for t in tensors)',
 'not all(t.is_cuda for t in tensors)',
 'grad.device != p.device or exp_avg.device != p.device',
 'grad.numel() < n_elements or exp_avg.numel() < n_elements',
 'not all(t.is_contiguous() for t in tensors)',
 'any(t.is_neg() for t in tensors)',
 'n_elements == 0',
 'any(t.data_ptr() % 16 != 0 for t in tensors)',
 'any(a[0] < b[1] and b[0] < a[1] for i, a in enumerate(ranges) for b in ranges[i + 1:])',
]
LEAN_GUARDS=[
 'p.layout != .strided || g.layout != .strided || m.layout != .strided',
 'p.dtype != .floating .fp32 || g.dtype != .floating .fp32 || m.dtype != .floating .fp32',
 '!(p.isCuda && g.isCuda && m.isCuda)',
 'g.device != p.device || m.device != p.device',
 'g.numel<p.numel || m.numel<p.numel',
 '!(p.isContiguous && g.isContiguous && m.isContiguous)',
 'p.isNeg || g.isNeg || m.isNeg',
 'p.numel=0',
 'p.dataPtr.toNat%16 != 0 || g.dataPtr.toNat%16 != 0 || m.dataPtr.toNat%16 != 0',
 '!(decide (DisjointRanges p g) && decide (DisjointRanges p m) && decide (DisjointRanges g m))',
]


def source_binding():
 assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()=='27d4f15342638339b2daea53f43752e30a28114a39d1b8dcdaf5e29b89072f45'
 pin=re.search(r'def candidateHash : String := "([0-9a-f]+)"',(HERE/'Proofs/Compilation.lean').read_text()).group(1)
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==pin
 kernel=function(SOURCE,'update_fn_kernel')
 assert hashlib.sha256(ast.dump(kernel).encode()).hexdigest()==KERNEL_HASH
 assert hashlib.sha256(ast.dump(function(SOURCE,'_numeric_scalar')).encode()).hexdigest()=='514261965af009b40980f16a47f390bbb82b1b20c556b7bd0b214b160c8c3b96'
 wrapper=function(SOURCE,'update_fn_repaired')
 actual=[n.test for n in wrapper.body if isinstance(n,ast.If)]
 assert list(map(ast.dump,actual))==[ast.dump(ast.parse(t,mode='eval').body) for t in GUARDS]
 lean=(HERE/'Contract.lean').read_text()
 assert re.findall(r'\bif (.*?) then',lean.split('def checkTensorInputs')[1].split('def checkInputs')[0])==LEAN_GUARDS
 for field in ('layout','dtype','cuda','sameDevice','enough','contiguous','noNeg','alignment','separate'):
  assert re.search(r'\n  '+field+r' :',lean)
 calls=[n.func.attr for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
 assert calls.count('view')==3
 assert not set(calls)&{'clone','copy_','contiguous','resolve_neg','detach','to','index_select','index_copy_'}
 forbidden={'_snapshot_words','_copyback_words','_byte_indices','_storage_bytes'}
 assert not forbidden&{n.name for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef)}
 assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
 lean_source=(HERE/'Kernels.lean').read_text()
 body=re.search(r'def sourceWithArgumentDeclarations\b.*?triton \{\n(.*?)\n\}',lean_source,re.S).group(1)
 body=re.sub(r'\$\(\((\d+) : ℝ\)\)',r'\1',body)
 body=body.replace('$(block)','BLOCK_SIZE').replace('$(chunk)','chunk_elements')
 declarations=re.findall(r'^  tl.param (\w+) : tl.float32$',body,re.M)
 assert declarations==['lr','wd','beta1','beta2']
 body=re.sub(r'^  tl.param \w+ : tl.float32\n','',body,flags=re.M)
 displayed=ast.parse(textwrap.dedent(body)).body
 class RegionNames(ast.NodeTransformer):
  def visit_Name(self,node):
   return ast.copy_location(ast.Name(id={'P':'p_ptr','G':'grad_ptr','M':'exp_avg_ptr'}.get(node.id,node.id),ctx=node.ctx),node)
 displayed=[RegionNames().visit(s) for s in displayed]
 expected=function(SOURCE,'update_fn_kernel').body
 class PromoteBool(ast.NodeTransformer):
  def visit_BinOp(self,node):
   self.generic_visit(node)
   if isinstance(node.op,ast.Mult) and isinstance(node.right,ast.Name) and node.right.id=='can_update':
    node.right=ast.parse('tl.where(can_update, 1, 0)',mode='eval').body
   return node
 expected=[PromoteBool().visit(s) for s in expected]
 assert list(map(ast.dump,displayed))==list(map(ast.dump,expected))
 assert '.surfaceBody.drop' not in lean_source and 'algebra.withDefaultPrecision .fp32' in lean_source
 candidate=re.search(r'def commutedWithArguments\b.*?triton \{\n(.*?)\n\}',lean_source,re.S).group(1)
 original=re.search(r'def sourceWithArgumentDeclarations\b.*?triton \{\n(.*?)\n\}',lean_source,re.S).group(1)
 assert candidate==original.replace('exp_avg = diff * beta2 + grad','exp_avg = grad + diff * beta2')
 print('Original and candidate hash bound; historical kernel/autotune AST unchanged; all tensor checks match Lean; only three alias views remain.')
 print('Existing candidate differs only at the final momentum addition; both full ordinary quotes retain runtime FP32 declarations.')
 print('All 19 canonical DSL statements bind to Python; nonexecuting FP32 scalar declarations retain runtime registers, and boolean multiplication promotes to FP32.')


class Tensor(helpers.TensorFacade):
 pointer_queries=[]
 def data_ptr(self):
  self.pointer_queries.append(self)
  return super().data_ptr()
 def view(self,*shape):return Tensor(self.data.view(*shape),self.is_cuda)
 def __getitem__(self,key):return Tensor(self.data[key],self.is_cuda)


def scalar_graph(p,g,m,lr,wd,b1,b2):
 diff=m-g;update=diff*b1+g
 return p*(1-lr*wd)+torch.where(update>0,-lr,lr)*(update!=0),diff*b2+g


def interpreter(path,name):
 kernel,pid=helpers.interpreter(path,name)
 tl=kernel.__globals__['tl']
 tl.program_id=lambda axis=0:torch.tensor(pid[0])
 old_load=tl.load
 tl.load=lambda p,mask,other=0.:old_load(p,mask,other)
 return kernel,pid


def host_checks():
 spec=importlib.util.spec_from_file_location('adam_direct_source',SOURCE)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 kernel,pid=interpreter(SOURCE,'update_fn_kernel')
 calls=[];contexts=[];validation_only=[False];fail_second=[False]
 @contextlib.contextmanager
 def device(owner):
  contexts.append(('enter',owner))
  try:yield
  finally:contexts.append(('exit',owner))
 class Launch:
  def __getitem__(self,grid):
   def run(p,g,m,lr,wd,b1,b2,total,count):
    calls.append((p,g,m,total,count,grid({'BLOCK_SIZE':128}),grid({'BLOCK_SIZE':1024})))
    if validation_only[0]:
     if fail_second[0]:
      if len(calls)==1:p.data[0]=123.;m.data[0]=456.
      else:raise RuntimeError('injected second chunk failure')
     return
    saved_p=p.data.view(torch.int32).clone();saved_m=m.data.view(torch.int32).clone()
    def execute(block):
     for row in range(grid({'BLOCK_SIZE':block})[0]):
      pid[0]=row;kernel(Pointer(p.data),Pointer(g.data),Pointer(m.data),lr,wd,b1,b2,total,count,block)
    for block in (128,1024,128):
     execute(block)
     p.data.view(torch.int32).copy_(saved_p);m.data.view(torch.int32).copy_(saved_m)
    execute(1024)
   return run
 module.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
  no_grad=contextlib.nullcontext,cuda=SimpleNamespace(device=device))
 module.update_fn_kernel=Launch()
 base=torch.arange(160,dtype=torch.float32)
 bad=[]
 for side in range(3):
  items=[(None,TypeError),(Tensor(torch.eye(2).to_sparse()),TypeError),
   (Tensor(torch.ones(8,dtype=torch.float64)),TypeError),(Tensor(torch.ones(8),cuda=False),ValueError),
   (Tensor(torch.ones(4,2).T),ValueError),(Tensor(torch._neg_view(torch.ones(8))),ValueError)]
  items += [(Tensor(base[o:o+8]),ValueError) for o in (1,2,3)]
  for item,error in items:
   tensors=[Tensor(torch.ones(8)) for _ in range(3)];tensors[side]=item;bad.append((*tensors,error))
 different=Tensor(torch.ones(8));different.device=torch.device('cuda:1')
 bad += [(Tensor(torch.ones(8)),different,Tensor(torch.ones(8)),ValueError),
  (Tensor(torch.ones(8)),Tensor(torch.ones(7)),Tensor(torch.ones(8)),ValueError),
  (Tensor(torch.ones(8)),Tensor(torch.ones(8)),Tensor(torch.ones(7)),ValueError)]
 for roles in ((0,1),(0,2),(1,2)):
  tensors=[Tensor(torch.ones(8)) for _ in range(3)];tensors[roles[1]]=tensors[roles[0]]
  bad.append((*tensors,ValueError))
 bad.append((Tensor(base[8:16]),Tensor(torch.ones(8)),Tensor(base[:24]),ValueError))
 for p,g,m,error in bad:
  calls.clear();contexts.clear()
  try:module.update_fn_repaired(p,g,m,.01,.02,.9,.99)
  except error:pass
  else:raise AssertionError('unsupported inputs admitted')
  assert not calls and not contexts
 good=[(torch.randn(n),torch.randn(n+3),torch.randn(n+7),(.01,.02,.9,.99)) for n in (1,3,127,128,129,1023,1024,1025,8193)]
 good += [(torch.randn(2,3),torch.randn(3,2),torch.randn(8),(.01,.02,.9,.99)),
  (torch.tensor(2.),torch.randn(4),torch.randn(3),(.01,.02,.9,.99)),
  (base[4:12],base[16:28],base[32:48],(.01,.02,.9,.99)),
  (torch.as_strided(base,(1,1),(99,77),4),torch.randn(2),torch.randn(2),(.01,.02,.9,.99)),
  (torch.tensor([float('nan'),float('inf'),-0.,float('-inf')]),torch.ones(4),torch.ones(4),(.01,.02,.9,.99)),
  (torch.randn(4),torch.randn(4),torch.randn(4),(float('nan'),-0.,float('inf'),-2.))]
 for pd,gd,md,scalars in good:
  calls.clear();contexts.clear();Tensor.pointer_queries.clear()
  p,g,m=Tensor(pd),Tensor(gd),Tensor(md);n=pd.numel()
  expected_p,expected_m=scalar_graph(pd.flatten().clone(),gd.flatten()[:n].clone(),md.flatten()[:n].clone(),*scalars)
  backing={t.untyped_storage()._cdata:raw_storage(t).view(torch.int32).clone() for t in (pd,gd,md)}
  original={id(t):(t.shape,t.stride(),t.storage_offset(),t.untyped_storage()._cdata) for t in (pd,gd,md)}
  result=module.update_fn_repaired(p,g,m,*scalars)
  assert result is None and len(calls)==1 and len(contexts)==2
  torch.testing.assert_close(pd.flatten(),expected_p,rtol=5e-6,atol=2e-6,equal_nan=True)
  torch.testing.assert_close(md.flatten()[:n],expected_m,rtol=5e-6,atol=2e-6,equal_nan=True)
  for t in (pd,gd,md):assert original[id(t)]==(t.shape,t.stride(),t.storage_offset(),t.untyped_storage()._cdata)
  changed={}
  for t in (pd,md):changed.setdefault(t.untyped_storage()._cdata,set()).update(range(t.storage_offset(),t.storage_offset()+n))
  for t in (pd,gd,md):
   actual=raw_storage(t).view(torch.int32);before=backing[t.untyped_storage()._cdata]
   mask=torch.tensor([i not in changed.get(t.untyped_storage()._cdata,set()) for i in range(actual.numel())])
   assert torch.equal(actual[mask],before[mask])
  kp,kg,km,total,count,_,_=calls[0]
  assert total==n==count
  for k,t in ((kp,pd),(kg,gd),(km,md)):
   assert k.data.data_ptr()==t.data_ptr() and k.data.untyped_storage()._cdata==t.untyped_storage()._cdata
 storage=torch.arange(80,dtype=torch.uint8).untyped_storage()[1:65]
 empty=torch.empty(0,dtype=torch.float32).set_(storage,0,(0,8),(8,1))
 calls.clear();contexts.clear();Tensor.pointer_queries.clear()
 assert module.update_fn_repaired(Tensor(empty),Tensor(empty),Tensor(empty),object(),None,[],{}) is None
 assert not calls and not contexts and not Tensor.pointer_queries
 class IntPayload(int):
  def __float__(self):raise AssertionError('subclass override used')
 class FloatPayload(float):
  def __float__(self):raise AssertionError('subclass override used')
 for value,expected in [(True,1.),(IntPayload(3),3.),(FloatPayload(-0.),-0.),(module.tl.constexpr(2),2.)]:
  actual=module._numeric_scalar(value,'test');assert actual==expected and math.copysign(1,actual)==math.copysign(1,expected)
 for value,error in [(object(),TypeError),(torch.tensor(1.),TypeError),(module.tl.constexpr(object()),TypeError),(10**1000,OverflowError)]:
  calls.clear();contexts.clear();p,g,m=(Tensor(torch.ones(8)) for _ in range(3))
  before=p.data.clone()
  try:module.update_fn_repaired(p,g,m,value,.02,.9,.99)
  except error:pass
  else:raise AssertionError('invalid scalar admitted')
  assert not calls and not contexts and torch.equal(p.data,before)
 validation_only[0]=True
 p,g,m=(Tensor(torch.empty(CHUNK+3)) for _ in range(3))
 calls.clear();contexts.clear();module.update_fn_repaired(p,g,m,.01,.02,.9,.99)
 assert [v[4] for v in calls]==[CHUNK,3] and all(v[3]==CHUNK+3 for v in calls)
 for offset,call in zip((0,CHUNK),calls):
  for k,t in zip(call[:3],(p,g,m)):assert k.data.data_ptr()==t.data.data_ptr()+4*offset
 assert calls[0][5]==(65535,) and calls[1][5]==(1,)
 fail_second[0]=True;calls.clear();contexts.clear()
 try:module.update_fn_repaired(p,g,m,.01,.02,.9,.99)
 except RuntimeError:pass
 else:raise AssertionError('failure injection failed')
 assert p.data[0]==123 and m.data[0]==456 and contexts[-1][0]=='exit'
 old,old_pid=interpreter(ORIGINAL,'update_fn_kernel')
 p=torch.arange(6,dtype=torch.float32).view(2,3).T;g=torch.tensor([[1.,-1.],[1.,-1.],[1.,-1.]]);m=torch.zeros(6)
 expected,_=scalar_graph(p.flatten().clone(),g.flatten(),m.clone(),.1,0.,.9,.99)
 old_pid[0]=0;old(Pointer(p),Pointer(g),Pointer(m),.1,0.,.9,.99,6,128)
 assert not torch.allclose(p.flatten(),expected)
 print(f'{len(bad)} tensor rejection cases and {len(good)} direct in-place cases passed; repeated autotune trials restored both outputs.')
 print('Empty/scalar ABI paths, full caller-byte complement, shared disjoint storage, both chunk boundaries and failure context restoration passed.')
 print('Original noncontiguous logical-order defect reproduced. CPU arithmetic and mocked autotuning do not establish GPU numerical conformance.')


if __name__=='__main__':
 source_binding();host_checks()
