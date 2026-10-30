"""Actual source in Triton's CPU interpreter; no GPU execution claim."""
import os
os.environ['TRITON_INTERPRET']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import contextlib,hashlib,importlib.util,io,json,pathlib,shutil,traceback,warnings
import torch,triton
D=pathlib.Path(__file__).resolve().parents[1];R=D.parents[3];E=pathlib.Path('/tmp/veritile-cross-entropy2')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def info(t):
 return dict(shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype),bytes=t.untyped_storage().nbytes(),is_neg=t.is_neg())
def raw(t):
 return torch.empty(0,dtype=torch.uint8).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).clone()
class TorchHost:
 def __init__(self,runner):self.runner=runner
 def __getattr__(self,name):return getattr(torch,name)
 def empty(self,*a,**k):
  t=torch.empty(*a,**k);t.fill_(float('nan'));self.runner.allocations.append(t);return t
 def empty_like(self,*a,**k):
  t=torch.empty_like(*a,**k);t.fill_(float('nan'));self.runner.allocations.append(t);return t
class Kernel:
 def __init__(self,runner,kernel,name):self.runner,self.kernel,self.name=runner,kernel,name
 def __getitem__(self,grid):
  def run(*args,**kwargs):
   names=self.kernel.arg_names
   bound=dict(zip(names,args));bound.update(kwargs)
   self.runner.launches.append(dict(kernel=self.name,grid=list(grid),block=bound['BLOCK_SIZE'],split=bound.get('SPLIT'),tensors={k:info(v) for k,v in bound.items() if isinstance(v,torch.Tensor)}))
   return self.kernel[grid](*args,**kwargs)
  return run
class Runner:
 def __init__(self,path,name):
  self.allocations=[];self.launches=[]
  spec=importlib.util.spec_from_file_location(name,path);self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
  self.module.torch=TorchHost(self)
  for name in ('cross_entropy_fwd_kernel','cross_entropy_bwd_kernel'):
   setattr(self.module,name,Kernel(self,getattr(self.module,name),name))
def reference(x,y,g,s,a,z,ignored,total,start,has):
 u=x.double()*a;l=torch.logsumexp(u,dim=1);h=s if has else 0.;shift=y.long()-start;owner=(shift>=0)&(shift<x.shape[1]);safe=shift.clamp(0,x.shape[1]-1)
 t=torch.where(owner,u.gather(1,safe[:,None]).flatten(),0.)
 loss=l-(1-h)*t-h*u.sum(1)/total+z*l*l;zz=z*l*l
 grad=(1+2*z*l[:,None])*torch.softmax(u,dim=1)-h/total
 grad.scatter_add_(1,safe[:,None],(-(1-h)*owner.double())[:,None]);grad=(g.double()*a)[:,None]*grad
 selected=y.long()!=ignored
 return torch.where(selected,loss,0.),l,torch.where(selected,zz,0.),torch.where(selected[:,None],grad,0.)
def case(name,x,y,*,g=None,s=0.,a=1.,z=0.,ignored=-1,total=None,start=0,block=4,has=False,split=False,lse=None,exceptional=False):
 if g is None:g=torch.ones(x.shape[0])
 if total is None:total=start+x.shape[1]
 inputs=[x,y,g]+([] if lse is None else [lse]);before=[raw(t) for t in inputs]
 r=Runner(D/'cross_entropy2_repaired.py','candidate_'+name)
 f=r.module.cross_entropy_fwd(x,y,s,a,z,ignored,total,start,block,has,split)
 assert all(t.shape==(x.shape[0],) and t.dtype==torch.float32 for t in f)
 grad=r.module.cross_entropy_bwd(g,x,f[1] if lse is None else lse,y,s,a,z,ignored,total,start,block,has)
 assert grad is r.allocations[-1] and grad.shape==x.shape and grad.is_contiguous()
 if not exceptional:
  expected=reference(x,y,g,s,a,z,ignored,total,start,has)
  for got,want in zip((*f,grad),expected):torch.testing.assert_close(got.double(),want,rtol=8e-5,atol=8e-5)
 else:
  mask=y.long()==ignored
  for t in (f[0],f[2],grad):assert torch.equal(t[mask],torch.zeros_like(t[mask]))
 for t,b in zip(inputs,before):assert torch.equal(raw(t),b),'caller bytes changed'
 return dict(case=name,inputs=[info(t) for t in inputs],forward=[info(t) for t in f],gradient=info(grad),full_caller_backing_frame=True,reference_checked=not exceptional,ignored_literal_zero=exceptional,launches=r.launches)
def main():
 out=E/(sys.argv[1] if len(sys.argv)>1 else 'cpu-controls001');out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/'probe.py');shutil.copyfile(D/'cross_entropy2_repaired.py',out/'candidate.py')
 torch.set_num_threads(1);records=[]
 def run(*a,**k):
  v=case(*a,**k);records.append(v);(out/'partial.json').write_text(json.dumps(records,indent=2));print(v['case'],'passed',flush=True)
 dense=lambda:torch.tensor([[0.,1.,2.],[2.,0.,1.]])
 labels=lambda:torch.tensor([0,1],dtype=torch.int32)
 for a in (1.,0.,-2.):
  for has in (False,True):
   for split in (False,True):run(f'tail_a{a}_has{has}_split{split}',dense(),labels(),a=a,s=.2,z=.1,has=has,split=split)
 for block in (1,2,4,8):run('blocks_'+str(block),dense(),labels(),block=block,s=.1,z=.1,has=True)
 run('column_stride',torch.tensor([[0.,9.,0.,7.],[2.,8.,1.,6.]])[:,::2],labels(),block=2)
 run('row_gap_offset',torch.arange(35,dtype=torch.float32).reshape(5,7)[1:5:2,2:5],labels(),block=2)
 run('transpose',torch.arange(6,dtype=torch.float32).reshape(3,2).T,labels(),block=2)
 run('broadcast_logits',torch.tensor([[0.,1.,2.]]).expand(2,3),labels())
 run('label_stride',dense(),torch.tensor([0,2,1],dtype=torch.int32)[::2])
 run('broadcast_labels',dense(),torch.tensor([1],dtype=torch.int32).expand(2))
 run('negative_logits',torch._neg_view(dense()),labels(),a=-1.)
 run('negative_labels',dense(),torch._neg_view(torch.tensor([0,-1],dtype=torch.int32)))
 run('negative_upstream',dense(),labels(),g=torch._neg_view(torch.tensor([1.,2.])))
 run('stride_upstream',dense(),labels(),g=torch.tensor([1.,99.,2.])[::2])
 lse=torch.logsumexp(dense(),1)
 run('negative_lse',dense(),labels(),lse=torch._neg_view(-lse))
 run('stride_lse',dense(),labels(),lse=torch.stack((lse,torch.full_like(lse,99.)),1).flatten()[::2])
 run('proper_shard_missing_target',dense(),torch.tensor([0,8],dtype=torch.int32),total=10,start=2,has=True,s=.2,z=.1,block=2)
 run('extreme_int32_labels',dense(),torch.tensor([-2**31,2**31-1],dtype=torch.int32),total=2**63-1,start=2**63-4,has=True,s=.2,z=.1)
 run('ignored_nonfinite',torch.tensor([[float('nan'),float('inf'),-float('inf')],[1.,2.,3.]]),torch.tensor([-1,0],dtype=torch.int32),g=torch.tensor([float('nan'),1.]),split=True,exceptional=True)
 run('ignored_int32_min',dense(),torch.tensor([-2**31,1],dtype=torch.int32),ignored=-2**31,total=2**63-1,start=2**63-4,has=True,s=.2)
 run('ignored_index_above_int32',dense(),torch.tensor([0,1],dtype=torch.int32),ignored=2**32,split=True)
 run('ignored_index_below_int32',dense(),torch.tensor([0,1],dtype=torch.int32),ignored=-2**32,split=True)
 run('zero_rows',torch.empty(0,3),torch.empty(0,dtype=torch.int32),block=2,split=True)
 storage=torch.UntypedStorage(27);b=torch.empty(0,dtype=torch.uint8).set_(storage,0,(27,),(1,));b.fill_(173);x=torch.empty(0).set_(storage,0,(2,3),(3,1));x.copy_(dense());run('partial_backing_bytes',x,labels())
 x=dense();y=x.view(torch.int32).flatten()[::3];run('mixed_int32_fp32_alias',x,y)
 invalid=[]
 for name,change in [('logit_dtype',{'x':dense().half()}),('label_dtype',{'y':labels().long()}),('zero_classes',{'x':torch.empty(2,0)}),('block_nonpower',{'block':3}),('bad_classes',{'total':2}),('smoothing',{'s':2.})]:
  k=dict(x=dense(),y=labels(),s=0.,a=1.,z=0.,ignored=-1,total=3,start=0,block=4,has=False,split=False);k.update(change)
  r=Runner(D/'cross_entropy2_repaired.py','invalid_'+name)
  try:r.module.cross_entropy_fwd(*[k[t] for t in ['x','y','s','a','z','ignored','total','start','block','has','split']])
  except (ValueError,TypeError) as e:invalid.append(dict(case=name,error=str(e),launches=len(r.launches)));assert not r.launches
  else:raise AssertionError(name)
 original=E/'original/cross_entropy2.py';trim=original.read_text().split('\n\ndef test_cross_entropy_kernels():')[0];p=out/'original_importable.py';p.write_text(trim)
 defects=[]
 def old(name,x,y,a=1.,block=2,split=False):
  r=Runner(p,'original_'+name)
  with contextlib.redirect_stdout(io.StringIO()):
   f=r.module.cross_entropy_fwd(x,y,0.,a,0.,-1,x.shape[1],0,block,False,split)
   g=r.module.cross_entropy_bwd(torch.ones(x.shape[0]),x,f[1],y,0.,a,0.,-1,x.shape[1],0,block,False)
  defects.append(dict(case=name,loss=f[0].tolist(),lse=f[1].tolist(),z_loss=f[2].tolist(),gradient=g.tolist(),returned_shapes=[list(t.shape) for t in f],reference=[v.tolist() for v in reference(x,y,torch.ones(x.shape[0]),0.,a,0.,-1,x.shape[1],0,False)],diagnostic_nan_fill_not_original_gpu_values=True))
 old('two_blocks_lse_handoff',torch.zeros(1,4),torch.tensor([0],dtype=torch.int32))
 old('incomplete_returns',torch.zeros(2,2),torch.tensor([0,0],dtype=torch.int32))
 old('split_unwritten_z_loss',torch.zeros(2,2),torch.tensor([0,0],dtype=torch.int32),split=True)
 old('column_stride',torch.tensor([[0.,1.,0.,7.]])[:,::2],torch.tensor([0],dtype=torch.int32))
 old('label_stride',torch.zeros(2,2),torch.tensor([0,1,0],dtype=torch.int32)[::2])
 old('zero_scale_tail',torch.zeros(1,3),torch.tensor([0],dtype=torch.int32),a=0.,block=4)
 old('negative_scale_tail',torch.zeros(1,3),torch.tensor([0],dtype=torch.int32),a=-1.,block=4)
 result=dict(profile='actual Triton CPU source interpreter plus actual Torch CPU metadata and host operations; no GPU',torch=torch.__version__,triton=triton.__version__,cuda_available=torch.cuda.is_available(),candidate_sha256=sha(D/'cross_entropy2_repaired.py'),original_sha256=sha(original),positive=records,rejected=invalid,original_defects=defects)
 (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=True)+'\n');print('PASS',len(records),'positive',len(invalid),'rejections',len(defects),'original controls')
if __name__=='__main__':
 try:
  with warnings.catch_warnings():warnings.simplefilter('ignore');main()
 except BaseException:
  traceback.print_exc();raise
