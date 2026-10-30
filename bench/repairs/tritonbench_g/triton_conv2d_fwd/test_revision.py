"""Bounded actual-source CPU interpreter controls, not native GPU conformance."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,itertools
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent
os.environ['TRITON_INTERPRET']='1';os.environ['TRITON_CACHE_DIR']='/tmp/conv2d-cpu-cache'
import torch
source=E/'triton_conv2d_fwd_repaired.py'
spec=importlib.util.spec_from_file_location('conv_cpu',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
old_cuda=torch.Tensor.is_cuda;torch.Tensor.is_cuda=property(lambda _:True)
old_device=torch.cuda.current_device;torch.cuda.current_device=lambda:None
wrapper=mod.conv2d_forward_kernel
class Launch:
 def __init__(self):self.rows=[]
 def __getitem__(self,grid):
  def run(*args):
   self.rows.append({'grid':grid,'input_stride':args[10:14],'weight_stride':args[14:18],'output_stride':args[18:22],'fp16':args[29],'tf32':args[30]})
   return wrapper[grid](*args)
  return run

def bytes(t):
 st=t.untyped_storage();return torch.empty(0,dtype=torch.uint8).set_(st,0,(st.nbytes(),),(1,)).clone()
rows=[]
try:
 torch.manual_seed(52)
 for name,N,Cg,Og,H,W,KH,KW,SH,SW,PH,PW,G,strided in [
  ('tail1',1,1,1,1,1,1,1,1,1,0,0,1,False),
  ('padding32',1,32,2,1,1,3,3,1,1,1,1,1,False),
  ('group-tail',2,33,35,3,4,2,2,1,2,1,1,2,False),
  ('strided',1,3,5,4,5,3,2,2,1,1,1,1,True),
  ('spatial-tail',1,2,3,12,12,1,1,1,1,0,0,1,False),
  ('offset',1,3,5,4,5,3,2,2,1,1,1,1,False),
  ('alias',1,1,1,1,1,1,1,1,1,0,0,1,False),
  ('empty-batch',0,3,5,4,5,3,2,2,1,1,1,1,False)]:
  for half,tf in itertools.product([False,True],repeat=2):
   a=torch.randint(-2,3,(N,G*Cg,H,W),dtype=torch.float32)
   b=torch.randint(-2,3,(G*Og,Cg,KH,KW),dtype=torch.float32)
   if strided:
    a=a.transpose(2,3).contiguous().transpose(2,3);b=b.transpose(2,3).contiguous().transpose(2,3)
   if name=='offset':
    aa=torch.empty(a.numel()+7);aa[7:]=a.flatten();a=aa[7:].view(a.shape)
    bb=torch.empty(b.numel()+3);bb[3:]=b.flatten();b=bb[3:].view(b.shape)
   if name=='alias':b=a
   aa,bb=bytes(a),bytes(b)
   selected=Launch();mod.conv2d_forward_kernel=selected
   out=mod.conv2d_forward(a,b,KH,KW,SH,SW,PH,PW,G,half,tf)
   expected=torch.nn.functional.conv2d(a,b,stride=(SH,SW),padding=(PH,PW),groups=G)
   assert torch.equal(out,expected),(name,half,tf,(out-expected).abs().max())
   assert torch.equal(aa,bytes(a)) and torch.equal(bb,bytes(b))
   assert out.dtype==torch.float32 and out.device==a.device and out.shape==expected.shape
   assert out is not a and out is not b
   if out.numel():assert out.untyped_storage().data_ptr() not in {a.untyped_storage().data_ptr(),b.untyped_storage().data_ptr()}
   assert len(selected.rows)==int(N>0)
   rows.append({'name':name,'fp16':half,'tf32':tf,'shape':list(out.shape),'launches':selected.rows,'exact_small_integer_match':True,'caller_bytes_preserved':True,'fresh_output':True})
   print(name,half,tf,'PASS',flush=True)
 # The host validator is part of the selected supported domain.
 a=torch.ones(1,2,2,2);b=torch.ones(2,2,1,1)
 reject=[]
 for name,aa,bb,args in [
  ('lazy-neg',torch._neg_view(a),b,(1,1,1,1,0,0,1,False,False)),
  ('wrong-weight',a,b,(2,1,1,1,0,0,1,False,False)),
  ('nonbool',a,b,(1,1,1,1,0,0,1,1,False)),
  ('zero-stride',a.expand(2,-1,-1,-1),b,(1,1,1,1,0,0,1,False,False)),
  ('overflow',a,b,(1,1,2**31,1,0,0,1,False,False)),
  ('dtype',a.double(),b,(1,1,1,1,0,0,1,False,False)),
  ('rank',a.flatten(),b,(1,1,1,1,0,0,1,False,False)),
  ('noninteger',a,b,(True,1,1,1,0,0,1,False,False)),
  ('negative-padding',a,b,(1,1,1,1,-1,0,1,False,False)),
  ('nondivisible',a,b,(1,1,1,1,0,0,3,False,False)),
  ('zero-stride-argument',a,b,(1,1,0,1,0,0,1,False,False)),
  ('grid-z',torch.ones(1,65536,1,1),torch.ones(65536,1,1,1),(1,1,1,1,0,0,65536,False,False)),
  ('grid-y',torch.ones(1,1,1,1),torch.ones(32*65535+1,1,1,1),(1,1,1,1,0,0,1,False,False))]:
  try:mod.conv2d_forward(aa,bb,*args)
  except (ValueError,TypeError) as e:reject.append({'case':name,'error':str(e)})
  else:raise AssertionError(('accepted invalid',name))
 report={'status':'PASS','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'engine':'Actual source wrapper and Triton CPU interpreter with CUDA residency and current-device observation shims only; actual allocations, strides, flags, default launch and masks','GPU_execution':False,'native_numerical_conformance':False,'cases':rows,'rejected_domain_cases':reject}
 (E/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
finally:
 torch.Tensor.is_cuda=old_cuda
 torch.cuda.current_device=old_device
