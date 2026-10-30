from pathlib import Path
import ast,contextlib,hashlib,importlib.util,json,os,sys,tempfile
sys.dont_write_bytecode=True
E=Path(tempfile.mkdtemp(prefix='veritile-int4-control-'));(E/'evidence').mkdir(exist_ok=True);os.environ['TRITON_INTERPRET']='1';os.environ['TRITON_CACHE_DIR']=str(E/'evidence/cpu-cache')
import torch
p=Path(__file__).resolve().with_name('matmul_dequantize_int4_repaired.py')
original=p.parents[4]/'bench/tritonbench_g/matmul_dequantize_int4/matmul_dequantize_int4.py'
tree=ast.parse(original.read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test')]
(E/'original.py').write_text(ast.unparse(tree))
current_tree=ast.parse(p.read_text())
source_kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul4_kernel')
current_kernel=next(n for n in current_tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul4_kernel')
assert ast.dump(source_kernel)==ast.dump(current_kernel), 'JIT source or configurations changed'
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load(p,'mdqi4_cpu');old=load(E/'original.py','mdqi4_original_cpu')
real_cuda=torch.Tensor.cuda;real_empty=torch.empty;torch.Tensor.cuda=lambda t,*a,**k:t

def raw(t):
 s=t.untyped_storage();return torch.empty(0,dtype=torch.uint8).set_(s,0,(s.nbytes(),),(1,))
def view(r,c,dtype,step=1):
 sr=max(1,c)*step+3;sc=step;base=torch.full((max(1,(r-1)*sr+(c-1)*sc+1)+13,),7,dtype=dtype);return base.as_strided((r,c),(sr,sc))
quant=[];gemm=[]
try:
 # Independent exact-integer weight examples, all groups span [0,15].
 for K,N,g in [(256,256,128),(32,16,8),(16,8,4)]:
  w=(torch.arange(K*N).reshape(K,N)%16).float()
  for n in range(N):
   for start in range(0,K,g):
    w[start:start+g,n]=torch.arange(g)%16
    w[start,n]=0;w[start+g-1,n]=15
  before=raw(w).clone();q,s,z,returned=m.quantize_int4(w,g)
  assert returned==g and q.dtype==z.dtype==torch.int32 and s.dtype==torch.float32
  expected=torch.zeros(K//8,N,dtype=torch.int64)
  for lane in range(8):expected|=w[lane::8].to(torch.int64)<<(4*lane)
  assert torch.equal(q,expected.to(torch.int32));assert torch.equal(s,torch.ones(K//g,N));assert torch.equal(z,torch.zeros(K//g,N//8,dtype=torch.int32))
  assert torch.equal(raw(w),before)
  quant.append({'shape':[K,N],'group_size':g,'status':'PASS','all_packed_words_and_scales_and_zeros':True,'input_frame':True})
 # Actual original helper with a deterministic allowed fresh seed. Interpose
 # only torch.empty used by its packed temporary; every increment at [0,0] is 0.
 w=torch.zeros(256,256);w[120::128,:]=15
 def poisoned(*a,**k):
  t=real_empty(*a,**k)
  if tuple(t.shape)==(256,32):t.fill_(1)
  return t
 torch.empty=poisoned
 bad=old.quantize_int4(w,128)[0];good=m.quantize_int4(w,128)[0]
 torch.empty=real_empty
 assert bad[0,0].item()==1 and good[0,0].item()==0
 witness={'case':'actual_original_seed_one','original_word00':1,'candidate_word00':0,'independent_target_word00':0}
 class CudaView(torch.Tensor):
  @property
  def is_cuda(self):return True
  @property
  def device(self):return torch.device('cuda',0)
 def host_empty(*args,**kwargs):
  if str(kwargs.get('device','cpu')).startswith('cuda'):
   kwargs['device']='cpu';return real_empty(*args,**kwargs).as_subclass(CudaView)
  return real_empty(*args,**kwargs)
 torch.empty=host_empty
 tuner=m.matmul4_kernel;kernel=tuner.fn
 class Selected:
  def __init__(self,config):self.config=config
  def __getitem__(self,grid):
   def invoke(*args,**kwargs):
    meta=self.config.kwargs
    dims=grid(meta) if callable(grid) else grid
    return kernel[dims](*args,**kwargs,**meta,num_warps=self.config.num_warps,num_stages=self.config.num_stages)
   return invoke
 torch.manual_seed(38)
 for ci,config in enumerate(tuner.configs):
  for nogroup in [False,True]:
   M,K,N=3,256,256;g=K if nogroup else 128;G=K//g
   a=torch.randint(-2,3,(M,K)).half();q=view(K//8,N,torch.int32,2);z=view(G,N//8,torch.int32,2);s=view(G,N,torch.float16,2)
   q.copy_(torch.randint(-(2**31),2**31-1,q.shape,dtype=torch.int32));z.copy_(torch.randint(-(2**31),2**31-1,z.shape,dtype=torch.int32));s.fill_(0.125)
   supplied=(ci%2)==0;c=view(M,N,torch.float16,2) if supplied else None
   a,q,s,z=[v.as_subclass(CudaView) for v in [a,q,s,z]]
   if c is not None:c=c.as_subclass(CudaView)
   inputs=[a,q,s,z]+([c] if c is not None else []);saved=[raw(x).clone() for x in inputs]
   qi=((q[(torch.arange(K)//8),:].long() >> ((torch.arange(K)%8)*4)[:,None])&15)
   zi=((z[(torch.arange(K)//g),:][:,torch.arange(N)//8].long() >> ((torch.arange(N)%8)*4)[None,:])&15)
   dequant=(qi-zi).double()*s[(torch.arange(K)//g),:].double()
   expected=(a.double()@dequant).half()
   m.matmul4_kernel=Selected(config)
   result=m.matmul_dequantize_int4_gptq(a,q,s,z,g,output=c)
   assert (result is None)==supplied
   out=c if supplied else result
   assert out.dtype==torch.float16 and out.shape==(M,N) and torch.equal(out,expected)
   for x,before in zip(inputs[:4],saved[:4]):assert torch.equal(raw(x),before)
   if supplied:
    mask=torch.zeros(raw(c).numel(),dtype=torch.bool)
    for r in range(M):
     for n in range(N):
      off=2*(c.storage_offset()+r*c.stride(0)+n*c.stride(1));mask[off:off+2]=True
    assert torch.equal(raw(c)[~mask],saved[4][~mask])
   gemm.append({'config_index':ci,'NO_GROUPS':nogroup,'supplied_output':supplied,'status':'PASS','all_outputs_and_caller_frame':True});print('GEMM',ci,nogroup,'PASS',flush=True)
 m.matmul4_kernel=tuner
 admitted=0;rejected=0
 def reject(fn):
  global rejected
  try:fn()
  except (TypeError,ValueError):rejected+=1
  else:raise AssertionError('unsupported argument admitted')
 def data():
  return [torch.ones(2,128,dtype=torch.float16).as_subclass(CudaView),
          torch.zeros(16,256,dtype=torch.int32).as_subclass(CudaView),
          torch.ones(1,256,dtype=torch.float16).as_subclass(CudaView),
          torch.zeros(1,32,dtype=torch.int32).as_subclass(CudaView)]
 a,q,s,z=data();m._check_matmul(a,q,s,z,128);admitted+=1
 for bad in (0,-1,True,1.0,64,256):reject(lambda bad=bad:m._check_matmul(a,q,s,z,bad))
 reject(lambda:m._check_matmul(a.float(),q,s,z,128))
 reject(lambda:m._check_matmul(a,q.long(),s,z,128))
 reject(lambda:m._check_matmul(a,q,s.float(),z,128))
 reject(lambda:m._check_matmul(a,q,s,z.long(),128))
 reject(lambda:m._check_matmul(a,q[:,:128],s,z,128))
 reject(lambda:m._check_matmul(a,q,s[:,:128],z,128))
 reject(lambda:m._check_matmul(a,q,s,z[:,:16],128))
 reject(lambda:m._check_matmul(a,q,s,z,128,torch.empty((2,128),dtype=torch.float16,device='cuda')))
 reject(lambda:m._check_matmul(a,q,s,z,128,torch.empty((2,256),dtype=torch.float32,device='cuda')))
 reject(lambda:m._check_matmul(a,q,s,z,128,s.expand(2,256)))
 reject(lambda:m._check_matmul(a,q,s,z,128,q.view(torch.float16)[:2,:256]))
 reject(lambda:m._check_matmul(a[:,::2],q,s,z,128))
 class Negative(CudaView):
  def is_neg(self):return True
 reject(lambda:m._check_matmul(a.as_subclass(Negative),q,s,z,128))
 class Conjugate(CudaView):
  def is_conj(self):return True
 reject(lambda:m._check_matmul(a.as_subclass(Conjugate),q,s,z,128))
 outbase=torch.full((1100,),7,dtype=torch.float16).as_subclass(CudaView)
 out=outbase.as_strided((2,256),(520,2),7)
 m._check_matmul(a,q,s,z,128,out);admitted+=1
 st=view(1,256,torch.float16,2).as_subclass(CudaView)
 m._check_matmul(a,q,st,z,128,out);admitted+=1
 for bad in (0,-8,True,2.0,3):reject(lambda bad=bad:m.quantize_int4(torch.ones(16,8),bad))
 for rank in (-1,True,0.5):reject(lambda rank=rank:m.quantize_int4(torch.ones(16,8),8,rank))
 reject(lambda:m.quantize_int4(torch.ones(16,8).half(),8))
 reject(lambda:m.quantize_int4(torch.ones(16,7),8))
 reject(lambda:m.quantize_int4(torch.zeros(16,8),8))
 reject(lambda:m.quantize_int4(torch.full((16,8),float('nan')),8))
 reject(lambda:m.quantize_int4(torch.full((16,8),float('inf')),8))
 reject(lambda:m.quantize_int4(torch.full((16,8),torch.finfo(torch.float32).smallest_normal*2**-23),8))
 overflow=torch.empty((16,8));overflow[::2]=3e38;overflow[1::2]=-3e38
 reject(lambda:m.quantize_int4(overflow,8))
 wbase=torch.zeros(400);w=wbase.as_strided((16,8),(20,2),5)
 w.copy_(torch.arange(16).view(16,1).expand(16,8));before=raw(w).clone()
 qout,sout,zout,group=m.quantize_int4(w,8);assert torch.equal(raw(w),before);admitted+=1
 print('GUARDS',admitted,'admissions',rejected,'rejections',flush=True)
 (E/'evidence/cpu-controls.json').write_text(json.dumps({'status':'PASS','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'quantizer_cases':quant,'original_counterexample':witness,'gemm_cases':gemm,'engine':'actual Torch CPU; quantizer CUDA transfer shim; actual Triton interpreter kernel and wrapper with exact explicit candidate chosen instead of autotune benchmarking','autotune_all_17_candidates_checked':True,'guard_admissions':admitted,'guard_rejections':rejected,'JIT_AST_equal_original':True,'GPU_execution':False,'native_numerical_conformance':False},indent=2)+'\n')
 print('Evidence:',E/'evidence/cpu-controls.json',flush=True)
finally:
 torch.Tensor.cuda=real_cuda;torch.empty=real_empty
