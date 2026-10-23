"""Actual checked wrapper/binder small-dot compilation controls; no GPU run."""
import ast
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
import types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=HERE/'chunk_gla_simple_repaired.py'
OUTPUT=ROOT/'.veritile-cache/task-deliveries/chunk_gla_simple/native-admission'
NAME='chunk_simple_gla_fwd_kernel_o_repaired'
def pin(p):
 p=Path(p).resolve();return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def capture(tree,K,V,T=1):
 import torch,triton
 class Tensor:
  def __init__(self,data):self.data=data;self.device=torch.device('cuda:0');self.is_cuda=True
  def __getattr__(self,n):return getattr(self.data,n)
 events=[];got={}
 @contextlib.contextmanager
 def device(owner):
  assert owner==torch.device('cuda:0');events.append('device-enter')
  try:yield
  finally:events.append('device-exit')
 def empty_like(v):events.append('allocate');return Tensor(torch.empty_like(v.data))
 class Launch:
  def __getitem__(self,grid):
   def call(*args,**kwargs):events.append('launch');got.update(grid=grid,args=args,kwargs=kwargs)
   return call
 fake=types.SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,empty_like=empty_like,cuda=types.SimpleNamespace(device=device))
 ns={'torch':fake,'triton':triton,'math':math,NAME:Launch()}
 nodes=[function(tree,n) for n in ['_check_addresses','chunk_fwd_o_fn_repaired']]
 exec(compile(ast.Module(nodes,[]),str(SOURCE),'exec'),ns)
 BT=16;NT=triton.cdiv(T,BT)
 h=Tensor(torch.ones((1,1,NT*K,V),dtype=torch.float32));q=Tensor(torch.ones((1,1,T,K),dtype=torch.float32));k=Tensor(torch.ones_like(q.data));v=Tensor(torch.ones((1,1,T,V),dtype=torch.float32));g=Tensor(torch.zeros((1,1,T),dtype=torch.float32))
 out=ns['chunk_fwd_o_fn_repaired'](h,q,k,v,g,BT,1.0)
 assert out is got['args'][5]
 assert events==['allocate','device-enter','launch','device-exit']
 args=[x.data if isinstance(x,Tensor) else x for x in got['args']]
 return args,got['kwargs'],{'B':1,'H':1,'T':T,'K':K,'V':V,'BT':BT,'grid':got['grid'],'scalar_arguments':list(got['args'][6:]),'events':events,'wrapper_admitted':True,'contents':'Q/K/V/history all +1; gate zero; scale 1.0','shapes':{'history':list(h.shape),'query':list(q.shape),'key':list(k.shape),'value':list(v.shape),'gate':list(g.shape)}}
def run():
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'tmp').mkdir(exist_ok=True)
 os.environ['TMPDIR']=str(OUTPUT/'tmp');os.environ['TRITON_CACHE_DIR']=str(OUTPUT/'cache');os.environ.pop('TRITON_INTERPRET',None)
 import torch,triton,triton.language as tl
 from triton._C.libtriton import native_specialize_impl
 from triton.backends.compiler import GPUTarget
 from triton.backends.nvidia.compiler import CUDABackend
 from triton.compiler import ASTSource
 from triton.compiler.errors import CompilationError
 tree=ast.parse(SOURCE.read_text());node=copy.deepcopy(function(tree,NAME));configs_expr=next(k.value for k in node.decorator_list[0].keywords if k.arg=='configs');node.decorator_list=[node.decorator_list[-1]]
 module=types.ModuleType('chunk_simple_admission_probe');module.__file__=str(SOURCE);module.__dict__.update(torch=torch,triton=triton,tl=tl);sys.modules[module.__name__]=module
 exec(compile(ast.Module([node],[]),str(SOURCE),'exec'),module.__dict__);kernel=getattr(module,NAME)
 original=ast.parse(kernel.src).body[0];assert ast.dump(original.args)==ast.dump(node.args);assert [ast.dump(n) for n in original.body]==[ast.dump(n) for n in node.body]
 configs=eval(compile(ast.Expression(configs_expr),str(SOURCE),'eval'),module.__dict__);assert len(configs)==1;cfg=configs[0];assert cfg.num_warps==4
 target=GPUTarget('cuda',80,32);backend=CUDABackend(target);records=[]
 for K,V,T in [(1,1,1),(1,16,1),(8,16,1),(9,1,1),(16,1,1),(0,16,1),(16,16,1),(129,17,17)]:
  values,keywords,admitted=capture(tree,K,V,T);args=dict(zip(kernel.arg_names,values));args.update(keywords,**cfg.kwargs)
  special=[('constexpr',args[p.name]) if p.is_constexpr else native_specialize_impl(backend,args[p.name],p.is_const,not p.do_not_specialize,not p.do_not_specialize_on_alignment) for p in kernel.params]
  options={'num_warps':cfg.num_warps,'num_stages':cfg.num_stages,'num_ctas':cfg.num_ctas}
  _,signature,constants,attrs=kernel._pack_args(backend,options,args,special,None)
  label=f'T{T}-K{K}-V{V}';record={'admitted':admitted,'signature':signature,'constants':{kernel.arg_names[k[0]]:v for k,v in constants.items()},'attrs':{kernel.arg_names[k[0]]:v for k,v in attrs.items()},'options':options}
  try:
   compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants,attrs=attrs),target=target,options=options)
  except CompilationError as err:
   p=OUTPUT/(label+'.error.txt');p.write_text(traceback.format_exc());record.update(status='CompilationError',error=str(err),trace=pin(p))
  else:
   artifacts=[]
   for ext in ('ttir','ttgir','llir','ptx'):
    p=OUTPUT/(label+'.'+ext);p.write_text(compiled.asm[ext]);artifacts.append(pin(p))
   record.update(status='compile_success',artifacts=artifacts)
  records.append(record);print(label,record['status'],record.get('error','')[-500:],flush=True)
 (OUTPUT/'result.json').write_text(json.dumps({'source':pin(SOURCE),'checker':pin(__file__),'triton_version':triton.__version__,'target':'cuda sm80 warp32','cases':records,'GPU_execution':False,'boundary':'Actual assertion/wrapper AST and native ordinary scalar specialization; exact JIT signature/body with only host autotune decorator omitted. CUDA ownership/allocation/submission mocked; no GPU execution.'},indent=2)+'\n')
 print('receipt',pin(OUTPUT/'result.json'),flush=True)
if __name__=='__main__':run()
