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
SOURCE=HERE/'chunk_gla_simple_min_dot_tile.py'
OUTPUT=ROOT/'.veritile-cache/task-deliveries/chunk_gla_simple_min_dot_tile/native-admission'
NAME='chunk_simple_gla_fwd_kernel_o_repaired'
def pin(p):
 p=Path(p).resolve();return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def capture(tree,K,V,T=1,BT=16,B=1,H=1):
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
 NT=triton.cdiv(T,BT)
 h=Tensor(torch.ones((B,H,NT*K,V),dtype=torch.float32));q=Tensor(torch.ones((B,H,T,K),dtype=torch.float32));k=Tensor(torch.ones_like(q.data));v=Tensor(torch.ones((B,H,T,V),dtype=torch.float32));g=Tensor(torch.zeros((B,H,T),dtype=torch.float32))
 out=ns['chunk_fwd_o_fn_repaired'](h,q,k,v,g,BT,1.0)
 if B*H*T*V == 0:
  assert events==['allocate'] and not got
  return None,None,{'B':B,'H':H,'T':T,'K':K,'V':V,'BT':BT,'events':events,'wrapper_admitted':True,'empty_return':True}
 assert out is got['args'][5]
 assert events==['allocate','device-enter','launch','device-exit']
 args=[x.data if isinstance(x,Tensor) else x for x in got['args']]
 return args,got['kwargs'],{'B':B,'H':H,'T':T,'K':K,'V':V,'BT':BT,'grid':got['grid'],'scalar_arguments':list(got['args'][6:]),'events':events,'wrapper_admitted':True,'contents':'Q/K/V/history all +1; gate zero; scale 1.0','shapes':{'history':list(h.shape),'query':list(q.shape),'key':list(k.shape),'value':list(v.shape),'gate':list(g.shape)}}
def run():
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'tmp').mkdir(exist_ok=True)
 os.environ['TMPDIR']=str(OUTPUT/'tmp');os.environ['TRITON_CACHE_DIR']=str(OUTPUT/'cache');os.environ.pop('TRITON_INTERPRET',None)
 import torch,triton,triton.language as tl
 from triton.runtime.jit import create_function_from_signature
 from triton.backends.compiler import GPUTarget
 from triton.backends.nvidia.compiler import CUDABackend
 from triton.compiler import ASTSource
 from triton.compiler.errors import CompilationError
 tree=ast.parse(SOURCE.read_text());node=copy.deepcopy(function(tree,NAME));configs_expr=next(k.value for k in node.decorator_list[0].keywords if k.arg=='configs');node.decorator_list=[node.decorator_list[-1]]
 module=types.ModuleType('chunk_simple_min_dot_tile_admission_probe');module.__file__=str(SOURCE);module.__dict__.update(torch=torch,triton=triton,tl=tl);sys.modules[module.__name__]=module
 exec(compile(ast.Module([node],[]),str(SOURCE),'exec'),module.__dict__);kernel=getattr(module,NAME)
 original=ast.parse(kernel.src).body[0];assert ast.dump(original.args)==ast.dump(node.args);assert [ast.dump(n) for n in original.body]==[ast.dump(n) for n in node.body]
 configs=eval(compile(ast.Expression(configs_expr),str(SOURCE),'eval'),module.__dict__);assert len(configs)==1;cfg=configs[0];assert cfg.num_warps==4
 target=GPUTarget('cuda',80,32);backend=CUDABackend(target);records=[]
 binder=create_function_from_signature(kernel.signature,kernel.params,backend)
 cases=[(1,1,1,16,1,1),(1,16,1,16,1,1),(8,16,1,16,1,1),(9,1,1,16,1,1),(16,1,1,16,1,1),(0,16,1,16,1,1),(16,16,1,16,1,1),(129,17,17,16,1,1)]
 cases += [(1,1,1,BT,1,1) for BT in (32,64,128,256,512,1024)]
 cases += [(0,1,17,16,1,1),(7,65,33,32,2,3),(65,1,17,16,1,1)]
 cases += [(1,1,1,16,0,1),(1,1,1,16,1,0),(1,1,0,16,1,1),(1,0,1,16,1,1)]
 for K,V,T,BT,B,H in cases:
  values,keywords,admitted=capture(tree,K,V,T,BT,B,H)
  label=f'B{B}-H{H}-T{T}-K{K}-V{V}-BT{BT}'
  if values is None:
   records.append({'admitted':admitted,'status':'empty_return_no_launch'});continue
  options={'num_warps':cfg.num_warps,'num_stages':cfg.num_stages,'num_ctas':cfg.num_ctas}
  args,special,bound_options=binder(*values,**keywords,**cfg.kwargs,**options)
  assert list(args)==kernel.arg_names and bound_options==options
  _,signature,constants,attrs=kernel._pack_args(backend,options,args,special,None)
  record={'admitted':admitted,'signature':signature,'constants':{kernel.arg_names[k[0]]:v for k,v in constants.items()},'attrs':{kernel.arg_names[k[0]]:v for k,v in attrs.items()},'options':options}
  try:
   compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants,attrs=attrs),target=target,options=options)
  except Exception as err:
   p=OUTPUT/(label+'.error.txt');p.write_text(traceback.format_exc());record.update(status=type(err).__name__,error=str(err),trace=pin(p))
  else:
   artifacts=[]
   for ext in ('ttir','ttgir','llir','ptx','cubin'):
    p=OUTPUT/(label+'.'+ext);(p.write_bytes(compiled.asm[ext]) if isinstance(compiled.asm[ext],bytes) else p.write_text(compiled.asm[ext]));artifacts.append(pin(p))
   record.update(status='compile_success',artifacts=artifacts,metadata={k:(str(v) if k=="target" else v) for k,v in compiled.metadata._asdict().items()})
  records.append(record);(OUTPUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print(label,record['status'],record.get('error','')[-500:],flush=True)
 (OUTPUT/'result.json').write_text(json.dumps({'source':pin(SOURCE),'checker':pin(__file__),'triton_version':triton.__version__,'target':'cuda sm80 warp32','cases':records,'GPU_execution':False,'boundary':'Actual assertion/wrapper AST and generated native binder with signature-order packing; exact JIT signature/body with only host autotune decorator omitted. CUDA ownership/allocation/submission mocked; no GPU execution.'},indent=2)+'\n')
 print('receipt',pin(OUTPUT/'result.json'),flush=True)
if __name__=='__main__':run()
