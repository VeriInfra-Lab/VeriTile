"""Execute exact original host/heuristic AST with Torch metadata and launch spies.
No GPU, kernel replacement value oracle, or numerical equality claim is made.
"""
from pathlib import Path
import ast,hashlib,itertools,json,logging,sys,types
sys.dont_write_bytecode=True
import torch,triton
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path(__file__).resolve().parent/'evidence'
SOURCE=ROOT/'bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py'
source=SOURCE.read_text();tree=ast.parse(source)
heur_names=['heur_tile_k','heur_tile_n_non_inner','heur_one_tile_per_cta','heur_num_warps_non_inner','heur_tile_n_inner','heur_num_warps_inner','heur_tile_n_bwd_non_inner','heru_tile_m']
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in heur_names+['softmax'] or isinstance(n,ast.ClassDef) and n.name=='Softmax']
module=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));ast_hash=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()
choices=[32,64,128,256,512,1024]
class Context:
 def save_for_backward(self,*tensors):self.saved_tensors=tensors
class Facade:
 def __init__(self,t,trace,tag):self.tensor=t;self.trace=trace;self.tag=tag;self.device=torch.device('cuda',0)
 @property
 def ndim(self):return self.tensor.ndim
 @property
 def shape(self):return self.tensor.shape
 @property
 def dtype(self):return self.tensor.dtype
 def numel(self):return self.tensor.numel()
 def contiguous(self):
  t=self.tensor.contiguous();self.trace.append({'op':'contiguous','tag':self.tag,'old_strides':list(self.tensor.stride()),'new_strides':list(t.stride())})
  return Facade(t,self.trace,self.tag+'.contiguous')
class Device:
 def __init__(self,device,trace):self.device=device;self.trace=trace
 def __enter__(self):self.trace.append({'op':'device_enter','device':str(self.device)})
 def __exit__(self,*exc):self.trace.append({'op':'device_exit'});return False

def meta_tensor(shape,noncontiguous=True):
 if noncontiguous and len(shape)>1:
  return torch.empty(tuple(reversed(shape)),device='meta').permute(tuple(reversed(range(len(shape)))))
 return torch.empty(shape,device='meta')

def run(shape,dim,indtype,outdtype,choice,sms=80):
 trace=[];launches=[];env={}
 class Function:
  @classmethod
  def apply(cls,*args):
   cls.last_ctx=Context();return cls.forward(cls.last_ctx,*args)
 class TorchProxy:
  autograd=types.SimpleNamespace(Function=Function)
  cuda=types.SimpleNamespace(device=lambda d:Device(d,trace))
  def empty_like(self,inp,dtype=None):
   trace.append({'op':'empty_like','shape':list(inp.shape),'dtype':str(dtype or inp.dtype)})
   return Facade(torch.empty_like(inp.tensor,dtype=dtype),trace,'allocation_'+str(len(trace)))
 class Kernel:
  def __init__(self,name):self.name=name
  def __getitem__(self,grid):
   def launch(*args):
    back='backward' in self.name;noninner='non_inner' in self.name
    if back:P,G,Y,M,N,*tail=args;K=tail[0] if tail else 1
    else:Y,X,M,N,*tail=args;K=tail[0] if tail else 1
    d={'M':M,'N':N,'K':K}
    if noninner:
     d['TILE_K']=choice if back else env['heur_tile_k'](d)
     d['TILE_N']=env['heur_tile_n_bwd_non_inner' if back else 'heur_tile_n_non_inner'](d)
    else:
     d['TILE_N']=choice if back else env['heur_tile_n_inner'](d)
     if back:d['TILE_M']=env['heru_tile_m'](d)
    d['ONE_TILE_PER_CTA']=env['heur_one_tile_per_cta'](d)
    d['num_warps']=4 if back else env['heur_num_warps_non_inner' if noninner else 'heur_num_warps_inner'](d)
    actual_grid=tuple(grid(d)) if callable(grid) else grid
    expected=(M,triton.cdiv(K,d['TILE_K']),1) if noninner else ((triton.cdiv(M,d['TILE_M']),1,1) if back else (M,1,1))
    assert actual_grid==expected
    assert d['ONE_TILE_PER_CTA']==(d['TILE_N']>=N)
    assert d['TILE_N']>0 and d['TILE_N']&(d['TILE_N']-1)==0
    if noninner:assert d['TILE_K']>0 and d['TILE_K']&(d['TILE_K']-1)==0
    launched_count=M*N*K if noninner else M*N
    row={'kernel':self.name,'meta':d,'grid':list(actual_grid),'output_extent':Y.numel(),'active_address_count':launched_count,'safe_extent':launched_count<=Y.numel(),'source_storage_dtype':str(Y.dtype)}
    launches.append(row);trace.append({'op':'launch','kernel':self.name})
   return launch
 env.update(torch=TorchProxy(),triton=triton,logging=logging,MAX_TILE_K=8192,NUM_SMS=sms)
 for name in ['softmax_kernel_inner','softmax_kernel_non_inner','softmax_backward_kernel_inner','softmax_backward_kernel_non_inner']:env[name]=Kernel(name)
 exec(compile(module,str(SOURCE),'exec'),env)
 x=Facade(meta_tensor(shape).to(dtype=indtype),trace,'input')
 result={'shape':list(shape),'dim':dim,'input_dtype':str(indtype),'requested_dtype':str(outdtype) if outdtype else None,'autotune_choice':choice,'NUM_SMS':sms}
 try:
  out=env['softmax'](x,dim,outdtype)
  ctx=env['Softmax'].last_ctx
  assert ctx.saved_tensors==(out,) and ctx.dim==dim%len(shape)
  assert tuple(out.shape)==tuple(shape) and out.dtype==(outdtype or indtype)
  grad=Facade(meta_tensor(shape).to(dtype=out.dtype),trace,'output_gradient')
  in_grad,no_dim,no_dtype=env['Softmax'].backward(ctx,grad)
  assert no_dim is None and no_dtype is None
  assert tuple(in_grad.shape)==tuple(shape) and in_grad.dtype==out.dtype
  result.update(status='host_replayed',saved_output_identity=True,saved_dim=ctx.dim,return_gradient_slots=[str(in_grad.dtype),None,None])
 except Exception as exc:result.update(status='host_error',exception_type=type(exc).__name__,message=str(exc))
 result.update(launches=launches,trace=trace)
 return result
storage=[torch.float16,torch.bfloat16,torch.float32,torch.float64]
shapes=[((10,),-1),((4,5),1),((4,5),0),((2,3,4),1),((2,65537),1),((2,8193,17),1),((130,7),1),((2,2049,33),1)]
rows=[run(shape,dim,i,o,c) for shape,dim in shapes for i in storage for o in storage+[None] for c in choices]
assert len(rows)==960 and all(r['status']=='host_replayed' and len(r['launches'])==2 and all(l['safe_extent'] for l in r['launches']) for r in rows)
boundaries=[run(shape,dim,torch.float32,None,32) for shape,dim in [((2,9,0),1),((2,0),0),((0,3),1),((2,0),1),((),-1),((2,3),2),((2,3),-3)]]
assert boundaries[0]['status']==boundaries[1]['status']=='host_replayed'
assert all(not l['safe_extent'] for r in boundaries[:2] for l in r['launches'])
assert all(r['status']=='host_error' and not r['launches'] for r in boundaries[2:])
heur_env={'triton':triton,'MAX_TILE_K':8192,'NUM_SMS':80}
heur_mod=ast.fix_missing_locations(ast.Module(body=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name in heur_names],type_ignores=[]))
exec(compile(heur_mod,str(SOURCE),'exec'),heur_env)
heur=[]
for sms,M,K in itertools.product([1,8,80,132],[0,1,2,79,80,81,1024,65536],[0,1,2,3,31,32,33,8191,8192,8193,65537]):
 heur_env['NUM_SMS']=sms;C=heur_env['heur_tile_k']({'M':M,'K':K});B=triton.cdiv(8192,C)
 assert C>0 and C<=8192 and C&(C-1)==0 and B*C==8192
 heur.append({'NUM_SMS':sms,'M':M,'K':K,'TILE_K':C,'TILE_N':B})
(OUT/'host-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'host-boundaries.json').write_text(json.dumps(boundaries,indent=2)+'\n')
(OUT/'heuristic-cases.json').write_text(json.dumps(heur,indent=2)+'\n')
receipt={'status':'passed','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'selected_host_ast_sha256':ast_hash,'host_ast_unmodified':True,'torch_version':torch.__version__,'triton_version':triton.__version__,'host_cases':len(rows),'heuristic_cases':len(heur),'boundary_cases':len(boundaries),'dtype_pairs':20,'autotune_choices':choices,'actual_kernels_executed':False,'gpu_executed':False,'metadata_implementation':'Torch meta tensors under CUDA-labelled facades; original host AST and all heuristic functions execute unchanged.','zero_tail_source_bug':'Positive M,N with K=0 enters inner kernels; active M*N addresses exceed zero storage extent in both forward and backward.','ordinary_errors':['M=0 or N=0 raises ZeroDivisionError after contiguous/empty_like; invalid dim/scalar raises AssertionError before allocation.'],'scope':'Host call wiring, metadata, configuration, grid and rejection/unsafe boundary evidence only; actual kernel values are established separately by Lean real-source execution proofs, not by spies.'}
(OUT/'host-probe-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
