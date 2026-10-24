"""Original wrapper AST + synthetic address spaces; no GPU execution claim."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,itertools,json
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
source=R/'bench/tritonbench_g/triton_mul2/triton_mul2.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=next((W/'evidence/pointer-replay').glob('pointer_probe*.so'))
spec=importlib.util.spec_from_file_location('pointer_probe',native);extract=importlib.util.module_from_spec(spec);spec.loader.exec_module(extract)
tree=ast.parse(source.read_text());wrappers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['triton_mul2','triton_mul2_inplace']]
assert len(wrappers)==2
class Tensor:
 def __init__(self,n,width,base,offset,owner,pinned=False):
  self.n=n;self.width=width;self.base=base;self.offset=offset;self.device=owner;self.pinned=pinned
  self.shape=(n,);self.strides=(1,);self.storage_bytes=(n+offset+3)*width
 def numel(self):return self.n
 def data_ptr(self):return self.base+self.offset*self.width if self.n else 0
def i32(n):return (n+2**31)%2**32-2**31
class Observation(Exception):pass

def one(mode,route,width,n,block,reverse=False,denied=None):
 owner='cuda:0' if route=='local' else 'cuda:1' if route=='peer' else 'cpu'
 x=Tensor(n,width,0x1000,2,owner,route=='pinned')
 original_metadata=copy.deepcopy(vars(x));current=0
 memory={0x80:239}
 if n<100:
  for k in range(x.storage_bytes):memory[x.base+k]=(37*k+19)%256
 before=dict(memory);mapping={};readable=set();writable=set();events=[];io={'reads':0,'writes':0};extracted=[]
 def view(t):
  base=t.base if t.device.startswith('cuda') else t.base+0x10000
  if t.n and t.pinned or t.n and t.device.startswith('cuda'):
   if t.n<100:
    for k in range(t.storage_bytes):mapping[base+k]=t.base+k;readable.add((current,base+k));writable.add((current,base+k))
  return base
 input_device_base=view(x)
 if denied=='read' and n:readable.discard((current,input_device_base+x.offset*width))
 if denied=='write' and n:writable.discard((current,input_device_base+x.offset*width))
 def zeros_like(t):
  events.append('zeros_like');out=Tensor(t.n,t.width,0x8000,0,t.device,False);out.storage_bytes=t.n*t.width
  if n<100:
   for k in range(out.storage_bytes):memory[out.base+k]=0
  view(out);return out
 class Kernel:
  def __init__(self,inplace):self.inplace=inplace
  def __getitem__(self,grid):
   def launch(*args):
    inp=args[0];out=inp if self.inplace else args[1];numel=args[-2];B=args[-1]
    events.append('grid_parse');g=extract.grid(grid[0])
    actual=[]
    for t in ([inp] if self.inplace else [inp,out]):
     events.append('pointer')
     if t.n and t.device=='cpu' and not t.pinned:raise Observation('pointer_setup')
     base=t.base if t.device.startswith('cuda') else t.base+0x10000
     ptr,calls=extract.pointer(t,base+t.offset*t.width)
     extracted.append({'raw':t.data_ptr(),'actual':ptr,'attribute_calls':calls,'owner':t.device,'current':current})
     actual.append(ptr)
    inptr=actual[0];outptr=actual[0] if self.inplace else actual[1]
    events.append('launch')
    if i32(numel)==-2**31:return
    pids=list(range(g));pids=pids[::-1] if reverse else pids;seen=set()
    for pid in pids:
     loaded=[]
     for lane in range(B):
      offset=i32(i32(pid)*B+lane)
      if offset>=i32(numel):continue
      ia=(inptr+offset*width)%2**64;oa=(outptr+offset*width)%2**64
      value=0
      for k in range(width):
       if (current,ia+k) not in readable:raise Observation('input_access')
       backing=mapping.get(ia+k)
       if backing not in memory:raise Observation('missing_input_bytes')
       io['reads']+=1;value|=memory[backing]<<(8*k)
      loaded.append((oa,(2*value)%(1<<(8*width))))
     writes=[]
     for oa,value in loaded:
      for k in range(width):
       if (current,oa+k) not in writable:raise Observation('output_access')
       backing=mapping.get(oa+k)
       if backing is None:raise Observation('output_access')
       if backing in seen:raise Observation('ownership')
       seen.add(backing);writes.append((backing,(value>>(8*k))&255))
     for backing,value in writes:memory[backing]=value;io['writes']+=1
   return launch
 def cdiv(a,b):events.append('cdiv');return (a+b-1)//b
 env={'torch':SimpleNamespace(zeros_like=zeros_like),'triton':SimpleNamespace(cdiv=cdiv),'mul2_kernel':Kernel(False),'mul2_inplace_kernel':Kernel(True)}
 exec(compile(ast.Module(body=wrappers,type_ignores=[]),str(source),'exec'),env)
 error=None;returned=None
 try:returned=env['triton_mul2_inplace' if mode=='inplace' else 'triton_mul2'](x,block)
 except (Observation,OverflowError,ZeroDivisionError) as e:error=type(e).__name__+':'+str(e)
 assert vars(x)==original_metadata
 if n==2**31 and block==1:
  assert error and error.startswith('OverflowError') and events[-1]=='grid_parse' and 'pointer' not in events
 elif mode=='outplace' and route=='pinned' and n:
  assert error=='Observation:pointer_setup' and io=={'reads':0,'writes':0}
 elif denied and n:
  assert error=='Observation:'+('input_access' if denied=='read' else 'output_access')
 else:
  assert error is None,error
  assert (returned is x)==(mode=='inplace') and returned.device==owner
  assert returned.shape==x.shape and returned.strides==x.strides
  if n<100:
   expected=dict(before)
   if mode=='outplace':
    for k in range(returned.storage_bytes):expected[returned.base+k]=0
   for i in range(n):
    raw=x.base+(x.offset+i)*width
    value=sum(before[raw+k]<<(8*k) for k in range(width));doubled=(2*value)%(1<<(8*width))
    dest=returned.base+(returned.offset+i)*width
    for k in range(width):expected[dest+k]=(doubled>>(8*k))&255
   assert memory==expected
  if n==0 or n==2**31:assert io=={'reads':0,'writes':0}
  if route=='peer' and n:assert all(e['owner']=='cuda:1' and e['current']==0 for e in extracted)
  if route=='pinned' and n:assert extracted[0]['actual']!=extracted[0]['raw']
 return {'mode':mode,'route':route,'width':width,'numel':n,'block':block,'reverse':reverse,'denied':denied,'events':events,'outcome':'returned' if error is None else error,'io':io,'extracted':extracted}
rows=[]
for mode,route,width,n,block,reverse in itertools.product(['outplace','inplace'],['local','peer','pinned'],[1,2,4,8],[0,1,3,17],[1,16],[False,True]):
 rows.append(one(mode,route,width,n,block,reverse))
for route,denied in itertools.product(['local','peer','pinned'],['read','write']):rows.append(one('inplace',route,1,3,16,False,denied))
for mode,block in itertools.product(['outplace','inplace'],[1,16]):rows.append(one(mode,'peer',1,2**31,block))
report={'status':'passed','original_source_sha256':sha(source),'original_wrapper_AST_unchanged':True,'scope':'Actual installed pointer extraction and CPython signed grid parsing with explicit CUDA attribute stub; original wrapper AST with synthetic byte address/access spaces and independent canonical result. No live CUDA accessibility or native numeric validation.','gpu_launches':0,'cuda_attribute_is_stub':True,'cases':len(rows),'rows':rows,'checks':['same-device and accessible peer routes','in-place host-pinned distinct device alias writes original backing','outplace pageable output pointer setup failure','per-byte permission denial independent of ownership','full tile loads before stores','exact in-place identity and outplace metadata','canonical backing result and exterior frame','null pointer extraction without backing map or permissions','N2^31/B1 grid parse before pointers','N2^31/B16 all-masked no access']}
(W/'evidence/pointer-execution-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
