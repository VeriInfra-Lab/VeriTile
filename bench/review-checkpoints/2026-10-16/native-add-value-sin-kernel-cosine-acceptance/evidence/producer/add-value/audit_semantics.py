import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,ctypes,hashlib,json,subprocess,types
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');O=W/'evidence/semantics';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
native=sorted((W/'native-attempts').glob('*/receipt.json'))[-1];receipt=load(native);latest={'receipt':str(native.relative_to(W)),'sha256':sha(native)};(W/'evidence/native-latest.json').write_text(json.dumps(latest,indent=2)+'\n')
for name,h in receipt['artifacts'].items():assert sha(native.parent/name)==h
counts={};ircount=0
for r in receipt['records']:
 assert r['exit_code']==0
 row=r['result'];d=r['dtype']
 if row['status']=='compiled':
  stage='compiled';text=(native.parent/(d+'.raw.ttir')).read_text()
  assert row['signature']['N']=='i32' and row['signature']['value']=='fp32'
  assert row['specialization'][-1]==['fp32',None] and row['constexprs']=={'(3,)':1024}
  for needle in ['arith.cmpi slt','arith.muli','tensor<1024xi32>','tt.addptr','tt.load','tt.store','%value']:assert needle in text,(d,needle)
  assert 'tt.assert' not in text and text.count('tt.load')==1 and text.count('tt.store')==1 and text.index('tt.load')<text.index('tt.store')
  if d=='bool':assert 'arith.cmpi ne' in text and 'arith.fptosi' in text
  if d=='float64':assert 'arith.extf' in text and 'tensor<1024xf64>' in text
  ircount+=1
 elif 'fnuz' in d:stage='architecture';assert 'not supported in this architecture' in row['exception']
 else:stage='binder';assert row['failure_stage']=='signature' and 'KeyError' in row['exception']
 counts[stage]=counts.get(stage,0)+1
assert counts=={'compiled':15,'architecture':2,'binder':5}
llvm=(native.parent/'float32.llir').read_text();ptx=(native.parent/'float32.ptx').read_text()
for needle in ['shl i32','icmp slt i32','sext i32','getelementptr float']:assert needle in llvm
for needle in ['shl.b32','setp.lt.s32','mul.wide.s32','add.s64']:assert needle in ptx
c=O/'arithmetic.c';c.write_text('''#include <stdint.h>
uint32_t off(uint32_t pid,uint32_t lane){return pid*1024u+lane;}
int mask(uint32_t n,uint32_t pid,uint32_t lane){return (int32_t)off(pid,lane)<(int32_t)n;}
uint64_t pointer(uint64_t data,uint32_t bytes,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)off(pid,lane)*bytes;}
''');cmd=['cc','-shared','-fPIC','-O2',str(c),'-o',str(O/'arithmetic.so')];compiled=subprocess.run(cmd,text=True,capture_output=True);(O/'build.log').write_text(compiled.stdout+compiled.stderr);assert compiled.returncode==0
lib=ctypes.CDLL(str(O/'arithmetic.so'));lib.mask.argtypes=[ctypes.c_uint32]*3;lib.mask.restype=ctypes.c_int;lib.pointer.argtypes=[ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32];lib.pointer.restype=ctypes.c_uint64
signed=lambda x:((x+2**31)%2**32)-2**31
extents=[0,1,1023,1024,1025,2**31-1,2**31,2**31+1,2**32-1,2**32,2**32+1,(2**31-1)*1024,(2**31-1)*1024+1,2**63-1]
check_count=0;samples=[]
for n in extents:
 grid=(n+1023)//1024
 pids=sorted({0,1,2**21-1,2**21,2**22-1,2**22,min(max(grid-1,0),2**31-2)})
 for pid in pids:
  for lane in range(1024):
   off=signed(pid*1024+lane);active=off<signed(n);assert bool(lib.mask(n%2**32,pid,lane))==active
   for width in [1,2,4,8]:
    for data in [0,32,2**33+32,2**64-16]:
     address=(data+off*width)%2**64;assert lib.pointer(data,width,pid,lane)==address;check_count+=1
   if lane in [0,1023]:samples.append({'n':n,'grid':grid,'grid_supported':grid<=2**31-1,'pid':pid,'lane':lane,'launched':pid<grid,'offset_i32':off,'mask':active,'pointer4_from_2pow33_plus32':(2**33+32+off*4)%2**64})
source=R/'bench/tritonbench_g/add_value/add_value.py';tree=ast.parse(source.read_text());wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='puzzle1');traces=[]
for n,xc,oc,allocation_error in [(n,True,True,False) for n in extents]+[(0,False,True,False),(2**31,False,True,False),(0,True,False,False),(2**63-1,True,True,True)]:
 events=[]
 class T:
  def __init__(self,name,cuda):self.name=name;self.cuda=cuda
  @property
  def is_cuda(self):events.append(['is_cuda',self.name,self.cuda]);return self.cuda
  def numel(self):events.append(['numel',self.name,n]);return n
 x=T('x',xc);out=T('allocated',oc)
 def alloc(a):
  events.append(['empty_like',a.name])
  if allocation_error:raise MemoryError('instrumented allocation failure')
  return out
 class K:
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    events.append(['launch',list(grid(kwargs)),args[2],kwargs]);assert args== (x,out,n) and kwargs=={'BLOCK_SIZE':1024,'value':10}
   return launch
 env={'torch':types.SimpleNamespace(Tensor=T,empty_like=alloc),'triton':types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),'puzzle1_kernel':K()}
 exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(source),'exec'),env)
 try:
  returned=env['puzzle1'](x);assert returned is out;outcome='returned'
 except (AssertionError,MemoryError) as e:outcome=type(e).__name__
 expected='MemoryError' if allocation_error else 'AssertionError' if not (xc and oc) else 'returned';assert outcome==expected
 if expected!='returned':assert not any(e[0]=='launch' for e in events)
 if not xc and not allocation_error:assert events==[['empty_like','x'],['is_cuda','x',False]]
 traces.append({'n':n,'events':events,'outcome':outcome,'scope':'exact unchanged wrapper AST with ordinary instrumented primitives; no native launch'})
metadata=[]
import torch
for tensor in [torch.empty((0,2**31)),torch.empty((2,0,3)),torch.empty(()),torch.arange(64).reshape(8,8).T,torch.arange(64)[7:19:2],torch.arange(64)[7:7]]:
 out=torch.empty_like(tensor)
 metadata.append({'shape':list(tensor.shape),'numel':tensor.numel(),'strides':list(tensor.stride()),'storage_offset':tensor.storage_offset(),'storage_bytes':tensor.untyped_storage().nbytes(),'data_ptr_zero':tensor.data_ptr()==0,'output_shape':list(out.shape),'output_strides':list(out.stride()),'output_offset':out.storage_offset(),'output_data_ptr_zero':out.data_ptr()==0})
for name,data in [('address-samples.json',samples),('wrapper-traces.json',traces),('metadata.json',metadata)]: (O/name).write_text(json.dumps(data,indent=2)+'\n')
result={'status':'passed','source_sha256':sha(source),'native_receipt':latest,'all22_stage_counts':counts,'raw_source_ir_checks':ircount,'compiled_C_vs_integer_reference_pointer_checks':check_count,'host_extents':extents,'original_wrapper_AST_cases':len(traces),'CPU_metadata_examples':len(metadata),'compiled_C_command':cmd,'native_PTX_arithmetic':'i32 shl,signed32 mask,signed32 wide multiply,64bit add','runtime_argument':'actual non-constexpr fp32 value10; original input remains unary','gpu_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
