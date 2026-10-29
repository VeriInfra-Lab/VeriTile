import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,ctypes,hashlib,json,subprocess,types
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');O=W/'evidence/semantics';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
latest=json.loads((W/'evidence/native-latest.json').read_text());native=W/latest['receipt'];receipt=json.loads(native.read_text());rows=json.loads(native.with_name('records.json').read_text())
for name,h in receipt['artifacts'].items():assert sha(native.parent/name)==h
extras={'complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2'};fp8={'float8_e4m3fn','float8_e5m2'}
counts={};ircount=0
for row in rows:
 x,y=row['x'],row['y']
 if x in extras or y in extras:stage='binder';assert row['frontend']=='rejected' and row['error_type']=='KeyError'
 elif 'fnuz' in x or 'fnuz' in y:stage='signature';assert row['frontend']=='rejected' and 'not supported in this architecture' in row['error']
 elif (x in fp8 and (y=='bool' or y.startswith(('int','uint')))) or (y in fp8 and (x=='bool' or x.startswith(('int','uint')))):stage='addition';assert row['frontend']=='rejected'
 elif (x==y and x in fp8) or (x=='float64' and y in fp8) or (y=='float64' and x in fp8):stage='backend';assert row['backend']=='rejected'
 else:stage='compiled';assert row['backend']=='compiled'
 counts[stage]=counts.get(stage,0)+1
 if row['frontend']=='generated':
  text=(native.parent/(x+'__'+y+'.raw.ttir')).read_text()
  for needle in ['arith.cmpi slt','arith.muli','tensor<16xi32>','tt.addptr','tt.load','tt.store']:assert needle in text,(x,y,needle)
  assert 'tt.assert' not in text
  assert text.index('tt.load')<text.rindex('tt.load')<text.index('tt.store')
  ircount+=1
llvm=(native.parent/'float32__float32.llir').read_text();ptx=(native.parent/'float32__float32.ptx').read_text()
for needle in ['shl i32','icmp slt i32','sext i32','getelementptr float']:assert needle in llvm
for needle in ['shl.b32','setp.lt.s32','mul.wide.s32','add.s64']:assert needle in ptx
c=O/'arithmetic.c';c.write_text('''#include <stdint.h>
uint32_t off(uint32_t pid,uint32_t lane){return pid*16u+lane;}
int mask(uint32_t n,uint32_t pid,uint32_t lane){return (int32_t)off(pid,lane)<(int32_t)n;}
uint64_t pointer(uint64_t data,uint32_t bytes,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)off(pid,lane)*bytes;}
''');cmd=['cc','-shared','-fPIC','-O2',str(c),'-o',str(O/'arithmetic.so')];compiled=subprocess.run(cmd,text=True,capture_output=True);(O/'build.log').write_text(compiled.stdout+compiled.stderr);assert compiled.returncode==0
lib=ctypes.CDLL(str(O/'arithmetic.so'));lib.mask.argtypes=[ctypes.c_uint32]*3;lib.mask.restype=ctypes.c_int;lib.pointer.argtypes=[ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32];lib.pointer.restype=ctypes.c_uint64
signed=lambda x:((x+2**31)%2**32)-2**31
extents=[0,1,15,16,17,2**31-1,2**31,2**31+1,2**32-1,2**32,2**32+1,(2**31-1)*16,(2**31-1)*16+1,2**63-1]
check_count=0;samples=[]
for n in extents:
 grid=(n+15)//16
 pids=sorted({0,1,2**27-1,2**27,2**28-1,2**28,min(max(grid-1,0),2**31-2)})
 for pid in pids:
  for lane in range(16):
   off=signed(pid*16+lane);active=off<signed(n);assert bool(lib.mask(n%2**32,pid,lane))==active
   for width in [1,2,4,8]:
    for data in [0,32,2**33+32,2**64-16]:
     address=(data+off*width)%2**64;assert lib.pointer(data,width,pid,lane)==address;check_count+=1
   if lane in [0,15]:samples.append({'n':n,'grid':grid,'grid_supported':grid<=2**31-1,'pid':pid,'lane':lane,'launched':pid<grid,'offset_i32':off,'mask':active,'pointer4_from_2pow33_plus32':(2**33+32+off*4)%2**64})
source=R/'bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py';original=source.read_text();tree=ast.parse(original);wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='custom_add');traces=[]
for n in extents+[None]:
 events=[]
 class T:
  def __init__(self,name):self.name=name
  @property
  def is_cuda(self):raise AssertionError('source has no CUDA assertion')
  def size(self,axis):
   events.append(['size',self.name,axis,n]);assert axis==0
   if n is None:raise IndexError('dimension specified as0 but tensor has no dimensions')
   return n
 x=T('x');y=T('y');out=T('allocated')
 def alloc(a):events.append(['empty_like',a.name]);return out
 class K:
  def __getitem__(self,grid):
   def launch(*args,**kwargs):events.append(['launch',list(grid),args[3],kwargs]);assert args[3]==n and kwargs=={'BLOCK':16}
   return launch
 env={'torch':types.SimpleNamespace(Tensor=T,empty_like=alloc),'triton':types.SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b),'_add_kernel':K()}
 exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(source),'exec'),env)
 if n is None:
  try:env['custom_add'](x,y)
  except IndexError:assert [e[0] for e in events]==['empty_like','size']
  else:raise AssertionError('rankzero must fail after allocation')
  traces.append({'size':None,'shape_class':'rankzero','events':events,'failure':'IndexError_after_allocation'})
 else:
  assert env['custom_add'](x,y) is out
  assert events[-1][1]==[(n+15)//16]
  traces.append({'size':n,'events':events,'returned_allocated':True,'scope':'exact original wrapper AST with successful ordinary primitive stubs; no native launch; no CUDA query or numel query'})

metadata=[]
import torch
for tensor in [torch.empty((2**31,0)),torch.empty((0,2**31)),torch.empty((2,0,3)),torch.empty(()),torch.arange(64).reshape(8,8).T,torch.arange(64)[7:19:2]]:
 out=torch.empty_like(tensor)
 metadata.append({'shape':list(tensor.shape),'numel':tensor.numel(),'strides':list(tensor.stride()),'storage_offset':tensor.storage_offset(),'storage_bytes':tensor.untyped_storage().nbytes(),'data_ptr_zero':tensor.data_ptr()==0,'output_shape':list(out.shape),'output_strides':list(out.stride()),'output_offset':out.storage_offset(),'output_data_ptr_zero':out.data_ptr()==0})
(O/'address-samples.json').write_text(json.dumps(samples,indent=2)+'\n');(O/'wrapper-traces.json').write_text(json.dumps(traces,indent=2)+'\n');(O/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
result={'status':'passed','source_sha256':sha(source),'native_receipt':latest,'all22_stage_counts':counts,'raw_source_ir_checks':ircount,'compiled_C_vs_integer_reference_pointer_checks':check_count,'host_extents':extents,'original_wrapper_AST_cases':len(traces),'CPU_metadata_examples':len(metadata),'compiled_C_command':cmd,'native_PTX_arithmetic':'i32 shl, signed32 mask, signed32 wide multiply,64bit add','gpu_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
