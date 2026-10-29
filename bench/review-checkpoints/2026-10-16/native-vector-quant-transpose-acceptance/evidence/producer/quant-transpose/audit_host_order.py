"""Run the exact original wrapper AST with ordinary host-success stubs and pinned C grid parser.
No tensor allocation, GPU execution, or numerical result is claimed.
"""
from pathlib import Path
import ast,ctypes,hashlib,json,types
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
O=W/'evidence/host-order';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=R/'bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py';tree=ast.parse(source.read_text());wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quantize_global_transpose')
lib=ctypes.PyDLL(str(W/'evidence/driver-order/parser-audit/parser.so'));parse=lib.parse_then_failure;parse.argtypes=[ctypes.py_object];parse.restype=ctypes.py_object;counter=ctypes.c_int.in_dll(lib,'pointer_attempted')
cases=[(1,1),(128,128),(2**38-128,1),(2**38-127,1),(2**38,1),(1,2**38),(2**20,2**25)]
records=[]
for M,N in cases:
 events=[]
 class T:
  def __init__(self,name,shape):self.name=name;self.shape=shape
  def abs(self):events.append(['abs',self.name]);return T('abs',(M,N))
  def max(self):events.append(['max',self.name]);return T('maximum',())
  def unsqueeze(self,axis):events.append(['unsqueeze',self.name,axis]);return T('absmax',(1,))
  def __rtruediv__(self,value):events.append(['reciprocal',value,self.name]);return T('inverse',(1,))
  def size(self,axis):return self.shape[axis]
  def stride(self,axis):return self.shape[1] if axis==0 else 1
 def empty(n,m,**kwargs):events.append(['empty',n,m,kwargs]);return T('output',(n,m))
 class K:
  def __getitem__(self,grid):
   def launch(*args,**kwargs):
    dims=grid({'BLOCK_M':128,'BLOCK_N':128});events.append(['launch_grid',list(dims)])
    gx=dims[0];packed=(gx,1,1,0,0,False,False,(4,1,0),None,None,None,None,None,(),b'',())
    parse(packed)
   return launch
 env={'torch':types.SimpleNamespace(empty=empty,int8='int8'),'triton':types.SimpleNamespace(cdiv=lambda x,y:(x+y-1)//y),'_quantize_global_transpose':K()}
 exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(source),'exec'),env)
 gx=((M+127)//128)*((N+127)//128)
 try:env['quantize_global_transpose'](T('CPU_input',(M,N)))
 except Exception as e:
  assert type(e).__name__==('ValueError' if gx<2**31 else 'OverflowError')
  assert counter.value==int(gx<2**31)
  assert [v[0] for v in events]==['abs','max','unsqueeze','reciprocal','empty','launch_grid']
  records.append({'M':M,'N':N,'host_numel':M*N,'grid_x':gx,'events':events,'first_error_type':type(e).__name__,'first_error':str(e),'downstream_pointer_stage_reached':bool(counter.value)})
 else:raise AssertionError('expected instrumented downstream pointer or real grid-parse failure')
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n')
review={'status':'passed','source_sha256':sha(source),'wrapper_AST_sha256':hashlib.sha256(ast.dump(wrapper,include_attributes=False).encode()).hexdigest(),'cases':len(records),'source_statements':[ast.unparse(n) for n in wrapper.body],'pinned_C_parser_sha256':sha(W/'evidence/driver-order/parser-audit/parser.c'),'parser_build_receipt_sha256':sha(W/'evidence/driver-order/parser-audit/receipt.json'),'ordinary_host_boundary':'Successful ordinary abs/max/unsqueeze/reciprocal/allocation metadata stubs; no claim these arbitrarily large allocations occur on a particular GPU. No source guard proves grid support from successful allocation.','observed_rejection':'Exact installed PyArg_ParseTuple raises OverflowError for grid>=2^31 before any downstream pointer instrumentation. For supported grids the downstream failure is deliberately instrumented, not a native pointer query.','gpu_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items() if k not in ['artifacts','source_statements']},indent=2))
