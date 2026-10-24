from pathlib import Path
import ast,ctypes,hashlib,json,math,subprocess,sys,sysconfig,types
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=W.parent/'cosine-native-finite-acceptance-revision1-wip';R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(P/'evidence/receipt.json')==(W/'producer-pin.txt').read_text().strip();O=W/'host-memory-001';O.mkdir()
source=R/'bench/tritonbench_g/cosine_compute/cosine_compute.py';wrapper=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cos')
import triton,torch
traces=[]
for n,fail in [(n,False) for n in [0,1,5,4096,2**31,2**31+1,2**40+1,2**52+1,2**63-1]]+[(0,True),(2**63-1,True)]:
 events=[]
 class Tensor:
  def numel(self):events.append(['numel',n]);return n
  @property
  def is_cuda(self):raise AssertionError('inserted CUDA guard')
  @property
  def device(self):raise AssertionError('inserted device context')
 x=Tensor();out=object()
 def alloc(t):
  assert t is x;events.append(['empty_like'])
  if fail:raise MemoryError('review allocation failure')
  return out
 def sqrt(v):events.append(['sqrt',v]);return math.sqrt(v)
 def ceil(v):events.append(['ceil',v.hex()]);return math.ceil(v)
 def power(v):events.append(['next_power',v]);return triton.next_power_of_2(v)
 def cdiv(a,b):events.append(['cdiv',a,b]);return triton.cdiv(a,b)
 class Kernel:
  def __getitem__(self,grid):
   def launch(a,b,n_arg,block):
    assert a is x and b is out and n_arg==n and grid==(triton.cdiv(n,block),1,1);events.append(['launch',grid,n,block])
   return launch
 env={'torch':types.SimpleNamespace(empty_like=alloc),'math':types.SimpleNamespace(sqrt=sqrt,ceil=ceil),'triton':types.SimpleNamespace(next_power_of_2=power,cdiv=cdiv),'cos_func':Kernel()};exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(source),'exec'),env)
 try:result=env['cos'](x);assert result is out;outcome='returned_same_output'
 except (MemoryError,ZeroDivisionError) as e:outcome=type(e).__name__
 expected=['empty_like'] if fail else ['empty_like','numel','sqrt','ceil','next_power','cdiv']+([] if n==0 else ['launch'])
 assert [e[0] for e in events]==expected;assert outcome==('MemoryError' if fail else 'ZeroDivisionError' if n==0 else 'returned_same_output')
 traces.append({'n':n,'allocation_failure':fail,'events':events,'outcome':outcome})
metadata=[]
for name in ['float32','bfloat16']:
 base=torch.empty(32,dtype=getattr(torch,name))
 tensors={'scalar':torch.empty((),dtype=getattr(torch,name)),'transpose':base[:24].reshape(4,6).T,'offset_slice':base[3:21:2],'broadcast':base[:1].expand(5),'empty_offset':base[17:17]}
 for label,x in tensors.items():
  out=torch.empty_like(x)
  def meta(t):return {'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),'n':t.numel(),'offset':t.storage_offset(),'data':t.data_ptr(),'base':t.untyped_storage().data_ptr(),'bytes':t.untyped_storage().nbytes()}
  a,b=meta(x),meta(out);assert a['dtype']==b['dtype'] and a['shape']==b['shape'] and b['offset']==0 and b['data']==(b['base'] if b['n'] else 0)
  assert not a['bytes'] or not b['bytes'] or a['base']+a['bytes']<=b['base'] or b['base']+b['bytes']<=a['base']
  metadata.append({'case':label,'input':a,'output':b})
# Exact current installed source slices, freshly built; this is C instrumentation.
driver=Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c');assert sha(driver)==sha(P/'evidence/inherited/annotated-i32-scope-review/driver.c.txt');text=driver.read_text();(O/'driver.c.txt').write_text(text)
body=text[text.index('static PyObject *launchKernel('):];parser=body[body.index('  // Parse the arguments.'):body.index('  // launch entry hook.')];assert body.index('PyArg_ParseTuple')<body.index('extractor.extract(params')
extract=text[text.index('bool extractI32('):];extract=extract[:extract.index('\n}')+2]
code='#include <Python.h>\n#include <stdint.h>\n#include <stdbool.h>\nint reached=0;\nPyObject *review_parse(PyObject *args){ reached=0;\n'+parser+'\nreached=1;PyBuffer_Release(&signature);PyErr_SetString(PyExc_ValueError,"review downstream marker");return NULL;}\n'+extract+'\nuint64_t review_ptr(uint64_t data,uint32_t width,uint32_t block,uint32_t pid,uint32_t lane){return data+(uint64_t)(int64_t)(int32_t)(pid*block+lane)*width;}\n'
(O/'instrument.c').write_text(code);cmd=['cc','-shared','-fPIC','-O2','-I'+sysconfig.get_path('include'),str(O/'instrument.c'),'-o',str(O/'instrument.so')];run=subprocess.run(cmd,capture_output=True,text=True);(O/'build.log').write_text(run.stdout+run.stderr);assert run.returncode==0
lib=ctypes.PyDLL(str(O/'instrument.so'));parse=lib.review_parse;parse.argtypes=[ctypes.py_object];parse.restype=ctypes.py_object;reached=ctypes.c_int.in_dll(lib,'reached');parser_rows=[]
for grid in [0,2**31-1,2**31]:
 args=(grid,1,1,0,0,False,False,(4,1,0),None,None,None,None,None,(),b'',())
 try:parse(args)
 except Exception as e:
  assert type(e).__name__==('ValueError' if grid<2**31 else 'OverflowError');assert reached.value==int(grid<2**31);parser_rows.append({'grid':grid,'exception':type(e).__name__,'downstream_reached':bool(reached.value)})
 else:raise AssertionError('expected instrumented error')
fn=lib.extractI32;fn.argtypes=[ctypes.c_void_p,ctypes.py_object];fn.restype=ctypes.c_bool;abi=[]
for n in [0,1,2**31-1,2**31,2**31+1,2**32,2**40,2**40+1,2**63-1]:
 out=ctypes.c_int32();assert fn(ctypes.byref(out),n);assert out.value==((n+2**31)%2**32)-2**31;abi.append({'n':n,'i32':out.value})
ptr=lib.review_ptr;ptr.argtypes=[ctypes.c_uint64]+[ctypes.c_uint32]*4;ptr.restype=ctypes.c_uint64;pointer_rows=[]
for n in [1,5,4096,2**31,2**31+1,2**32+1,2**40]:
 block=triton.next_power_of_2(math.ceil(math.sqrt(n)));grid=triton.cdiv(n,block)
 for pid in sorted({0,grid-1,min(grid-1,2**31//block)}):
  for lane in sorted({0,block-1}):
   offset=((pid*block+lane+2**31)%2**32)-2**31
   for width in [2,4]:
    for base in [32,2**33+32,2**64-16]:
     value=ptr(base,width,block,pid,lane);assert value==(base+offset*width)%2**64;pointer_rows.append({'n':n,'block':block,'pid':pid,'lane':lane,'width':width,'data':base,'address':value})
for name,data in [('wrapper-traces.json',traces),('metadata.json',metadata),('parser.json',parser_rows),('i32.json',abi),('pointers.json',pointer_rows)]:(O/name).write_text(json.dumps(data,indent=2)+'\n')
report={'status':'passed','wrapper_AST_traces':len(traces),'CPU_metadata_cases':len(metadata),'exact_current_C_parser_cases':len(parser_rows),'exact_current_C_i32_extractor_cases':len(abi),'compiled_C_pointer_comparisons':len(pointer_rows),'dtype_widths':['BF16:2','FP32:4'],'source_sha256':sha(source),'driver_sha256':sha(driver),'parser_source_sha256':hashlib.sha256(parser.encode()).hexdigest(),'extractor_source_sha256':hashlib.sha256(extract.encode()).hexdigest(),'parser_reachability':'infrastructure only; oversized cosine blocks reject compilation before such overflowing grids','GPU_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}))
