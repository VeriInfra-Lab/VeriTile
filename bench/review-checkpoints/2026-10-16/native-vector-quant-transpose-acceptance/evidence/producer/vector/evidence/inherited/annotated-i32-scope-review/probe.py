import sys;sys.dont_write_bytecode=True
from pathlib import Path
import ast,ctypes,datetime,hashlib,importlib.util,json,os,re,shutil,struct,subprocess,sysconfig,traceback,types
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');N=2**31
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
os.environ['TRITON_CACHE_DIR']=str(W/'cache');os.environ['TMPDIR']=str(W/'tmp');(W/'tmp').mkdir(exist_ok=True)
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
tasks=[('vector_addition','add_kernel','add',1024,'BLOCK_SIZE','n_elements',3),('vector_addition_custom','_add_kernel','custom_add',16,'BLOCK','size',3),('add_value','puzzle1_kernel','puzzle1',1024,'BLOCK_SIZE','N',2)]
driver=Path('/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/driver.c');text=driver.read_text();extract=re.search(r'bool extractI32\(void \*ptr, PyObject \*obj\) \{.*?\n\}',text,re.S)[0]
(W/'driver.c.txt').write_bytes(driver.read_bytes())
(W/'extract_i32.c').write_text('#include <Python.h>\n#include <stdint.h>\n#include <stdbool.h>\n'+extract+'\n')
cmd=[shutil.which('cc'),'-shared','-fPIC','-I'+sysconfig.get_paths()['include'],str(W/'extract_i32.c'),'-o',str(W/'extract_i32.so')]
build=subprocess.run(cmd,text=True,capture_output=True);(W/'extract-build.log').write_text(build.stdout+build.stderr);assert build.returncode==0
lib=ctypes.PyDLL(str(W/'extract_i32.so'));lib.extractI32.argtypes=[ctypes.c_void_p,ctypes.py_object];lib.extractI32.restype=ctypes.c_bool
word=ctypes.c_int32(0);ok=lib.extractI32(ctypes.byref(word),N);assert ok and word.value==-N
extraction={'source':str(driver),'source_sha256':sha(driver),'exact_function':extract,'compiler_command':cmd,'compiler_exit_code':build.returncode,'sizeof_long':ctypes.sizeof(ctypes.c_long),'python_argument':N,'success':ok,'stored_signed_i32':word.value,'stored_bits_hex':hex(ctypes.c_uint32(word.value).value)}
(W/'driver-extraction.json').write_text(json.dumps(extraction,indent=2)+'\n')
records=[]
for task,kernel_name,wrapper_name,block,block_name,n_name,pointers in tasks:
 row={'task':task,'N_python':N,'block':block,'grid_x':N//block,'native_signed_bound':word.value};records.append(row)
 source=R/'bench/tritonbench_g'/task/(task+'.py');original=source.read_text();prefix=original.split('################################################################')[0];path=W/(task+'_source.py');path.write_text(prefix)
 f=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in [kernel_name,wrapper_name]}
 assert f(original)==f(prefix);row.update(original_source=str(source),source_sha256=sha(source),production_ast_unchanged=True)
 spec=importlib.util.spec_from_file_location('review_'+task,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);kernel=getattr(module,kernel_name)
 arguments=[jit.MockTensor(torch.float32) for _ in range(pointers)]+[N];kw={block_name:block};
 if task=='add_value':kw['value']=10
 binder=jit.create_function_from_signature(kernel.signature,kernel.params,backend);params,specialization,options=binder(*arguments,**kw);options,signature,constexprs,attrs=kernel._pack_args(backend,options,params,specialization,options)
 assert signature[n_name]=='i32' and specialization[pointers][0]=='i32'
 row.update(signature=signature,specialization=specialization,constexprs={str(k):v for k,v in constexprs.items()},params_N=params[n_name],options=vars(options))
 (W/(task+'-binder.json')).write_text(json.dumps(row,indent=2)+'\n');print(task,'binder accepted',N,'as i32',flush=True)
 src=ASTSource(kernel,signature,constexprs,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
 raw=str(src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context));(W/(task+'.raw.ttir')).write_text(raw)
 assert 'arith.cmpi slt' in raw and 'tt.assert' not in raw
 compiled=triton.compile(src,target=target,options=vars(options))
 for ext in ['ttir','ttgir','llir','ptx']:(W/(task+'.'+ext)).write_text(compiled.asm[ext])
 row['compiled']='TTIR_LLVM_PTX';row['gpu_launches']=0
 # Execute unchanged original wrapper AST against ordinary fallible allocation
 # and launch stubs; no bound check exists and the exact allocated object returns.
 events=[]
 class T:
  def __init__(self,name):self.name=name;self.is_cuda=True
  def numel(self):events.append(['numel',self.name,N]);return N
  def size(self,dim):assert dim==0;events.append(['size',self.name,dim,N]);return N
 inp=T('input');other=T('second');out=T('allocated_output')
 class K:
  def __getitem__(self,grid):
   def call(*args,**kwargs):
    g=grid(kwargs) if callable(grid) else grid;events.append(['launch',list(g),args[pointers],kwargs]);assert g==(N//block,)
   return call
 def alloc(x):events.append(['empty_like',x.name]);return out
 wrapper=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==wrapper_name)
 env={'torch':types.SimpleNamespace(Tensor=T,empty_like=alloc),'triton':types.SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b),kernel_name:K()}
 exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(source),'exec'),env)
 returned=env[wrapper_name](inp,other) if task!='add_value' else env[wrapper_name](inp)
 assert returned is out
 row['original_wrapper_trace']=events;row['returns_allocated_output']=True
 row['all_lane_argument']={'programs':N//block,'pid_interval':[0,N//block-1],'lane_interval':[0,block-1],'signed_index_interval':[0,N-1],'overflow':False,'mask':'signed_index < -2147483648','all_masks_false':True,'input_loads':0,'output_stores':0,'result':'allocated output unchanged; host returns normally if ordinary allocator/launch primitives succeed'}
 (W/(task+'-result.json')).write_text(json.dumps(row,indent=2)+'\n');print(task,'compiled and wrapper returned',flush=True)
# A real zero-byte CPU Torch tensor realizes the custom first-axis metadata;
# the original custom wrapper has no CUDA assertion. No large allocation used.
x=torch.empty((N,0));y=torch.empty_like(x)
assert x.size(0)==N and x.numel()==0 and x.data_ptr()==0 and y.data_ptr()==0
zero={'shape':list(x.shape),'strides':list(x.stride()),'numel':x.numel(),'storage_bytes':x.untyped_storage().nbytes(),'data_ptr':x.data_ptr(),'output_shape':list(y.shape),'output_data_ptr':y.data_ptr(),'scope':'actual CPU metadata construction only; no GPU launch'}
(W/'custom-zero-byte-metadata.json').write_text(json.dumps(zero,indent=2)+'\n')
result={'status':'passed','torch_version':torch.__version__,'triton_version':triton.__version__,'driver_extraction':extraction,'records':records,'custom_zero_byte_example':zero,'gpu_launches':0,'large_tensor_allocations':0}
(W/'probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','tasks':len(records),'gpu_launches':0}))
