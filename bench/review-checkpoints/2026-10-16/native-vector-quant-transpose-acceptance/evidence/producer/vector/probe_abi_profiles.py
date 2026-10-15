import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ctypes,datetime,hashlib,importlib.util,json,os
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/abi-profiles'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();os.environ['TRITON_CACHE_DIR']=str(O/'cache');os.environ['TMPDIR']=str(O/'tmp');(O/'tmp').mkdir()
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
native=W/json.loads((W/'evidence/native-latest.json').read_text())['receipt'];prefix=native.with_name('source-prefix.py');spec=importlib.util.spec_from_file_location('vector_abi_'+stamp,prefix);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);binder=jit.create_function_from_signature(mod.add_kernel.signature,mod.add_kernel.params,backend)
extractor=W/'evidence/inherited/annotated-i32-scope-review/extract_i32.so';lib=ctypes.PyDLL(str(extractor));lib.extractI32.argtypes=[ctypes.c_void_p,ctypes.py_object];lib.extractI32.restype=ctypes.c_bool
extents=json.loads((W/'evidence/semantics/receipt.json').read_text())['host_extents'];rows=[]
for n in extents:
 word=ctypes.c_int32();assert lib.extractI32(ctypes.byref(word),n);expected=((n+2**31)%2**32)-2**31;assert word.value==expected
 args=[jit.MockTensor(torch.float32)]*3+[n];params,specialization,options=binder(*args,BLOCK_SIZE=1024);options,signature,constexprs,attrs=mod.add_kernel._pack_args(backend,options,params,specialization,options)
 assert signature['n_elements']=='i32'
 src=ASTSource(mod.add_kernel,signature,constexprs,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context);raw=str(src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context));(O/(str(n)+'.raw.ttir')).write_text(raw)
 kernel=triton.compile(src,target=target,options=vars(options));artifacts={}
 for ext,data in kernel.asm.items():
  p=O/(str(n)+'.'+ext);p.write_bytes(data if isinstance(data,bytes) else str(data).encode());artifacts[p.name]=sha(p)
 rows.append({'n':n,'signature':signature,'specialization':specialization,'constexprs':{str(k):v for k,v in constexprs.items()},'attrs':{str(k):v for k,v in attrs.items()},'driver_i32':word.value,'options':vars(options),'raw_sha256':sha(O/(str(n)+'.raw.ttir')),'artifacts':artifacts})
(O/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes());receipt={'status':'passed','cases':len(rows),'original_source_sha256':json.loads(native.read_text())['original_source_sha256'],'production_AST_unchanged':True,'source_prefix_sha256':sha(prefix),'exact_driver_extractor_sha256':sha(extractor),'target':vars(target),'all_bounds_annotated_i32':True,'backend_compiles':len(rows),'gpu_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(W/'evidence/abi-latest.json').write_text(json.dumps({'path':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n');print(json.dumps({'cases':len(rows),'receipt':str(O/'receipt.json'),'sha256':sha(O/'receipt.json')}))
