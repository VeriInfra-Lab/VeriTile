"""Offline original-AST compilation and actual binder inventory; no GPU launch."""
from pathlib import Path
import ast,concurrent.futures,datetime,hashlib,importlib.util,json,os,subprocess,sys,traceback
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
SOURCE=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/triton_mul2/triton_mul2.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DTYPES=['bool','uint8','int8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
KERNELS=['mul2_kernel','mul2_inplace_kernel']
if len(sys.argv)>1 and sys.argv[1]=='worker':
 work=Path(sys.argv[2]);name=sys.argv[3];kernel_name=sys.argv[4];dtype_name=sys.argv[5];block=int(sys.argv[6]);full=sys.argv[7]=='full'
 os.environ['TRITON_CACHE_DIR']=str(work/'cache'/name)
 import torch,triton,triton.runtime.jit as jit
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 from triton.backends.nvidia.compiler import CUDABackend
 from triton._C.libtriton import ir
 spec=importlib.util.spec_from_file_location('mul2_original',work/'source.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 kernel=getattr(module,kernel_name);target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
 row={'name':name,'kernel':kernel_name,'dtype':dtype_name,'block':block,'n':2049,'mode':'full' if full else 'frontend','target':{'backend':'cuda','arch':90,'warp_size':32}}
 stage='signature'
 try:
  dtype=getattr(torch,dtype_name);binder=jit.create_function_from_signature(kernel.signature,kernel.params,backend)
  args=[jit.MockTensor(dtype)]*(2 if kernel_name=='mul2_kernel' else 1)+[2049,block]
  params,specialization,kwargs=binder(*args)
  options,signature,constexprs,attrs=kernel._pack_args(backend,kwargs,params,specialization,kwargs)
  row.update(signature=signature,specialization=specialization,constexprs={str(k):v for k,v in constexprs.items()},attrs={str(k):v for k,v in attrs.items()},options=vars(options),parameters=[{'name':p.name,'annotation':p.annotation,'annotation_type':p.annotation_type,'is_constexpr':p.is_constexpr} for p in kernel.params])
  assert signature['n_elements']=='i32'
  stage='frontend';src=ASTSource(kernel,signature,constexprs,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
  raw=src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
  (work/(name+'.raw.ttir')).write_text(str(raw));row['status']='frontend_passed'
  if full:
   stage='backend';compiled=triton.compile(src,target=target,options=vars(options))
   for ext in ['ttir','ttgir','llir','ptx']:(work/(name+'.'+ext)).write_text(compiled.asm[ext])
   row['status']='compiled'
 except Exception:
  row.update(status='rejected',failure_stage=stage,exception=traceback.format_exc())
 (work/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
else:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');work=W/'native-attempts'/stamp;work.mkdir(parents=True)
 original=SOURCE.read_text();prefix=original.split('################################################################')[0]
 selected=KERNELS+['triton_mul2','triton_mul2_inplace']
 nodes=lambda s:[ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in selected]
 assert nodes(original)==nodes(prefix)
 (work/'source.py').write_text(prefix);(work/'probe_native.py').write_bytes(Path(__file__).read_bytes())
 jobs=[]
 for kernel in KERNELS:
  for dtype in DTYPES:jobs.append((kernel+'-'+dtype+'-b16',kernel,dtype,16,'full'))
  for block in [1,2,3,0,-16,512,1024,1025,1048576,1048577,2147483648]:
   jobs.append((kernel+'-float32-b'+str(block),kernel,'float32',block,'full' if block in [512,1024] else 'frontend'))
 def run(job):
  name,kernel,dtype,block,mode=job;cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(work),name,kernel,dtype,str(block),mode]
  with (work/(name+'.log')).open('w') as log:
   try:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180);code=r.returncode
   except subprocess.TimeoutExpired:code=124
  row={'name':name,'command':cmd,'exit_code':code,'log':name+'.log','log_sha256':sha(work/(name+'.log'))}
  record=work/(name+'.json')
  if record.exists():row.update(result=json.loads(record.read_text()),record_sha256=sha(record))
  else:row['status']='process_failure'
  return row
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for row in pool.map(run,jobs):records.append(row);print(row['name'],row['exit_code'],row.get('result',{}).get('status','process_failure'),flush=True)
 import torch,triton,triton.runtime.jit as jit,triton.language.semantic as semantic,triton.language.core as core
 import triton.backends.nvidia.driver as driver
 for label,path in [('jit',jit.__file__),('semantic',semantic.__file__),('core',core.__file__),('driver',driver.__file__),('driver_c',Path(driver.__file__).with_suffix('.c'))]:
  (work/(label+'.txt')).write_bytes(Path(path).read_bytes())
 receipt={'task':'triton_mul2','source':str(SOURCE),'source_sha256':sha(SOURCE),'production_ast_unchanged':True,'torch_version':torch.__version__,'torch_commit':torch.version.git_version,'triton_version':triton.__version__,'records':records,'gpu_launches':0,'artifacts':{str(p.relative_to(work)):sha(p) for p in sorted(work.iterdir()) if p.is_file()}}
 p=work/'receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(p),'sha256':sha(p)}))
