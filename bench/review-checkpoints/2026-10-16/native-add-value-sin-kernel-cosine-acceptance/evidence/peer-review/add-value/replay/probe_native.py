"""Unchanged annotated source, actual JIT binder, isolated offline CUDA compiles."""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,struct,subprocess,sys,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/add_value/add_value.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['bool','uint8','int8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
if len(sys.argv)>1 and sys.argv[1]=='worker':
 work=Path(sys.argv[2]);name=sys.argv[3]
 os.environ['TRITON_CACHE_DIR']=str(work/'cache'/name)
 import torch,triton
 import triton.runtime.jit as jit
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 from triton.backends.nvidia.compiler import CUDABackend
 from triton._C.libtriton import ir
 spec=importlib.util.spec_from_file_location('add_value_source',work/'source.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 kernel=module.puzzle1_kernel;target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
 row={'dtype':name,'target':{'backend':'cuda','arch':90,'warp_size':32},'value_python':10,'N_python':2049}
 stage='signature'
 try:
  dtype=getattr(torch,name)
  binder=jit.create_function_from_signature(kernel.signature,kernel.params,backend)
  params,specialization,options=binder(jit.MockTensor(dtype),jit.MockTensor(dtype),2049,BLOCK_SIZE=1024,value=10)
  options,signature,constexprs,attrs=kernel._pack_args(backend,options,params,specialization,options)
  row.update(signature=signature,specialization=specialization,constexprs={str(k):v for k,v in constexprs.items()},attrs={str(k):v for k,v in attrs.items()},options=vars(options),parameters=[{'name':p.name,'annotation':p.annotation,'annotation_type':p.annotation_type,'is_constexpr':p.is_constexpr} for p in kernel.params])
  assert signature['value']=='fp32' and specialization[-1]==('fp32',None)
  assert constexprs=={(3,):1024}
  stage='frontend'
  src=ASTSource(kernel,signature,constexprs,attrs)
  context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
  raw=src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
  (work/(name+'.raw.ttir')).write_text(str(raw))
  stage='backend'
  compiled=triton.compile(src,target=target,options=vars(options))
  for ext in ['ttir','ttgir','llir','ptx']:(work/(name+'.'+ext)).write_text(compiled.asm[ext])
  row.update(status='compiled',value_word_fp32=struct.unpack('<I',struct.pack('<f',10))[0])
 except Exception:
  row.update(status='rejected',failure_stage=stage,exception=traceback.format_exc())
 (work/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n')
 print(json.dumps(row))
else:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 work=ROOT/'native-attempts'/stamp;work.mkdir(parents=True)
 original=SOURCE.read_text();prefix=original.split('################################################################')[0]
 nodes=lambda s:[ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in ['puzzle1_kernel','puzzle1']]
 assert nodes(original)==nodes(prefix)
 (work/'source.py').write_text(prefix);(work/'probe_native.py').write_bytes(Path(__file__).read_bytes())
 records=[]
 for name in names:
  cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(work),name]
  with (work/(name+'.log')).open('w') as log:
   result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  row={'dtype':name,'command':cmd,'exit_code':result.returncode,'log':name+'.log','log_sha256':sha(work/(name+'.log'))}
  record=work/(name+'.json')
  if record.exists():row.update(result=json.loads(record.read_text()),record_sha256=sha(record))
  else:row.update(status='process_failure')
  records.append(row);print(name,result.returncode,row.get('result',{}).get('status','process_failure'),flush=True)
 import torch,triton,triton.runtime.jit as jit,triton.language.semantic as semantic
 import triton.backends.nvidia.driver as driver
 for label,source in [('jit',jit.__file__),('semantic',semantic.__file__),('driver',driver.__file__)]:
  (work/(label+'.py.txt')).write_bytes(Path(source).read_bytes())
 result={'task':'add_value','source':str(SOURCE),'source_sha256':sha(SOURCE),'production_ast_unchanged':True,'torch_version':torch.__version__,'torch_commit':torch.version.git_version,'triton_version':triton.__version__,'records':records,'gpu_launches':0,'artifacts':{str(p.relative_to(work)):sha(p) for p in sorted(work.iterdir()) if p.is_file()}}
 receipt=work/'receipt.json';receipt.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'receipt':str(receipt),'sha256':sha(receipt)}))
