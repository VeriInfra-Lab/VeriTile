from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
work=root/'native-attempts'/stamp;work.mkdir(parents=True)
os.environ['TRITON_CACHE_DIR']=str(work/'cache')
import triton,torch
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py')
original=source.read_text();prefix=original.split('################################################################')[0]
copied=work/'quantize_global.py';copied.write_text(prefix)
nodes=lambda s:[ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in ['_quantize_global_transpose','quantize_global_transpose']]
assert nodes(original)==nodes(prefix)
spec=importlib.util.spec_from_file_location('quantize_finite_'+stamp,copied);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
function=module._quantize_global_transpose.fn
(work/'probe_native.py').write_bytes(Path(__file__).read_bytes())
records=[]
for name,x,scale in [('bool','u1','fp32'),('u8','u8','fp32'),('i8','i8','fp32'),('i16','i16','fp32'),('i32','i32','fp32'),('i64','i64','fp32'),('fp16','fp16','fp16'),('bf16','bf16','bf16'),('fp32','fp32','fp32'),('fp32-i64-n','fp32','fp32'),('fp64','fp64','fp64'),('complex32','complex32','fp16'),('complex64','complex64','fp32'),('complex128','complex128','fp64')]:
 for ci,config in enumerate(module._quantize_global_transpose.configs):
  rec={'name':name,'signature':{'A':'*'+x,'absmax_inv_ptr':'*'+scale,'B':'*i8',**{k:('i64' if name=='fp32-i64-n' else 'i32') for k in ['stride_am','stride_an','stride_bn','stride_bm','M','N']}},'constexprs':config.kwargs,'options':{k:v for k,v in config.all_kwargs().items() if k not in config.kwargs},'target':{'backend':'cuda','arch':90,'warp_size':32}}
  rec['config_index']=ci
  tag=name+'-config'+str(ci)
  try:
   compiled=triton.compile(ASTSource(function,rec['signature'],constexprs=config.kwargs),target=GPUTarget('cuda',90,32),options=rec['options'])
   rec['artifacts']={}
   for stage in ['ttir','ttgir','llir','ptx']:
    artifact=work/(tag+'.'+stage);artifact.write_text(compiled.asm[stage]);rec['artifacts'][stage]={'path':str(artifact),'sha256':sha(artifact)}
   rec['status']='compiled'
   rec['arithmetic_ir']=[l.strip() for l in compiled.asm['ttir'].splitlines() if any(s in l for s in ['mulf','extf','truncf','sitofp','uitofp','llrint','trunci'])]
  except Exception:
   rec['status']='failed';rec['error']=traceback.format_exc()
   (work/(tag+'.log')).write_text(rec['error'])
  records.append(rec);print(name,config.kwargs,rec['status'],flush=True)
result={'torch_version':torch.__version__,'torch_commit':torch.version.git_version,'triton_version':triton.__version__,'source':str(source),'source_sha256':sha(source),'copied_prefix':str(copied),'copied_prefix_sha256':sha(copied),'production_ast_unchanged':True,'records':records,'gpu_launches':0}
r=work/'receipt.json';r.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'receipt':str(r),'sha256':sha(r)}))
