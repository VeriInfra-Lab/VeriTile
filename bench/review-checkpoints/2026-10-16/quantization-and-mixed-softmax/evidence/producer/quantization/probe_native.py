"""Compile original fp32 signatures offline; never launch a GPU kernel."""
from pathlib import Path
import ast, datetime, hashlib, importlib.util, json, os, sys, traceback
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
attempt=root/'native-attempts'/stamp
attempt.mkdir(parents=True)
os.environ['TRITON_CACHE_DIR']=str(attempt/'cache')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records=[]
for task,name,signature,constants in [
 ('quantize_global','_quantize_global',
  {'x_ptr':'*fp32','absmax_inv_ptr':'*fp32','output_ptr':'*i8','n_elements':'i32'},
  {'BLOCK_SIZE':1024}),
 ('quant_transpose_kernel','_quantize_global_transpose',
  {'A':'*fp32','absmax_inv_ptr':'*fp32','B':'*i8','stride_am':'i32','stride_an':'i32',
   'stride_bn':'i32','stride_bm':'i32','M':'i32','N':'i32'},
  {'BLOCK_M':128,'BLOCK_N':128,'GROUP_M':8}),
]:
 source=repo/f'bench/tritonbench_g/{task}/{task}.py'
 original=source.read_text()
 prefix=original.split('################################################################')[0]
 copied=attempt/f'{task}.py';copied.write_text(prefix)
 original_node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==name)
 copied_node=next(n for n in ast.parse(prefix).body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(original_node)==ast.dump(copied_node)
 record={'task':task,'original_source':str(source),'original_sha256':sha(source),
   'copied_prefix':str(copied),'copied_prefix_sha256':sha(copied),
   'original_kernel_ast_sha256':hashlib.sha256(ast.dump(original_node).encode()).hexdigest(),
   'kernel_ast_unchanged':True,'signature':signature,'constexprs':constants,
   'target':{'backend':'cuda','arch':90,'warp_size':32},'options':{'num_warps':4,'enable_fp_fusion':False},
   'triton_version':triton.__version__,'gpu_launches':0}
 try:
  spec=importlib.util.spec_from_file_location('quantize_stage_probe_'+task,copied)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  function=getattr(module,name).fn
  compiled=triton.compile(ASTSource(function,signature,constexprs=constants),
      target=GPUTarget('cuda',90,32),options=record['options'])
  for stage in ('ttir','ttgir','llir','ptx'):
   artifact=attempt/f'{task}.{stage}';artifact.write_text(compiled.asm[stage])
   record[stage]={'path':str(artifact),'sha256':sha(artifact)}
  ttir=compiled.asm['ttir']
  record['llrintf_present']='__nv_llrintf' in ttir
  record['int64_to_int8_truncation_present']='arith.trunci' in ttir and 'i64' in ttir and 'i8' in ttir
  record['fp32_multiplies']=ttir.count('arith.mulf')
  assert record['llrintf_present'] and record['int64_to_int8_truncation_present'] and record['fp32_multiplies']==2
  record['status']='compiled'
 except Exception:
  record['status']='failed';record['error']=traceback.format_exc()
 records.append(record)
receipt={'kind':'original_fp32_offline_admission','records':records,
 'scope':'Two selected original kernel signatures compiled offline. No host wrapper, launch, runtime numerical conformance, or whole-program acceptance.',
 'gpu_launches':0,'whole_original_program_accepted':0,
 'status':'passed' if all(r['status']=='compiled' for r in records) else 'failed'}
out=attempt/'receipt.json';out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'status':receipt['status'],'records':[{k:r[k] for k in ('task','status')} for r in records]}))
for r in records:
 if 'error' in r:print(r['error'])
sys.exit(receipt['status']!='passed')
