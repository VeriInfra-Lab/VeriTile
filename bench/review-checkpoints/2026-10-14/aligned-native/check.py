import ast,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path(__file__).resolve().parent
source=ROOT/'bench/tritonbench_g/attention_kernel_aligned/attention_kernel_aligned.py'
raw=source.read_bytes();tree=ast.parse(raw);lines=raw.decode().splitlines(True)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fwd_kernel_aligned')
code='import triton\nimport triton.language as tl\n'+''.join(lines[node.decorator_list[0].lineno-1:node.end_lineno])
assert ast.dump(ast.parse(code).body[-1],include_attributes=False)==ast.dump(node,include_attributes=False)
(OUT/'original.py').write_text(code);(OUT/'source.py').write_bytes(raw)
os.environ['TRITON_CACHE_DIR']=str(OUT/'cache')
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
spec=importlib.util.spec_from_file_location('aligned_source',OUT/'original.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
results=[]
for dtype,kind in [('fp16',tl.float16),('bf16',tl.bfloat16)]:
 dest=OUT/dtype;dest.mkdir(exist_ok=True)
 sig={n:'*'+dtype for n in ['Q','K','V','B0','Out']};sig['sm_scale']='fp32'
 for n in ['stride_qh','stride_qm','stride_qk','stride_kh','stride_kn','stride_kk','stride_vh','stride_vk','stride_vn','stride_oh','stride_om','stride_on','stride_b0h','stride_b0m','Z','H','N_CTX','P_SEQ']:sig[n]='i32'
 const={'OUT_DTYPE':kind,'BIAS_LAST_SIZE':64,'B0_NUMEL':128,'BLOCK_DMODEL':64,'BLOCK_M':64,'BLOCK_N':64}
 options={'num_warps':4,'num_stages':2}
 result={'dtype':dtype,'signature':sig,'constexprs':{k:str(v) for k,v in const.items()},'options':options}
 try:
  kernel=triton.compile(ASTSource(mod._fwd_kernel_aligned,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options=options)
  for suffix,data in kernel.asm.items():(dest/('kernel.'+suffix)).write_bytes(data if isinstance(data,bytes) else data.encode())
  result['compiled']=True
 except Exception as exc:
  result.update(compiled=False,error=str(exc));(dest/'error.txt').write_text(str(exc)+'\n')
 results.append(result)
receipt={'source_sha256':hashlib.sha256(raw).hexdigest(),'triton_version':triton.__version__,'target':'cuda sm90','original_kernel_ast_unchanged':True,'kernel_executed':False,'results':results}
receipt['artifact_sha256']={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and 'cache' not in p.relative_to(OUT).parts and p.name!='receipt.json'}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['results']))
