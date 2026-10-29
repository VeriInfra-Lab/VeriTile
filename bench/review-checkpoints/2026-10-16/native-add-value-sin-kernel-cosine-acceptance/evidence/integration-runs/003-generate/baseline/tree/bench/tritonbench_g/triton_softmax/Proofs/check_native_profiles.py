"""Recheck unchanged TritonBench-G softmax JIT profiles without a GPU launch."""
import ast, hashlib, importlib.util, json, os
from pathlib import Path
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton-softmax-native-cache'
import triton
from triton.compiler import ASTSource, compile
from triton.backends.compiler import GPUTarget
root=Path(__file__).resolve().parents[4]
source=root/'bench/tritonbench_g/triton_softmax/triton_softmax.py'
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='softmax_kernel')
module=ast.Module(body=[ast.Import(names=[ast.alias(name='triton')]), ast.Import(names=[ast.alias(name='triton.language',asname='tl')]),fn],type_ignores=[])
path=Path('/tmp/triton_softmax_native_kernel.py');path.write_text(ast.unparse(module)+'\n')
spec=importlib.util.spec_from_file_location('triton_softmax_native_kernel',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
profiles=[];floats=['fp16','bf16','fp32','fp64','fp8e4nv','fp8e5'];discrete=['i1','i8','i16','i32','i64','u8','u16','u32','u64']
for inp,out in [(a,b) for a in floats for b in floats]+[(a,a) for a in discrete]:
 row={'input':inp,'output':out,'signature':{'output_ptr':'*'+out,'input_ptr':'*'+inp,'input_row_stride':'i32','output_row_stride':'i32','n_cols':'i32'},'constexprs':{'BLOCK_SIZE':8}}
 try:
  result=compile(ASTSource(m.softmax_kernel,row['signature'],constexprs=row['constexprs']),target=GPUTarget('cuda',90,32),options={'num_warps':4})
  ttir=result.asm['ttir'];row.update(status='compiled',ttir_sha256=hashlib.sha256(ttir.encode()).hexdigest(),ptx_bytes=len(result.asm['ptx'].encode()),precision_lines=[l.strip() for l in ttir.splitlines() if any(s in l for s in ['tt.load','tt.store','tt.reduce','arith.extf','arith.truncf','math.exp','arith.divf'])])
 except Exception as exc:row.update(status='rejected',error=type(exc).__name__+': '+str(exc))
 profiles.append(row);print(inp,out,row['status'],flush=True)
receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'jit_ast_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'triton_version':triton.__version__,'target':{'backend':'cuda','arch':90,'warp_size':32},'scope':'Unchanged softmax_kernel AST. Offline compilation only at BLOCK_SIZE=8 and four warps, signed i32 runtime strides/count. No GPU execution, shape-general native admission, fixed-width overflow, IEEE theorem, host grid/race or independent Lean replay claim. Rejected discrete same-storage profiles document frontend failures, not executions.','results':profiles}
(root/'bench/tritonbench_g/triton_softmax/source_dtype_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
