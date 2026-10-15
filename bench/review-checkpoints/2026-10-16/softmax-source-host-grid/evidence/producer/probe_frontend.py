from pathlib import Path
import ast,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');OUT=Path(__file__).resolve().parent/'evidence'
S=ROOT/'bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py'
names=['softmax_kernel_inner','softmax_kernel_non_inner','softmax_backward_kernel_inner','softmax_backward_kernel_non_inner','prev_multiple_of','next_multiple_of']
nodes=[n for n in ast.parse(S.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
bodies={n.name:hashlib.sha256(ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False).encode()).hexdigest() for n in nodes}
for n in nodes:n.decorator_list=[ast.Attribute(value=ast.Name(id='triton',ctx=ast.Load()),attr='jit',ctx=ast.Load())]
module=ast.fix_missing_locations(ast.Module(body=ast.parse('import triton\nimport triton.language as tl\n').body+nodes,type_ignores=[]))
extract=OUT/'source_jit_functions.py';extract.write_text(ast.unparse(module)+'\n')
actual=ast.parse(extract.read_text());assert bodies=={n.name:hashlib.sha256(ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False).encode()).hexdigest() for n in actual.body if isinstance(n,ast.FunctionDef)}
spec=importlib.util.spec_from_file_location('softmax_host_frontend_probe',extract);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({});context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
rows=[]
for name in names[:4]:
 back='backward' in name;noninner='non_inner' in name
 for inp in ['fp16','bf16','fp32','fp64']:
  for out in ([inp] if back else ['fp16','bf16','fp32','fp64']):
   for one in [False,True]:
    if back:sig={'out_ptr':'*'+inp,'out_grad_ptr':'*'+inp,'in_grad_ptr':'*'+inp,'M':'i32','N':'i32'}
    else:sig={'output_ptr':'*'+out,'input_ptr':'*'+inp,'M':'i32','N':'i32'}
    if noninner:sig['K']='i32'
    const={'TILE_N':32,'ONE_TILE_PER_CTA':one}
    if noninner:const['TILE_K']=32
    elif back:const['TILE_M']=32
    row={'kernel':name,'input':inp,'output':out,'one_tile':one,'signature':sig,'constexprs':const}
    try:
     src=ASTSource(getattr(mod,name),sig,constexprs=const)
     irtext=str(src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context))
     row.update(status='frontend_generated',ttir_sha256=hashlib.sha256(irtext.encode()).hexdigest())
    except Exception as exc:row.update(status='frontend_error',exception_type=type(exc).__name__,message=str(exc))
    expect=one or inp!='fp64'
    assert (row['status']=='frontend_generated')==expect,row
    rows.append(row)
(OUT/'frontend-profile-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
result={'status':'passed','source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'triton_version':triton.__version__,'function_body_ast_identical':True,'kernel_and_helper_body_sha256':bodies,'only_decorator_change':'heuristics/autotune wrappers removed; original functions remain @triton.jit and unchanged body AST. Host selection wrappers are checked separately by probe_host.py.','profiles':len(rows),'generated':sum(r['status']=='frontend_generated' for r in rows),'rejected':sum(r['status']=='frontend_error' for r in rows),'complete_declared_profile_partition':'one=true admits every modeled Storage; one=false rejects fp64 input/storage and admits fp16/bf16/fp32.','target':vars(target),'native_backend_execution':False,'gpu_execution':False}
(OUT/'frontend-profile-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
