"""Exact original JIT AST / finite frontend acceptance. No GPU or full compiler run."""
import ast,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path(__file__).resolve().parent/'evidence'
os.environ['TRITON_CACHE_DIR']=str(OUT/'triton-cache')
import triton,triton.language as tl
from triton.language.semantic import TritonSemantic
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
S=ROOT/'bench/tritonbench_g/matrix_transpose/matrix_transpose.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
node=next(n for n in ast.parse(S.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
lines=S.read_text().splitlines(keepends=True)
text='import triton\nimport triton.language as tl\n'+''.join(lines[min([node.lineno]+[x.lineno for x in node.decorator_list])-1:node.end_lineno])
extract=OUT/'extracted_kernel.py';extract.write_text(text)
assert ast.dump(node,include_attributes=False)==ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)),include_attributes=False)
spec=importlib.util.spec_from_file_location('matrix_acceptance_source',extract);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({})
context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
sig={'M':'*fp16','Out':'*fp16','matrix_stridex':'i32','matrix_stridey':'i32','out_stridex':'i32','out_stridey':'i32'}

def run(m,d):
 row={'SIZE_M':m,'D_HEAD':d}
 try:
  src=ASTSource(mod.kernel,sig,constexprs={'SIZE_M':m,'D_HEAD':d})
  module=src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context)
  irtext=str(module)
  row.update(status='frontend_generated',ttir_sha256=hashlib.sha256(irtext.encode()).hexdigest(),load_count=irtext.count('tt.load'),store_count=irtext.count('tt.store'),transpose_count=irtext.count('tt.trans'))
  assert row['load_count']==row['store_count']==row['transpose_count']==1
  if (m,d) in [(1,1),(16,16),(32,64),(1,2**20),(2**10,2**10),(2**20,1)]:
   p=OUT/f'source-{m}x{d}.ttir';p.write_text(irtext);row['ttir_path']=p.name
 except Exception as exc:
  row.update(status='frontend_error',exception_type=type(exc).__name__,message=str(exc))
 return row
rows=[run(2**a,2**b) for a in range(21) for b in range(21-a)]
assert len(rows)==231 and all(r['status']=='frontend_generated' for r in rows),[r for r in rows if r['status']!='frontend_generated']
# All individually legal exponent pairs, including the entire > tensor cap complement.
rejected=[run(2**a,2**b) for a in range(21) for b in range(21) if a+b>20]
assert len(rejected)==210 and all(r['status']=='frontend_error' and 'maximum tensor numel' in r['message'] and 'matrix_ptr =' in r['message'] for r in rejected)
boundaries=[(0,16),(16,0),(3,16),(16,3),(2**21,16),(16,2**21),(2**32,16),(16,2**32),(0,0),(3,3),(2**21,0),(0,2**32),(-1,16),(16,-1)]
rejected += [run(m,d) for m,d in boundaries]
assert all(r['status']=='frontend_error' for r in rejected)
(OUT/'frontend-success.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'frontend-rejections.json').write_text(json.dumps(rejected,indent=2)+'\n')
excerpts={'TritonSemantic.arange':inspect.getsource(TritonSemantic.arange),'validate_block_shape':inspect.getsource(tl.core.validate_block_shape),'TRITON_MAX_TENSOR_NUMEL':tl.core.TRITON_MAX_TENSOR_NUMEL,'block_type.__init__':inspect.getsource(tl.core.block_type.__init__),'broadcast_impl_shape':inspect.getsource(TritonSemantic.broadcast_impl_shape)}
(OUT/'trusted-frontend-primitives.json').write_text(json.dumps(excerpts,indent=2)+'\n')
r={'status':'passed','source_sha256':sha(S),'kernel_ast_identical':True,'source_ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'triton_version':triton.__version__,'primitive_files':{p:sha(p) for p in [inspect.getfile(TritonSemantic),tl.core.__file__]},'signature':sig,'target':vars(target),'options':vars(opts),'successful_pairs':231,'all_individually_legal_pairs':441,'product_limit_rejections':210,'additional_boundary_rejections':len(boundaries),'successful_domain':'m=2^a,d=2^b,a>=0,b>=0,a+b<=20','frontend_only':True,'native_backend_execution':False,'gpu_execution':False,'artifacts':{p.name:sha(p) for p in OUT.glob('*') if p.is_file()}}
(OUT/'frontend-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['status','successful_pairs','product_limit_rejections','additional_boundary_rejections']}))
