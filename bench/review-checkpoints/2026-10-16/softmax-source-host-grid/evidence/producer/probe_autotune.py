"""Exact installed Autotuner/Heuristics execution, original decorated source AST.
A frontend-only JIT sink replaces GPU launching; benchmark repetition count and
costs are explicit test inputs. Ordinary CompilationError is not substituted.
"""
from pathlib import Path
import ast,hashlib,importlib.util,inspect,json,sys,types
sys.dont_write_bytecode=True
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
from triton.runtime.autotuner import Autotuner,Heuristics
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');OUT=Path(__file__).resolve().parent/'evidence'
S=ROOT/'bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py'
tree=ast.parse(S.read_text());nodes=[]
for n in tree.body:
 if isinstance(n,ast.ClassDef):break
 if isinstance(n,ast.FunctionDef):nodes.append(n)
preamble=ast.parse('import triton\nimport triton.language as tl\nMAX_TILE_K=8192\nNUM_SMS=80\n').body
module=ast.fix_missing_locations(ast.Module(body=preamble+nodes,type_ignores=[]))
path=OUT/'autotune_source.py';path.write_text(ast.unparse(module)+'\n')
selected_ast=ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
parsed_nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)]
assert selected_ast==ast.dump(ast.Module(body=parsed_nodes,type_ignores=[]),include_attributes=False)
spec=importlib.util.spec_from_file_location('softmax_exact_autotune_source',path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({});context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
rows=[]
for noninner,N in [(False,n) for n in [1,32,33,1024,1025]]+[(True,n) for n in [1,2,8,33]]:
 for storage in ['fp16','bf16','fp32','fp64']:
  name='softmax_backward_kernel_'+('non_inner' if noninner else 'inner')
  tuner=getattr(mod,name);assert isinstance(tuner,Autotuner) and isinstance(tuner.fn,Heuristics)
  jit=tuner.fn.fn;tile_key='TILE_K' if noninner else 'TILE_N'
  assert [c.kwargs[tile_key] for c in tuner.configs]==[32,64,128,256,512,1024]
  assert not tuner.reset_to_zero and not tuner.restore_value and not tuner.user_defined_pre_hook and not tuner.user_defined_post_hook
  assert tuner.prune_configs({})==tuner.configs
  M,K=3,17
  args=[types.SimpleNamespace(dtype=storage) for _ in range(3)]+[M,N]+([K] if noninner else [])
  key=tuple([M,N]+([K] if noninner else [])+[storage]*3)
  signature={'out_ptr':'*'+storage,'out_grad_ptr':'*'+storage,'in_grad_ptr':'*'+storage,'M':'i32','N':'i32'}
  if noninner:signature['K']='i32'
  for selected in [None,32,64,128,256,512,1024]:
   trace=[];cache={};tuner.cache.clear();tuner.cache_results=False
   if selected is not None:tuner.cache[key]=next(c for c in tuner.configs if c.kwargs[tile_key]==selected)
   def sink(*unused,**meta):
    const={k:v for k,v in meta.items() if k in ['TILE_N','TILE_M','TILE_K','ONE_TILE_PER_CTA']}
    entry={'tile':meta[tile_key],'constexprs':const};trace.append(entry)
    token=tuple(sorted(const.items()))
    try:
     if token not in cache:
      src=ASTSource(jit,signature,constexprs=const)
      ttir=str(src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context))
      cache[token]=hashlib.sha256(ttir.encode()).hexdigest()
     entry.update(status='frontend_generated',ttir_sha256=cache[token])
    except Exception as exc:
     entry.update(status='frontend_error',exception_type=type(exc).__name__,message=str(exc));raise
   jit.run=sink
   def bench(call,quantiles):
    call();call()
    return [float(2048-trace[-1]['tile'])]*3
   tuner._do_bench=bench
   result={'kernel':name,'M':M,'N':N,'K':K if noninner else 1,'storage':storage,'cache':'cold' if selected is None else 'hit','cached_tile':selected}
   try:tuner.run(*args,grid=(1,1,1));result['status']='frontend_route_completed'
   except Exception as exc:result.update(status='frontend_route_rejected',exception_type=type(exc).__name__)
   tiles=[e['tile'] for e in trace]
   candidates=[32,64,128,256,512,1024] if selected is None else [selected]
   first_bad=next((c for c in candidates if storage=='fp64' and N>(max(1,1024//c) if noninner else c)),None)
   if first_bad is None:
    assert result['status']=='frontend_route_completed'
    assert tiles==([c for c in candidates for _ in range(2)]+[1024] if selected is None else [selected])
   else:
    assert result['status']=='frontend_route_rejected' and result['exception_type']=='CompilationError'
    assert tiles==([c for c in candidates[:candidates.index(first_bad)] for _ in range(2)]+[first_bad] if selected is None else [selected])
    assert key not in tuner.cache if selected is None else True
   result.update(trace=trace,first_rejected_tile=first_bad,selected_final_tile=tuner.best_config.kwargs[tile_key] if result['status']=='frontend_route_completed' else None)
   rows.append(result)
(OUT/'autotune-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
runtime=Path(inspect.getfile(Autotuner));runtime_copy=OUT/'installed_autotuner.py.txt';runtime_copy.write_bytes(runtime.read_bytes())
receipt={'status':'passed','source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'source_function_and_decorator_ast_identical':True,'selected_ast_sha256':hashlib.sha256(selected_ast.encode()).hexdigest(),'runtime_path':str(runtime),'runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),'triton_version':triton.__version__,'cases':len(rows),'cold_cases':sum(r['cache']=='cold' for r in rows),'rejected':sum(r['status']=='frontend_route_rejected' for r in rows),'trial_order':[32,64,128,256,512,1024],'benchmark_calls_per_candidate':2,'benchmark_costs':'2048 - tile, deterministic synthetic timing','runtime_methods_executed':['Autotuner.run','Autotuner._bench','Autotuner.prune_configs','Heuristics.run'],'jit_sink':'original JIT body AST compiled by ASTSource.make_ir; actual CompilationError propagates through real runtime methods','cold_fp64_domains':{'inner':'N <= 32','noninner':'N <= 1'},'cache_hit_fp64_domain':'N <= selected TILE_N','native_backend_execution':False,'gpu_execution':False,'scope':'Runtime trial ordering, source configuration/heuristic wiring and compiler rejection. Synthetic timing and frontend-only JIT sink do not establish real timing, backend resource availability, or GPU value behavior.'}
(OUT/'autotune-probe-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
