#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,sys,types,os,traceback
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/logsumexp_fwd/logsumexp_fwd.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def serial(x):
 if x is None or isinstance(x,(str,int,float,bool)):return x
 if hasattr(x,'_asdict'):return serial(x._asdict())
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return list(map(serial,x))
 return repr(x)
def environment():
 import triton,triton.backends.nvidia.compiler as nc
 base=Path(triton.__file__).resolve().parent
 files={p.resolve() for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 files.add(Path(sys.executable).resolve())
 tools={str(a):str(Path(nc.get_ptxas(a).path).resolve()) for a in [80,90]}
 files.update(map(Path,tools.values()))
 libdevice=Path(nc.knobs.nvidia.libdevice_path or base/'backends/nvidia/lib/libdevice.10.bc').resolve();files.add(libdevice)
 return {'triton':triton.__version__,'python':sys.version,'ptxas':tools,'libdevice':str(libdevice),'file_sha256':{str(p):sha(p) for p in sorted(files)}}
def main():
 out=ROOT/'native';out.mkdir(exist_ok=False);os.environ['TRITON_CACHE_DIR']=str(out/'fresh-cache');sys.pycache_prefix=str(out/'pycache')
 data=SOURCE.read_bytes();text=data.decode();fn=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='logsumexp_fwd_kernel')
 assert len(fn.decorator_list)==3 and ast.unparse(fn.decorator_list[-1])=='triton.jit'
 segment=''.join(text.splitlines(keepends=True)[fn.decorator_list[-1].lineno-1:fn.end_lineno])
 decorated=''.join(text.splitlines(keepends=True)[fn.decorator_list[0].lineno-1:fn.end_lineno])
 generated=('import triton\nimport triton.language as tl\n\n'+segment).encode();(out/'original.py').write_bytes(data);(out/'original_jit.py').write_bytes(generated);(out/'original_decorated.txt').write_text(decorated)
 gen=next(x for x in ast.parse(generated).body if isinstance(x,ast.FunctionDef));body=ast.FunctionDef(name=fn.name,args=fn.args,body=fn.body,decorator_list=[fn.decorator_list[-1]],returns=fn.returns,type_comment=fn.type_comment)
 assert ast.dump(gen,include_attributes=False)==ast.dump(body,include_attributes=False)
 cfg=fn.decorator_list[0];assert isinstance(cfg,ast.Call) and ast.unparse(cfg.func)=='triton.autotune'
 configs=next(x.value for x in cfg.keywords if x.arg=='configs');warps=[]
 for config in configs.elts:
  assert ast.unparse(config.func)=='triton.Config' and ast.literal_eval(config.args[0])=={}
  assert len(config.keywords)==1 and config.keywords[0].arg=='num_warps';warps.append(ast.literal_eval(config.keywords[0].value))
 assert warps==[1,2,4,8,16,32]
 host=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='logsumexp_fwd');(out/'host-exact.py').write_text(''.join(text.splitlines(keepends=True)[host.lineno-1:host.end_lineno]))
 dump(out/'input-binding.json',{'path':str(SOURCE),'source_sha256':sha(SOURCE),'jit_sha256':sha(out/'original_jit.py'),'function_body_AST_equal':True,'jit_decorator_retained':True,'autotune_and_heuristics_bypassed_for_exact_profile_selection':True,'original_autotune_warps':warps,'first_line':fn.decorator_list[-1].lineno,'last_line':fn.end_lineno,'signature_names':[x.arg for x in fn.args.args],'profile_membership':'All six original autotune num_warps choices; D=17/B=32 satisfies min(next_power_of_2(D),65536). HAS_SCALE false uses constexpr None, true runtime fp32 scale. Explicit num_stages=3 captured default; fusion variants are characterization, not host source options. No restriction of theorem D/B/pids/scale domain.'})
 import triton
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 before=environment();dump(out/'environment-before.json',before)
 module=types.ModuleType('logsumexp_checked_source');module.__file__=str(out/'original_jit.py');sys.modules[module.__name__]=module;exec(compile(generated,module.__file__,'exec'),module.__dict__);jit=module.logsumexp_fwd_kernel
 assert jit.src==segment.split('@triton.jit\n',1)[1]
 rows=[]
 for arch in [80,90]:
  for warps in [1,2,4,8,16,32]:
   for scaled in [False,True]:
    for fusion in [False,True]:
     label=f'sm{arch}-fp32-d17-b32-w{warps}-scale{int(scaled)}-fusion{int(fusion)}';folder=out/label;folder.mkdir()
     signature={'x':'*fp32','z':'*fp32'};constants={'D':17,'B':32,'HAS_SCALE':scaled}
     if scaled:signature['scale']='fp32'
     else:constants['scale']=None
     options={'num_warps':warps,'num_stages':3,'enable_fp_fusion':fusion};row={'id':label,'arch':arch,'signature':signature,'constexprs':constants,'options':options,'native_execution':False}
     try:
      k=triton.compile(ASTSource(jit,signature=signature,constexprs=constants),target=GPUTarget('cuda',arch,32),options=options)
      artifacts={}
      for name,body in k.asm.items():
       if not isinstance(body,(str,bytes)):continue
       p=folder/('kernel.'+name);p.write_bytes(body.encode() if isinstance(body,str) else body);artifacts[p.relative_to(out).as_posix()]=sha(p)
      assert all(label+'/kernel.'+x in artifacts for x in ['ttir','ttgir','llir','ptx','cubin'])
      row.update(status='COMPILED_ONLY',metadata=serial(k.metadata),artifact_sha256=artifacts)
     except Exception as e:
      (folder/'error.txt').write_text(traceback.format_exc());row.update(status='COMPILE_ERROR',error=repr(e))
     rows.append(row);dump(out/'running-results.json',rows);print(label,row['status'],flush=True)
 after=environment();dump(out/'environment-after.json',after);assert before==after and SOURCE.read_bytes()==data
 dump(out/'receipt.json',{'schema':1,'source_sha256':sha(SOURCE),'input_binding_sha256':sha(out/'input-binding.json'),'records':rows,'compiled':sum(r['status']=='COMPILED_ONLY' for r in rows),'native_launches':0,'environment_stable':True,'script_sha256':sha(__file__)})
 assert all(x['status']=='COMPILED_ONLY' for x in rows)
if __name__=='__main__':main()
