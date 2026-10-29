#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,sys,types,os,traceback
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/token_softmax_bloom/token_softmax_bloom.py')
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
 data=SOURCE.read_bytes();text=data.decode();fn=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name=='_fwd_kernel_token_softmax')
 assert len(fn.decorator_list)==1 and ast.unparse(fn.decorator_list[0])=='triton.jit'
 segment=''.join(text.splitlines(keepends=True)[fn.decorator_list[0].lineno-1:fn.end_lineno])
 generated=('import triton\nimport triton.language as tl\n\n'+segment).encode();(out/'original.py').write_bytes(data);(out/'original_jit.py').write_bytes(generated)
 gen=next(x for x in ast.parse(generated).body if isinstance(x,ast.FunctionDef));assert ast.dump(gen,include_attributes=False)==ast.dump(fn,include_attributes=False)
 (out/'host.py').write_text(''.join(text.splitlines(keepends=True)[38:57]))
 dump(out/'input-binding.json',{'path':str(SOURCE),'source_sha256':sha(SOURCE),'jit_sha256':sha(out/'original_jit.py'),'function_AST_equal':True,'first_line':fn.decorator_list[0].lineno,'last_line':fn.end_lineno,'signature_names':[x.arg for x in fn.args.args],'profile_membership':'width 32 / 2048 / 4096; host warp branches 4 / 8 / 16; input/output storage matched; metadata int32; all runtime strides signed32. Source host permits further profiles; these probes do not bound theorem inputs.'})
 import triton
 from triton.compiler import ASTSource
 from triton.backends.compiler import GPUTarget
 before=environment();dump(out/'environment-before.json',before)
 module=types.ModuleType('softmax_bloom_checked_source');module.__file__=str(out/'original_jit.py');sys.modules[module.__name__]=module;exec(compile(generated,module.__file__,'exec'),module.__dict__);jit=module._fwd_kernel_token_softmax
 assert jit.src==segment.split('@triton.jit\n',1)[1]
 rows=[]
 for arch in [80,90]:
  for dtype in ['fp16','bf16','fp32']:
   for width,warps in [(32,4),(2048,8),(4096,16)]:
    for fusion in [False,True]:
     label=f'sm{arch}-{dtype}-b{width}-w{warps}-fusion{int(fusion)}';folder=out/label;folder.mkdir()
     signature={x:('*'+dtype if x in ['Logics','Prob_Out'] else '*i32' if x in ['B_Start_Loc','B_Seqlen'] else 'i32') for x in jit.arg_names if x!='BLOCK_SIZE'}
     options={'num_warps':warps,'num_stages':3,'enable_fp_fusion':fusion};row={'id':label,'arch':arch,'signature':signature,'constexprs':{'BLOCK_SIZE':width},'options':options,'native_execution':False}
     try:
      k=triton.compile(ASTSource(jit,signature=signature,constexprs={'BLOCK_SIZE':width}),target=GPUTarget('cuda',arch,32),options=options)
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
