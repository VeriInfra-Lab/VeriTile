"""Compile the actual repaired JIT binding on SM80/SM90; no CUDA execution."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
A=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-parallel-retention-lowering'));opts=parser.parse_args();D=opts.output_dir;D.mkdir(parents=True,exist_ok=True)
source=A/'parallel_retention_attention_repaired.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
spec=importlib.util.spec_from_file_location('retention_checked',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
for sm,B,H,T,DK,DV in [(80,1,1,1,1,1),(90,1,2,65,3,17),(80,1,2,129,32,64),(90,1,1,65,129,129)]:
 for mode in ['fwd','bwd']:
  name=f'sm{sm}_{mode}_{B}_{H}_{T}_{DK}_{DV}';os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name)
  fn=getattr(m,f'parallel_retention_{mode}_kernel');target=GPUTarget('cuda',sm,32);backend=CUDABackend(target)
  q=torch.empty(B,H,T,DK);v=torch.empty(B,H,T,DV);BK=min(128,max(16,triton.next_power_of_2(DK)));BV=min(128,max(16,triton.next_power_of_2(DV)));NK=triton.cdiv(DK,BK);NV=triton.cdiv(DV,BV)
  o=torch.empty(NK,B,H,T,DV);dq=torch.empty(NV,B,H,T,DK);args=[q,q,v,o] if mode=='fwd' else [q,q,v,v,dq,dq,o]
  args += [T*DK,DK,1,T*DV,DV,1,B,H,T,DK**-0.5]
  options={'num_warps':4,'num_stages':3 if DK<=64 else 2,'enable_fp_fusion':True}
  row={'name':name,'mode':mode,'sm':sm,'shape':[B,H,T,DK,DV],'source_sha256':sha(source)}
  try:
   binder=create_function_from_signature(fn.signature,fn.params,backend);bound,specialization,bopts=binder(*args,BTL=128 if mode=='fwd' else 64,BTS=32,BK=BK,BV=BV,DK=DK,DV=DV,**options)
   parsed,sig,const,attrs=fn._pack_args(backend,bopts,bound,specialization,bopts)
   code=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options);artifacts={}
   for kind in ['ttir','ttgir','llir','ptx']:
    path=D/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
   ptx=code.asm['ptx'];assert 'mma.sync' not in ptx and 'wgmma' not in ptx and 'tcgen05' not in ptx
   assert 'fma.rn.f32' in ptx and 'st.global' in ptx and 'ld.global' in ptx and 'ex2.approx.ftz.f32' in ptx
   row.update(status='compiled',signature=sig,constants={str(a):str(v) for a,v in const.items()},options=options,artifacts=artifacts)
  except Exception as ex:
   row.update(status='failed',error=str(ex));(D/(name+'.error.txt')).write_text(traceback.format_exc())
  rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
  (D/'report.json').write_text(json.dumps({'status':'PASS' if all(r['status']=='compiled' for r in rows) else 'FAIL','triton':triton.__version__,'actual_JIT_binder':True,'GPU_execution':False,'results':rows},indent=2)+'\n')
assert all(r['status']=='compiled' for r in rows)

# A supported singleton feature shape is a concrete original compile failure.
import ast,types
p=A.parents[2]/'tritonbench_g/parallel_retention_attention/parallel_retention_attention.py';tree=ast.parse(p.read_text(),str(p));tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
m=types.ModuleType('retention_original_control');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(tree,str(p),'exec'),m.__dict__)
fn=m.parallel_retention_fwd_kernel;backend=CUDABackend(GPUTarget('cuda',80,32));tensor=torch.empty(1,1,1,1);args=[tensor]*4+[1]*9+[1.0]
binder=create_function_from_signature(fn.signature,fn.params,backend)
bound,specialization,opts=binder(*args,BTL=128,BTS=32,BK=1,BV=1,DK=1,DV=1,num_warps=4,num_stages=3)
parsed,sig,const,attrs=fn._pack_args(backend,opts,bound,specialization,opts)
try:triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=GPUTarget('cuda',80,32),options={'num_warps':4,'num_stages':3})
except Exception as ex:
 text=str(ex);assert 'shape' in text.lower() or 'at least' in text.lower(),text
 (D/'original_minimum_tile_error.txt').write_text(text+'\n');print('original BK=BV=1 compilation failed as expected:',type(ex).__name__)
else:raise AssertionError('original unexpectedly compiled')

report=json.loads((D/'report.json').read_text())
report['original_minimum_tile_control']={'status':'failed_as_expected','path':str(D/'original_minimum_tile_error.txt'),'sha256':sha(D/'original_minimum_tile_error.txt')}
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
