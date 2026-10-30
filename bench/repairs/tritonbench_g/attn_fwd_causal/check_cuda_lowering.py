"""Offline SM80 compilation with actual JIT specialization; no GPU execution."""
from pathlib import Path
import argparse,ast,dataclasses,hashlib,importlib.util,json,os,sys,traceback,types
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-attn-causal-lowering'));opts=parser.parse_args()
A=Path(__file__).resolve().parent;D=opts.output_dir;D.mkdir(parents=True,exist_ok=True)
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
source=A/'attn_fwd_causal_repaired.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=GPUTarget('cuda',80,32);backend=CUDABackend(target);rows=[]
def compile_case(fn,B,H,N,offset=0):
 q=torch.empty(B*H*N*128+offset,dtype=torch.float32)[offset:].view(B,H,N,128);k=torch.empty_like(q);v=torch.empty_like(q)
 qs=torch.empty(B,H,triton.cdiv(N,128));ks=torch.empty(B,H,triton.cdiv(N,64));out=torch.empty_like(q,dtype=torch.float16)
 args=[q,k,v,qs,ks,out,*q.stride(),*k.stride(),*v.stride(),*out.stride(),B,H]
 options={'num_warps':8,'num_stages':2,'enable_fp_fusion':True}
 binder=create_function_from_signature(fn.signature,fn.params,backend)
 bound,specialization,bopts=binder(*args,N_CTX=N,HEAD_DIM=128,BLOCK_M=128,BLOCK_N=64,STAGE=3,**options)
 parsed,sig,const,attrs=fn._pack_args(backend,bopts,bound,specialization,bopts)
 return triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options),sig,const,options
for B,H,N,offset in [(1,1,1,0),(1,1,63,0),(1,1,64,0),(1,1,65,0),(1,1,128,0),(1,1,129,1),(2,3,257,0),(16,16,256,0)]:
 name=f'B{B}_H{H}_N{N}_offset{offset}';spec=importlib.util.spec_from_file_location('causal_'+name,source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 fn=m._attn_fwd;os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name);row={'name':name,'shape':[B,H,N,128],'source_sha256':sha(source)}
 try:
  code,sig,const,options=compile_case(fn,B,H,N,offset);artifacts={}
  for kind in ['ttir','ttgir','llir','ptx']:
   path=D/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
  ir=code.asm['ttir'];ptx=code.asm['ptx']
  assert 'inputPrecision = tf32' in ir and 'math.exp2' in ir
  assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
  assert 'mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16' in ptx
  assert 'ex2.approx.ftz.f32' in ptx and ('cvt.rn.f16.f32' in ptx or 'cvt.rn.f16x2.f32' in ptx)
  assert 'st.global' in ptx and code.metadata.shared<=163840
  row.update(status='compiled',signature=sig,constants={fn.arg_names[a[0]]:str(v) for a,v in const.items()},options=options,artifacts=artifacts,metadata=code.metadata._asdict())
 except Exception as ex:
  row.update(status='failed',error=str(ex));(D/(name+'.error.txt')).write_text(traceback.format_exc())
 rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
 (D/'report.json').write_text(json.dumps({'status':'PASS' if all(r['status']=='compiled' for r in rows) else 'FAIL','triton':triton.__version__,'actual_JIT_binder':True,'GPU_execution':False,'target':dataclasses.asdict(target),'results':rows},indent=2,default=str)+'\n')
assert all(r['status']=='compiled' for r in rows)
original=A.parents[2]/'tritonbench_g/attn_fwd_causal/attn_fwd_causal.py'
tree=ast.parse(original.read_text(),str(original));tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
m=types.ModuleType('original_causal_cuda');m.__file__=str(original);sys.modules[m.__name__]=m;exec(compile(tree,str(original),'exec'),m.__dict__)
try:compile_case(m._attn_fwd,1,1,128)
except Exception as ex:
 assert 'same dtype' in str(ex) or 'dtype' in str(ex),(type(ex),str(ex));(D/'original_fp32_error.txt').write_text(str(ex)+'\n')
 print('original FP32 mixed-PV compilation rejected as expected')
else:raise AssertionError('original FP32 unexpectedly compiled')
