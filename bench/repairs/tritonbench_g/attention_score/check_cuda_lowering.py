from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-attention-score-lowering'));opts=parser.parse_args()
A=Path(__file__).resolve().parent;D=opts.output_dir;D.mkdir(parents=True,exist_ok=True)
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
source=A/'attention_score_repaired.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=GPUTarget('cuda',80,32);backend=CUDABackend(target);rows=[]
for B,N,NK,DIM,window,comp in [(16,1,1,16,False,False),(16,1,1,16,True,False),(16,1,1,16,True,True),(16,17,16,32,True,False),(16,16,17,64,True,True),(32,31,33,16,False,False),(32,33,32,64,True,True),(64,64,64,128,True,False),(64,65,64,16,False,False),(64,65,65,64,True,True),(64,129,97,128,True,False)]:
 name=f'B{B}_N{N}_K{NK}_D{DIM}_W{int(window)}_C{int(comp)}';spec=importlib.util.spec_from_file_location('score_'+name,source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 fn=m._score_kernel.fn;binder=create_function_from_signature(fn.signature,fn.params,backend);Z,H,HK=2,4,2;R=N+7
 q=torch.empty(Z,H,N,DIM);k=torch.empty(Z,HK,NK,DIM);norm=torch.empty(Z,H,R);out=torch.empty(Z,H,NK)
 args=[q,k,norm,1/(DIM**0.5),out,*q.stride(),*k.stride(),*out.stride(),Z,H,HK,N,R,NK,(1 if N==1 else 2147483647) if window else None,(1 if N==1 else 17) if window else None]
 options={'num_warps':4,'num_stages':3,'enable_fp_fusion':True}
 bound,specialization,binder_options=binder(*args,SLIDING_WINDOW=window,COMPLEMENT_SLIDING_WINDOW=comp,IS_EVEN_M=N%B==0,IS_EVEN_N=NK%B==0,BLOCK_M=B,BLOCK_N=B,BLOCK_DMODEL=DIM,**options)
 parsed,sig,const,attrs=fn._pack_args(backend,binder_options,bound,specialization,binder_options)
 row={'name':name,'source_sha256':sha(source),'shape':[Z,H,HK,N,R,NK,DIM],'block':B,'window':window,'complement':comp,'signature':sig,'constants':{fn.arg_names[a[0]]:str(v) for a,v in const.items()},'options':options}
 os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name)
 try:
  code=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options);artifacts={}
  for kind in ['ttir','ttgir','llir','ptx']:
   path=D/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
  ir=code.asm['ttir'];ptx=code.asm['ptx']
  assert 'inputPrecision = tf32' in ir and 'xf32>' in ir and 'math.exp2' in ir
  assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
  assert 'ex2.approx.ftz.f32' in ptx and 'st.global' in ptx
  if window:assert 'xi64>' in ir
  row.update(status='compiled',artifacts=artifacts,metadata=code.metadata._asdict())
 except Exception as ex:
  row.update(status='failed',error=str(ex));(D/(name+'.error.txt')).write_text(traceback.format_exc())
 rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
 (D/'report.json').write_text(json.dumps({'status':'PASS' if all(r['status']=='compiled' for r in rows) else 'FAIL','triton':triton.__version__,'actual_JIT_binder':True,'GPU_execution':False,'target':dataclasses.asdict(target),'results':rows},indent=2,default=str)+'\n')
assert all(r['status']=='compiled' for r in rows)
