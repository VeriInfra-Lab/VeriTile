"""Offline current-source SM90/SM80 profiles, using actual JIT argument binding."""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-lightning-lowering'));opts=p.parse_args()
A=Path(__file__).resolve().parent;D=opts.output_dir;D.mkdir(parents=True,exist_ok=True)
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
source=A/'lightning_attention_repaired.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
for arch,B,H,N,K,E,offset in [(90,1,1,1,16,16,0),(90,1,1,64,16,32,0),(90,1,1,65,32,64,0),(90,2,3,129,16,32,1),(90,2,8,128,64,128,0),(80,1,1,65,16,32,0)]:
 target=GPUTarget('cuda',arch,32);backend=CUDABackend(target)
 for stage,fnname in [('forward','_fwd_kernel'),('intra','_bwd_intra_kernel'),('inter','_bwd_inter_kernel')]:
  name=f'SM{arch}_B{B}_H{H}_N{N}_D{K}_E{E}_offset{offset}_{stage}'
  spec=importlib.util.spec_from_file_location('lightning_'+name,source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);fn=getattr(m,fnname)
  q=torch.empty(B*H*N*K+offset,dtype=torch.float32)[offset:].view(B,H,N,K);k=torch.empty_like(q);v=torch.empty(B,H,N,E)
  o=torch.empty_like(v);g=torch.empty_like(v);dq=torch.empty_like(q);dk=torch.empty_like(k);dv=torch.empty_like(v)
  tensors=[q,k,v,o] if stage=='forward' else [q,k,v,g,dq,dk,dv]
  kwargs={'BLOCK':64,'NUM_BLOCK':triton.cdiv(N,64)}
  if stage=='forward':kwargs['BLOCK_MODEL']=min(E,32)
  else:kwargs.update(CBLOCK=32,NUM_CBLOCK=2)
  options={'num_warps':4,'num_stages':3,'enable_fp_fusion':True};binder=create_function_from_signature(fn.signature,fn.params,backend)
  bound,specialization,bopts=binder(*tensors,B,H,N,K,E,**kwargs,**options)
  parsed,sig,const,attrs=fn._pack_args(backend,bopts,bound,specialization,bopts)
  os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name);row={'name':name,'source_sha256':sha(source),'shape':[B,H,N,K,E],'stage':stage,'target':dataclasses.asdict(target),'signature':sig,'constants':{fn.arg_names[a[0]]:str(x) for a,x in const.items()},'options':options}
  try:
   code=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options);artifacts={}
   for kind in ['ttir','ttgir','llir','ptx']:
    path=D/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
   ptx=code.asm['ptx'];ir=code.asm['ttir']
   if stage!='inter' or N>64:
    assert 'inputPrecision = tf32' in ir and ('wgmma.mma_async' in ptx or 'mma.sync.aligned.m16n8k8' in ptx)
    assert 'st.global' in ptx and 'xf16>' not in ir
   row.update(status='compiled',metadata=code.metadata._asdict(),artifacts=artifacts)
  except Exception as ex:
   row.update(status='failed',error=str(ex));(D/(name+'.error.txt')).write_text(traceback.format_exc())
  rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
  (D/'report.json').write_text(json.dumps({'status':'PASS' if all(r['status']=='compiled' for r in rows) else 'FAIL','triton':triton.__version__,'actual_JIT_binder':True,'GPU_execution':False,'results':rows},indent=2,default=str)+'\n')
assert all(r['status']=='compiled' for r in rows)
