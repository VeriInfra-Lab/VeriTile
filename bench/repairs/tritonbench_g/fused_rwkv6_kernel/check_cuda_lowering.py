"""Offline compilation through the actual JIT argument binder, with no GPU run."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
A=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-rwkv-lowering'));opts=parser.parse_args();D=opts.output_dir;D.mkdir(parents=True,exist_ok=True)
source=A/'fused_rwkv6_kernel_repaired.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();spec=importlib.util.spec_from_file_location('rwkv_checked_lowering',source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);rows=[]
for sm,B,H,T,K,V,initial,final,reverse in [(80,1,1,1,1,1,False,False,False),(90,2,3,4,8,8,True,True,False),(80,1,2,3,17,19,False,True,True),(90,1,2,4,65,37,True,True,True),(80,1,1,0,3,5,False,True,False),(90,1,1,0,3,5,True,True,True),(80,1,1,2,32,32,True,False,False),(90,1,1,2,33,65,False,False,False)]:
 name=f'sm{sm}_{B}_{H}_{T}_{K}_{V}_i{int(initial)}f{int(final)}r{int(reverse)}';os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name);target=GPUTarget('cuda',sm,32);backend=CUDABackend(target);fn=m.fused_recurrent_rwkv6_fwd_kernel
 q=torch.empty(B,H,T,K);v=torch.empty(B,H,T,V);u=torch.empty(H,K);BK=min(triton.next_power_of_2(K),32);BV=min(triton.next_power_of_2(V),32);o=torch.empty(triton.cdiv(K,BK),B,H,T,V);h0=torch.empty(B,H,K,V) if initial else None;ht=torch.empty(B,H,K,V) if final else None
 args=[q,q,v,q,u,o,h0,ht,T*K,T*V,0.5];options={'num_warps':1,'num_stages':1,'enable_fp_fusion':True};row={'name':name,'sm':sm,'shape':[B,H,T,K,V],'initial':initial,'final':final,'reverse':reverse,'source_sha256':sha(source)}
 try:
  binder=create_function_from_signature(fn.signature,fn.params,backend);bound,specialization,bopts=binder(*args,B=B,H=H,T=T,K=K,V=V,BK=BK,BV=BV,USE_INITIAL_STATE=initial,STORE_FINAL_STATE=final,REVERSE=reverse,**options)
  parsed,sig,const,attrs=fn._pack_args(backend,bopts,bound,specialization,bopts);code=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options);artifacts={}
  for kind in ['ttir','ttgir','llir','ptx']:
   path=D/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
  assert 'mma' not in code.asm['ptx'] and 'st.global' in code.asm['ptx']
  if T>1 or (T>0 and final):assert 'ex2.approx' in code.asm['ptx']
  row.update(status='compiled',signature=sig,constants={str(a):str(v) for a,v in const.items()},options=options,artifacts=artifacts)
 except Exception as ex:
  row.update(status='failed',error=str(ex));(D/(name+'.error.txt')).write_text(traceback.format_exc())
 rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
 (D/'report.json').write_text(json.dumps({'status':'PASS' if all(r['status']=='compiled' for r in rows) else 'FAIL','triton':triton.__version__,'actual_JIT_binder':True,'GPU_execution':False,'results':rows},indent=2)+'\n')
assert all(r['status']=='compiled' for r in rows)
