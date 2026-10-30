"""Offline SM80/SM90 compilation through the actual JIT binder; no GPU run."""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('/tmp/veritile-fused-rotary-lowering'));opts=parser.parse_args()
A=Path(__file__).resolve().parent;D=opts.output_dir;D.mkdir(parents=True,exist_ok=True)
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
source=A/'fused_rotary_embedding_repaired.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[]
for sm,T,HQ,HK,W,B,X,new in [(80,1,1,1,2,1,2,False),(90,1,4,2,6,3,3,True),(80,3,6,2,16,4,16,False),(90,3,8,2,96,8,8,True),(80,2,8,2,128,16,128,False),(90,2,8,2,128,16,8,True),(80,2,8,2,256,16,16,True),(90,2,8,2,512,16,512,False)]:
 name=f'sm{sm}_T{T}_H{HQ}_{HK}_D{W}_B{B}_X{X}_new{int(new)}';spec=importlib.util.spec_from_file_location('rotary_'+name,source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 fn=m.decoding_fused_rotary_embedding_kernel;os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name);target=GPUTarget('cuda',sm,32);backend=CUDABackend(target)
 q=torch.empty(T,HQ,W);k=torch.empty(T,HK,W);v=torch.empty_like(k);c=torch.empty(T,W//2+1);s=torch.empty(T,W//2+2)
 kc=torch.empty((8,HK,W//X,B,X) if new else (8,HK,B,W));vc=torch.empty(8,HK,B,W);table=torch.empty(T,4,dtype=torch.int32);length=torch.empty(T,dtype=torch.int32)
 strides=kc.stride()[-3:] if new else (0,kc.stride(2),kc.stride(3))
 args=[q,k,v,c,s,kc,vc,table,length,X,*q.stride(),*k.stride(),*v.stride(),*c.stride(),*s.stride(),length.stride(0),kc.stride(0),kc.stride(1),*strides,*vc.stride(),*table.stride(),B]
 options={'num_warps':16 if W>=512 else 8 if W>=256 else 4,'enable_fp_fusion':True}
 row={'name':name,'sm':sm,'shape':[T,HQ,HK,W,B,X,new],'source_sha256':sha(source)}
 try:
  binder=create_function_from_signature(fn.signature,fn.params,backend)
  bound,specialization,bopts=binder(*args,KV_GROUP_NUM=HQ//HK,HEAD_DIM=W,BLOCK_HALF=triton.next_power_of_2(W//2),BLOCK_DIM=triton.next_power_of_2(W),**options)
  parsed,sig,const,attrs=fn._pack_args(backend,bopts,bound,specialization,bopts)
  code=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options);artifacts={}
  for kind in ['ttir','ttgir','llir','ptx']:
   path=D/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
  assert 'fma.rn.f32' in code.asm['ptx'] and 'mul.f32' in code.asm['ptx'] and 'ld.global' in code.asm['ptx'] and 'st.global' in code.asm['ptx']
  assert '.ftz.f32' not in code.asm['ptx']
  row.update(status='compiled',signature=sig,constants={fn.arg_names[a[0]]:str(v) for a,v in const.items()},options=options,artifacts=artifacts,metadata=code.metadata._asdict())
 except Exception as ex:
  row.update(status='failed',error=str(ex));(D/(name+'.error.txt')).write_text(traceback.format_exc())
 rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
 (D/'report.json').write_text(json.dumps({'status':'PASS' if all(r['status']=='compiled' for r in rows) else 'FAIL','triton':triton.__version__,'actual_JIT_binder':True,'GPU_execution':False,'results':rows},indent=2,default=str)+'\n')
assert all(r['status']=='compiled' for r in rows)
