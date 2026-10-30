"""Offline SM80 compilation of every original block profile; no GPU run."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import triton
from triton import language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import native_specialize_impl
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(args.output/'cache')
src=Path(__file__).resolve().parents[1]/'Whole001/streamk_matmul_repaired.py'
spec=importlib.util.spec_from_file_location('streamk_compile',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
target=GPUTarget('cuda',80,32);backend=CUDABackend(target);records=[]
profiles=[(16,128,128,32),(32,64,64,16),(8,32,32,32),(4,256,256,32)]
for profile,(P,BM,BN,BK) in enumerate(profiles):
 for wide in (False,True) if profile==2 else (False,):
  M,K,N=(BM*3+1,BK*3+1,BN*2+1) if not wide else (32769,2**30+1,32769)
  if wide:P=65536
  tiles=triton.cdiv(M,BM)*triton.cdiv(N,BN);I=triton.cdiv(K,BK)
  stream=tiles%P
  if tiles-stream>P:stream+=P
  J=stream*I
  scalars=dict(M=M,N=N,K=K,stride_am=K,stride_ak=1,stride_bk=N,stride_bn=1,stride_cm=N,stride_cn=1,
    total_programs_streamk=P,total_full_tiles_streamk=J//P,total_partial_tiles_streamk=J%P,
    iters_per_tile=I,total_tiles_streamk=stream)
  for name in ('first_wave','reduce_streamk','full_tiles'):
   fn=getattr(mod,name);out=args.output/f'{profile}-{int(wide)}-{name}';out.mkdir(exist_ok=True)
   const=dict(BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,ACC_TYPE=tl.float32)
   const={k:v for k,v in const.items() if k in fn.arg_names}
   sig={};attrs={};specialization={}
   for i,key in enumerate(fn.arg_names):
    if key in ('A','B','C','partials'):sig[key]='*fp32';attrs[(i,)]=backend.parse_attr('D')
    elif key in const:sig[key]='constexpr'
    else:
     kind,attr=native_specialize_impl(backend,scalars[key],False,True,True)
     specialization[key]=[kind,attr];sig[key]=kind
     if kind=='constexpr':const[key]=attr
     elif attr:attrs[(i,)]=backend.parse_attr(attr)
   compiled=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=dict(num_warps=4,num_stages=3))
   artifacts={}
   for ext in ('ttir','ttgir','llir','ptx'):
    path=out/f'kernel.{ext}';path.write_text(compiled.asm[ext]);artifacts[ext]=hashlib.sha256(path.read_bytes()).hexdigest()
   ir,ptx=compiled.asm['ttir'],compiled.asm['ptx']
   assert 'i64' in ir and 'tt.addptr' in ir
   adds=[line.strip() for line in ir.splitlines() if '= tt.addptr' in line]
   assert all('xi64>' in line or ', i64' in line for line in adds),adds
   if name!='reduce_streamk':
    assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
    assert 'tf32.f32' in ptx
    assert 'arith.addf' not in ir
    assert 'tt.dot' in ir
    assert '%acc_' in next(line for line in ir.splitlines() if 'tt.dot ' in line)
   else:
    assert 'tt.atomic_rmw fadd' in ir and 'add.f32' in ptx
   if name=='full_tiles':assert 'arith.truncf' in ir and 'arith.extf' in ir
   else:assert 'arith.truncf' not in ir
   records.append(dict(kernel=name,profile=[P,BM,BN,BK],shape=[M,K,N],wide=wide,signature=sig,specialization=specialization,
      pointer_offsets_i64=True,artifacts=artifacts))
result=dict(candidate_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),triton=triton.__version__,target='sm80',native_gpu_execution=False,records=records)
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(compiled_variants=len(records),target='sm80',pointer_offsets_i64=True,native_gpu_execution=False)))
