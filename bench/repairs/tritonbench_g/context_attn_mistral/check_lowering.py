"""Selected current-source SM90 offline compiler evidence; no GPU execution."""
from pathlib import Path
import importlib.util,os,json,re,hashlib
out=Path('/tmp/veritile-context-mistral-lowering');out.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
path=Path(__file__).with_name('context_attn_mistral_repaired.py')
spec=importlib.util.spec_from_file_location('context_mistral_lowering',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fn=m._fwd_kernel_repaired
records=[]
for dim in [16,32,64,128]:
 for ones in [False,True]:
  tag=f'D{dim}-ones{int(ones)}'
  sig={n:'i32' for n in fn.arg_names if not n.startswith('BLOCK_')}
  for n in ['Q','K','V','Out']:sig[n]='*fp32'
  for n in ['B_Start_Loc','B_Seqlen']:sig[n]='*i32'
  sig['sm_scale']='fp32';sig['sliding_window']='i64'
  const={'BLOCK_DMODEL':dim,'BLOCK_M':128,'BLOCK_N':128}
  if ones:
   for n in ['kv_group_num','stride_qd','stride_kd','stride_vd','stride_od']:const[n]=1
  warps=4 if dim<=64 else 8
  kernel=triton.compile(ASTSource(fn,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options={'num_warps':warps,'num_stages':1,'enable_fp_fusion':True})
  ptx=kernel.asm['ptx'];(out/(tag+'.ptx')).write_text(ptx)
  assert 'wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32' in ptx
  assert f'wgmma.mma_async.sync.aligned.m64n{dim}k8.f32.tf32.tf32' in ptx
  assert 'ex2.approx.f32' in ptx and 'st.global' in ptx
  assert '0fFF800000' in ptx  # exact folded zeros-minus-infinity initializer
  modes=[line.rsplit(',',3)[-3].strip() for line in ptx.splitlines() if 'wgmma.mma_async' in line]
  true_modes=set(re.findall(r'mov\.pred\s+(%p\d+), -1;',ptx))
  assert '0' in modes and any(mode in true_modes for mode in modes)
  assert all(mode=='0' or mode in true_modes for mode in modes)
  # All PV instructions use the live seed, including an actual zero payload.
  pv_lines=[line for line in ptx.splitlines() if 'wgmma.mma_async' in line and re.search(r'\}, \{',line)]
  assert pv_lines and all(line.rsplit(',',3)[-3].strip() in true_modes for line in pv_lines)
  records.append({'tag':tag,'warps':warps,'pass':True})
  print(tag,flush=True)
(out/'report.json').write_text(json.dumps({'triton':triton.__version__,'target':'SM90','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':records},indent=2))
