"""Offline compiler evidence only; no GPU execution."""
from pathlib import Path
import importlib.util,os,json
out=Path('/tmp/veritile-token_attn_llama2-lowering');out.mkdir(exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(out/'cache')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
records=[]
for name in ['token_attn_llama2']:
 path=Path(__file__).with_name(name+'_repaired.py')
 spec=importlib.util.spec_from_file_location(name+'_candidate',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 fn=getattr(m,'_fwd_kernel_token_att1' if name.endswith('llama2') else '_fwd_kernel_token_att2')
 ptrs=(['Q','K','Att_Out','B_Loc','B_Start_Loc','B_Seqlen'] if name.endswith('llama2') else ['Prob','V','Out','Req_to_tokens','B_req_idx','B_Start_Loc','B_Seqlen']+(['B_Att_Start_Loc','B_Att_Seqlen'] if name.endswith('mistral') else []))
 for dim in [16,32,64,128]:
  for ones in [False,True]:
   tag=f'{name}-D{dim}-ones{int(ones)}'
   sig={n:'i32' for n in fn.arg_names if n not in ['BLOCK_DMODEL','BLOCK_N']}
   for n in ptrs:sig[n]='*fp32' if n in ['Q','K','Att_Out','Prob','V','Out'] else '*i32'
   if 'sm_scale' in sig:sig['sm_scale']='fp32'
   const={'BLOCK_DMODEL':dim,'BLOCK_N':32 if name.endswith('llama2') else 128}
   if ones:
    for n in ['kv_group_num','stride_b_loc_s','stride_qd','stride_kd','att_stride_bs','stride_b_req_idx','stride_b_seq_len','stride_b_start_loc','stride_b_att_start','stride_b_att_len','stride_req_to_tokens_s','stride_pbs','stride_vd','stride_od']:
     if n in sig:const[n]=1
   kernel=triton.compile(ASTSource(fn,signature=sig,constexprs=const),target=GPUTarget('cuda',80,32),options={'num_warps':2 if name.endswith('llama2') else 4,'num_stages':1,'enable_fp_fusion':True})
   ptx=kernel.asm['ptx'];(out/(tag+'.ptx')).write_text(ptx)
   assert 'st.global' in ptx and 'ld.global' in ptx
   rec={'tag':tag,'pass':True,'fma':sum('fma.rn.f32' in l for l in ptx.splitlines()),'mul':sum('mul.f32' in l for l in ptx.splitlines()),'add':sum('add.f32' in l for l in ptx.splitlines()),'warps':2 if name.endswith('llama2') else 4}
   records.append(rec)
   print(tag,flush=True)
(out/'report.json').write_text(json.dumps({'triton':triton.__version__,'target':'SM80','cases':records},indent=2))
