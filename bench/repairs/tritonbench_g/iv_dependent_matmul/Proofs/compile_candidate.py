"""Offline Triton compile of actual candidate. No GPU execution."""
import argparse, hashlib, importlib.util, json, os, re
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import native_specialize_impl

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',choices=['boundary','wide-metadata'],default='boundary');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(a.output/'cache')
src=Path(__file__).resolve().parents[1]/'iv_dependent_matmul_i64.py'
spec=importlib.util.spec_from_file_location('iv_repair_candidate',src)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
fn=mod.iv_dependent_matmul_kernel_i64
T=GPUTarget('cuda',80,32);backend=CUDABackend(T);records=[]
shapes=[(65537,32,32768),(35,65,37),(1,65537,32768)] if a.profile=='boundary' else [(2**31+1,1,1),(1,2**31+1,1),(1,1,2**31+1)]
for M,K,N in shapes:
 for mode in ['pre_load','post_load','post_pre_mixed','post_load_two_iters','post_load_three_iters']:
    out=a.output/f'{M}-{K}-{N}-{mode}';out.mkdir(exist_ok=True)
    const={'BLOCK_SIZE_M':32,'BLOCK_SIZE_N':32,'BLOCK_SIZE_K':32,'type':mode}
    scalars={'M':M,'N':N,'K':K,'stride_am':max(K,1),'stride_ak':1,'stride_bk':max(N,1),'stride_bn':1,'stride_cm':max(N,1),'stride_cn':1}
    sig={};attrs={};specialization={}
    for i,name in enumerate(fn.arg_names):
        if name in ('a_ptr','b_ptr','c_ptr'):
            sig[name]='*fp32';attrs[(i,)]=backend.parse_attr('D')
        elif name in const: sig[name]='constexpr'
        else:
            kind,attribute=native_specialize_impl(backend,scalars[name],False,True,True)
            specialization[name]=[kind,attribute];sig[name]=kind
            if kind=='constexpr':const[name]=attribute
            elif attribute:attrs[(i,)]=backend.parse_attr(attribute)
    options={'num_warps':4,'num_stages':4 if mode=='post_load_three_iters' else 3}
    compiled=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=T,options=options)
    artifacts={}
    for ext in ('ttir','ttgir','llir','ptx'):
        f=out/('kernel.'+ext);f.write_text(compiled.asm[ext]);artifacts[ext]={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    ir=compiled.asm['ttir']
    assert 'i64' in ir and 'tt.addptr' in ir
    assert 'arith.truncf' in ir and 'arith.extf' in ir
    assert 'tt.dot' in ir and 'tf32' in ir
    assert 'arith.addf' not in ir
    assert '%accumulator_' in next(line for line in ir.splitlines() if 'tt.dot ' in line)
    ptx=compiled.asm['ptx']
    assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
    assert 'cvt.rn.f16x2.f32' in ptx and 'cvt.f32.f16' in ptx and 'tf32.f32' in ptx
    addptr_lines=[line.strip() for line in ir.splitlines() if '= tt.addptr' in line]
    # Every surviving tensor pointer addition receives i64 lane offsets.
    assert all('xi64>' in line or ', i64' in line for line in addptr_lines),addptr_lines
    records.append({'shape':[M,K,N],'selector':mode,'signature':sig,'specialization':specialization,'options':options,
      'all_ttir_pointer_offsets_i64':True,'half_narrow_then_fp32_store':True,'compiled_dot_graph':'tt.dot with accumulator; distinct from source Dot+Add',
      'artifacts':artifacts})
result={'profile':a.profile,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'triton':triton.__version__,'target':'sm80','native_gpu_execution':False,'records':records}
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'compiled_variants':len(records),'all_ttir_pointer_offsets_i64':True,'target':'sm80','triton':triton.__version__,'native_gpu_execution':False}))
