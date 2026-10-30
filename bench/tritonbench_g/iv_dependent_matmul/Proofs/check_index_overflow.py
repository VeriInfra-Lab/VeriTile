"""Compile-only audit of the unchanged source's large-allocation index types.
No GPU allocation or native execution is performed.
"""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import native_specialize_impl

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--inner', type=int, default=32)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
os.environ['TRITON_CACHE_DIR'] = str(a.output/'cache')
source = Path(__file__).resolve().parents[1]/'iv_dependent_matmul.py'
text = source.read_text()
extracted = a.output/'original_kernel.py'
extracted.write_text(text[:text.index('def iv_dependent_matmul_wrapper(')])
spec = importlib.util.spec_from_file_location('iv_overflow_original_kernel',extracted)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fn=module.iv_dependent_matmul_kernel
M,N,K=65537,32768,a.inner
target=GPUTarget('cuda',80,32)
backend=CUDABackend(target)
constants={'BLOCK_SIZE_M':32,'BLOCK_SIZE_N':32,'BLOCK_SIZE_K':32,'type':'pre_load'}
scalars={'M':M,'N':N,'K':K,'stride_am':max(K,1),'stride_ak':1,'stride_bk':N,'stride_bn':1,'stride_cm':N,'stride_cn':1}
sig={};attrs={};specialization={}
for index,name in enumerate(fn.arg_names):
    if name in ('a_ptr','b_ptr','c_ptr'):
        sig[name]='*fp32';attrs[(index,)]=backend.parse_attr('D')
    elif name in constants:
        sig[name]='constexpr'
    else:
        kind,attribute=native_specialize_impl(backend,scalars[name],False,True,True)
        specialization[name]=[kind,attribute]
        sig[name]=kind
        if kind=='constexpr': constants[name]=attribute
        elif attribute: attrs[(index,)]=backend.parse_attr(attribute)
compiled=triton.compile(ASTSource(fn,sig,constexprs=constants,attrs=attrs),target=target,
                       options={'num_warps':4,'num_stages':3})
files={}
for ext in ('ttir','ttgir','llir','ptx'):
    path=a.output/('kernel.'+ext)
    path.write_text(compiled.asm[ext]);files[ext]=str(path)
pid=2048*1024
row=(pid//1024)*32
logical=row*N
signed=((logical+2**31)%2**32)-2**31
result={
 'status':'compile_only', 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'triton':triton.__version__,'target':'sm80','signature':sig,'constants':constants,
 'actual_scalar_specialization':specialization,'attrs':{str(k):v for k,v in attrs.items()},
 'case':{'M':M,'N':N,'K':K,'BM':32,'BN':32,'BK':32,'selector':'pre_load',
         'stridesA':[max(K,1),1],'stridesB':[N,1],'stridesC':[N,1],
         'output_elements':M*N,'output_bytes':4*M*N,'grid':2049*1024,'pid':pid,'row':row,'col':0,
         'mask_active':row<M,'ideal_element_offset':logical,'signed_i32_element_offset':signed,
         'relative_byte_offset':4*signed},
 'artifacts':files,
 'native_gpu_executed':False,
}
assert 0<=pid<2049*1024 and row<M and signed<0
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
