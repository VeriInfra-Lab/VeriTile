"""Scoped SM80 compilation using the actual JIT binder; no GPU execution."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OUT=ROOT/'.veritile-cache/kernel-evidence/softmax_reducev_neginf_checked/compiler';OUT.mkdir(parents=True,exist_ok=True)
source=HERE/'softmax_reducev_neginf_checked.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('softmax_reducev_neginf_compiler_source',source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for dim,unaligned in [(1,False),(16,False),(128,False),(16,True)]:
    name=f'd{dim}_offset{int(unaligned)}';os.environ['TRITON_CACHE_DIR']=str(OUT/'cache'/name)
    target=GPUTarget('cuda',80,32);backend=CUDABackend(target);fn=m._fwd_kernel
    def tensor(shape,dtype=torch.float32):
        if unaligned:
            size=1
            for n in shape:size*=n
            return torch.empty(size+1,dtype=dtype)[1:].reshape(shape)
        return torch.empty(shape,dtype=dtype)
    q=tensor((2,129));v=tensor((129,2,dim));o=tensor((2,2,dim))
    loc=tensor((2,129),torch.int32);starts=tensor((2,),torch.int32);lengths=tensor((2,),torch.int32)
    args=[q,v,o,loc,starts,lengths,129,129,1,2*dim,dim,1,2*dim,dim,1,129,1,0]
    options={'num_warps':1,'num_stages':3,'enable_fp_fusion':True}
    binder=create_function_from_signature(fn.signature,fn.params,backend)
    bound,specialization,bopts=binder(*args,BLOCK_DMODEL=dim,BLOCK_N=64,**options)
    parsed,sig,const,attrs=fn._pack_args(backend,bopts,bound,specialization,bopts)
    code=triton.compile(ASTSource(fn,sig,constexprs=const,attrs=attrs),target=target,options=options)
    artifacts={}
    for kind in ['ttir','ttgir','llir','ptx']:
        path=OUT/(name+'.'+kind);path.write_text(code.asm[kind]);artifacts[kind]={'path':str(path),'sha256':sha(path)}
    ptx=code.asm['ptx']
    assert '0fFF7FFFFF' in ptx or '0fFF7fffff' in ptx, 'missing exact FP32 floor'
    assert 'ex2.approx' in ptx and 'fma.rn.f32' in ptx and 'st.global' in ptx
    assert 'f64' not in ptx and 'mma' not in ptx
    rows.append(dict(name=name,status='compiled',source_sha256=sha(source),signature=sig,constants={str(k):str(v) for k,v in const.items()},attrs=str(attrs),options=options,artifacts=artifacts))
    print(name,'compiled',flush=True)
    (OUT/'report.json').write_text(json.dumps(dict(status='PASS',triton=triton.__version__,actual_JIT_binder=True,GPU_execution=False,results=rows),indent=2)+'\n')
