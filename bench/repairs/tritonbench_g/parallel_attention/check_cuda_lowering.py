"""Offline compile through Triton's real specialization binder; no GPU launch."""
from pathlib import Path
import hashlib,importlib.util,json,os,re,sys,time,traceback
D=Path(__file__).resolve().parent
ART=Path('/tmp/parallel-attention-lowering');ART.mkdir(exist_ok=True)
os.environ.update(TRITON_CACHE_DIR='/tmp/parallel-attention-triton-cache',CUDA_CACHE_PATH='/tmp/parallel-attention-cuda-cache',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import torch,triton
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
source=D/'parallel_attention_repaired.py'
spec=importlib.util.spec_from_file_location('actual_binder_parallel_attention',source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
rows=[]
for label,(B,H,T,DK,DV),scaled in [(label+('-scaled' if scaled else '-noscale'),shape,scaled) for label,shape in [('tiny',(1,1,1,1,1)),('ragged',(1,2,33,17,19)),('ordinary',(2,2,137,48,48)),('sharded',(1,2,65,127,131))] for scaled in [True,False]]:
    BK=min(128,max(16,triton.next_power_of_2(DK)));BV=min(128,max(16,triton.next_power_of_2(DV)))
    stages=2
    q=torch.empty(B,H,T,DK);v=torch.empty(B,H,T,DV)
    for kind in ['forward','backward']:
        fn=m.parallel_rebased_fwd_kernel if kind=='forward' else m.parallel_rebased_bwd_kernel
        binder=create_function_from_signature(fn.signature,fn.params,backend)
        NK=triton.cdiv(DK,BK);NV=triton.cdiv(DV,BV)
        tensor_args=[q,q,v,torch.empty(NK,B,H,T,DV),torch.empty(NK,B,H,T)] if kind=='forward' else [q,q,v,v,torch.empty(B,H,T),torch.empty(NV,B,H,T,DK),torch.empty(NV,B,H,T,DK),torch.empty(NK,B,H,T,DV)]
        args=tensor_args+[T*DK,DK,1,T*DV,DV,1,DK**-.5 if scaled else 1]
        kwargs={'BTL':128 if kind=='forward' else 64,'BTS':32,'BK':BK,'BV':BV,'B':B,'H':H,'T':T,'K':DK,'V':DV,'num_warps':4,'num_stages':stages}
        bound,specialization,opts=binder(*args,**kwargs)
        _,signature,constants,attrs=fn._pack_args(backend,opts,bound,specialization,opts)
        folder=ART/('actual-binder-'+label+'-'+kind);folder.mkdir(exist_ok=True)
        record={'kind':kind,'shape':[B,H,T,DK,DV],'source_sha256':sha(source),'binding':'create_function_from_signature plus JITFunction._pack_args','signature':signature,'constants':{fn.arg_names[p[0]]:x for p,x in constants.items()},'attrs':str(attrs),'target':['cuda',90,32],'num_warps':4,'num_stages':stages,'GPU_execution':False,'native_numerical_conformance':False}
        before=time.monotonic()
        try:
            code=triton.compile(ASTSource(fn,signature,constexprs=constants,attrs=attrs),target=target,options={'num_warps':4,'num_stages':stages})
            artifacts={}
            for ext in ['source','ttir','ttgir','llir','ptx','cubin']:
                value=source.read_bytes() if ext=='source' else code.asm[ext]
                path=folder/('kernel.py' if ext=='source' else 'kernel.'+ext)
                path.write_bytes(value if isinstance(value,bytes) else value.encode())
                artifacts[ext]={'path':str(path),'sha256':sha(path)}
            record.update(success=True,artifacts=artifacts,fp_fusion=code.metadata.enable_fp_fusion,instructions=sorted(set(re.findall(r'^\s*(?:@!?%\w+\s+)?((?:fma|mul|add|sub|ex2|lg2|cvt|mma|wgmma)\.[a-z0-9_.]+)\s',code.asm['ptx'],re.M))))
        except Exception:
            record.update(success=False,error=traceback.format_exc())
        record['seconds']=time.monotonic()-before
        receipt=folder/'receipt.json';receipt.write_text(json.dumps(record,indent=2)+'\n')
        rows.append({'path':str(receipt),'sha256':sha(receipt)})
        print(json.dumps({'profile':folder.name,'success':record['success'],'seconds':record['seconds']}),flush=True)
        if not record['success']:print(record['error'],flush=True);raise SystemExit(1)
(D/'cuda-lowering.json').write_text(json.dumps({'source_sha256':sha(source),'compiler':triton.__version__,'profile_count':len(rows),'profiles':rows,'actual_argument_binder':True,'GPU_execution':False,'native_numerical_conformance':False},indent=2)+'\n')
