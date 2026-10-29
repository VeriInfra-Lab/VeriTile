from pathlib import Path
import dataclasses,hashlib,importlib.util,json,os,sys,traceback
sys.dont_write_bytecode=True
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'binder-artifacts')
args=parser.parse_args()
A=Path(__file__).resolve().parents[1];D=args.output_dir;D.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_ALWAYS_COMPILE']='1'
import triton,torch,triton.language as tl
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=A/'attention_kernel_repaired.py'
target=GPUTarget('cuda',80,32);backend=CUDABackend(target);rows=[]
for dtype in [torch.float16,torch.bfloat16]:
 for N,DIM,BM,BN,W,S in [(1,16,16,16,8,4),(129,128,128,128,4,4),(129,128,128,128,8,4),(97,64,128,16,4,1),(65,32,16,128,8,4)]:
    name=f'{str(dtype).split(".")[-1]}_N{N}_D{DIM}_M{BM}_K{BN}_W{W}_S{S}'
    spec=importlib.util.spec_from_file_location(name,source);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    fn=m._fwd_kernel_aligned;binder=create_function_from_signature(fn.signature,fn.params,backend)
    Z,H=2,3;shape=(Z,H,N,DIM)
    q=torch.empty(shape,dtype=dtype);k=torch.empty_like(q);v=torch.empty_like(q);out=torch.empty_like(q);bias=torch.empty(Z,H,N,128,dtype=dtype)
    args=[q,k,v,bias,0.1,out,*q.stride(),*k.stride(),*v.stride(),*out.stride(),*bias.stride()[:3],Z,H,N,0]
    options={'num_warps':W,'num_stages':S,'enable_fp_fusion':True}
    bound,specialization,binder_options=binder(*args,OUT_DTYPE=tl.float16 if dtype==torch.float16 else tl.bfloat16,BIAS_LAST_SIZE=64,B0_NUMEL=128,BLOCK_DMODEL=DIM,BLOCK_M=BM,BLOCK_N=BN,**options)
    parsed,sig,constants,attrs=fn._pack_args(backend,binder_options,bound,specialization,binder_options)
    row={'name':name,'source':str(source),'source_sha256':sha(source),'shape':shape,'signature':sig,'constants':{fn.arg_names[a[0]]:str(v) for a,v in constants.items()},'attrs':{str(a):v for a,v in attrs.items()},'requested_options':options,'actual_JIT_signature_binder':True,'descriptor_device':'CPU descriptor used for offline CUDA target compilation'}
    os.environ['TRITON_CACHE_DIR']=str(D/'cache'/name)
    try:
        code=triton.compile(ASTSource(fn,sig,constexprs=constants,attrs=attrs),target=target,options=options)
        artifacts={}
        for kind in ['ttir','ttgir','llir','ptx']:
            f=D/(name+'.'+kind);f.write_text(code.asm[kind]);artifacts[kind]={'path':str(f),'sha256':sha(f)}
        row.update(status='compiled',artifacts=artifacts,metadata=code.metadata._asdict())
    except Exception as e:
        row.update(status='compile_rejected',error=str(e));(D/(name+'.error.txt')).write_text(traceback.format_exc())
    rows.append(row);print(name,row['status'],row.get('error',''),flush=True)
    (D/'report.json').write_text(json.dumps({'status':'PASS' if all(x['status']=='compiled' for x in rows) else 'FAIL','triton':triton.__version__,'target':dataclasses.asdict(target),'actual_JIT_signature_binder':True,'actual_CUDA_JIT_launch':False,'GPU_execution':False,'FP_numerical_execution':False,'results':rows},indent=2,default=str)+'\n')
