"""Offline actual-JIT-binder specializations for the admitted layout fixtures.

Uses actual Tensor shapes/strides/offsets and default compiler options. It does
not run on a GPU or interpret floating-point instruction results.
"""
from pathlib import Path
import dataclasses,hashlib,importlib.util,json,os,re,sys,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DEST=ROOT/'.veritile-cache/task-deliveries/triton_conv2d_fwd/layout-lowering'
DEST.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_INTERPRET']='0'
os.environ['TRITON_ALWAYS_COMPILE']='1'
os.environ['TRITON_CACHE_DIR']=str(DEST/'cache')
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(shape,strides,offset):
    span=1+sum((d-1)*s for d,s in zip(shape,strides))
    backing=torch.empty(span+offset,dtype=torch.float32)
    return backing.as_strided(shape,strides,offset)

source=HERE/'triton_conv2d_fwd_repaired.py'
spec=importlib.util.spec_from_file_location('conv_layout_binder',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
fn=module.conv2d_forward_kernel
target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
binder=create_function_from_signature(fn.signature,fn.params,backend)
controls=DEST.parent/'layout-controls.json'
data=json.loads(controls.read_text())
assert data['source']['sha256']==sha(source)
profiles=[]
for case in data['cases']:
    X=fixture(case['input_shape'],case['input_strides'],case['input_offset'])
    W=fixture(case['weight_shape'],case['weight_strides'],case['weight_offset'])
    Y=torch.empty(case['output_shape'],dtype=torch.float32)
    N,C,H,WI=X.shape;O=W.shape[0];OH,OW=Y.shape[2:]
    KH,KW,SH,SW,PH,PW,G=case['arguments']
    values=[X,W,Y,N,C,H,WI,O,OH,OW,*X.stride(),*W.stride(),*Y.stride(),
        KH,KW,SH,SW,PH,PW,G,case['fp16'],case['tf32'],128,32,32]
    bound,specialization,options=binder(*values)
    _,signature,constants,attrs=fn._pack_args(backend,options,bound,specialization,options)
    start=time.monotonic()
    code=triton.compile(ASTSource(fn,signature,constexprs=constants,attrs=attrs),target=target)
    label=f"{case['name']}-half{int(case['fp16'])}-tf32{int(case['tf32'])}"
    artifacts={}
    for kind in ('ttir','ttgir','llir','ptx'):
        path=DEST/(label+'.'+kind);path.write_text(code.asm[kind])
        artifacts[kind]={'path':str(path),'sha256':sha(path)}
    opcodes={}
    for op in re.findall(r'^\s*(?:@!?%\w+\s+)?((?:fma|add|mul|cvt|mma|wgmma)\.[\w.]+)\s',code.asm['ptx'],re.M):
        opcodes[op]=opcodes.get(op,0)+1
    profile={'label':label,'shape':dict(zip(('N','C','O','H','W','OH','OW','KH','KW','SH','SW','PH','PW','groups'),
        (N,C,O,H,WI,OH,OW,KH,KW,SH,SW,PH,PW,G))),
        'strides':{'input':list(X.stride()),'weight':list(W.stride()),'output':list(Y.stride())},
        'storage_offsets':{'input':X.storage_offset(),'weight':W.storage_offset(),'output':Y.storage_offset()},
        'fp16':case['fp16'],'tf32':case['tf32'],'grid':[(N*OH*OW+127)//128,(O+31)//32,G],
        'signature':signature,'constants':{fn.arg_names[k[0]]:v for k,v in constants.items()},
        'attrs':{str(k):v for k,v in attrs.items()},'artifacts':artifacts,'opcodes':opcodes,
        'metadata':{k:v for k,v in code.metadata._asdict().items() if isinstance(v,(str,int,bool,float))},
        'seconds':time.monotonic()-start,'status':'compiled'}
    profiles.append(profile)
    print(json.dumps({'label':label,'opcodes':opcodes,'seconds':profile['seconds']}),flush=True)
report={'status':'PASS','source_sha256':sha(source),'script_sha256':sha(Path(__file__)),
    'controls':{'path':str(controls),'sha256':sha(controls)},'target':dataclasses.asdict(target),
    'triton_version':triton.__version__,'actual_source_JIT_binder':True,'default_compiler_options':True,
    'scope':'Actual source JIT binder called on CPU Tensor metadata fixtures, then explicit offline ASTSource compilation; not a CUDA runtime launch trace.',
    'GPU_execution':False,'native_numerical_conformance':False,'profiles':profiles}
(DEST/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(DEST/'report.json')
