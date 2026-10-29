"""Capture the actual source JIT binder and default compiler options; no GPU run."""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,os,re,sys,time
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=Path('/tmp/conv2d-checked-lowering'))
D=parser.parse_args().output_dir.resolve()
D.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_ALWAYS_COMPILE']='1'
os.environ['TRITON_CACHE_DIR']=str(D/'cache')
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
source=E/'triton_conv2d_fwd_repaired.py'
spec=importlib.util.spec_from_file_location('conv_repaired_capture',source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
fn=module.conv2d_forward_kernel
target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
binder=create_function_from_signature(fn.signature,fn.params,backend)
profiles=[]
for shape_id,(batch,cg,og,height,width,kh,kw,sh,sw,ph,pw,groups) in enumerate([
 (1,3,5,4,5,3,2,2,1,1,1,1),
 (2,33,35,3,4,2,2,1,2,1,1,2),
 (1,2,3,12,12,1,1,1,1,0,0,1),
]):
 for half in [False,True]:
  for tf32 in [False,True]:
   C,O=cg*groups,og*groups
   oh=(height+2*ph-kh)//sh+1;ow=(width+2*pw-kw)//sw+1
   X=torch.empty(batch,C,height,width,dtype=torch.float32)
   W=torch.empty(O,cg,kh,kw,dtype=torch.float32)
   Y=torch.empty(batch,O,oh,ow,dtype=torch.float32)
   values=[X,W,Y,batch,C,height,width,O,oh,ow,*X.stride(),*W.stride(),*Y.stride(),kh,kw,sh,sw,ph,pw,groups,half,tf32,128,32,32]
   bound,specialization,options=binder(*values)
   _,signature,constants,attrs=fn._pack_args(backend,options,bound,specialization,options)
   start=time.monotonic()
   code=triton.compile(ASTSource(fn,signature,constexprs=constants,attrs=attrs),target=target)
   label=f'shape{shape_id}-half{int(half)}-tf32{int(tf32)}'
   artifacts={}
   for kind in ['ttir','ttgir','llir','ptx']:
    path=D/(label+'.'+kind);path.write_text(code.asm[kind])
    artifacts[kind]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
   opcodes={}
   for op in re.findall(r'^\s*(?:@!?%\w+\s+)?((?:fma|add|mul|cvt|mma|wgmma)\.[\w.]+)\s',code.asm['ptx'],re.M):
    opcodes[op]=opcodes.get(op,0)+1
   profile={'label':label,'shape':{'N':batch,'C':C,'O':O,'H':height,'W':width,'OH':oh,'OW':ow,'KH':kh,'KW':kw,'SH':sh,'SW':sw,'PH':ph,'PW':pw,'groups':groups},'fp16':half,'tf32':tf32,
    'grid':[(batch*oh*ow+127)//128,(O+31)//32,groups],
    'signature':signature,'constants':{fn.arg_names[k[0]]:v for k,v in constants.items()},'attrs':{str(k):v for k,v in attrs.items()},
    'metadata':{k:v for k,v in code.metadata._asdict().items() if isinstance(v,(str,int,bool,float))},
    'artifacts':artifacts,'opcodes':opcodes,'seconds':time.monotonic()-start,'status':'compiled'}
   profiles.append(profile)
   print(json.dumps({'label':label,'opcodes':opcodes,'seconds':profile['seconds']}),flush=True)
report={'status':'PASS','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'target':dataclasses.asdict(target),'triton_version':triton.__version__,'actual_source_JIT_binder':True,'default_compiler_options':True,'GPU_execution':False,'native_numerical_conformance':False,'profiles':profiles}
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(E/'cuda-lowering.json').write_text(json.dumps(report,indent=2)+'\n')
