"""Scoped actual-wrapper/JIT-binder checks. No GPU or compiler execution."""
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ.pop('TRITON_INTERPRET', None)
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import triton
from triton.runtime.jit import create_function_from_signature
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name('attention_score_tile_checked.py')
OLD = SOURCE.parent.parent / 'attention_score/attention_score_repaired.py'
OUT = ROOT / '.veritile-cache/kernel-evidence/attention_score_tile_checked'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old_ast, new_ast = ast.parse(OLD.read_text()), ast.parse(SOURCE.read_text())
old_functions = {n.name:n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
new_functions = {n.name:n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
unchanged = ['_score_kernel','_check_pointer_domain','get_score']
assert all(ast.dump(old_functions[n]) == ast.dump(new_functions[n]) for n in unchanged)
static_if = new_functions['_check_launch_domain'].body.pop(1)
assert ast.dump(old_functions['_check_launch_domain']) == ast.dump(new_functions['_check_launch_domain'])
assert isinstance(static_if, ast.If)
assert ast.unparse(static_if.test) == 'block * d > 1 << 20'

spec = importlib.util.spec_from_file_location('attention_score_tile_checked_capture', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
heuristics = module._score_kernel
function = heuristics.fn
backend = CUDABackend(GPUTarget('cuda',80,32))
binder = create_function_from_signature(function.signature,function.params,backend)
torch.set_num_threads(1)
rows=[]

cases=[]
for block in (16,32,64):
    boundary=(1<<20)//block
    cases += [(f'B{block}_boundary',block,boundary,1,1,1,1,1,None,False,False,False),
              (f'B{block}_oversize',block,boundary*2,1,1,1,1,1,None,False,False,True),
              (f'B{block}_empty_query',block,boundary*2,1,1,1,0,1,None,False,False,False)]
cases += [
    ('empty_batch',64,32768,0,1,1,1,1,None,False,False,False),
    ('empty_heads',64,32768,1,0,1,1,1,None,False,False,False),
    ('empty_keys',64,32768,1,1,1,1,0,None,False,False,False),
    ('grouped_query_tail_window',16,16,1,4,2,17,19,(3,7),False,False,False),
    ('grouped_complement',32,16,1,4,2,17,19,(-2,5),True,False,False),
    ('read_alias_offset',64,16,1,1,1,1,1,None,False,False,False),
    ('single_retry',64,16,1,1,1,1,1,None,False,True,False),
]
for name,block,d,z,h,hk,n,nk,window,complement,retry,reject in cases:
    q=torch.zeros(z,h,n,d);k=torch.zeros(z,hk,nk,d);m=torch.zeros(z,h,n)
    if name=='read_alias_offset':
        q=torch.zeros(d+1)[1:].view(1,1,1,d);k=q
    snaps=[x.clone() for x in (q,k,m)]
    launches=[];domains=[]
    class Capture:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                launches.append((tuple(grid(kwargs)),args,kwargs))
                if retry and len(launches)==1:
                    raise triton.OutOfResources(100,1,'shared memory')
            return launch
    def trace(frame,event,arg):
        if frame.f_code is module._check_launch_domain.__code__ and event=='call':
            domains.append({'block':frame.f_locals['block'],'d':frame.f_locals['d']})
        return trace
    module._BLOCK_M=module._BLOCK_N=block
    error=None;returned=None
    sys.settrace(trace)
    try:
        with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)), \
             patch.object(torch.cuda,'device',lambda _:contextlib.nullcontext()), \
             patch.object(module,'_score_kernel',Capture()):
            returned=module.get_score(q,k,m,window,complement)
    except ValueError as e:
        error=str(e)
    finally:
        sys.settrace(None)
    assert all(torch.equal(a,b) for a,b in zip((q,k,m),snaps))
    row={'case':name,'block':block,'d':d,'domain_calls':domains,'launches':[],'error':error}
    if reject:
        assert error=='source tile exceeds Triton static tensor element limit' and not launches
    else:
        assert error is None
        assert list(returned.shape)==[z,h,nk] and returned.dtype==torch.float32
        assert torch.equal(returned,torch.zeros_like(returned))
        expected=0 if n==0 or returned.numel()==0 else 2 if retry else 1
        assert len(launches)==len(domains)==expected
        for grid,args,kwargs in launches:
            assert args[0] is q and args[1] is k and args[2] is m and args[4] is returned
            env=dict(zip(function.arg_names,args));env.update(kwargs)
            options={key:value(env) for key,value in heuristics.values.items()}
            bound,specialization,binder_options=binder(*args,**kwargs,**options)
            _,signature,constants,attrs=function._pack_args(backend,binder_options,bound,specialization,binder_options)
            row['launches'].append({'grid':grid,'block':kwargs['BLOCK_M'],'d':kwargs['BLOCK_DMODEL'],
                'actual_binder':True,'signature':signature,'heuristics':options,
                'window':window,'complement':complement,'scale':args[3]})
        if retry: assert [x['block'] for x in domains]==[64,32]
    rows.append(row)

old_packet=ROOT/'.veritile-cache/kernel-evidence/attention_score/delivery.json'
packet=json.loads(old_packet.read_text())
assert sha(OLD)==packet['selected_source']['sha256']
controls=[]
for control in packet['controls']:
    for artifact in control['artifacts'].values():
        assert sha(Path(artifact['path']))==artifact['sha256']
    controls.append({'case':control['case'],'status':control['status'],'artifacts':control['artifacts'],
                     'reuse_basis':'byte-identical JIT function AST including heuristic decorators'})
report={'source':str(SOURCE),'source_sha256':sha(SOURCE),'only_change':'nonempty per-attempt block*d<=1048576 guard',
 'unchanged_ast_functions':unchanged,'cases':rows,'reused_compiler_controls':controls,
 'original_diagnosis':{'path':str(old_packet),'sha256':sha(old_packet)},
 'gpu_execution':False,'kernel_interpreter_execution':False,'new_compilations':0,'numerical_experiments':[],
 'boundary':'Actual Python wrapper and Torch objects; only is_cuda, CUDA context and launch replaced. Captured arguments enter actual JIT binder. Static boundary cases establish admission, not device resource capacity.'}
(OUT/'candidate-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS {len(rows)} wrapper cases; {sum(len(r["launches"]) for r in rows)} actual binder captures; 2 compiler controls reused; no GPU.')
