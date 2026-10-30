"""Exercise the actual public candidate, ordinary Tensor objects, and JIT binder."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,json,os,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'.veritile-cache/kernel-evidence/attn_fwd_triton_tile_checked'
SOURCE=Path(__file__).with_name('attn_fwd_triton_tile_checked.py')
OLD=SOURCE.parent.parent/'attn_fwd_triton/attn_fwd_triton_repaired.py'
sys.dont_write_bytecode=True
os.environ.pop('TRITON_INTERPRET',None)
os.environ['TRITON_CACHE_DIR']=str(OUT/'triton-cache')
os.environ['OMP_NUM_THREADS']='1'
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a,b=ast.parse(OLD.read_text()),ast.parse(SOURCE.read_text())
launch=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name=='_launch_contract')
guard=next(x for x in launch.body if isinstance(x,ast.If) and ast.unparse(x.test)=='128 * bd > 1048576')
assert ast.dump(guard.test)==ast.dump(ast.parse('128 * bd > 1048576',mode='eval').body)
launch.body.remove(guard)
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
spec=importlib.util.spec_from_file_location('attn_static_candidate',SOURCE)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
function=mod._attn_fwd_repaired
backend=CUDABackend(GPUTarget('cuda',90,32))
binder=create_function_from_signature(function.signature,function.params,backend)
torch.set_num_threads(1)
rows=[]
def one(name,shape,layout='contiguous',reject=None,scales='valid'):
    q,k,v=[torch.zeros(shape,dtype=torch.float32) for _ in range(3)]
    if layout=='aliased': k=v=q
    elif layout=='broadcast':
        q=torch.zeros((1,1,1,1),dtype=torch.float32).expand(shape);k=q;v=q
    elif layout=='independent':
        n,d=shape[2:]
        q=torch.zeros((1,1,d,n),dtype=torch.float32).transpose(2,3)
        k=torch.zeros((1,1,n,d*2),dtype=torch.float32)[...,::2]
        v=torch.zeros((1,1,n*2,d),dtype=torch.float32)[:,:,::2,:]
    qs=torch.ones((shape[0]*shape[1]*triton.cdiv(shape[2],128),),dtype=torch.float32)
    ks=torch.ones((shape[0]*shape[1]*triton.cdiv(shape[2],64),),dtype=torch.float32)
    if scales=='unpacked':
        qs=torch.ones((max(2,qs.numel())*2,),dtype=torch.float32)[::2]
        ks=torch.ones((max(2,ks.numel())*2,),dtype=torch.float32)[::2]
    elif scales=='short': qs=torch.ones((0,),dtype=torch.float32);ks=qs
    elif scales=='negative_view': qs=torch._neg_view(qs)
    elif scales=='signed': qs=-qs;ks=-ks
    inputs=(q,k,v,qs,ks);snap=[x.clone() for x in inputs]
    allocations=[];launches=[];devices=[]
    raw_empty,raw_empty_like=torch.empty,torch.empty_like
    def empty(*args,**kw):
        t=raw_empty(*args,**kw);allocations.append(('M',t));return t
    def empty_like(*args,**kw):
        t=raw_empty_like(*args,**kw);allocations.append(('O',t));return t
    def device(d): devices.append(d);return contextlib.nullcontext()
    class Capture:
        def __getitem__(self,grid):
            def call(*args,**kw): launches.append((grid,args,kw))
            return call
    exc=None;returned=None
    with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)),patch.object(torch.cuda,'device',device),\
         patch.object(torch,'empty',empty),patch.object(torch,'empty_like',empty_like),\
         patch.object(mod,'_attn_fwd_repaired',Capture()):
        try: returned=mod.forward_repaired(*inputs)
        except (ValueError,AssertionError,TypeError) as error: exc=error
    assert all(torch.equal(x,y) for x,y in zip(inputs,snap))
    row={'case':name,'shape':shape,'layout':layout,'scales':scales,'allocations':[k for k,_ in allocations],
         'input_snapshots_unchanged':True,'exception':None if exc is None else {'type':type(exc).__name__,'message':str(exc)}}
    if reject:
        assert exc is not None and reject in str(exc),row
        assert not launches and not devices
        row['status']='rejected_before_launch'
        if allocations: assert [k for k,_ in allocations]==['O','M']
    else:
        assert exc is None,row
        assert [k for k,_ in allocations]==['O','M']
        assert returned is allocations[0][1] and returned.dtype==torch.bfloat16
        assert allocations[1][1].dtype==torch.float32 and tuple(allocations[1][1].shape)==shape[:3]
        row.update(return_is_actual_output=True,output_dtype='BF16',workspace_dtype='FP32')
        if q.numel()==0:
            assert not launches and not devices
            row['status']='empty_return_before_launch_checks'
        else:
            assert len(launches)==1 and devices==[q.device]
            grid,args,kw=launches[0]
            assert all(args[i] is inputs[i] for i in range(5)) and args[5] is returned
            assert args[6:22]==tuple(s for t in (q,k,v,returned) for s in t.stride())
            bound,specialization,options=binder(*args,**kw)
            _,signature,constants,attrs=function._pack_args(backend,options,bound,specialization,options)
            assert 128*kw['BLOCK_DMODEL']<=1048576
            row.update(status='actual_binder_pass',grid=grid,signature=signature,
                constants={function.arg_names[key[0]]:str(value) for key,value in constants.items()},
                specialization=str(specialization),attrs=str(attrs),launch_options=kw,
                source_envelope=mod._launch_contract(*inputs,returned),
                strides={key:list(t.stride()) for key,t in zip(('q','k','v','qs','ks','o'),(*inputs,returned))})
    rows.append(row);print(name,row['status'],flush=True)
for d in (1,16,17,8192): one('D'+str(d),(1,1,1,d))
one('overflow_D8193',(1,1,1,8193),reject='static tensor element limit')
for name,n,d in [('tail',65,31),('two_blocks',129,16)]: one(name,(1,1,n,d))
for layout in ('aliased','broadcast','independent'): one(layout,(1,1,129,17),layout)
one('signed_scales',(1,1,129,16),scales='signed')
one('unpacked_scales',(1,1,129,16),reject='packed-flat',scales='unpacked')
one('negative_view',(1,1,1,16),reject='resolved FP32',scales='negative_view')
for name,shape in [('zero_B',(0,1,1,8193)),('zero_H',(1,0,1,8193)),('zero_N',(1,1,0,8193)),('zero_D',(1,1,129,0))]:
    one(name,shape,scales='unpacked')
one('empty_short_scales',(1,1,0,8193),scales='short')
report={'task':'attn_fwd_triton_tile_checked','status':'PASS','source':{'path':str(SOURCE),'sha256':sha(SOURCE)},
 'canonical':{'path':str(OLD),'sha256':sha(OLD)},'script_sha256':sha(Path(__file__)),
 'sole_ast_change':ast.unparse(guard),'cases':rows,'actual_binder_count':sum(r['status']=='actual_binder_pass' for r in rows),
 'boundary':'Ordinary CPU Torch tensors; only CUDA flag, device context and final launcher replaced. All checks/allocations and installed JIT binder actual. No GPU/interpreter execution or numerical experiment.',
 'gpu_execution':False,'interpreter_execution':False,'triton_version':triton.__version__}
(OUT/'candidate-check.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print('PASS',len(rows),'cases')
