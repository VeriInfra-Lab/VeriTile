"""Focused actual-source layout controls; no native numerical conformance.

CUDA residency/current-device queries are adapted for the CPU interpreter. All
Tensor storage/strides, wrapper guards, allocations and kernel ASTs are actual.
The two negative controls reject an incorrect dense-layout interpretation.
"""
from pathlib import Path
import hashlib,importlib.util,itertools,json,os,sys
sys.dont_write_bytecode=True
os.environ['TRITON_INTERPRET']='1'
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
CACHE=ROOT/'.veritile-cache/task-deliveries/triton_conv2d_fwd'
os.environ['TRITON_CACHE_DIR']=str(CACHE/'interpreter-cache')
import torch

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def backing(t):
    s=t.untyped_storage()
    return torch.empty(0,dtype=torch.uint8).set_(s,0,(s.nbytes(),),(1,)).clone()
def dense_interpretation(t):
    strides=[];stride=1
    for size in reversed(t.shape):strides.append(stride);stride*=size
    return t.as_strided(t.shape,tuple(reversed(strides)),t.storage_offset())
def offset_copy(t):
    raw=torch.empty(t.numel()+1,dtype=t.dtype)
    result=raw.as_strided(t.shape,t.stride(),1)
    result.copy_(t)
    return result

def cases():
    x=torch.arange(12,dtype=torch.float32).reshape(1,2,2,3)
    x=x.transpose(2,3).contiguous().transpose(2,3)
    w=torch.tensor([1.,0.]).reshape(1,2,1,1)
    yield 'input-transposed',x,w,(1,1,1,1,0,0,1),'input'
    x=torch.zeros(1,2,2,2);x[0,0,0,1]=1
    w=torch.arange(1,25,dtype=torch.float32).reshape(3,2,2,2)
    w=w.transpose(2,3).contiguous().transpose(2,3)
    yield 'weight-transposed',x,w,(2,2,1,1,0,0,1),'weight'
    x=(torch.arange(120,dtype=torch.float32)%5-2).reshape(2,4,3,5)
    w=(torch.arange(72,dtype=torch.float32)%3-1).reshape(6,2,2,3)
    x=offset_copy(x.contiguous(memory_format=torch.channels_last))
    w=offset_copy(w.contiguous(memory_format=torch.channels_last))
    yield 'channels-last-offset',x,w,(2,3,1,2,1,1,2),None
    x=(torch.arange(288,dtype=torch.float32)%5-2).reshape(1,2,12,12)
    w=torch.arange(6,dtype=torch.float32).reshape(3,2,1,1)
    x=offset_copy(x.contiguous(memory_format=torch.channels_last))
    w=offset_copy(w.contiguous(memory_format=torch.channels_last))
    yield 'channels-last-spatial-tail',x,w,(1,1,1,1,0,0,1),None

source=HERE/'triton_conv2d_fwd_repaired.py'
spec=importlib.util.spec_from_file_location('conv_layout_controls',source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
old_cuda=torch.Tensor.is_cuda
old_current=torch.cuda.current_device
torch.Tensor.is_cuda=property(lambda _:True)
torch.cuda.current_device=lambda:None
rows=[];negative=[]
try:
    for name,x,w,args,fault in cases():
        kh,kw,sh,sw,ph,pw,groups=args
        expected=torch.nn.functional.conv2d(x,w,stride=(sh,sw),padding=(ph,pw),groups=groups)
        if fault:
            wrong=torch.nn.functional.conv2d(dense_interpretation(x) if fault=='input' else x,
                dense_interpretation(w) if fault=='weight' else w,
                stride=(sh,sw),padding=(ph,pw),groups=groups)
            mismatch=(expected!=wrong).nonzero()[0].tolist()
            index=tuple(mismatch)
            negative.append({'name':name,'incorrect_assumption':fault+' has dense NCHW strides',
                'index':mismatch,'actual_stride_value':expected[index].item(),'wrong_dense_value':wrong[index].item()})
        for half,tf32 in itertools.product([False,True],repeat=2):
            before_x,before_w=backing(x),backing(w)
            out=mod.conv2d_forward(x,w,*args,half,tf32)
            assert torch.equal(out,expected),(name,half,tf32)
            assert torch.equal(backing(x),before_x) and torch.equal(backing(w),before_w)
            assert out.shape==expected.shape and out.dtype==torch.float32 and out.device==x.device
            assert out.untyped_storage().data_ptr() not in {x.untyped_storage().data_ptr(),w.untyped_storage().data_ptr()}
            rows.append({'name':name,'fp16':half,'tf32':tf32,'input_shape':list(x.shape),
                'input_strides':list(x.stride()),'input_offset':x.storage_offset(),
                'weight_shape':list(w.shape),'weight_strides':list(w.stride()),'weight_offset':w.storage_offset(),
                'arguments':list(args),'output_shape':list(out.shape),
                'full_output_match':True,'input_backings_preserved':True})
            print(name,half,tf32,'PASS',flush=True)
finally:
    torch.Tensor.is_cuda=old_cuda
    torch.cuda.current_device=old_current
CACHE.mkdir(parents=True,exist_ok=True)
dest=CACHE/'layout-controls.json'
dest.write_text(json.dumps({'source':{'path':str(source),'sha256':sha(source)},
    'script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))},
    'evidence':'actual wrapper/kernel CPU-interpreter source controls on exactly representable small integers',
    'GPU_execution':False,'native_numerical_conformance':False,'cases':rows,
    'rejected_model_simplifications':negative},indent=2)+'\n')
print(dest)
