"""Actual repaired Triton CPU source; libdevice scalar leaves use NumPy/math.
This does not validate GPU arithmetic or compiler instruction conformance.
"""
import os
os.environ['TRITON_INTERPRET']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,contextlib,hashlib,importlib.util,json,math,sys,warnings
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import torch
from triton.runtime import interpreter
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('linear_source',D/'triton_linear_activation_repaired.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for path,digest in json.loads((D/'source_pin.json').read_text())['files'].items():
    assert hashlib.sha256((D.parents[3]/path).read_bytes()).hexdigest()==digest
raw=m.kernel_fma.fn.fn
configs=m.kernel_fma.configs
libdevice=m.tl.extra.cuda.libdevice
lib_tanh,lib_erf=libdevice.tanh,libdevice.erf

def scalar_arg(x):
    # Interpreter constexpr arithmetic can wrap a runtime tensor in fast_gelu.
    return x.value if isinstance(x,m.tl.constexpr) else x

def kernel_ast(path):
    return [ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name in ('kernel_fma','tanh','relu','gelu','fast_gelu')]
assert kernel_ast(D/'triton_linear_activation_repaired.py')==kernel_ast(D/'triton_linear_activation_conditional.py')

def bytes_of(t):
    return torch.empty(0,dtype=torch.uint8).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).clone()

def scalar_intrinsic(self,lib,path,symbol,args,ret,pure):
    assert symbol in ('__nv_tanhf','__nv_erff'),symbol
    data=np.tanh(args[0].data) if symbol=='__nv_tanhf' else np.vectorize(math.erf,otypes=[np.float32])(args[0].data)
    return interpreter.TensorHandle(data.astype(np.float32),ret.scalar)

class Ctx:
    def __init__(self,fail=False):self.fail=fail
    def save_for_backward(self,*args):
        if self.fail:raise RuntimeError('save_for_backward')
        self.saved=args

class Runner:
    def __init__(self,index=18):
        self.config=configs[index];self.allocations=[];self.launches=[];self.current=7;self.fail=None
    @contextlib.contextmanager
    def device(self,owner):
        if self.fail=='enter':raise RuntimeError('device enter')
        previous=self.current;self.current=owner
        try:yield
        finally:
            self.current=previous
            if self.fail=='exit':raise RuntimeError('device exit')
    def empty(self,*args,**kw):
        if self.fail=='allocation':raise RuntimeError('allocation')
        kw['device']='cpu';out=torch.empty(*args,**kw);out.fill_(31.25);self.allocations.append(out);return out
    def __getattr__(self,name):return getattr(torch,name)
    @property
    def cuda(self):return self
    def __getitem__(self,grid):
        def invoke(*args,**kw):
            if self.fail=='before':raise RuntimeError('setup')
            self.launches.append((args,kw,self.current))
            kw.update(self.config.kwargs)
            kw['K_LOAD_MASK_NEEDED']=args[7]%kw['BLOCK_K']==0
            raw[grid(kw)](*args,**kw,num_warps=self.config.num_warps,num_stages=self.config.num_stages)
            if self.fail=='after':raise RuntimeError('post-launch')
        return invoke

def activation(x,name):
    if name=='tanh':return x.tanh()
    if name=='relu':return torch.maximum(torch.tensor(0.),x)
    if name=='gelu':return x*.5*(1+torch.erf(x/math.sqrt(2)))
    if name=='fast_gelu':return x*.5*(1+torch.tanh(math.sqrt(2/math.pi)*(x+.044715*x*x*x)))
    return x

cases=0

def case(index,shape,n,act='',amp=False,bias_flag=True,tail=0):
    global cases
    k=shape[-1];rows=math.prod(shape[:-1]);runner=Runner(index)
    x=((torch.arange(math.prod(shape),dtype=torch.float32).reshape(shape)%7)-3)/16
    w=((torch.arange(n*k,dtype=torch.float32).reshape(n,k)%11)-5)/16
    bias=torch.arange(n,dtype=torch.float32)/16 if bias_flag else None
    backing=bytearray(rows*n*4+tail)
    saved=torch.frombuffer(backing,dtype=torch.float32,count=rows*n).view(rows,n) if rows*n else torch.empty(rows,n)
    saved.fill_(23.)
    kept=[(t,bytes_of(t)) for t in (x,w,bias) if t is not None]
    saved_before=bytes_of(saved);ctx=Ctx()
    with patch.object(m,'torch',runner),patch.object(m,'kernel_fma',runner), \
         patch.object(torch.Tensor,'is_cuda',property(lambda _:True)), \
         patch.object(libdevice,'tanh',lambda x:lib_tanh(scalar_arg(x),_semantic=interpreter.interpreter_semantic)), \
         patch.object(libdevice,'erf',lambda x:lib_erf(scalar_arg(x),_semantic=interpreter.interpreter_semantic)), \
         patch.object(interpreter.InterpreterBuilder,'create_extern_elementwise',scalar_intrinsic):
        if amp:
            xx=x.half();ww=w.half();bb=bias.half() if bias is not None else None
            out=m.LinearLayer._forward_compute.__wrapped__(ctx,xx,ww,bb,act,m._SavedBuffer(saved))
        else:out=m.LinearLayer.forward(ctx,x,w,bias,act,saved)
    actual_x=x.half().float() if amp else x
    actual_w=w.half().float() if amp else w
    z=actual_x.reshape(rows,k)@actual_w.T
    if bias is not None:z=z+(bias.half().float() if amp else bias)
    expected=activation(z,act).to(torch.float16 if amp else torch.float32).reshape(shape[:-1]+(n,))
    expected_saved=z.half().float() if amp else z
    torch.testing.assert_close(out,expected,atol=5e-5 if not amp else 2e-3,rtol=3e-5 if not amp else 2e-3,equal_nan=True)
    torch.testing.assert_close(saved,expected_saved,atol=5e-5 if not amp else 2e-3,rtol=3e-5 if not amp else 2e-3,equal_nan=True)
    assert out.shape==expected.shape and runner.current==7
    assert len(runner.launches)==(1 if rows and n else 0)
    if runner.launches:
        args,kw,owner=runner.launches[0]
        assert args[1] is saved if not amp else args[1] is not saved
        assert owner==x.device
    assert ctx.saved[0].shape==w.shape and ctx.saved[2].shape==x.shape
    for t,old in kept:assert torch.equal(bytes_of(t),old)
    assert torch.equal(bytes_of(saved)[rows*n*4:],saved_before[rows*n*4:])
    cases+=1

for i in range(98):case(i,(1,1),1)
for i in (0,14,18,42,97):
    bk=configs[i].kwargs['BLOCK_K']
    for k in (0,bk-1,bk+3):
        for act in ('','tanh','relu','gelu','fast_gelu','unknown',None):
            for amp in (False,True):case(i,(3,k),5,act,amp)
for shape,n in [((0,3),4),((2,0,3),4),((2,3),0),((2,3,5),7)]:
    for amp in (False,True):case(18,shape,n,'relu',amp)
for tail in (1,2,3):case(18,(3,5),7,'relu',False,True,tail)

rejected=[]
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    x=torch.empty(2,3);w=torch.empty(4,3)
    invalid=[('nonTensor',[],w,None,'',None),('cpu_meta',torch.empty(2,3,device='meta'),w,None,'',None),
      ('sparse',x.to_sparse(),w,None,'',None),('negative',torch._neg_view(x),w,None,'',None),
      ('offset',torch.empty(7)[1:].view(2,3),w,None,'',None),
      ('unaligned',torch.frombuffer(bytearray(25),dtype=torch.float32,offset=1,count=6).view(2,3),w,None,'',None)]
    shared=bytearray(64)
    left=torch.frombuffer(shared,dtype=torch.float32,offset=0,count=6).view(2,3)
    right=torch.frombuffer(shared,dtype=torch.float32,offset=4,count=12).view(4,3)
    invalid.append(('overlap_different_bases',left,right,None,'',None))
    for name,*args in invalid:
        try:m._validate_inputs(*args)
        except (ValueError,TypeError):rejected.append(name)
        else:raise AssertionError(name)

failures=[]
for stage in ('allocation','enter','before','after','exit','copyback_before','copyback_after','save'):
    runner=Runner();runner.fail=stage
    x=torch.ones(2,3);w=torch.ones(4,3);saved=torch.full((2,4),23.);ctx=Ctx(stage=='save')
    inputs=[(v,bytes_of(v)) for v in (x,w)]
    original_copy=torch.Tensor.copy_
    def copy(self,source,*args,**kw):
        if self is saved and stage=='copyback_before':raise RuntimeError('copyback')
        result=original_copy(self,source,*args,**kw)
        if self is saved and stage=='copyback_after':raise RuntimeError('copyback')
        return result
    with patch.object(m,'torch',runner),patch.object(m,'kernel_fma',runner), \
         patch.object(torch.Tensor,'is_cuda',property(lambda _:True)),patch.object(torch.Tensor,'copy_',copy):
        try:
            if stage.startswith('copyback'):
                m.LinearLayer._forward_compute.__wrapped__(ctx,x.half(),w.half(),None,'',m._SavedBuffer(saved))
            else:m.LinearLayer.forward(ctx,x,w,None,'',saved)
        except RuntimeError:pass
        else:raise AssertionError(stage)
    assert runner.current==7
    expected=23. if stage in ('allocation','enter','before','copyback_before') else 3.
    assert torch.all(saved==expected),(stage,saved)
    for v,old in inputs:assert torch.equal(bytes_of(v),old)
    failures.append(stage)
print(f'{cases} actual CPU source/reference/frame cases; {len(rejected)} additional rejections; {len(failures)} ordinary failure prefixes passed')
