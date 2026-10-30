"""Actual checked production Triton CPU source and actual host guard controls.
No GPU TF32/WGMMA or native exceptional-value conformance is claimed.
"""
import os
os.environ['TRITON_INTERPRET']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast,contextlib,hashlib,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('leaky_source',D/'matmul_leakyrelu_fp8_scalar_cast.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert hashlib.sha256((D/'matmul_leakyrelu_fp8_scalar_cast.py').read_bytes()).hexdigest() == '185ae6684de33dfff4c28b33c11f824338e585f3a283712bf4f32f2d76829457'
raw=m.matmul_kernel.fn
cfg=m.matmul_kernel.configs[0]
assert cfg.num_stages==2 and cfg.num_warps==8

def raw_bytes(t):
    return torch.empty(0,dtype=torch.uint8).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).clone()
class Runner:
    def __init__(self):self.current=7;self.launches=[];self.allocations=[];self.fail=None
    def __getattr__(self,name):return getattr(torch,name)
    @property
    def cuda(self):return self
    @contextlib.contextmanager
    def device(self,owner):
        if self.fail=='enter':raise RuntimeError('enter')
        previous=self.current;self.current=owner
        try:yield
        finally:
            self.current=previous
            if self.fail=='exit':raise RuntimeError('exit')
    def empty(self,*args,**kw):
        if self.fail=='allocation':raise RuntimeError('allocation')
        kw['device']='cpu';out=torch.empty(*args,**kw);out.fill_(9.);self.allocations.append(out);return out
    def __getitem__(self,grid):
        def launch(*args,**kw):
            if self.fail=='before':raise RuntimeError('before')
            kw.update(cfg.kwargs)
            assert self.current==args[0].device
            self.launches.append((args,kw))
            raw[grid(kw)](*args,**kw,num_warps=cfg.num_warps,num_stages=cfg.num_stages)
            if self.fail=='after':raise RuntimeError('after')
        return launch

cases=0

def case(M,N,K,activation='',layout='dense'):
    global cases
    run=Runner()
    a=((torch.arange(M*K,dtype=torch.float32).reshape(M,K)%7)-3)/16
    if layout=='tail':
        backing=bytearray(M*K*4+3)
        a=torch.frombuffer(backing,dtype=torch.float32,count=M*K).view(M,K);a.fill_(.125)
    if layout=='offset':
        backing=torch.full((M*K+4,),3.);a=backing[2:2+M*K].view(M,K)
    if layout=='transpose':b=(((torch.arange(N*K,dtype=torch.float32).reshape(N,K)%11)-5)/16).T
    elif layout=='broadcast':b=torch.arange(N,dtype=torch.float32).div(16).expand(K,N)
    elif layout=='alias':
        assert M==K==N;b=a.T
    elif layout=='offset':
        backing=torch.arange(K*(N+2)+6,dtype=torch.float32).div(32)
        b=backing.as_strided((K,N),(N+2,1),3)
    else:b=((torch.arange(K*N,dtype=torch.float32).reshape(K,N)%11)-5)/16
    before=[(t,raw_bytes(t)) for t in (a,b)]
    with patch.object(m,'torch',run),patch.object(m,'matmul_kernel',run),patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
        out=m.matmul(a,b,activation)
    expected=a@b
    if activation=='leaky_relu':expected=torch.where(expected>=0,expected,.01*expected)
    torch.testing.assert_close(out,expected.half(),atol=2e-3,rtol=2e-3,equal_nan=True)
    assert out is run.allocations[0] and out.dtype==torch.float16 and out.shape==(M,N)
    assert out.stride()==(max(N,1),1) and out.storage_offset()==0
    assert run.current==7 and len(run.launches)==(1 if M and N and K else 0)
    assert out.untyped_storage().data_ptr() not in [t.untyped_storage().data_ptr() for t in (a,b)] or out.numel()==0
    for t,old in before:assert torch.equal(raw_bytes(t),old)
    cases+=1

for M,N,K in [(1,1,1),(3,5,0),(0,3,5),(3,0,5),(5,7,1),(5,7,63),(5,7,64),(5,7,65),(129,3,3),(3,257,3),(1025,3,1)]:
    for activation in ('','leaky_relu','fallback'):case(M,N,K,activation)
for layout in ('transpose','broadcast','offset','tail'):
    for activation in ('','leaky_relu'):case(3,5,67,activation,layout)
for activation in ('','leaky_relu'):case(3,3,3,activation,'alias')

rejected=[]
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    a=torch.empty(2,3);b=torch.empty(3,4)
    invalid=[('not_tensor',[],b,''),('sparse',a.to_sparse(),b,''),('rank',a.view(6),b,''),
      ('half_fixture',a.half(),b,''),('bfloat',a,b.bfloat16(),''),('shape',a,torch.empty(2,4),''),
      ('contiguous',a.T,torch.empty(2,4),''),('neg',torch._neg_view(a),b,''),
      ('activation',a,b,None),('device',torch.empty(2,3,device='meta'),b,''),
      ('unaligned',torch.frombuffer(bytearray(25),dtype=torch.float32,offset=1,count=6).view(2,3),b,'')]
    for name,*args in invalid:
        try:m._validate_inputs(*args)
        except (ValueError,TypeError):rejected.append(name)
        else:raise AssertionError(name)
try:m._validate_inputs(torch.empty(2,3),torch.empty(3,4),'')
except ValueError:rejected.append('cpu')
else:raise AssertionError('cpu')

failures=[]
for stage in ('allocation','enter','before','after','exit'):
    run=Runner();run.fail=stage;a=torch.ones(2,3);b=torch.ones(3,4)
    before=[(v,raw_bytes(v)) for v in (a,b)]
    with patch.object(m,'torch',run),patch.object(m,'matmul_kernel',run),patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
        try:m.matmul(a,b)
        except RuntimeError:pass
        else:raise AssertionError(stage)
    assert run.current==7
    for v,old in before:assert torch.equal(raw_bytes(v),old)
    failures.append(stage)

# Original wrapper accepted lazy negative views; raw Triton pointer loads do
# not apply the view's logical sign. Both repaired operands reject that input.
a=torch.arange(6,dtype=torch.float32).reshape(2,3);b=torch.ones(3,4)
assert not torch.equal(torch._neg_view(a)@b,a@b)
print(f'{cases} actual CPU source/reference/frame cases; {len(rejected)} rejections; {len(failures)} ordinary failure prefixes; 1 original logical-view counterexample passed')

report={'complete_cpu_source_executions':cases, 'rejected_inputs':rejected, 'ordinary_failure_controls':failures, 'gpu_execution':False, 'native_tf32_numerical_conformance':False}
(Path('.veritile-cache/kernel-evidence/matmul_leakyrelu_fp8_scalar_cast')/'cpu-source-controls.json').write_text(json.dumps(report,indent=2)+'\n')
