"""Actual CPU JIT/metadata/frame controls; no GPU numerical-conformance claim."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('kl_compute_support',D.parent/'cross_entropy1/Proofs/probe.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
torch=support.torch;torch.set_num_threads(1)
SOURCE=D/'kldiv_compute_repaired.py';ORIGINAL=D.parents[2]/'tritonbench_g/kldiv_compute/kldiv_compute.py'
for path,digest in json.loads((D/'source_pin.json').read_text())['files'].items():
    assert hashlib.sha256((D.parents[3]/path).read_bytes()).hexdigest()==digest
def kernel_ast(path):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='kldivergence_kernel')
assert ast.dump(kernel_ast(SOURCE),include_attributes=False)==ast.dump(kernel_ast(ORIGINAL),include_attributes=False)
class Kernel:
    def __init__(self,run,kernel):self.run=run;self.kernel=kernel
    def __getitem__(self,grid):
        def invoke(*args,**kwargs):
            self.run.launches.append((grid,args,kwargs))
            return self.kernel[grid](*args,**kwargs)
        return invoke
class Runner:
    def __init__(self,name,path=SOURCE):
        self.allocations=[];self.launches=[]
        spec=importlib.util.spec_from_file_location('kl_compute_'+name,path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.module.torch=support.TorchHost(self)
        self.module.kldivergence_kernel=Kernel(self,self.module.kldivergence_kernel)
x0=lambda:torch.tensor([1.,2.,3.]);y0=lambda:torch.tensor([2.,1.,4.])
r=Runner('guards');m=r.module;rejected=[];cases=[]
def reject(name,f,kind=(TypeError,ValueError)):
    try:f()
    except kind:rejected.append(name)
    else:raise AssertionError('admitted '+name)
reject('CPU',lambda:m.kldivergence(x0(),y0()))
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    for name,x,y in [
        ('not tensor',[],y0()),('fp16',x0().half(),y0()),('fp64',x0(),y0().double()),
        ('integer',x0().int(),y0()),('sparse',x0().to_sparse(),y0()),
        ('shape',x0().view(1,3),y0()),('device',torch.empty(3,device='meta'),y0()),
        ('x stride',torch.arange(6.)[::2],y0()),('y stride',x0(),torch.arange(6.)[::2]),
        ('negative x bit',torch._neg_view(x0()),y0()),('negative y bit',x0(),torch._neg_view(y0())),
        ('extent',torch.empty(1<<31,device='meta'),torch.empty(1<<31,device='meta')),
        ('x alignment',torch.frombuffer(bytearray(13),dtype=torch.float32,offset=1,count=3),y0()),
        ('y alignment',x0(),torch.frombuffer(bytearray(13),dtype=torch.float32,offset=1,count=3))]:
        reject(name,lambda x=x,y=y:m.kldivergence(x,y))
    def case(name,x,y):
        expected=x.double()*torch.log(x.double()/y.double())
        backing={t.untyped_storage()._cdata:(t,support.storage_bytes(t)) for t in (x,y)}
        run=Runner(name)
        with patch.object(torch.Tensor,'contiguous',side_effect=AssertionError('unexpected input packing')),patch.object(torch.Tensor,'resolve_neg',side_effect=AssertionError('unexpected normalization')):
            result=run.module.kldivergence(x,y)
        assert result is run.allocations[0] and len(run.allocations)==1
        assert result.shape==x.shape and result.dtype==torch.float32 and result.device==x.device
        assert result.untyped_storage()._cdata not in backing
        assert len(run.launches)==(1 if x.numel() else 0)
        if x.numel():
            args=run.launches[0][1];assert args[0] is x and args[1] is y and args[2] is result
        torch.testing.assert_close(result.double(),expected,atol=2e-5,rtol=2e-5,equal_nan=True)
        for t,old in backing.values():assert torch.equal(old,support.storage_bytes(t))
        cases.append(name);print(name,'passed',flush=True)
    case('scalar',torch.tensor(2.),torch.tensor(3.))
    case('empty',torch.empty(0),torch.empty(0))
    case('empty_rank3',torch.empty(2,0,3),torch.empty(2,0,3))
    for n in (1,3,1023,1024,1025,2049):case('tail_'+str(n),torch.linspace(.1,2,n),torch.linspace(2,.1,n))
    case('offsets',torch.arange(1.,14.)[2:8].view(2,3),torch.arange(1.,16.)[4:10].view(2,3))
    x=torch.arange(1.,8.);case('identical_inputs',x,x)
    x=torch.arange(1.,10.);case('overlapping_reads',x[:6],x[2:8])
    for tail in (1,2,3):case('partial_byte_tail_'+str(tail),support.partial_float_storage(tail),torch.ones(2,3))
    for shape,stride in [((1,3),(77,1)),((2,1),(1,7)),((0,4),(19,7))]:
        x=torch.empty_strided(shape,stride).fill_(2);y=torch.empty_strided(shape,stride).fill_(3)
        case('contiguous_degenerate_'+str(shape),x,y)
    case('nonfinite',torch.tensor([0.,-0.,-1.,float('nan'),float('inf'),1.,1e-40]),
         torch.tensor([1.,1.,1.,2.,2.,0.,1.]))
assert not r.launches and not r.allocations
# The unchanged original ignores logical stride and reads the first two words.
original_body=ORIGINAL.read_text().split('def test_kldivergence',1)[0]
with tempfile.TemporaryDirectory(prefix='veritile-kl-original-') as temp:
    path=Path(temp)/'original.py';path.write_text(original_body)
    run=Runner('original_stride',path)
    with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
        x=torch.tensor([1.,1.]);y=torch.tensor([1.,1.,2.])[::2]
        out=run.module.kldivergence(x,y)
    assert torch.equal(out,torch.zeros(2))
    assert not torch.allclose(out,x*torch.log(x/y))
print('original stride counterexample reproduced',flush=True)
print(f'{len(rejected)} rejected; {len(cases)} actual source/reference/frame cases passed')
