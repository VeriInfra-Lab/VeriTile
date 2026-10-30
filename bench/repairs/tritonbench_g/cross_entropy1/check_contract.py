"""Checked wrapper controls using the actual JIT CPU interpreter and autograd.
Only Tensor.is_cuda and the CUDA context are adapted; production guards remain intact.
This checks source behavior and frames, not installed GPU numerical conformance.
"""
import ast
import hashlib
import importlib.util
import pathlib
import sys
from unittest.mock import patch
sys.dont_write_bytecode = True
D = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ce1_probe', D/'Proofs/probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
torch = probe.torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)
source = D/'cross_entropy1_repaired.py'
probe.SOURCE = source
conditional = D/'cross_entropy1_conditional.py'
assert hashlib.sha256(conditional.read_bytes()).hexdigest() == 'edfc7f0c6adb8c9a2050660c04d43034083fdefb58f068192208d91f676278d5'
def kernels(path):
    return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name.endswith('_kernel')}
assert kernels(source) == kernels(conditional)
# Removing the two checks must recover the exact accepted forward/backward bodies.
def host(path, strip):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CrossEntropyLoss')
    if strip:
        for f in cls.body:
            if isinstance(f,ast.FunctionDef):
                f.body=[n for n in f.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                    and isinstance(n.value.func,ast.Name) and n.value.func.id.startswith('_check_'))]
    return ast.dump(cls,include_attributes=False)
assert host(source,True)==host(conditional,False)
runner=probe.Runner('guard_controls')
m=runner.module
dense=lambda: torch.tensor([[0.,1.,2.],[2.,0.,1.]])
labels=lambda: torch.tensor([0,1])
rejected=[]
def reject(name, thunk, exception=(ValueError,TypeError,ZeroDivisionError)):
    try: thunk()
    except exception: rejected.append(name)
    else: raise AssertionError('admitted '+name)
def check(x=None,y=None,s=0.,z=0.,ignored=-100,inplace=False,group=None):
    m._check_forward(dense() if x is None else x,labels() if y is None else y,s,z,ignored,inplace,group)
reject('CPU device',lambda:check())
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    for name,kw in [
        ('non Tensor',{'x':[]}),('FP64',{'x':dense().double()}),('int32 labels',{'y':labels().int()}),
        ('rank1',{'x':torch.arange(3.)}),('labels shape',{'y':torch.tensor([0])}),
        ('column stride',{'x':torch.arange(12.).reshape(2,6)[:,::2]}),
        ('strided labels',{'y':torch.tensor([0,9,1])[::2]}),
        ('negative logits',{'x':torch._neg_view(dense())}),('negative labels',{'y':torch._neg_view(labels())}),
        ('integer smoothing',{'s':0}),('integer z',{'z':0}),('nan smoothing',{'s':float('nan')}),
        ('infinite smoothing',{'s':float('inf')}),('negative smoothing',{'s':-.1}),('large smoothing',{'s':1.1}),
        ('nan z',{'z':float('nan')}),('infinite z',{'z':float('inf')}),('negative z',{'z':-.1}),
        ('boolean ignored',{'ignored':True}),('small ignored',{'ignored':-(1<<63)-1}),
        ('large ignored',{'ignored':1<<63}),('integer inplace',{'inplace':1}),('process group',{'group':object()}),
        ('negative label',{'y':torch.tensor([-1,1])}),('large label',{'y':torch.tensor([0,3])}),
        ('empty classes',{'x':torch.empty(2,0)}),
    ]: reject(name,lambda kw=kw:check(**kw))
    for name,dy in [('upstream type',[]),('upstream dtype',torch.ones(2,dtype=torch.float64)),
                    ('upstream shape',torch.ones(3)),('negative upstream',torch._neg_view(torch.ones(2)))]:
        reject(name,lambda dy=dy:m._check_backward(dy,dense()))
    cases=[]
    def case(*args,**kwargs):
        result=probe.run_case(*args,**kwargs);cases.append(result);print(result['case'],'passed',flush=True)
    case('dense',dense(),labels())
    case('smoothing_sum_broadcast_upstream',dense(),labels(),smoothing=.1,sum_loss=True)
    case('smoothing_z',dense(),labels(),smoothing=.2,scale=.1,dy=torch.tensor([1.,2.]))
    case('mixed_ignored',dense(),torch.tensor([-100,1]),smoothing=.1,scale=.2)
    case('int64_min_ignored',dense(),torch.tensor([-(1<<63),1]),ignored=-(1<<63))
    case('contiguous_offset',torch.arange(12.).reshape(4,3)[1:3],labels(),inplace=True)
    case('strided_upstream',dense(),labels(),dy=torch.tensor([1.,9.,2.])[::2])
    x=dense();case('inplace_upstream_alias',x,labels(),dy=x[:,0],inplace=True)
    y=labels();case('inplace_label_storage_alias',y.view(torch.float32).reshape(2,2),y,inplace=True)
    case('single_class',torch.tensor([[2.],[1.]]),torch.tensor([0,0]),scale=.2)
    case('empty_batch',torch.empty(0,3),torch.empty(0,dtype=torch.int64),dy=torch.empty(0))
    for tail in [1,2,3]:
        case('partial_backing_'+str(24+tail),probe.partial_float_storage(tail),labels(),inplace=tail==2)
    case('ignored_nan_inf',torch.tensor([[float('nan'),1.],[float('inf'),0.],[0.,float('-inf')],[float('nan'),1.]]),
         torch.tensor([-100,-100,-100,0]),exceptional=True)
    case('backward_tail_4097',torch.zeros(1,4097),torch.tensor([4096]),smoothing=.1,scale=.2)
    case('forward_local_split_65537',torch.zeros(1,65537),torch.tensor([65536]),smoothing=.1,scale=.2)
    # Saved version failure remains before guard/snapshot/launch in backward.
    version=probe.Runner('saved_version')
    x=dense().requires_grad_();loss=version.module.cross_entropy_loss(x,labels())
    with torch.no_grad():x.add_(1)
    reject('saved tensor version',lambda:torch.autograd.grad(loss.sum(),x),RuntimeError)
    assert len(version.launches)==1
print(f'{len(rejected)} rejected; {len(cases)} checked source/autograd/frame cases passed; no GPU numerical claim')
