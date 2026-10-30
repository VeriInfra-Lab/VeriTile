"""CPU interpreter checks of guarded CE2 source and exact caller frames."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ce2_probe',D/'Proofs/probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
torch=probe.torch;torch.set_num_threads(1)
checked=D/'cross_entropy2_repaired.py';conditional=D/'cross_entropy2_conditional.py'
assert hashlib.sha256(conditional.read_bytes()).hexdigest()=='741315fb177ec4cf0c7325c8244c87a8cbc95921511d81e41468d304ce5e8b6a'
def asts(path,strip):
    nodes=[]
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.FunctionDef) and (n.name.endswith('_kernel') or n.name in ('cross_entropy_fwd','cross_entropy_bwd')):
            if strip:
                class Checks(ast.NodeTransformer):
                    def visit_Expr(self,node):
                        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_check_row':return None
                        return self.generic_visit(node)
                    def visit_Name(self,node):
                        if node.id=='_checked_validate':node.id='_validate'
                        return node
                n=Checks().visit(n)
            nodes.append(ast.dump(n,include_attributes=False))
    return nodes
assert asts(checked,True)==asts(conditional,False)
dense=lambda:torch.tensor([[0.,1.,2.],[2.,0.,1.]])
labels=lambda:torch.tensor([0,1],dtype=torch.int32)
r=probe.Runner(checked,'guard_checks');m=r.module
rejected=[]
def reject(name,thunk):
    try:thunk()
    except (TypeError,ValueError):rejected.append(name)
    else:raise AssertionError('admitted '+name)
def forward(**changes):
    k=dict(x=dense(),y=labels(),s=0.,a=1.,z=0.,ignored=-1,total=3,start=0,block=4,has=False,split=False);k.update(changes)
    return m.cross_entropy_fwd(*[k[n] for n in ('x','y','s','a','z','ignored','total','start','block','has','split')])
reject('CPU',lambda:forward())
records=[]
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)), patch.object(torch.cuda,'current_device',lambda:None):
    with patch.object(torch.cuda,'current_device',lambda:0):
        reject('different current CUDA launch device',lambda:forward())
    for name,k in [
        ('non Tensor',{'x':[]}),('FP64',{'x':dense().double()}),('int64 labels',{'y':labels().long()}),
        ('rank1',{'x':torch.arange(3.)}),('zero classes',{'x':torch.empty(2,0)}),('label shape',{'y':labels()[:1]}),
        ('column stride',{'x':torch.arange(12.).reshape(2,6)[:,::2]}),('label stride',{'y':torch.tensor([0,9,1],dtype=torch.int32)[::2]}),
        ('negative view',{'x':torch._neg_view(dense())}),('negative label view',{'y':torch._neg_view(labels())}),
        ('integer smoothing',{'s':0}),('integer logit scale',{'a':1}),('integer square scale',{'z':0}),
        ('nan smoothing',{'s':float('nan')}),('infinite scale',{'a':float('inf')}),('nan z',{'z':float('nan')}),
        ('negative smoothing',{'s':-.1}),('large smoothing',{'s':1.1}),('negative z',{'z':-.1}),
        ('boolean ignored',{'ignored':False}),('small ignored',{'ignored':-(1<<63)-1}),('large ignored',{'ignored':1<<63}),
        ('noninteger total',{'total':3.}),('zero total',{'total':0}),('large total',{'total':1<<63}),
        ('negative start',{'start':-1}),('invalid shard',{'start':1}),
        ('boolean block',{'block':True}),('zero block',{'block':0}),('nonpower block',{'block':3}),('large block',{'block':131072}),
        ('integer smoothing flag',{'has':1}),('integer split flag',{'split':0}),
        ('negative label',{'y':torch.tensor([-2,1],dtype=torch.int32)}),('large label',{'y':torch.tensor([0,3],dtype=torch.int32)}),
    ]:reject(name,lambda k=k:forward(**k))
    for name,g,l in [('upstream type',[],torch.ones(2)),('upstream dtype',torch.ones(2,dtype=torch.float64),torch.ones(2)),
                     ('upstream shape',torch.ones(3),torch.ones(2)),('upstream stride',torch.ones(4)[::2],torch.ones(2)),
                     ('upstream negative',torch._neg_view(torch.ones(2)),torch.ones(2)),('LSE dtype',torch.ones(2),torch.ones(2,dtype=torch.float64)),
                     ('LSE stride',torch.ones(2),torch.ones(4)[::2])]:
        reject(name,lambda g=g,l=l:m.cross_entropy_bwd(g,dense(),l,labels(),0.,1.,0.,-1,3,0,4,False))
    def case(*args,**kwargs):
        item=probe.case(*args,**kwargs);records.append(item);print(item['case'],'passed',flush=True)
    for scale in (1.,0.,-2.):
        for has in (False,True):
            for split in (False,True):
                case(f'tail_scale{scale}_smoothing{has}_split{split}',dense(),labels(),a=scale,s=.2,z=.1,has=has,split=split)
    for block in (1,2,8,8192,65536):case('width_'+str(block),dense(),labels(),block=block,s=.1,z=.2,has=True)
    case('contiguous_offset',torch.arange(12.).reshape(4,3)[1:3],labels(),block=2)
    case('proper_shard_nonowned_targets',dense(),torch.tensor([0,8],dtype=torch.int32),total=10,start=2,has=True,s=.2,z=.1,block=2)
    case('ignored_int32_min',dense(),torch.tensor([-(1<<31),1],dtype=torch.int32),ignored=-(1<<31),total=(1<<63)-1,start=(1<<63)-4,has=True,s=.2)
    case('large_ignored_never_truncates',dense(),labels(),ignored=1<<32,split=True)
    case('ignored_nonfinite',torch.tensor([[float('nan'),float('inf'),-float('inf')],[1.,2.,3.]]),
         torch.tensor([-1,0],dtype=torch.int32),g=torch.tensor([float('nan'),1.]),split=True,exceptional=True)
    case('empty_batch',torch.empty(0,3),torch.empty(0,dtype=torch.int32),block=2,split=True)
    storage=torch.UntypedStorage(27);raw=torch.empty(0,dtype=torch.uint8).set_(storage,0,(27,),(1,));raw.fill_(173)
    x=torch.empty(0).set_(storage,0,(2,3),(3,1));x.copy_(dense());case('partial_backing_tail',x,labels())
    backing=torch.tensor([0,1,2,0,1,2],dtype=torch.int32);case('shared_int32_FP32_storage',backing.view(torch.float32).reshape(2,3),backing[:2])
    # A legitimate standalone backward uses the supplied LSE, even when it is
    # deliberately different from the forward logsumexp.
    x=dense();y=labels();g=torch.tensor([1.,2.]);lse=torch.tensor([.25,1.5]);before=[probe.raw(t) for t in (x,y,g,lse)]
    grad=m.cross_entropy_bwd(g,x,lse,y,.2,-1.,.1,-1,3,0,2,True)
    probability=torch.exp(-x.double()-lse.double()[:,None]);want=(1+.2*lse.double()[:,None])*probability-.2/3
    want.scatter_add_(1,y.long()[:,None],torch.full((2,1),-.8,dtype=torch.float64));want*=(-g.double())[:,None]
    torch.testing.assert_close(grad.double(),want,rtol=8e-5,atol=8e-5)
    for t,b in zip((x,y,g,lse),before):assert torch.equal(probe.raw(t),b)
    records.append({'case':'standalone_supplied_LSE'})
assert not r.launches[:-1], 'guard rejection launched a kernel'
print(f'{len(rejected)} rejected; {len(records)} source/reference/frame cases passed; no GPU conformance claim')
