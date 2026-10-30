"""CPU execution of the actual four-loop source; no native numerical claim."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('flaggems_cpu',HERE.parent/'softmax_flaggems/check_contract.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
Tensor,Language=h.Tensor,h.Language
class JIT:
    def __init__(self,f,tl,calls):self.f,self.tl,self.calls=f,tl,calls
    def __call__(self,*args,**kwargs):return self.f(*args,**kwargs)
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            fusion=kwargs.pop('enable_fp_fusion',True)
            warps=kwargs.pop('num_warps',4);stages=kwargs.pop('num_stages',3)
            self.calls.append((grid,args[-1],fusion,warps,stages))
            for row in range(grid[0]):
                self.tl.pids=(row,0)
                with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
                    self.f(*args,**kwargs)
        return launch

def load(path):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
    class Cast(ast.NodeTransformer):
        def visit_Attribute(self,n):
            self.generic_visit(n)
            if n.attr=='element_ty' and isinstance(n.value,ast.Attribute) and n.value.attr=='dtype':
                return ast.copy_location(ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='float32',ctx=ast.Load()),n)
            return n
    module=ast.fix_missing_locations(Cast().visit(ast.Module(body=nodes,type_ignores=[])))
    tl,calls,allocs=Language(),[],[]
    tri=SimpleNamespace(next_power_of_2=lambda n:1<<(max(n,1)-1).bit_length())
    tri.jit=lambda f:JIT(f,tl,calls)
    def empty_like(x):
        out=Tensor(torch.empty_like(x.value));allocs.append(out);return out
    @contextmanager
    def device(d):assert d=='cuda:0';yield
    tor=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,empty_like=empty_like,cuda=SimpleNamespace(device=device))
    ns={'torch':tor,'triton':tri,'tl':tl};exec(compile(module,str(path),'exec'),ns)
    return ns,calls,allocs

def binding():
    old=ast.parse((ROOT/'bench/tritonbench_g/softmax_optimize/softmax_optimize.py').read_text())
    new=ast.parse((HERE/'softmax_optimize_repaired.py').read_text())
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='softmax_kernel_online_v2')
    b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='softmax_kernel_online_v2_repaired')
    b.name=a.name
    class Restore(ast.NodeTransformer):
        def visit_arg(self,n):
            if n.arg in ('M','N'):n.annotation=None
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and n.args and ast.unparse(n.args[0])=='tl.int64':return n.func.value
            return n
        def visit_For(self,n):
            self.generic_visit(n)
            if any(isinstance(k,ast.Assign) and ast.unparse(k.targets[0])=='old_mass' for k in n.body):
                oldmass=next(k for k in n.body if isinstance(k,ast.Assign) and ast.unparse(k.targets[0])=='old_mass')
                newmass=next(k for k in n.body if isinstance(k,ast.Assign) and ast.unparse(k.targets[0])=='new_mass')
                assert ast.unparse(oldmass.value)=='tl.where(new_m == -float(\'inf\'), z, tl.exp(m - new_m) * z)'
                assert ast.unparse(newmass.value)=='tl.where(inp == -float(\'inf\'), 0.0, tl.exp(inp - new_m))'
                n.body.remove(oldmass);n.body.remove(newmass)
                k=next(k for k in n.body if isinstance(k,ast.Assign) and ast.unparse(k.targets[0])=='new_z')
                k.value=ast.parse('tl.exp(m - new_m) * z + tl.exp(inp - new_m)',mode='eval').body
            return n
    assert ast.dump(Restore().visit(a))==ast.dump(Restore().visit(b))
    kernel=(HERE/'Kernels.lean').read_text()
    assert kernel.count('for start_n in range')==4 and kernel.count('old_mass = tl.where')==2
    assert 'import bench.repairs' not in kernel

def tensor(value):return Tensor(torch.as_tensor(value,dtype=torch.float32).clone())
def main():
    binding();new,calls,allocs=load(HERE/'softmax_optimize_repaired.py')
    old,_,_=load(ROOT/'bench/tritonbench_g/softmax_optimize/softmax_optimize.py')
    accepted=rejected=counterexamples=0
    for m,n in [(0,0),(0,3),(2,0),(1,1),(2,3),(3,7),(1,33),(1,4095),(2,4096),(2,4097),(1,8193)]:
        x=tensor(torch.linspace(-1,1,m*n).reshape(m,n));before=x.value.clone();count=len(calls)
        y=new['softmax_repaired'](x)
        torch.testing.assert_close(y.value,torch.softmax(before,dim=1),rtol=4e-6,atol=3e-7)
        assert y is allocs[-1] and torch.equal(x.value,before)
        assert len(calls)-count==int(m*n>0);accepted+=1
    for value in [[[float('-inf'),0.,1.]],[[0.,float('-inf'),1.]],[[float('inf'),1.,0.]],[[float('nan'),1.,0.]],[[float('-inf')]*3]]:
        x=tensor(value);y=new['softmax_repaired'](x)
        torch.testing.assert_close(y.value,torch.softmax(x.value,dim=1),equal_nan=True);accepted+=1
    x=Tensor(torch.arange(32,dtype=torch.float32)[3:9].view(2,3));new['softmax_repaired'](x);accepted+=1
    x=tensor(torch.empty(0,3));x.metadata['forbid_pointer']=True;new['softmax_repaired'](x);assert x.pointer_queries==0;accepted+=1
    def reject(x):
        nonlocal rejected
        count=len(calls);alloccount=len(allocs)
        try:new['softmax_repaired'](x)
        except (TypeError,ValueError):rejected+=1
        else:raise AssertionError('unsupported tensor admitted')
        assert len(calls)==count and len(allocs)==alloccount
    reject(None);reject(tensor(1.));reject(tensor([1.,2.]));reject(tensor(torch.ones(1,2,3)))
    for meta in ({'dtype':torch.float16},{'dtype':torch.float64},{'layout':torch.sparse_coo},{'is_cuda':False},
                 {'contiguous':False},{'negative':True},{'numel':2**31},{'pointer':1},{'pointer':2},{'pointer':3}):
        x=tensor(torch.ones(2,3));x.metadata.update(meta);reject(x)
    for value in [[[1.,2.,3.]],[[float('-inf'),0.,1.,2.]]]:
        x=tensor(value);y=old['softmax'](x)
        assert torch.isnan(y.value).all();counterexamples+=1
    x=tensor(torch.empty(1,0))
    try:old['softmax'](x)
    except IndexError:counterexamples+=1
    else:raise AssertionError('missing original negative-offset failure')
    assert all(f and w==4 and s==3 for _,_,f,w,s in calls)
    print(f'passed: {accepted} admissions, {rejected} rejections, {counterexamples} original counterexamples; {len(calls)} launches')
    print('all four source loops, finite and exceptional words, offset/empty matrices; CPU only')
if __name__=='__main__':main()
