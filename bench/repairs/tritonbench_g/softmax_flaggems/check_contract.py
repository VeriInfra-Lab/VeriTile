"""CPU interpretation of the actual four kernel bodies and autograd wrapper."""
import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('ksoftmax_cpu',HERE.parent/'ksoftmax_triton/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
Tensor,Pointer,Array=helpers.Tensor,helpers.Pointer,helpers.Array

class Language(helpers.Language):
    def load(self,pointer,mask=None,other=None,**unused):
        if isinstance(pointer,Tensor):pointer=Pointer(pointer,0)
        if mask is None:mask=np.ones(pointer.offsets.shape,dtype=bool)
        return super().load(pointer,mask,2.0 if other is None else other)
    def store(self,pointer,value,mask=None):
        if isinstance(pointer,Tensor):pointer=Pointer(pointer,0)
        if mask is None:mask=np.ones(pointer.offsets.shape,dtype=bool)
        return super().store(pointer,value,mask)
    def full(self,shape,value,dtype):return np.full(shape,value,dtype=dtype).view(Array)
    def zeros(self,shape,dtype):return np.zeros(shape,dtype=dtype).view(Array)
    def maximum(self,x,y):return np.maximum(x,y).view(Array)
    def cdiv(self,a,b):return (a+b-1)//b
class JIT:
    def __init__(self,f,tl,calls,namespace):self.f,self.tl,self.calls,self.ns=f,tl,calls,namespace
    def __call__(self,*args,**kw):return self.f(*args,**kw)
    def __getitem__(self,grid):
        def launch(*args,**kw):
            fusion=kw.pop('enable_fp_fusion',True)
            name=self.f.__name__;backward='backward' in name;noninner='non_inner' in name
            pos=3 if backward else 2;M,N=args[pos:pos+2];K=args[pos+2] if noninner else 1
            configs=(kw.pop("_cpu_tile"),) if "_cpu_tile" in kw else ((32,64,128,256,512,1024) if backward else (None,))
            for tile in configs:
                meta={'M':M,'N':N,'K':K,'NUM_SMS':kw.get('NUM_SMS',80)}
                if noninner:
                    C=tile if backward else self.ns['heur_tile_k'](meta)
                    B=max(1,1024//C) if backward else (8192+C-1)//C
                    meta.update(TILE_K=C,TILE_N=B)
                    kernelkw={'TILE_K':C,'TILE_N':B,'ONE_TILE_PER_CTA':B>=N}
                    if not backward and 'NUM_SMS' in self.f.__annotations__:kernelkw['NUM_SMS']=meta['NUM_SMS']
                else:
                    B=tile if backward else self.ns['heur_tile_n_inner'](meta)
                    C=max(1,1024//B) if backward else 1
                    meta.update(TILE_M=C,TILE_N=B)
                    kernelkw={'TILE_N':B,'ONE_TILE_PER_CTA':B>=N}
                    if backward:kernelkw['TILE_M']=C
                dims=grid(meta) if callable(grid) else grid
                self.calls.append((name,dims,B,C,B>=N,fusion))
                for pm in range(dims[0]):
                    for pk in range(dims[1]):
                        self.tl.pids=(pm,pk)
                        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                            self.f(*args,**kernelkw)
        return launch

class Context:
    def save_for_backward(self,*args):self.saved_tensors=args
class Function:
    @classmethod
    def apply(cls,*args):
        ctx=Context();out=cls.forward(ctx,*args);out.metadata['ctx']=ctx;return out

def load(path):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and not (isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))]
    class Projection(ast.NodeTransformer):
        def visit_Attribute(self,n):
            self.generic_visit(n)
            if n.attr=='element_ty' and isinstance(n.value,ast.Attribute) and n.value.attr=='dtype':
                return ast.copy_location(ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='float32',ctx=ast.Load()),n)
            return n
    module=ast.fix_missing_locations(Projection().visit(ast.Module(body=nodes,type_ignores=[])))
    tl,calls,ns=Language(),[],{}
    tri=SimpleNamespace(next_power_of_2=lambda n:1<<(max(n,1)-1).bit_length(),cdiv=lambda a,b:(a+b-1)//b,Config=lambda *a,**k:None)
    tri.jit=lambda f:JIT(f,tl,calls,ns)
    tri.heuristics=lambda *a,**k:lambda f:f
    tri.autotune=lambda *a,**k:lambda f:f
    allocations=[]
    def empty_like(x,dtype=None):
        out=Tensor(torch.empty_like(x.value,dtype=dtype));allocations.append(out);return out
    @contextmanager
    def device(d):assert d=='cuda:0';yield
    tor=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,empty_like=empty_like,
        cuda=SimpleNamespace(device=device,get_device_properties=lambda d:SimpleNamespace(multi_processor_count=80)),
        autograd=SimpleNamespace(Function=Function))
    ns.update(torch=tor,triton=tri,tl=tl,logging=logging,MAX_TILE_K=8192,NUM_SMS=80)
    # Under the original wrapper, contiguous() should be a no-op on our checked
    # inputs; preserve the mocked descriptor through this existing API operation.
    exec(compile(module,str(path),'exec'),ns)
    return ns,calls,allocations

def binding():
    a=ast.parse((ROOT/'bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py').read_text())
    b=ast.parse((HERE/'softmax_flaggems_repaired.py').read_text())
    class Original(ast.NodeTransformer):
        def visit_arg(self,n):
            if n.arg in ('M','N','K'):n.annotation=None
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and n.args and ast.unparse(n.args[0])=='tl.int64':return n.func.value
            return n
    for name in ('softmax_kernel_inner','softmax_kernel_non_inner','softmax_backward_kernel_inner','softmax_backward_kernel_non_inner'):
        x=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
        y=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if name=='softmax_kernel_non_inner':y.args.args=[p for p in y.args.args if p.arg!='NUM_SMS']
        if 'backward' in name:
            for n in ast.walk(y):
                if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load':
                    others=[k for k in n.keywords if k.arg=='other'];assert len(others)==1 and ast.literal_eval(others[0].value)==0
                    n.keywords.remove(others[0])
        assert ast.dump(Original().visit(x))==ast.dump(Original().visit(y)),name

def tensor(x):return Tensor(torch.as_tensor(x,dtype=torch.float32).clone())
def main():
    binding();new,calls,allocs=load(HERE/'softmax_flaggems_repaired.py')
    old,oldcalls,_=load(ROOT/'bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py')
    accepted=rejected=failures=0
    # Wrapper cases preserve arbitrary selected axes, all four kernel routes,
    # all six backward tuning choices, forward multi-tile loops and empty axes.
    for shape,dim in [((3,),-1),((2,3),0),((2,3),1),((2,3,7),1),((2,3,7),-1),
                      ((1,32769),1),((1,8193,2),1),((2,0,3),0),((0,2,3),1),((2,3,0),-1)]:
        count=int(np.prod(shape));x=tensor(torch.linspace(-1,1,count).reshape(shape))
        y=new['softmax'](x,dim);expected=torch.softmax(x.value,dim)
        torch.testing.assert_close(y.value,expected,rtol=3e-6,atol=3e-7)
        assert y is not x and y.shape==x.shape and y.ctx.saved_tensors==(y,)
        g=tensor(torch.linspace(-.5,.5,count).reshape(shape))
        grad,dimgrad,dtypegrad=new['Softmax'].backward(y.ctx,g)
        expectedgrad=expected*(g.value-(g.value*expected).sum(dim,keepdim=True))
        torch.testing.assert_close(grad.value,expectedgrad,rtol=1e-4,atol=3e-7)
        assert dimgrad is None and dtypegrad is None
        accepted+=2
    x=tensor(torch.arange(12).reshape(2,2,3));y=new['softmax'](x,1,torch.float32);accepted+=1
    # Every empty input bypasses pointer inspection and kernels; allocations are
    # still the original API's fresh empty forward/gradient outputs.
    x=tensor(torch.empty(0,3));x.metadata['forbid_pointer']=True
    before=len(calls);y=new['softmax'](x,1);g=tensor(torch.empty(0,3));g.metadata['forbid_pointer']=True
    new['Softmax'].backward(y.ctx,g);assert len(calls)==before and x.pointer_queries==g.pointer_queries==0;accepted+=1
    def reject(x,dim=-1,dtype=None):
        nonlocal rejected
        before=len(calls)
        try:new['softmax'](x,dim,dtype)
        except (TypeError,ValueError):rejected+=1
        else:raise AssertionError('unsupported input admitted')
        assert len(calls)==before
    reject(None)
    for meta in ({'dtype':torch.float16},{'dtype':torch.float64},{'layout':torch.sparse_coo},{'is_cuda':False},
                 {'contiguous':False},{'negative':True},{'numel':2**31},{'pointer':1}):
        x=tensor(torch.ones(2,3));x.metadata.update(meta);reject(x)
    for dim in (-3,2,True,1.0):reject(tensor(torch.ones(2,3)),dim)
    reject(tensor(1.0));reject(tensor(torch.ones(2,3)),1,torch.float16)
    x=tensor(torch.ones(2,3));y=new['softmax'](x,1)
    for meta in ({'dtype':torch.float16},{'contiguous':False},{'negative':True},{'device':'cuda:1'},{'shape':(3,2)},{'pointer':1}):
        g=tensor(torch.ones(2,3));g.metadata.update(meta)
        try:new['Softmax'].backward(y.ctx,g)
        except (TypeError,ValueError):rejected+=1
        else:raise AssertionError('bad gradient admitted')
    # Real source undefined padding may be any word. Choose 2: it pollutes the
    # old backward reduction, whereas explicit zero padding is deterministic.
    for inner in (False,True):
        o=tensor(torch.full((1,3,2) if not inner else (1,3),1/3));g=tensor(torch.ones_like(o.value));d=tensor(torch.zeros_like(o.value))
        fn='softmax_backward_kernel_inner' if inner else 'softmax_backward_kernel_non_inner'
        args=(o,g,d,1,3) if inner else (o,g,d,1,3,2)
        old[fn][lambda meta:(1,1,1)](*args,_cpu_tile=32)
        assert not torch.allclose(d.value,torch.zeros_like(d.value));failures+=1
    # Original division by zero (M or N) and zero-sized K inner out-of-bounds.
    for shape,dim in [((0,2),1),((2,0),1),((2,0),0)]:
        x=tensor(torch.empty(shape));x.metadata['contiguous']=True
        x.contiguous=lambda:x
        try:old['softmax'](x,dim)
        except (ZeroDivisionError,IndexError):failures+=1
        else:raise AssertionError('missing original empty failure')
    assert all(c[-1] for c in calls)
    assert len({(c[0],c[4]) for c in calls})==8
    print(f'passed: {accepted} forward/backward cases, {rejected} rejections, {failures} original counterexamples; {len(calls)} launches')
    print('all four kernels, both branches, six backward configurations; CPU only')

if __name__=='__main__':main()
