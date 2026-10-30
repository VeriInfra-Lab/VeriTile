"""CPU actual-source checks. CUDA metadata/launch are mocked; numerics are not certified."""
import ast
import re
import textwrap
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('masked_cpu_helpers',HERE.parent/'masked_add_cuda/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
Tensor,Array=helpers.Tensor,helpers.Array

class Language(helpers.Language):
    def arange(self,start,stop):
        if stop<=0 or stop&(stop-1):
            raise ValueError('tl.arange requires a positive power-of-two block')
        return super().arange(start,stop)
    def load(self,pointer,mask,**unused):
        return super().load(pointer,mask)
    def sigmoid(self,x):
        return (1/(1+np.exp(-x))).view(Array)
    def maximum(self,x,y):
        return np.maximum(x,y).view(Array)

class JIT:
    def __init__(self,f,tl,calls):self.f,self.tl,self.calls=f,tl,calls
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            fusion=kwargs.pop('enable_fp_fusion',None)
            dims=grid(kwargs) if callable(grid) else grid
            self.calls.append((dims,kwargs['BLOCK_SIZE'],fusion))
            for pid in range(dims[0]):
                self.tl.pid=pid
                with np.errstate(over='ignore',invalid='ignore'):
                    self.f(*args,**kwargs)
        return launch

def load(path):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('fused_add_mul_activation_kernel','fused_add_mul_activation_torch')]
    tl,calls=Language(),[]
    tri=SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b,next_power_of_2=lambda n:1<<(max(n,1)-1).bit_length())
    tri.jit=lambda f:JIT(f,tl,calls)
    @contextmanager
    def device(d):
        assert d=='cuda:0'
        yield
    tor=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,cuda=SimpleNamespace(device=device))
    ns={'torch':tor,'triton':tri,'tl':tl}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['fused_add_mul_activation_torch'],ns['fused_add_mul_activation_kernel'],calls,nodes[0]

def operands(n,w=3,extra=0):
    return Tensor(torch.linspace(-1,1,n)),Tensor(torch.linspace(-0.5,0.5,w)),Tensor(torch.linspace(-2,2,n+extra))

def reference(x,b,i):
    return torch.sigmoid(0.5*i.value.flatten()[:x.numel()]+x.value.flatten()+b.value.flatten()[torch.arange(x.numel())%b.numel()])

def source_binding():
    python = ast.parse((HERE/'fused_activation_repaired.py').read_text())
    functions = {node.name: node for node in python.body if isinstance(node, ast.FunctionDef)}
    lean = (HERE/'Kernels.lean').read_text()
    assert 'let x_ptr : Region .fp32' in lean and 'let bias_ptr : Region .fp32' in lean
    assert 'let in_ptr : Region .fp32' in lean and '.mk [' not in lean
    for lean_name, native_name in [('kernel', 'fused_add_mul_activation_kernel'),
                                    ('commutedKernel', 'fused_add_mul_activation_commuted_kernel')]:
        body = re.search(r'def '+lean_name+r'\b.*?triton \{\n(.*?)\n\}', lean, re.S).group(1)
        body = re.sub(r'\$\((\w+)\)', r'\1', body)
        body = body.replace('if activation {', 'if activation == "sigmoid":')
        body = body.replace('} else {', 'elif activation == "relu":')
        body = '\n'.join(line for line in body.splitlines() if line.strip() != '}')
        displayed = ast.parse(textwrap.dedent(body)).body
        assert list(map(ast.dump, displayed)) == list(map(ast.dump, functions[native_name].body))
    # The candidate changes only the outer bias addition.
    lhs = functions['fused_add_mul_activation_kernel']
    rhs = functions['fused_add_mul_activation_commuted_kernel']
    lhs.name = rhs.name
    lhs.body[7].value = ast.parse('tmp1 + (multiplier * tmp3 + tmp0)', mode='eval').body
    assert ast.dump(lhs) == ast.dump(rhs)
    print('Typed canonical DSL/Python sources agree; candidate changes only the outer bias addition.')

def main():
    source_binding()
    run,kernel,calls,src=load(HERE/'fused_activation_repaired.py')
    original,_,_,orig=load(ROOT/'bench/tritonbench_g/fused_activation/fused_activation.py')
    assert ast.dump(src)==ast.dump(orig)
    accepted=rejected=failures=0
    for n,w in [(0,0),(1,1),(3,2),(7,3),(1023,17),(2048,13),(2049,7),(4099,5000)]:
        args=operands(n,w,3)
        expected=reference(*args) if n else args[0].value.clone()
        before=len(calls)
        returned=run(*args)
        assert returned is args[0]
        torch.testing.assert_close(returned.value,expected,rtol=2e-6,atol=1e-7)
        assert len(calls)-before==(n>0)
        if n:assert calls[-1][1]==min(2048,1<<(n-1).bit_length()) and calls[-1][2] is True
        accepted+=1
    for shape in [(),(1,1),(2,3)]:
        n=int(np.prod(shape));x,b,i=operands(n)
        x.value=x.value.reshape(shape)
        expected=reference(x,b,i).reshape(shape)
        assert run(x,b,i) is x
        torch.testing.assert_close(x.value,expected)
        accepted+=1
    x=Tensor(torch.arange(8,dtype=torch.float32)[1:6]);b=Tensor(torch.ones(7)[2:5]);i=Tensor(torch.arange(9,dtype=torch.float32)[2:8])
    expected=reference(x,b,i);assert run(x,b,i) is x;torch.testing.assert_close(x.value,expected);accepted+=1
    # Read-only operands may alias; no output alias is introduced.
    x,b,i=operands(5,5);i=b
    expected=reference(x,b,i);run(x,b,i);torch.testing.assert_close(x.value,expected);accepted+=1
    # Distinct contiguous offset views may overlap in read-only backing storage.
    backing=torch.arange(8,dtype=torch.float32);snapshot=backing.clone()
    x,b,i=Tensor(torch.ones(5)),Tensor(backing[1:4]),Tensor(backing[:5])
    expected=reference(x,b,i);run(x,b,i);torch.testing.assert_close(x.value,expected)
    assert torch.equal(backing,snapshot);accepted+=1
    x,b,i=operands(0,0)
    for t in (x,b,i):t.metadata['forbid_pointer']=True
    assert run(x,b,i) is x and not any(t.pointer_queries for t in (x,b,i));accepted+=1
    # Original low-level ReLU branch remains complete and source-identical.
    x,b,i=operands(8,3);expected=torch.maximum(torch.zeros(8),0.5*i.value+x.value+b.value[torch.arange(8)%3])
    kernel[(1,)](x,b,i,3,8,multiplier=0.5,activation='relu',BLOCK_SIZE=8)
    torch.testing.assert_close(x.value,expected);accepted+=1
    def reject(args):
        nonlocal rejected
        before=len(calls)
        try:run(*args)
        except (TypeError,ValueError):rejected+=1
        else:raise AssertionError('unsupported input accepted')
        assert len(calls)==before
    reject((None,*operands(3)[1:]))
    for slot,meta in [(0,{'layout':torch.sparse_coo}),(0,{'dtype':torch.float16}),(1,{'dtype':torch.float64}),
                      (2,{'is_cuda':False}),(1,{'device':'cuda:1'}),(0,{'contiguous':False}),
                      (2,{'negative':True}),(2,{'numel':1}),(0,{'numel':2**31}),(1,{'numel':2**31})]:
        args=operands(3);args[slot].metadata.update(meta);reject(args)
    reject(operands(3,0))
    for slot in range(3):
        for rem in (1,2,3):
            raw=torch.zeros(64,dtype=torch.uint8)
            value=torch.empty(0,dtype=torch.float32).set_(raw.untyped_storage()[rem:rem+12],0,(3,),(1,))
            args=list(operands(3));args[slot]=Tensor(value);reject(args)
    # Original empty launch and non-power-of-two block are structural failures.
    for n in (0,3):
        try:original(*operands(n))
        except (ZeroDivisionError,ValueError):failures+=1
        else:raise AssertionError('missing original block failure')
    for args in [(Tensor(torch.ones(4)),Tensor(torch.ones(1)),Tensor(torch.ones(2))),operands(4,0)]:
        reject(args)
        try:original(*args)
        except IndexError:failures+=1
        else:raise AssertionError('missing original input/bias bounds failure')
    backing=torch.tensor([1.,77.,2.]);args=(Tensor(backing[::2]),Tensor(torch.zeros(1)),Tensor(torch.zeros(2)))
    reject(args);original(*args);assert backing[1]!=77 and backing[2]==2;failures+=1
    x,b,i=operands(4);x.value=torch._neg_view(torch.arange(4,dtype=torch.float32))
    expected=reference(x,b,i);reject((x,b,i));original(x,b,i);assert not torch.allclose(x.value,expected);failures+=1
    backing=torch.ones(2050);args=(Tensor(backing[1:]),Tensor(torch.zeros(1)),Tensor(backing[:-1]))
    expected=reference(*args).clone();reject(args);original(*args);assert not torch.allclose(args[0].value,expected);failures+=1
    backing=torch.ones(2049);args=(Tensor(backing),Tensor(backing[:1]),Tensor(torch.zeros(2049)))
    expected=reference(*args).clone();reject(args);original(*args);assert not torch.allclose(args[0].value,expected);failures+=1
    assert 'FPEquiv' not in (HERE/'Correct.lean').read_text()
    assert '.Correct' not in (HERE/'FPEquiv.lean').read_text()
    print(f'passed: {accepted} admissions/branches, {rejected} rejections, {failures} original counterexamples')
    print('kernel AST unchanged; CPU source interpretation only; no native numerical conformance claim')

if __name__=='__main__':main()
