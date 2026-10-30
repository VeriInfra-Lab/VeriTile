"""Focused source/contract checks using real CPU storage and Triton's interpreter.
CUDA context is mocked. These checks do not establish GPU numerical conformance.
"""
import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

os.environ['TRITON_INTERPRET'] = '1'
os.environ.setdefault('TRITON_CACHE_DIR', '/tmp/veritile-int-scaled-interpreter')
sys.dont_write_bytecode = True
import torch
from triton.runtime import interpreter

# The interpreter has no inline-assembly handler. Interpret only the single
# audited PTX integer add; the kernel's loads, masks, dot and stores still run
# through Triton's interpreter. Native lowering is checked separately.
def integer_add(self, assembly, constraints, values, result_type, pure, pack):
    assert assembly=='add.u32 $0, $1, $2;' and constraints=='=r,r,r'
    assert pure and pack==1 and len(values)==2
    result=self.create_add(values[0],values[1])
    return SimpleNamespace(get_result=lambda i: result if i==0 else None)
interpreter.InterpreterBuilder.create_inline_asm=integer_add

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE/'int_scaled_matmul_repaired.py'
ORIGINAL = ROOT/'bench/tritonbench_g/int_scaled_matmul/int_scaled_matmul.py'

def dump(node): return ast.dump(node, include_attributes=False)
def expr(text): return ast.parse(text, mode='eval').body
def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)

def check_source():
    for scaled, name in [(False,'matmul_kernel_with_block_pointers'),
                         (True,'scaled_matmul_kernel_with_block_pointers')]:
        old, new = function(ORIGINAL,name),function(SOURCE,name)
        loop=next(n for n in new.body if isinstance(n,ast.For))
        carry='acc' if scaled else 'accumulator'
        update=next(n for n in loop.body if isinstance(n,ast.Assign)
                    and isinstance(n.targets[0],ast.Name) and n.targets[0].id==carry)
        assert dump(update.value)==dump(expr('tl.inline_asm_elementwise("add.u32 $0, $1, $2;", '
            f'constraints="=r,r,r", args=[{carry}, tl.dot(a, b)], dtype=tl.int32, is_pure=True, pack=1)'))
        loop.body[loop.body.index(update)]=ast.AugAssign(target=ast.Name(carry,ast.Store()),op=ast.Add(),value=expr('tl.dot(a,b)'))
        if not scaled:
            loads = [n for n in ast.walk(new) if isinstance(n, ast.Call) and dump(n.func)==dump(expr('tl.load'))]
            assert len(loads)==2
            for load in loads:
                padding = next(k for k in load.keywords if k.arg=='padding_option')
                assert dump(padding.value)==dump(expr('"zero"'))
                load.keywords.remove(padding)
        else:
            replacements = {'ram':'rm % M','rbn':'rn % N',
                            'xindex':'idx_n * stride_cn + idx_m * stride_cm'}
            for item in new.body:
                if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name):
                    key=item.targets[0].id
                    if key in replacements:
                        assert dump(item.value)==dump(expr(replacements[key]))
                        item.value=next(n.value for n in old.body if isinstance(n,ast.Assign)
                                        and isinstance(n.targets[0],ast.Name) and n.targets[0].id==key)
                    if key=='tmp0':
                        row=item.value.args[0].right.args[0]
                        assert dump(row)==dump(expr('idx_m * stride_s1m'))
                        item.value.args[0].right.args[0]=expr('idx_m')
        assert dump(new)==dump(old), 'unreviewed source arithmetic/precision change'
    guard=function(SOURCE,'_checked_inputs')
    guards=[dump(n.test) for n in guard.body if isinstance(n,ast.If)]
    expected=[
        'any(not isinstance(t, torch.Tensor) for t in tensors)',
        'any(t.layout != torch.strided for t in tensors)',
        'a.dtype != torch.int8 or b.dtype != torch.int8 or c.dtype != torch.int32',
        'scaled and scales1.dtype != torch.float32',
        'any(not t.is_cuda for t in tensors)', 'any(t.device != a.device for t in tensors)',
        'any(t.ndim != 2 for t in tensors)', 'K != K_b or tuple(c.shape) != (M, N)',
        'scaled and tuple(scales1.shape) != (M, 1)',
        'any(not t.is_contiguous() for t in tensors)', 'any(t.is_neg() for t in tensors)',
        'not isinstance(config, Config)',
        'set(config.kwargs) != {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M"}',
        'any(type(v) is not int for v in (BM, BN, BK, G, config.num_warps, config.num_stages, config.num_ctas))',
        'any(v < minimum or v & (v - 1) for v, minimum in ((BM, 16), (BN, 16), (BK, 32)))',
        'max(BM * BN, BM * BK, BK * BN) > 2**20 or G <= 0',
        'config.num_warps not in (1, 2, 4, 8, 16, 32) or config.num_stages <= 0 or config.num_ctas != 1',
        'M == 0 or N == 0','any(v >= 2**31 for v in envelope)',
    ]
    assert guards==[dump(expr(t)) for t in expected]
    envelope=next(n.value for n in guard.body if isinstance(n,ast.Assign)
                  and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='envelope')
    assert dump(envelope)==dump(expr('(M + BM - 1, N + BN - 1, K + BK - 1, G * gn, gm * gn, '
                                    'pm * K + pk + BK, (pk + BK) * N + pn, pm * N + pn)'))
    alltests=[dump(n.test) for n in ast.walk(guard) if isinstance(n,ast.If)]
    for test in ['t.numel() == 0','start % width != 0 or end > 2**64',
                 'start < end and c_start < end and start < c_end']:
        assert dump(expr(test)) in alltests
    forbidden={'clone','contiguous','resolve_neg','resolve_conj','flatten','reshape'}
    for scaled,name in [(False,'int_matmul_kernel'),(True,'int_scaled_matmul_kernel')]:
        wrapper=function(SOURCE,name)
        assert not any(isinstance(n,ast.Attribute) and n.attr in forbidden for n in ast.walk(wrapper))
        launch=next(n for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript))
        assert [dump(x) for x in launch.args]==[dump(expr(x)) for x in
            (['a','b','c','scales1','M','N','K','K','1','N','1','N','1','1','1'] if scaled else
             ['a','b','c','M','N','K','K','1','N','1','N','1'])]
        if scaled:
            assert dump(next(k.value for k in launch.keywords if k.arg=='EVEN_K'))==dump(expr('K % config.kwargs["BLOCK_K"] == 0'))
        assert isinstance(wrapper.body[-1],ast.Return) and dump(wrapper.body[-1].value)==dump(expr('c'))
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text().split('namespace')[0]
    assert '.revision001.Correct' not in (HERE/'FPEquiv.lean').read_text()
    assert '.revision001.FPEquiv' not in (HERE/'Correct.lean').read_text()
    for path in (HERE/'Proofs').glob('*.lean'):
        for line in path.read_text().splitlines():
            if line.startswith('import '):
                assert not line.endswith(('.revision001.Correct','.revision001.FPEquiv'))
    contract=(HERE/'Contract.lean').read_text()
    for field in ['IndexEnvelope','ByteWindow','Separate','ConfigContract','Empty','InputContract']:
        assert field in contract

class Facade:
    def __init__(self,tensor,**overrides):
        self.data=tensor
        self.device=torch.device('cuda:0')
        self.is_cuda=True
        self.__dict__.update(overrides)
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    @property
    def shape(self): return getattr(self,'shape_override',self.data.shape)
    @property
    def ndim(self): return len(self.shape)
    def numel(self):
        import math
        return math.prod(self.shape)
    def element_size(self): return self.data.element_size()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.numel()>0, 'empty pointer queried'
        return getattr(self,'pointer_override',self.data.data_ptr())

class Launcher:
    def __init__(self,kernel): self.kernel,self.calls=kernel,0
    def __getitem__(self,grid):
        def run(*args,**kwargs):
            self.calls+=1
            return self.kernel[grid](*(x.data if isinstance(x,Facade) else x for x in args),**kwargs)
        return run

def unaligned(remainder,shape,dtype):
    n=shape[0]*shape[1]
    storage=torch.empty(4*n+4,dtype=torch.uint8).untyped_storage()[remainder:remainder+4*n]
    return torch.empty(0,dtype=dtype).set_(storage,0,shape,(shape[1],1))

def run():
    check_source()
    spec=importlib.util.spec_from_file_location('int_scaled_checked',SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for name in ['matmul_kernel_with_block_pointers','scaled_matmul_kernel_with_block_pointers']:
        setattr(module,name,Launcher(getattr(module,name)))
    module.torch=SimpleNamespace(Tensor=Facade,strided=torch.strided,int8=torch.int8,int32=torch.int32,
        float32=torch.float32,cuda=SimpleNamespace(device=lambda _:nullcontext()))
    config=lambda **kw:module.Config(**(dict(BLOCK_M=16,BLOCK_N=16,BLOCK_K=32,GROUP_M=2)|kw))
    A=torch.arange(12,dtype=torch.int8).reshape(3,4)
    B=torch.ones(4,5,dtype=torch.int8)
    C=torch.full((3,5),-777,dtype=torch.int32)
    S=torch.ones(3,1)
    base=[Facade(x) for x in [A,B,C,S]]
    rejects=0
    def reject(args=None,cfg=None,scaled=True):
        nonlocal rejects
        args=base if args is None else args
        before=[x.data.clone() if isinstance(x,Facade) and x.layout==torch.strided else None for x in args]
        calls=module.matmul_kernel_with_block_pointers.calls+module.scaled_matmul_kernel_with_block_pointers.calls
        a,b,c,s=args
        try:
            if scaled: module.int_scaled_matmul_kernel(a,b,s,c,config() if cfg is None else cfg)
            else: module.int_matmul_kernel(a,b,c,config() if cfg is None else cfg)
        except (TypeError,ValueError): pass
        else: raise AssertionError('unsupported input admitted')
        assert calls==module.matmul_kernel_with_block_pointers.calls+module.scaled_matmul_kernel_with_block_pointers.calls
        for x,saved in zip(args,before):
            if saved is not None: torch.testing.assert_close(x.data,saved,rtol=0,atol=0,equal_nan=True)
        rejects+=1
    for i in range(4):
        for value in [None,Facade(base[i].data.to_sparse()),Facade(base[i].data.double()),
                      Facade(base[i].data,is_cuda=False),Facade(base[i].data,device=torch.device('cuda:1')),
                      Facade(base[i].data.reshape(-1)),Facade(torch._neg_view(base[i].data))]:
            args=base.copy();args[i]=value;reject(args)
    for i,value in [(0,Facade(A.T)),(1,Facade(B.T)),(2,Facade(C.T)),(3,Facade(torch.ones(3,2))),
                    (0,Facade(torch.ones(3,4,dtype=torch.int8).T.contiguous().T)),
                    (1,Facade(torch.ones(4,5,dtype=torch.int8).T.contiguous().T)),
                    (2,Facade(torch.ones(3,5,dtype=torch.int32).T.contiguous().T))]:
        args=base.copy();args[i]=value;reject(args)
    for cfg in [object(),config(BLOCK_K=16),config(BLOCK_M=24),config(GROUP_M=0),config(GROUP_M=True),
                config(BLOCK_M=2048,BLOCK_N=2048),config(num_warps=3),config(num_stages=0),config(num_ctas=2),
                config(extra=1),config(BLOCK_K=-32)]: reject(cfg=cfg)
    for remainder in [1,2,3]:
        for i,dtype in [(2,torch.int32),(3,torch.float32)]:
            args=base.copy();args[i]=Facade(unaligned(remainder,base[i].shape,dtype));reject(args)
    raw=torch.arange(80,dtype=torch.uint8)
    reject([Facade(raw[:12].view(torch.int8).reshape(3,4)),base[1],
            Facade(raw[:60].view(torch.int32).reshape(3,5)),base[3]])
    reject([base[0],base[1],base[2],Facade(C.view(torch.float32).reshape(-1)[:3].reshape(3,1))])
    for i in [0,1,2,3]:
        args=base.copy();args[i]=Facade(base[i].data,pointer_override=2**64-base[i].element_size());reject(args)
    reject(cfg=config(GROUP_M=2**31))
    reject([Facade(A,shape_override=(2**31,4)),base[1],Facade(C,shape_override=(2**31,5)),
            Facade(S,shape_override=(2**31,1))])
    # Explicit source envelope boundary, checked without allocating huge buffers.
    def envelope(k):
        pk=(k+31)//32*32
        return max(16,k+31,32,16*k+pk+32,pk+48,32)
    lo,hi=0,2**31
    while lo+1<hi:
        mid=(lo+hi)//2
        if envelope(mid)<2**31:lo=mid
        else:hi=mid
    def wide(k):
        return [Facade(A,shape_override=(1,k),pointer_override=2**32),
                Facade(B,shape_override=(k,1),pointer_override=2**34),
                Facade(C,shape_override=(1,1),pointer_override=2**36),
                Facade(S,shape_override=(1,1),pointer_override=2**38)]
    a,b,c,s=wide(lo)
    assert module._checked_inputs(a,b,c,config(),s,scaled=True)==(1,1,lo)
    reject(wide(hi))
    torch.manual_seed(123)
    cases=[]
    for m,n,k in [(1,1,0),(3,5,7),(3,5,2),(19,17,34),(65,33,65),(48,35,64),(0,5,7),(3,0,7)]:
        cases.append((torch.randint(-128,128,(m,k),dtype=torch.int8),torch.randint(-128,128,(k,n),dtype=torch.int8)))
    cases += [(torch.arange(20,dtype=torch.int8).reshape(5,4)[1:4],B),
              (torch.as_strided(torch.arange(4,dtype=torch.int8),(1,4),(2**40,1)),B)]
    shared=torch.arange(16,dtype=torch.int8).reshape(4,4);cases.append((shared,shared))
    trials=0
    for a,b in cases:
        m,k=a.shape;n=b.shape[1]
        for tile in [config(),config(BLOCK_M=32,BLOCK_N=32,BLOCK_K=64,GROUP_M=3)]:
            for scaled in [False,True]:
                storage=torch.full((m*n+6,),-999,dtype=torch.int32)
                c=storage[2:2+m*n].reshape(m,n)
                scale=torch.linspace(-1.5,1.5,max(m,1),dtype=torch.float32)[:m].reshape(m,1)
                fa,fb,fc,fs=[Facade(x) for x in [a,b,c,scale]]
                sa,sb=a.clone(),b.clone()
                expected=(a.to(torch.int64)@b.to(torch.int64)).to(torch.int32)
                if scaled: expected=(expected.float()*scale).to(torch.int32)
                out=(module.int_scaled_matmul_kernel(fa,fb,fs,fc,tile) if scaled else
                     module.int_matmul_kernel(fa,fb,fc,tile))
                assert out is fc
                torch.testing.assert_close(c,expected,rtol=0,atol=0)
                torch.testing.assert_close(a,sa,rtol=0,atol=0);torch.testing.assert_close(b,sb,rtol=0,atol=0)
                assert torch.all(storage[:2]==-999) and torch.all(storage[2+m*n:]==-999)
                trials+=1
    # Undefined conversions stay admitted; no CPU result is promoted to a GPU law.
    for value in [float('nan'),float('inf'),-float('inf'),2**32]:
        s=Facade(torch.full((3,1),value,dtype=torch.float32));c=Facade(C.clone())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            assert module.int_scaled_matmul_kernel(base[0],base[1],s,c,config()) is c
        trials+=1
    # All -128 gives exactly 2**31. The original fused satfinite recurrence
    # yields INT32_MAX; the repaired explicit add wraps to INT32_MIN.
    large=Facade(torch.full((1,131072),-128,dtype=torch.int8))
    right=Facade(torch.full((131072,1),-128,dtype=torch.int8))
    for scaled in [False,True]:
        c=Facade(torch.zeros((1,1),dtype=torch.int32));s=Facade(torch.ones((1,1)))
        out=(module.int_scaled_matmul_kernel(large,right,s,c,config()) if scaled else
             module.int_matmul_kernel(large,right,c,config()))
        assert out is c and c.data.item()==-2147483648
        trials+=1
    # Dense offset values exceeding signed32 are rejected before source arithmetic.
    assert hi==lo+1 and envelope(lo)<2**31<=envelope(hi)
    print(f'int_scaled_matmul: AST/guard binding passed; {rejects} prelaunch rejections; '
          f'{trials} CPU-interpreted calls; source-envelope boundary {lo}/{hi}. No GPU execution.')

if __name__=='__main__': run()
