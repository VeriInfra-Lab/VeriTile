"""Checked host admission and exact-source CPU autograd controls."""
import ast
import contextlib
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
os.environ['TRITON_INTERPRET']='1'
import torch
import triton
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'chunk_linear_attn_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/chunk_linear_attn/chunk_linear_attn.py'
NAMES=['chunk_linear_attn_fwd_kernel_h','chunk_linear_attn_fwd_kernel_o',
       'chunk_linear_attn_bwd_kernel_dh','chunk_linear_attn_bwd_kernel_dqkv']

class PaddingOnly(ast.NodeTransformer):
    def visit_Call(self,n):
        self.generic_visit(n)
        if isinstance(n.func,ast.Attribute) and n.func.attr=='load':
            n.keywords=[k for k in n.keywords if k.arg!='padding_option']
        return n

def source_checks():
    tree=ast.parse(SOURCE.read_text())
    original=ast.parse(ORIGINAL.read_text())
    for name in NAMES:
        old=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        if name=='chunk_linear_attn_bwd_kernel_dh':
            new.args.args=[a for a in new.args.args if a.arg not in ('dht','dh0','USE_FINAL_GRADIENT','STORE_INITIAL_GRADIENT')]
            new.body=[n for n in new.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id in ('USE_FINAL_GRADIENT','STORE_INITIAL_GRADIENT'))]
        assert ast.dump(PaddingOnly().visit(old))==ast.dump(PaddingOnly().visit(new))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    for fn in [n for n in cls.body if isinstance(n,ast.FunctionDef)]:
        assert any(isinstance(n,ast.With) and ast.unparse(n.items[0].context_expr)=='torch.cuda.device(q.device)' for n in fn.body)
    wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chunk_linear_attn')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('clone','contiguous','resolve_neg','copy_') for n in ast.walk(wrapper))
    backward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
    assert sum(isinstance(n,ast.Attribute) and n.attr=='contiguous' for n in ast.walk(backward))==2
    assert any(isinstance(n,ast.Return) and isinstance(n.value,ast.Tuple) and len(n.value.elts)==6 for n in ast.walk(backward))
    for name in NAMES:
        kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        for n in ast.walk(kernel):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load':
                opts={k.arg:ast.literal_eval(k.value) for k in n.keywords if k.arg in ('other','padding_option')}
                assert opts.get('other')==0 or opts.get('padding_option')=='zero'

class Tensor:
    def __init__(self,data,*,shape=None,cuda=True,device='cuda:1',negative=False,conjugate=False,pointer=None,strides=None):
        self.data=data;self.shape=tuple(data.shape) if shape is None else tuple(shape);self.ndim=len(self.shape)
        self.dtype=data.dtype;self.layout=data.layout;self.device=torch.device(device);self.is_cuda=cuda
        self.negative=negative;self.conjugate=conjugate;self.pointer=pointer;self.strides=strides
    def is_neg(self):return self.negative or self.data.is_neg()
    def is_conj(self):return self.conjugate or self.data.is_conj()
    def is_contiguous(self):return self.data.is_contiguous()
    def stride(self):
        if self.strides is not None:return self.strides
        if self.shape==tuple(self.data.shape):return self.data.stride()
        return tuple(math.prod(self.shape[i+1:]) for i in range(len(self.shape)))
    def data_ptr(self):return self.data.data_ptr() if self.pointer is None else self.pointer

def host_checks():
    calls=[]
    class Apply:
        @staticmethod
        def apply(*args):calls.append(args);return ('output','final' if args[-1] else None)
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32),
        'triton':triton,'math':math,'Optional':Optional,'Tuple':Tuple,'ChunkLinearAttentionFunction':Apply}
    fns=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name not in NAMES]
    exec(compile(ast.Module(body=fns,type_ignores=[]),str(SOURCE),'exec'),ns)
    invoke=ns['chunk_linear_attn'];x=torch.ones(1,2,17,17);q=Tensor(x);v=Tensor(torch.ones(1,2,17,19));bad=[]
    for index in range(3):
        data=x if index<2 else v.data
        for tensor in [object(),Tensor(data.half()),Tensor(data,cuda=False),Tensor(data,device='cuda:2'),
                       Tensor(data,negative=True),Tensor(data,conjugate=True),Tensor(data,pointer=1),
                       Tensor(data,pointer=2**63-4),Tensor(data,strides=(1,1,1,1))]:
            args=[q,q,v];args[index]=tensor;bad.append((args,{}))
    for kw,value in [('normalize',1),('output_final_state',1),('scale',True),('scale',float('nan')),
                     ('scale',float('inf')),('scale',1e39),('scale',10**500),('initial_state',object()),
                     ('initial_state',Tensor(torch.ones(1,2,17,18)))]:bad.append(([q,q,v],{kw:value}))
    for shape in [(1,2,0,17),(1,2,17,15),(1,65536,1,17),(1,1,2**31,17)]:
        t=Tensor(x,shape=shape);vv=Tensor(v.data,shape=(*shape[:3],19));bad.append(([t,t,vv],{}))
    for args,kwargs in bad:
        before=len(calls)
        try:invoke(*args,**kwargs)
        except (TypeError,ValueError):pass
        else:raise AssertionError('expected admission rejection')
        assert len(calls)==before
    admitted=0
    for normalize in [False,True]:
        for final in [False,True]:
            for scale in [None,-0.,1e-50,.5]:
                out=invoke(q,q,v,scale,Tensor(torch.ones(1,2,17,19)),final,normalize)
                assert out==('output','final' if final else None)
                assert type(calls[-1][3]) is float
                if scale is not None:assert math.copysign(1,calls[-1][3])==math.copysign(1,scale)
                admitted+=1
    cot=ns['_check_cotangent'];cot(Tensor(torch.ones(1,1,1,1).expand(1,2,17,19)),q,v.shape)
    for badcot in [object(),Tensor(v.data.half()),Tensor(v.data,cuda=False),Tensor(v.data,negative=True),Tensor(torch.ones(1))]:
        try:cot(badcot,q,v.shape)
        except (TypeError,ValueError):pass
        else:raise AssertionError('cotangent rejection')
    return len(bad)+5,admitted


def reference(q,k,v,scale,initial,final):
    B,H,T,K=q.shape;V=v.shape[-1]
    state=torch.zeros(B,H,K,V) if initial is None else initial
    output=[]
    for t in range(T):
        state=state+k[:,:,t,:,None]*v[:,:,t,None,:]
        output.append((q[:,:,t,:,None]*state).sum(-2)*scale)
    return torch.stack(output,dim=2),state if final else None


def interpreter_checks(module):
    # CPU fixture changes only the CUDA admission/context shell. All four JIT
    # kernels, the custom autograd function, casts, copies and torch sum execute.
    old_device=module.torch.cuda.device;old_cot=module._check_cotangent
    current=['caller'];events=[]
    @contextlib.contextmanager
    def device(d):
        previous=current[0];current[0]=d;events.append(('enter',d))
        try:yield
        finally:current[0]=previous;events.append(('exit',previous))
    module.torch.cuda.device=device
    module._check_cotangent=lambda t,ref,shape: None if t is None else (
        None if t.dtype==torch.float32 and tuple(t.shape)==tuple(shape) else (_ for _ in ()).throw(ValueError('cotangent')))
    count=0
    try:
        for T,K,V in [(1,16,16),(3,17,17),(17,32,33),(65,65,17)]:
            for seed in [False,True]:
                for mode in ['output','final','both']:
                    final=mode!='output'
                    inputs=[torch.ones(1,1,T,K,requires_grad=True),torch.ones(1,1,T,K,requires_grad=True),
                            torch.ones(1,1,T,V,requires_grad=True)]
                    initial=torch.ones(1,1,K,V,requires_grad=True) if seed else None
                    expected_inputs=[x.detach().clone().requires_grad_() for x in inputs]
                    expected_initial=initial.detach().clone().requires_grad_() if seed else None
                    out,ht=module.ChunkLinearAttentionFunction.apply(*inputs,.5,initial,final)
                    want,wht=reference(*expected_inputs,.5,expected_initial,final)
                    torch.testing.assert_close(out,want,rtol=0,atol=0)
                    if final:torch.testing.assert_close(ht,wht,rtol=0,atol=0)
                    loss=out.sum() if mode=='output' else ht.sum() if mode=='final' else out.sum()+ht.sum()
                    expected_loss=want.sum() if mode=='output' else wht.sum() if mode=='final' else want.sum()+wht.sum()
                    loss.backward();expected_loss.backward()
                    for got,wanted in zip(inputs+([initial] if seed else []),expected_inputs+([expected_initial] if seed else [])):
                        if wanted.grad is None:assert got.grad is not None and torch.count_nonzero(got.grad)==0
                        else:torch.testing.assert_close(got.grad,wanted.grad,rtol=0,atol=0)
                    assert current[0]=='caller';count+=1
        assert all(events[i][0]=='enter' and events[i+1][0]=='exit' for i in range(0,len(events),2))
    finally:module.torch.cuda.device=old_device;module._check_cotangent=old_cot
    return count

if __name__=='__main__':
    source_checks();rejected,admitted=host_checks()
    spec=importlib.util.spec_from_file_location('linear_checked',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controls=interpreter_checks(module)
    print(f'PASS: explicit zero padding and unchanged arithmetic; {rejected} rejects; {admitted} admits; {controls} exact CPU autograd controls')
    print(f'PASS: terminal cotangent, initial gradient, zero-stride incoming cotangents, tails and six slots; torch={torch.__version__}; Triton={triton.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')
