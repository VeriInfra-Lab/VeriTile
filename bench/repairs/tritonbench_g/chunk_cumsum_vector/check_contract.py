"""Actual-source CPU controls, guard bindings and masked-NaN regression."""
import ast
import contextlib
import importlib.util
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import torch
import triton.runtime.interpreter as interpreter

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chunk_cumsum_vector_repaired.py"
NAME = "chunk_global_cumsum_vector"
CONFIGS = [(bt, warps) for bt in (16, 32, 64) for warps in (2, 4, 8)]
GUARDS = [
    ("s.layout != torch.strided", "s.layout != .strided"),
    ("s.dtype != torch.float32", "s.dtype != .floating .fp32"),
    ("dtype is not None and dtype != torch.float32", "dtype != none && dtype != some (.floating .fp32)"),
    ("s.ndim != 4", "s.shape.length != 4"),
    ("not s.is_cuda", "!s.isCuda"),
    ("not s.is_contiguous()", "!s.isContiguous"),
    ("s.is_neg()", "s.isNeg"),
    ("s.numel() != 0 and (B * H > 65535 or triton.cdiv(S, 32) >= 2**31)",
     "s.numel > 0 && (rows s > 65535 || featureBlocks s >= 2^31)"),
    ("s.numel() != 0 and s.data_ptr() % 4 != 0", "s.numel > 0 && s.dataPtr.toNat % 4 != 0"),
]


def fn(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_checks():
    wrapper = fn(SOURCE, NAME)
    parse = lambda x: ast.dump(ast.parse(x, mode="eval").body)
    assert [ast.dump(n.test) for n in wrapper.body if isinstance(n, ast.If)] == [
        parse("not isinstance(s, torch.Tensor)"), *(parse(p) for p, _ in GUARDS), parse("z.numel() == 0")]
    contract = (HERE / "Contract.lean").read_text().split("def checkTensorInput",1)[1].split("def checkInput",1)[0]
    assert re.findall(r"\bif (.*?) then",contract) == [x for _, x in GUARDS]
    kernel = fn(SOURCE, NAME+"_kernel")
    loop = next(n for n in kernel.body if isinstance(n,ast.For))
    load = next(n for n in ast.walk(loop) if isinstance(n,ast.Call) and ast.unparse(n.func)=="tl.load")
    assert [(k.arg,ast.unparse(k.value)) for k in load.keywords] == [("mask","active"),("other","0")]
    lean = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in lean and "mask=active, other=0" in lean
    assert "allow_tf32=False" in SOURCE.read_text() and "allow_tf32=false" in lean
    assert "with torch.cuda.device(s.device):" in SOURCE.read_text()
    assert not any(x in SOURCE.read_text() for x in (".clone(",".contiguous(",".resolve_neg("))


class TensorFacade:
    def __init__(self, data, *, shape=None, device="cuda:0", cuda=True):
        self.data = data
        self.shape = tuple(data.shape if shape is None else shape)
        self.ndim = len(self.shape)
        self.dtype, self.layout = data.dtype,data.layout
        self.device,self.is_cuda = torch.device(device),cuda
    def numel(self): return math.prod(self.shape)
    def stride(self,i): return self.data.stride(i)
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.numel()>0
        return self.data.data_ptr()


def host_checks():
    calls,allocations=[],[]
    current=[torch.device("cuda:0")]
    @contextlib.contextmanager
    def device(d):
        before=current[0];current[0]=d
        try: yield
        finally: current[0]=before
    def allocate(s,dtype):
        z=TensorFacade(torch.empty_like(s.data,dtype=dtype),shape=s.shape,device=s.device)
        allocations.append(z)
        return z
    class Kernel:
        def __getitem__(self,grid):
            def launch(s,z,**kwargs):
                assert s.device==z.device==current[0]
                assert grid==((s.shape[3]+31)//32,s.shape[0]*s.shape[1])
                assert kwargs["s_s_d"]==s.stride(3) and kwargs["s_z_d"]==z.stride(3)
                calls.append((s,z))
            return launch
    n=fn(SOURCE,NAME);n.returns=None
    for a in n.args.args:a.annotation=None
    ns={"torch":SimpleNamespace(Tensor=TensorFacade,strided=torch.strided,float32=torch.float32,
        empty_like=allocate,cuda=SimpleNamespace(device=device)),
        "triton":SimpleNamespace(cdiv=lambda n,d:(n+d-1)//d),NAME+"_kernel":Kernel()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),str(SOURCE),"exec"),ns)
    wrapper=ns[NAME]
    x=torch.arange(64,dtype=torch.float32).reshape(1,2,4,8)
    bad=[object(),TensorFacade(x.to(torch.float16)),TensorFacade(x.reshape(8,8)),
         TensorFacade(x,cuda=False),TensorFacade(x.transpose(2,3)),TensorFacade(torch._neg_view(x)),
         TensorFacade(x,shape=(65536,1,1,1)),TensorFacade(x,shape=(1,1,1,(2**31-1)*32+1))]
    for offset in (1,2,3):
        storage=torch.arange(32,dtype=torch.uint8).untyped_storage()[offset:offset+16]
        bad.append(TensorFacade(torch.empty(0,dtype=torch.float32).set_(storage,0,(1,1,1,4),(4,4,4,1))))
    for s in bad:
        before=(len(calls),len(allocations))
        try:wrapper(s)
        except (ValueError,TypeError):pass
        else:raise AssertionError("expected rejection")
        assert before==(len(calls),len(allocations))
    try:wrapper(TensorFacade(x),torch.float16)
    except TypeError:pass
    else:raise AssertionError("expected output dtype rejection")
    good=[TensorFacade(x),TensorFacade(torch.arange(80,dtype=torch.float32)[4:68].reshape(1,2,4,8)),
          TensorFacade(torch.as_strided(x,(1,2,4,8),(999,32,8,1))),
          TensorFacade(x,shape=(65535,1,1,1)),TensorFacade(x,shape=(1,1,1,(2**31-1)*32)),
          TensorFacade(torch.empty(1,1,0,1),shape=(2**32,1,0,2**40)),
          TensorFacade(torch.empty(0,2,3,4)),TensorFacade(torch.empty(2,0,3,4)),
          TensorFacade(torch.empty(2,3,0,4)),TensorFacade(torch.empty(2,3,4,0)),
          TensorFacade(torch.tensor([float("nan"),float("inf")]).reshape(1,1,1,2)),
          TensorFacade(x,device="cuda:1")]
    for s in good:
        before=len(calls)
        z=wrapper(s)
        assert z is allocations[-1] and z.shape==s.shape and z.device==s.device
        assert len(calls)==before+(s.numel()>0)
        if s.numel():assert calls[-1][0] is s
    assert current[0]==torch.device("cuda:0")
    return len(bad)+1,len(good)


def launch(kernel,s,bt,warps):
    b,h,t,f=s.shape;z=torch.empty_like(s)
    kernel[((f+31)//32,b*h)](s,z,H=h,T=t,S=f,BS=32,BT=bt,num_warps=warps,
        s_s_b=s.stride(0),s_s_h=s.stride(1),s_s_t=s.stride(2),s_s_d=s.stride(3),
        s_z_b=z.stride(0),s_z_h=z.stride(1),s_z_t=z.stride(2),s_z_d=z.stride(3))
    return z


def controls():
    kernel=load_module(SOURCE,"checked_vector").chunk_global_cumsum_vector_kernel.fn
    old_load=interpreter.InterpreterBuilder.create_masked_load
    poisoned=[0]
    def poison(self,ptrs,mask,other,*args):
        if other is None:
            poisoned[0]+=1
            other=interpreter.TensorHandle(np.full(ptrs.data.shape,np.nan,dtype=np.float32),ptrs.get_element_ty())
        return old_load(self,ptrs,mask,other,*args)
    interpreter.InterpreterBuilder.create_masked_load=poison
    try:
        count=0
        for bt,w in CONFIGS:
            for t,f in [(1,1),(4,5),(bt-1,31),(bt,32),(bt+1,33),(2*bt+1,65)]:
                x=(torch.arange(2*t*f+7,dtype=torch.float32)%7-3)[3:3+2*t*f].reshape(1,2,t,f)
                z=launch(kernel,x,bt,w)
                torch.testing.assert_close(z,x.cumsum(2),rtol=0,atol=0)
                count+=1
        assert poisoned[0]==0,"explicit other=0 must bypass unspecified padding"
        # Exact same actual source with only other=0 removed reproduces the defect.
        variant=Path("/tmp/veritile-vector-unspecified-control.py")
        variant.write_text(SOURCE.read_text().replace("mask=active, other=0","mask=active"))
        prior=load_module(variant,"unchecked_padding_vector").chunk_global_cumsum_vector_kernel.fn
        x=torch.ones(1,1,3,5)
        assert torch.isnan(launch(prior,x,16,2)).all() and poisoned[0]>0
        assert torch.isfinite(launch(kernel,x,16,2)).all()
    finally:interpreter.InterpreterBuilder.create_masked_load=old_load
    # Original flattened-head addressing goes beyond this transposed view's allocation.
    x=torch.empty(2,2,16,32).transpose(0,1)
    assert x.stride()==(512,1024,32,1) and 2*x.stride(1)==x.numel()
    return count


if __name__=="__main__":
    source_checks();rejected,admitted=host_checks();count=controls()
    print(f"PASS: source/guard binding; {rejected} rejection cases; {admitted} admission cases; {count} CPU Triton-interpreter controls")
    print("PASS: NaN-poison padding regression and original head-stride counterexample; no GPU execution")
