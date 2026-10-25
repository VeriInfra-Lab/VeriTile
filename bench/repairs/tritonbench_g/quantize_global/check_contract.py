"""Execute actual Python source AST on CPU; kernel contract and existing repair source checks."""
import ast
import contextlib
import re
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
SOURCE = HERE / "quantize_global_repaired.py"
spec = importlib.util.spec_from_file_location("add_contract_helpers", HERE.parent / "add_example/check_contract.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

class Tensor(helpers.TensorFacade):
    def abs(self): return Tensor(self.data.abs(), str(self.device))
    def amax(self): return Tensor(self.data.amax(), str(self.device))
    def reshape(self, *shape): return Tensor(self.data.reshape(*shape), str(self.device))

def function(name):
    return next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)

def main():
    pid, blocks, calls = [0], [1024, 2048, 1024], []
    class Pointer:
        def __init__(self, tensor, offsets=0): self.tensor, self.offsets = tensor, offsets
        def __add__(self, offsets): return Pointer(self.tensor, offsets)
    def load(ptr, mask=None, other=0.0):
        if mask is None: return ptr.tensor.reshape(-1)[ptr.offsets]
        result = torch.full_like(ptr.offsets, other, dtype=torch.float32)
        result[mask] = helpers.raw_storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offsets[mask]]
        return result
    def store(ptr, value, mask):
        ptr.tensor.reshape(-1)[ptr.offsets[mask]] = value[mask].to(ptr.tensor.dtype)
    tl = SimpleNamespace(int64=torch.int64, float32=torch.float32,
        program_id=lambda axis: torch.tensor(pid[0], dtype=torch.int32),
        arange=lambda a,b: torch.arange(a,b,dtype=torch.int32), load=load,store=store,where=torch.where,
        div_rn=torch.div, extra=SimpleNamespace(cuda=SimpleNamespace(libdevice=SimpleNamespace(llrint=lambda x:torch.round(x).to(torch.int64)))))
    kernel = function("_quantize_global_repaired")
    kernel.decorator_list=[]
    for arg in kernel.args.args: arg.annotation=None
    kns={"tl":tl}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel],type_ignores=[])),str(SOURCE),"exec"),kns)
    class Launch:
        def __getitem__(self, grid):
            def run(x, maximum, out, n, base):
                calls.append("launch")
                for block in blocks:
                    for p in reversed(range(grid({"BLOCK_SIZE":block})[0])):
                        pid[0]=p
                        kns["_quantize_global_repaired"](Pointer(x.data),Pointer(maximum.data),Pointer(out.data),n,base,block)
            return run
    def allocate(shape,device,dtype):
        calls.append("allocation")
        return Tensor(torch.empty(shape,dtype=dtype),str(device))
    def zeros(shape,device,dtype):
        calls.append("zeros")
        return Tensor(torch.zeros(shape,dtype=dtype),str(device))
    fake=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,int8=torch.int8,
        empty=allocate,zeros=zeros,isfinite=lambda x:torch.isfinite(x.data),cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    ns={"torch":fake,"triton":SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b),"_quantize_global_repaired":Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function("quantize_global")],type_ignores=[])),str(SOURCE),"exec"),ns)
    guards=[ast.unparse(n.test) for n in function("quantize_global").body if isinstance(n,ast.If)]
    assert guards==["not isinstance(x, torch.Tensor)","x.layout != torch.strided or x.dtype != torch.float32","not x.is_cuda",
        "not x.is_contiguous() or x.is_neg()","x.numel() >= 2 ** 63","x.numel() > 0 and x.data_ptr() % 4 != 0"]
    lean=(HERE/"Contract.lean").read_text()
    checks=lean.split("def checkTensorInputs",1)[1].split("def checkInputs",1)[0]
    assert re.findall(r"\bif (.*?) then",checks)==[
        'x.layout != .strided || x.dtype != .floating .fp32','!x.isCuda',
        '!x.isContiguous || x.isNeg','x.numel≥2^63',
        'x.numel>0 && x.dataPtr.toNat%4 != 0','x.numel != 0 && !allFinite']
    inner=[n for n in ast.walk(function("quantize_global")) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='not torch.isfinite(logical).all().item()']
    assert len(inner)==1
    assert 'finite : x.numel=0 ∨ allFinite=Bool.true' in lean
    assigns={n.targets[0].id:ast.unparse(n.value) for n in function("_quantize_global_repaired").body if isinstance(n,ast.Assign)}
    assert assigns["offsets"]=="pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64) + base"
    assert assigns["denominator"]=="tl.where(absmax == 0.0, 1.0, absmax)"
    assert assigns["normalized"]=="tl.div_rn(x, denominator)"
    assert assigns["output"]=="tl.extra.cuda.libdevice.llrint(bounded)"
    assert list(assigns)==['pid','offsets','mask','safe_offsets','x','absmax','denominator',
        'normalized','scaled','lower_bounded','bounded','output']
    assert assigns['pid']=='tl.program_id(axis=0).to(tl.int64)'
    assert assigns['mask']=='offsets < n_elements'
    assert assigns['safe_offsets']=='tl.where(mask, offsets, 0)'
    assert assigns['x']=='tl.load(x_ptr + safe_offsets, mask=mask, other=0.0).to(tl.float32)'
    assert assigns['absmax']=='tl.load(absmax_ptr).to(tl.float32)'
    assert assigns['scaled']=='127.0 * normalized'
    assert assigns['lower_bounded']=='tl.where(scaled < -127.0, -127.0, scaled)'
    assert assigns['bounded']=='tl.where(lower_bounded > 127.0, 127.0, lower_bounded)'
    stores=[n for n in ast.walk(function('_quantize_global_repaired')) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='store']
    assert len(stores)==1 and ast.unparse(stores[0])=='tl.store(output_ptr + safe_offsets, output, mask=mask)'
    # Compare the complete original quotation, including source casts,
    # intrinsic identity, runtime ABI and actual caller value bindings.
    check_manifest(ROOT, HERE / 'jit-correspondence.json')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ("clone","resolve_neg","contiguous","copy_") for n in ast.walk(function("quantize_global")))
    assert ".Proofs." not in (HERE/"Kernels.lean").read_text()
    f=torch.finfo(torch.float32)
    cases=[torch.tensor(-2.),torch.empty(0),torch.empty(2,0),torch.zeros(17),torch.tensor([f.tiny, -f.tiny, 0]),
        torch.tensor([f.max,-f.max,1.0]),torch.tensor([2**-149,-2**-149,0]),torch.tensor([127.,.5,1.5,2.5,-.5,-1.5,-2.5])]
    cases.extend(torch.linspace(-13,15,n) for n in (1,1023,1024,1025,2047,2048,2049,4097))
    for device in ("cuda:0","cuda:1"):
        for data in cases:
            backing=torch.zeros(data.numel()+6)
            backing[2:2+data.numel()]=data.reshape(-1)
            x=Tensor(backing[2:2+data.numel()].reshape(data.shape),device)
            before=backing.clone()
            calls.clear()
            q,m=ns["quantize_global"](x)
            maximum=data.abs().amax() if data.numel() else torch.tensor(0.)
            expected=torch.round((127.0*(data/torch.where(maximum==0,1.,maximum))).clamp(-127,127)).to(torch.int8)
            torch.testing.assert_close(q.data,expected,rtol=0,atol=0)
            torch.testing.assert_close(m.data,maximum.reshape(1),rtol=0,atol=0)
            torch.testing.assert_close(backing,before,rtol=0,atol=0)
            assert q.device==x.device and m.device==x.device and q.shape==x.shape and m.shape==(1,)
            assert calls==(["allocation","launch"] if data.numel() else ["allocation","zeros"])
    bad=[[],Tensor(torch.eye(2).to_sparse()),Tensor(torch.ones(3,dtype=torch.float64)),Tensor(torch.ones(3),"cpu",False),
        Tensor(torch.ones(2,3).T),Tensor(torch._neg_view(torch.ones(3)))]
    bad.extend(Tensor(helpers.unaligned_tensor(i)) for i in (1,2,3))
    class Huge(Tensor):
        def numel(self):return 2**63
    bad.append(Huge(torch.empty(0)))
    for x in bad:
        calls.clear()
        try:ns["quantize_global"](x)
        except (TypeError,ValueError):pass
        else:raise AssertionError("invalid input admitted")
        assert calls==[]
    for value in (float("nan"),float("inf"),-float("inf")):
        calls.clear()
        try:ns["quantize_global"](Tensor(torch.tensor([value])))
        except ValueError:pass
        else:raise AssertionError("nonfinite admitted")
        assert calls==["allocation"]
    # Original reciprocal graph has concrete zero/tiny-scale exceptional cases.
    zero=torch.tensor(0.,dtype=torch.float32)
    tiny=torch.tensor(2**-149,dtype=torch.float32)
    assert torch.isinf(1.0/zero) and torch.isnan(zero*(1.0/zero))
    assert torch.isinf(1.0/tiny) and torch.isinf(tiny*(1.0/tiny))
    assert torch.equal(torch.round(127.0*(tiny/tiny)).to(torch.int8),torch.tensor(127,dtype=torch.int8))
    # Native segment endpoints can be checked without allocating enormous tensors.
    for n in (0,1,2**30-1,2**30,2**30+1,2**31,2**63-1):
        segments=(n+2**30-1)//2**30
        for k in set([0,max(0,segments-1)]):
            if k>=segments:continue
            count=min(n-k*2**30,2**30)
            for block in (1024,2048):
                grid=(count+block-1)//block
                assert grid<=2**20
                last=k*2**30+(grid-1)*block+(block-1)
                assert last<2**63
    print(f"PASS: {2*len(cases)} complete CPU AST output-pair/frame cases; {len(bad)+3} rejections; both autotune blocks, repeated trials, segmented native endpoints.")
    print("CUDA mocked; no GPU/native-conformance claim. Configuration equivalence has no numerical assumptions.")
if __name__=="__main__":main()
