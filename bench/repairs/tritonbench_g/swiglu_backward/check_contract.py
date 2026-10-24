"""CPU replay of the unchanged source AST and both checked return paths."""
import ast
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "swiglu_backward_checked.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/swiglu_backward/swiglu_backward.py"
spec = importlib.util.spec_from_file_location("swiglu_fwd_fixtures", HERE.parent / "swiglu_fwd/check_contract.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
Tensor, Pointer, function = fixtures.Tensor, fixtures.Pointer, fixtures.function
BLOCKS = fixtures.BLOCKS


def load_host():
    kernel, ids = fixtures.cpu_kernel(SOURCE, "_swiglu_bwd_kernel")
    events = []
    @contextlib.contextmanager
    def device(d):
        events.append(("device",d))
        yield
    def empty_like(t):
        events.append(("allocation",tuple(t.shape)))
        return t.wrap(torch.empty_like(t.data))
    class Launch:
        def __getitem__(self, grid):
            def run(x,y,dout,out,dx,dy,sx,sy,sd,so,sdx,sdy,n):
                recompute = out is not None
                pointers = [Pointer(t.data) if t is not None else None for t in (x,y,dout,out,dx,dy)]
                for block in BLOCKS:
                    actual = grid({"BLOCK_N":block})
                    events.append(("launch",block,recompute,actual))
                    for row in reversed(range(actual[0])):
                        ids[0] = row
                        for col in reversed(range(actual[1])):
                            ids[1] = col
                            kernel(*pointers,sx,sy,sd,so,sdx,sdy,n,block,recompute)
            return run
    ns = {"torch": SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
          empty_like=empty_like,cuda=SimpleNamespace(device=device)),
          "triton":SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b),"_swiglu_bwd_kernel":Launch()}
    fns = [function(SOURCE,"_check_payload"),function(SOURCE,"swiglu_backward_checked")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=fns,type_ignores=[])),str(SOURCE),"exec"),ns)
    return ns["swiglu_backward_checked"],events


def check_source():
    assert ast.dump(function(SOURCE,"_swiglu_bwd_kernel")) == ast.dump(function(ORIGINAL,"_swiglu_bwd_kernel"))
    original, checked = function(ORIGINAL,"_swiglu_bwd"),function(SOURCE,"swiglu_backward_checked")
    for stmt in original.body:
        assert any(ast.dump(stmt)==ast.dump(x) for x in checked.body),ast.unparse(stmt)
    assert ast.dump(function(SOURCE,"_check_payload")) == ast.dump(function(fixtures.SOURCE,"_check_payload"))
    fields = ["positive","sameDevice","rows32","columns32","stride32","padded32",
              "gradientUnique","outputUnique","alignment","gradientSeparate","outputSeparate"]
    contract = (HERE/"Contract.lean").read_text()
    for field in fields: assert f"  {field} :" in contract
    assert '.Proofs.' not in (HERE/"Kernels.lean").read_text()


def check_host():
    backward,events = load_host()
    torch.manual_seed(41)
    cases = [torch.randn(3,66),torch.randn(2,3,258),torch.randn(3,132)[:,::2],
             torch.randn(2,3,66).transpose(0,1),torch.randn(2,160)[:,5:135],
             torch.empty(0,66),torch.randn(66),torch.randn(2,2)]
    for data in cases:
        n = data.shape[-1]//2
        dout = torch.randn(*data.shape[:-1],2*n)[...,::2]
        x,y = data[...,:n],data[...,n:]
        sigmoid = torch.sigmoid(x)
        expected = torch.cat((sigmoid*(1+x*(1-sigmoid))*y*dout,x*sigmoid*dout),-1)
        out_expected = x*sigmoid*y
        for recompute in (False,True):
            before,up_before = data.clone(),dout.clone()
            rows = data.numel()//(2*n)
            grad_backing = torch.full((rows*(2*n+9)+16,),-99.)
            grad = torch.as_strided(grad_backing,(rows,2*n),(2*n+9,1),4)
            out_backing = torch.full((rows*(n+7)+16,),-98.)
            out = torch.as_strided(out_backing,(rows,n),(n+7,1),3)
            events.clear()
            result = backward(Tensor(data),Tensor(dout),Tensor(grad),recompute,Tensor(out) if recompute else object())
            gradient,output = result if recompute else (result,None)
            torch.testing.assert_close(gradient.data,expected,rtol=2e-6,atol=1e-6)
            torch.testing.assert_close(data,before,rtol=0,atol=0)
            torch.testing.assert_close(dout,up_before,rtol=0,atol=0)
            assert gradient.untyped_storage().data_ptr()==grad_backing.untyped_storage().data_ptr()
            if recompute:
                torch.testing.assert_close(output.data,out_expected,rtol=1e-6,atol=1e-7)
                assert output.untyped_storage().data_ptr()==out_backing.untyped_storage().data_ptr()
            for backing,width,stride,base,sentinel in ((grad_backing,2*n,2*n+9,4,-99),
                                                       (out_backing,n,n+7,3,-98)):
                touched = {base+r*stride+q for r in range(rows) for q in range(width)}
                if sentinel == -98 and not recompute: touched=set()
                assert all(backing[k]==sentinel for k in range(backing.numel()) if k not in touched)
            assert [e[1] for e in events if e[0]=="launch"]==list(BLOCKS)
            assert ("device",1) in events
    a = Tensor(torch.randn(3,8)); d = Tensor(torch.randn(3,4))
    bad = [([],d,None,False,None,TypeError),
        (Tensor(torch.randn(3,8,dtype=torch.float64)),d,None,False,None,TypeError),
        (Tensor(torch._neg_view(a.data)),d,None,False,None,ValueError),
        (a,Tensor(torch._neg_view(d.data)),None,False,None,ValueError),
        (a,Tensor(d.data,"cuda:0"),None,False,None,ValueError),
        (a,d,a,False,None,ValueError),(a,d,None,True,a.wrap(a.data[:,:4]),ValueError),
        (a,d,Tensor(torch.zeros(1,8).expand(3,8)),False,None,ValueError),
        (a,d,None,True,Tensor(torch.zeros(1,4).expand(3,4)),ValueError),
        (a,Tensor(torch.randn(3,5)),None,False,None,AssertionError),
        (Tensor(torch.randn(3,7)),d,None,False,None,ValueError)]
    g=Tensor(torch.randn(3,8))
    bad.append((a,d,g,True,g.wrap(g.data[:,:4]),ValueError))
    for xy,dout,grad,recompute,out,error in bad:
        events.clear()
        try: backward(xy,dout,grad,recompute,out)
        except error: pass
        else: raise AssertionError("unsupported call admitted")
        assert not any(e[0]=="launch" for e in events)
    # Ordinary allocation branches and the intentionally ignored out argument.
    for recompute in (False,True):
        result=backward(a,d,recompute_output=recompute,out=None if recompute else object())
        assert isinstance(result,tuple)==recompute
    print(f"Unchanged source AST; {2*len(cases)} supplied-output paths × six configurations, two allocation paths and {len(bad)} rejections passed.")
    print(f"CPU storage execution only; CUDA mocked. CUDA available: {torch.cuda.is_available()}.")


def check_physical_aliases():
    backward,events = load_host()
    # Distinct Tensor storage wrappers can expose overlapping physical bytes.
    backing = bytearray(4*258)
    xy = torch.frombuffer(backing,dtype=torch.float32,count=256,offset=0).reshape(2,128)
    gradient = torch.frombuffer(backing,dtype=torch.float32,count=256,offset=4).reshape(2,128)
    xy.copy_(torch.linspace(-1,1,256).reshape_as(xy))
    upstream = torch.ones(2,64)
    old_guard_admitted = gradient.untyped_storage().data_ptr() not in {
        xy.untyped_storage().data_ptr(),upstream.untyped_storage().data_ptr()}
    assert old_guard_admitted
    before = xy.clone()
    kernel,ids = fixtures.cpu_kernel(SOURCE,"_swiglu_bwd_kernel")
    x,y = xy.chunk(2,-1)
    dx,dy = gradient.chunk(2,-1)
    ids[0],ids[1] = 0,0
    kernel(*(Pointer(t) if t is not None else None for t in (x,y,upstream,None,dx,dy)),
           128,128,64,0,128,128,64,32,False)
    assert not torch.equal(xy,before),"overlap witness must modify a live input"
    events.clear()
    try: backward(Tensor(xy),Tensor(upstream),Tensor(gradient))
    except ValueError as error: assert "byte range" in str(error)
    else: raise AssertionError("partially overlapping distinct storage was admitted")
    assert not events,"rejection must precede allocation and launch"

    # Read-only aliases and disjoint slices of the same allocation remain valid.
    data = torch.randn(3,16)
    shared = torch.full((3*16+3*8,),-99.)
    gradient = shared[:3*16].reshape(3,16)
    output = shared[3*16:].reshape(3,8)
    before = data.clone()
    result,out = backward(Tensor(data),Tensor(data[:,:8]),Tensor(gradient),True,Tensor(output))
    gate,value = before.chunk(2,-1)
    sig = torch.sigmoid(gate)
    expected = torch.cat((sig*(1+gate*(1-sig))*value*gate,gate*sig*gate),-1)
    torch.testing.assert_close(result.data,expected,rtol=2e-6,atol=1e-6)
    torch.testing.assert_close(out.data,gate*sig*value,rtol=2e-6,atol=1e-6)
    torch.testing.assert_close(data,before,rtol=0,atol=0)
    witness = {"old_guard_admitted":old_guard_admitted,"distinct_storage_base_delta_bytes":4,
        "shape":[2,128],"source_program":[0,0],"block":32,"recompute":False,
        "kernel_wrote_live_input":True,"repaired_guard_rejects_before_launch":True,
        "readonly_alias_and_disjoint_same_storage_outputs_pass":True,
        "execution":"CPU source-AST replay with CUDA metadata facade; no GPU execution"}
    (HERE/"overlap_counterexample.json").write_text(json.dumps(witness,indent=2)+"\n")
    print("Physical overlap counterexample reproduced; repaired guard and supported aliases passed.")

if __name__=="__main__":
    check_source()
    check_host()
    check_physical_aliases()
