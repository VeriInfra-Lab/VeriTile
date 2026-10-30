from pathlib import Path
import ast, contextlib, hashlib, importlib.util, json, os, sys, warnings
from unittest.mock import patch
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
os.environ["TRITON_CACHE_DIR"] = "/tmp/veritile-linear-triton-cache"
import torch
import numpy as np
from torch._subclasses.fake_tensor import FakeTensorMode
S = D
candidate = S / "triton_linear_activation_repaired.py"
original = D.parents[2] / "tritonbench_g/triton_linear_activation/triton_linear_activation.py"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
warnings.filterwarnings("ignore", category=FutureWarning)
spec = importlib.util.spec_from_file_location("tla_control", candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tree0, tree1 = ast.parse(original.read_text()), ast.parse(candidate.read_text())
def node(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
def ast_same(name):
    return ast.dump(node(tree0, name), include_attributes=False) == ast.dump(node(tree1, name), include_attributes=False)
assert all(ast_same(n) for n in ["tanh", "relu", "gelu", "fast_gelu", "get_configs_io_bound"])
def configs_ast(tree):
    fun = node(tree, "kernel_fma")
    call = next(d for d in fun.decorator_list if isinstance(d, ast.Call) and ast.unparse(d.func) == "triton.autotune")
    return ast.dump(next(k.value for k in call.keywords if k.arg == "configs"))
assert configs_ast(tree0) == configs_ast(tree1)
configs = module.kernel_fma.configs
assert len(configs) == 98
# Instantiate the original class alone: no original import-time GPU fixture runs.
original_ns = dict(vars(module))
exec(compile(ast.Module(body=[node(tree0, "LinearLayer")], type_ignores=[]), str(original), "exec"), original_ns)
OriginalLayer = original_ns["LinearLayer"]

class StorageProxy:
    def __init__(self, storage): self.storage = storage
    def data_ptr(self): return self.storage._cdata
    def __getattr__(self, key): return getattr(self.storage, key)
def identify(t):
    if t is not None:
        st = t.untyped_storage()
        st.data_ptr = lambda st=st: st._cdata * 2**40
        t.data_ptr = lambda st=st: st._cdata * 2**40
    return t
class Ctx:
    def save_for_backward(self, *xs): self.saved = xs

host_records = []
def host_case(shape, n, enabled, ambient, bias_flag, saved_flag, activation, original_case=False):
    record = {"shape": list(shape), "n": n, "autocast": enabled, "ambient": str(ambient),
              "bias": bias_flag, "saved": saved_flag, "activation": activation, "original": original_case}
    current = [7]
    with FakeTensorMode():
        x = identify(torch.empty(shape, device="cuda:0", dtype=torch.float32))
        w = identify(torch.empty((n, shape[-1]), device="cuda:0", dtype=torch.float32))
        rows = int(np.prod(shape[:-1]))
        bias = identify(torch.empty((n,), device="cuda:0")) if bias_flag else None
        saved = identify(torch.empty((rows, n), device="cuda:0")) if saved_flag else None
        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kw):
                    record["kernel"] = {"x_dtype": str(args[2].dtype), "w_dtype": str(args[3].dtype),
                        "saved_dtype": str(args[1].dtype) if args[1] is not None else None,
                        "saved_is_caller": args[1] is saved, "grid": list(grid({"BLOCK_M":128,"BLOCK_N":256})),
                        "device": current[0], "has_bias": kw["HAS_BIAS"], "has_saved": kw["SHOULD_SAVE_ACT_INPUTS"]}
                return launch
        @contextlib.contextmanager
        def device(owner):
            previous = current[0]; current[0] = owner.index
            try: yield
            finally: current[0] = previous
        real_copy = torch.Tensor.copy_
        def copy(dst, src, *args, **kw):
            record["copy"] = {"caller": dst is saved, "src_dtype": str(src.dtype), "dst_dtype": str(dst.dtype)}
            return real_copy(dst, src, *args, **kw)
        ctx = Ctx()
        kernel = Kernel()
        original_ns["kernel_fma"] = kernel
        with patch.object(module, "kernel_fma", kernel), patch.object(torch.cuda, "device", device), \
             patch.object(torch.cuda, "is_available", lambda: True), patch.object(torch.cuda, "is_bf16_supported", lambda: True), \
             patch.object(torch.Tensor, "copy_", copy):
            with torch.autocast("cuda", enabled=enabled, dtype=ambient):
                layer = OriginalLayer if original_case else module.LinearLayer
                out = layer.forward(ctx, x, w, bias, activation, saved)
        dtype = torch.float16 if enabled else torch.float32
        assert out.dtype == dtype and tuple(out.shape) == tuple(shape[:-1]) + (n,)
        assert ctx.saved[0].dtype == dtype and ctx.saved[2].dtype == dtype
        assert (ctx.saved[2] is x) == (not enabled)
        assert (ctx.saved[1] is None) == (not bias_flag)
        assert current[0] == 7
        if rows and n:
            assert "kernel" in record and record["kernel"]["x_dtype"] == str(dtype)
            if saved_flag:
                assert record["kernel"]["saved_dtype"] == str(dtype)
                assert record["kernel"]["saved_is_caller"] == (not enabled)
            if not original_case: assert record["kernel"]["device"] == 0
        else:
            assert "kernel" not in record
        assert ("copy" in record) == (enabled and saved_flag and not original_case)
        record.update(output_dtype=str(out.dtype), output_shape=list(out.shape), restored_device=current[0],
                      ctx_x_same=ctx.saved[2] is x, ctx_order=["weight", "bias_or_none", "x_original_rank"])
    host_records.append(record)

for enabled, ambient in [(False, torch.float16), (True, torch.float16), (True, torch.bfloat16)]:
    for shape, n in [((2,3),4), ((2,3,5),7), ((0,3),4), ((2,0,3),4), ((2,3),0), ((2,0),4)]:
        for bias, saved in [(False,False),(True,False),(False,True),(True,True)]:
            host_case(shape,n,enabled,ambient,bias,saved,"fast_gelu")
host_case((256,128),256,True,torch.float16,False,True,"",True)

invalid_records=[]
with FakeTensorMode():
    def tensor(shape, **kw): return identify(torch.empty(shape,device=kw.pop("device","cuda:0"),dtype=kw.pop("dtype",torch.float32),**kw))
    cases = [
        ("rank1", lambda: (tensor((3,)),tensor((4,3)),None,"",None)),
        ("dtype_half",lambda: (tensor((2,3),dtype=torch.float16),tensor((4,3)),None,"",None)),
        ("device_mismatch",lambda: (tensor((2,3)),tensor((4,3),device="cuda:1"),None,"",None)),
        ("weight_shape",lambda: (tensor((2,3)),tensor((4,2)),None,"",None)),
        ("bias_shape",lambda: (tensor((2,3)),tensor((4,3)),tensor((3,)),"",None)),
        ("saved_shape",lambda: (tensor((2,3)),tensor((4,3)),None,"",tensor((4,2)))),
        ("activation_type",lambda: (tensor((2,3)),tensor((4,3)),None,2,None)),
        ("index_guard",lambda: (tensor((2**31,0)),tensor((1,0)),None,"",None)),
    ]
    for name, factory in cases:
        try: module._validate_inputs(*factory())
        except (TypeError,ValueError) as error: invalid_records.append({"name":name,"error":str(error)})
        else: raise AssertionError(name)
    x=tensor((4,3))
    try: module._validate_inputs(x,x,None,"",None)
    except ValueError as error: invalid_records.append({"name":"nonempty_alias","error":str(error)})
    else: raise AssertionError("alias accepted")
    x=identify(torch.empty_strided((1,3),(99,1),device="cuda:0"))
    try: module._validate_inputs(x,tensor((4,3)),None,"",None)
    except ValueError as error: invalid_records.append({"name":"singleton_stride","error":str(error)})
    else: raise AssertionError("singleton stride accepted")

# CPU-only witness for the actual original hidden copy: initial caller 7 is
# not changed by writing its half conversion; explicit repaired copyback fixes it.
caller_saved=torch.full((256,256),7.0,dtype=torch.float32)
hidden=caller_saved.to(torch.float16);hidden.zero_()
assert torch.all(caller_saved==7)
before_witness=float(caller_saved[0,0])
caller_saved.copy_(hidden)
assert torch.all(caller_saved==0)
witness={"original_caller":before_witness,"hidden_store":0.0,"repaired_caller":float(caller_saved[0,0]),
         "shape":[256,256],"k":128,"execution":"CPU cast/copy plus actual custom_fwd FakeTensor control; no GPU kernel"}

geometry=[]
for index, conf in enumerate(configs):
    bm,bn,bk=(conf.kwargs[k] for k in ["BLOCK_M","BLOCK_N","BLOCK_K"])
    assert conf.kwargs["SPLIT_K"]==1
    shapes=[(1,1,0),(33,65,1),(10*bm+1,bn+3,bk+1),(257,131,129),(0,4,3),(3,0,3)]
    for M,N,K in shapes:
        gm=(M+bm-1)//bm;gn=(N+bn-1)//bn
        counts=np.zeros((M,N),dtype=np.int16);pairs=[]
        for pid in range(gm*gn):
            width=8*gn;group=pid//width;size=min(gm-group*8,8)
            tr=group*8+pid%size;tc=pid%width//size
            pairs.append((tr,tc))
            rr=tr*bm+np.arange(bm);cc=tc*bn+np.arange(bn)
            assert int(rr[-1])*max(N,1)+int(cc[-1])<2**31
            assert (int(rr[-1]%M)*max(K,1)+((K+bk-1)//bk)*bk+bk-1)<2**31
            valid_r=rr[rr<M];valid_c=cc[cc<N]
            assert np.array_equal(valid_r%M,valid_r) and np.array_equal(valid_c%N,valid_c)
            counts[np.ix_(valid_r,valid_c)]+=1
        assert len(set(pairs))==gm*gn
        assert set(pairs)=={(r,c) for r in range(gm) for c in range(gn)}
        assert np.all(counts==1)
        geometry.append({"config":index,"M":M,"N":N,"K":K,"programs":gm*gn,"cells":M*N})
    # Independent dense matrix expected value; source chronological masked K loop.
    for K in [0,1,bk-1,bk,bk+3,2*bk+1]:
        M,N=5,7
        X=(np.arange(M*K).reshape(M,K)%7-3).astype(np.int64)
        W=(np.arange(N*K).reshape(N,K)%11-5).astype(np.int64)
        bias=np.arange(N,dtype=np.int64)-3
        actual=np.broadcast_to(bias,(M,N)).copy()
        for remaining in range(K,0,-bk):
            offset=K-remaining
            for lane in range(bk):
                active=(K%bk==0) or lane<remaining
                assert active == (offset+lane<K)
                if active: actual+=X[:,offset+lane,None]*W[None,:,offset+lane]
        assert np.array_equal(actual, X@W.T+bias)

print(f"{len(host_records)} actual AMP/host cases; {len(invalid_records)} rejected; {len(geometry)} grouped geometry and {98*6} independent dense cases passed")
