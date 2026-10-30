"""Actual source AST CPU facade and actual host FakeTensor controls.

NumPy dot is diagnostic only; this does not test native FMA rounding or GPU.
"""
import ast
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

A = Path(__file__).resolve().parent
S = A / "Whole001/attention_fwd_triton1_repaired.py"
OLD = A.parents[3] / "bench/tritonbench_g/attention_fwd_triton1/attention_fwd_triton1.py"
rows = []


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


class Pointer:
    def __init__(self, array, base=0, label=""):
        self.array, self.base, self.label = array, base, label

    def __add__(self, offset):
        return Pointer(self.array, self.base + int(offset), self.label)


class BlockPointer:
    def __init__(self, base, shape, strides, offsets, block_shape, order):
        self.base, self.shape, self.strides = base, shape, strides
        self.offsets, self.block_shape, self.order = offsets, block_shape, order
        self.dtype = SimpleNamespace(element_ty=np.float32)

    def lanes(self, boundary_check):
        for lane in np.ndindex(self.block_shape):
            coords = tuple(int(a + b) for a, b in zip(self.offsets, lane))
            address = self.base.base + sum(c * s for c, s in zip(coords, self.strides))
            valid = all(0 <= coords[axis] < self.shape[axis] for axis in boundary_check)
            yield lane, int(address), valid


class TL:
    float32, int64 = np.float32, np.int64
    pid = 0
    stores = []

    @classmethod
    def program_id(cls, axis):
        assert axis == 0
        return np.asarray(cls.pid, dtype=np.int32).view(Tensor)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype).view(Tensor)

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def make_block_ptr(*args):
        return BlockPointer(*args)

    @staticmethod
    def load(pointer, boundary_check=(), padding_option=None):
        result = np.zeros(pointer.block_shape, dtype=np.float32).view(Tensor)
        for lane, address, valid in pointer.lanes(boundary_check):
            if valid:
                assert 0 <= address < pointer.base.array.size, (pointer.base.label, address, pointer.base.array.size)
                result[lane] = pointer.base.array[address]
            else:
                assert padding_option == "zero"
        return result

    @classmethod
    def store(cls, pointer, value, boundary_check=()):
        for lane, address, valid in pointer.lanes(boundary_check):
            if valid:
                assert 0 <= address < pointer.base.array.size, (pointer.base.label, address, pointer.base.array.size)
                pointer.base.array[address] = value[lane]
                cls.stores.append((pointer.base.label, address))

    @staticmethod
    def dot(a, b, allow_tf32):
        assert allow_tf32 is False
        return np.asarray(a @ b, dtype=np.float32).view(Tensor)


def kernel(path):
    tree = ast.parse(path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "attention_fwd_kernel")
    fn.decorator_list = []
    for arg in fn.args.args:
        arg.annotation = None
    env = {"tl": TL}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), env)
    return env[fn.name]


repaired = kernel(S)
original = kernel(OLD)
rng = np.random.default_rng(20261018)
for length in (0, 1, 31, 32, 33, 64, 65):
    for store in (False, True):
        for ifcond in (False, True):
            batch, heads, dim = 2, 2, 16
            chunks = (length + 31) // 32
            shape = (batch, heads, length, dim)
            q, k, v = [rng.integers(-1, 2, size=shape).astype(np.float32) for _ in range(3)]
            history = np.full((batch, heads, chunks, dim, dim), 1234, dtype=np.float32)
            output = np.full(shape, -1234, dtype=np.float32)
            arrays = [Pointer(x.reshape(-1), label=label) for x, label in zip((q, k, v, history, output), "qkvho")]
            TL.stores = []
            for pid in range(batch * heads):
                TL.pid = pid
                repaired(*arrays, length * dim, dim, 1, chunks * dim * dim, dim,
                         length, dim ** -0.5, 32, dim, chunks, store, ifcond)
            expected = np.empty_like(output)
            for b in range(batch):
                for h in range(heads):
                    for row in range(length):
                        end = min((row // 32 + 1) * 32, length)
                        expected[b, h, row] = ((q[b, h, row].astype(np.float64) * dim ** -0.5)
                                               @ k[b, h, :end].astype(np.float64).T) @ v[b, h, :end].astype(np.float64)
                    for chunk in range(chunks):
                        desired = k[b, h, :chunk * 32].T @ v[b, h, :chunk * 32] if store else np.full((dim, dim), 1234)
                        np.testing.assert_array_equal(history[b, h, chunk], desired)
            np.testing.assert_array_equal(output, expected)
            assert len(TL.stores) == output.size + (history.size if store else 0)
            assert len(set(TL.stores)) == len(TL.stores)
            rows.append({"case": "actual_AST_dense_tail_and_side_output", "shape": shape,
                         "store": store, "ifcond": ifcond, "status": "PASS",
                         "output_writes": output.size, "history_writes": history.size if store else 0})

# Preserve the source-defect witness: an unmasked T33 load/store is not bounded.
try:
    TL.pid = 0
    original(*[Pointer(np.zeros(33 * 16, dtype=np.float32), label=x) for x in "qkv"],
             Pointer(np.zeros(2 * 16 * 16, dtype=np.float32), label="h"),
             Pointer(np.zeros(33 * 16, dtype=np.float32), label="o"),
             33 * 16, 16, 1, 2 * 16 * 16, 16, 33, 0.25, 32, 16, 2, True, True)
except AssertionError as error:
    rows.append({"case": "original_T33_unmasked_failure", "status": "PASS", "failure": str(error)})
else:
    raise AssertionError("old source unexpectedly bounded")

spec = importlib.util.spec_from_file_location("aft1_host", S)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


current_device, launch_devices, fail_launch = [0], [], [False]
class DeviceContext:
    def __init__(self, device): self.device=torch.device(device)
    def __enter__(self): self.previous=current_device[0];current_device[0]=self.device.index
    def __exit__(self,*exc): current_device[0]=self.previous
class TorchHost:
    cuda=SimpleNamespace(device=DeviceContext)
    def __getattr__(self,name):return getattr(torch,name)
module.torch=TorchHost()


class Spy:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert current_device[0]==args[0].device.index
            launch_devices.append(current_device[0])
            if fail_launch[0]:raise RuntimeError('mock submission failure')
            self.calls.append((grid, args, kwargs))
        return launch


def reject(label, thunk):
    try:
        thunk()
    except ValueError as error:
        rows.append({"case": label, "status": "PASS", "rejection": str(error)})
    else:
        raise AssertionError(label)


with FakeTensorMode(), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for shape in ((2, 2, 33, 64), (2, 2, 0, 16), (0, 2, 33, 16), (2, 0, 33, 16)):
        inputs = [torch.empty(shape, dtype=torch.float32, device="cuda") for _ in range(3)]
        spy = Spy()
        module.attention_fwd_kernel = spy
        out, history = module._forward_with_state(*inputs, True, True)
        assert out.shape == inputs[0].shape and out.dtype == torch.float32
        assert len(spy.calls) == int(all(shape[:3]))
        if spy.calls:
            grid, args, kwargs = spy.calls[0]
            assert grid == (shape[0] * shape[1],)
            assert args[5:10] == (shape[2] * shape[3], shape[3], 1, kwargs["NT"] * shape[3] ** 2, shape[3])
        rows.append({"case": "actual_host_dense_or_empty", "shape": shape, "status": "PASS"})
    mismatch=[torch.empty((1,1,33,16),device="cuda:1") for _ in range(3)]
    module.attention_fwd_kernel=Spy()
    module._forward_with_state(*mismatch,True,True)
    assert current_device[0]==0 and launch_devices[-1]==1
    fail_launch[0]=True
    try:module._forward_with_state(*mismatch)
    except RuntimeError:pass
    else:raise AssertionError('expected mock submission failure')
    finally:fail_launch[0]=False
    assert current_device[0]==0 and launch_devices[-1]==1
    rows.append({"case":"owning_device_and_exception_restoration","status":"PASS"})
    q = torch.empty((1, 1, 32, 64), device="cuda")
    k = torch.empty((1, 1, 32, 128), device="cuda")[..., :64]
    reject("independent_K_stride_rejected_explicitly", lambda: module._forward_with_state(q, k, q))
    transposed = torch.empty((2, 2, 32, 64), device="cuda").transpose(0, 1)
    reject("batch_head_transpose_rejected_explicitly", lambda: module._forward_with_state(transposed, transposed, transposed))
    feature = torch.empty((1, 1, 64, 32), device="cuda").transpose(2, 3)
    reject("time_feature_transpose_rejected_explicitly", lambda: module._forward_with_state(feature, feature, feature, True))
    reject("lazy_neg_rejected_explicitly", lambda: module._forward_with_state(torch._neg_view(q), q, q))
    reject("non_FP32_rejected", lambda: module._forward_with_state(q.to(torch.float16), q.to(torch.float16), q.to(torch.float16)))
    singleton = q.as_strided(q.shape, (10**12, 10**12, 64, 1))
    spy = Spy()
    module.attention_fwd_kernel = spy
    module.AttentionFunction.forward(None, singleton, q, q)
    assert spy.calls[0][1][5:10] == (2048, 64, 1, 4096, 64)
    rows.append({"case": "canonical_singleton_metadata", "status": "PASS"})

reject("padded_machine_metadata", lambda: module._launch_domain(1, 1, (1 << 31) - 1, 16))
reject("flattened_grid_machine_metadata", lambda: module._launch_domain(1 << 30, 4, 1, 16))
module._launch_domain(2, 2, 33, 64)

reject("non_tensor", lambda: module._forward_with_state(None, None, None))
reject("non_strided", lambda: module._forward_with_state(torch.sparse_coo_tensor(size=(1,1,32,16)), None, None))
for base in (-4, 1, (1 << 64)-4):
    reject("invalid_actual_pointer", lambda base=base: module._check_pointers((SimpleNamespace(data_ptr=lambda: base),), (1,)))
module._check_pointers((SimpleNamespace(data_ptr=lambda: 4096),), (1024,))

import copy
dump = lambda n: ast.dump(n, include_attributes=False)
old = ast.parse(OLD.read_text())
new = ast.parse(S.read_text())
kernel = lambda module: next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "attention_fwd_kernel")
nk, ok = kernel(new), kernel(old)
restored = copy.deepcopy(nk)
assert ast.unparse(restored.body[0].value) == "tl.program_id(0).to(tl.int64)"
restored.body[0].value = ast.parse("tl.program_id(0)", mode="eval").body
loop = next(n for n in restored.body if isinstance(n, ast.For))
assert ast.unparse(loop.iter) == "range(0, NT)"
loop.iter = ast.parse("range(0, tl.cdiv(T, BT))", mode="eval").body
ph = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "p_h")
assert ast.unparse(ph.value.args[2]) == "(s_ht, 1)"
ph.value.args[2] = ast.parse("(s_ht, s_qd)", mode="eval").body
load_axes = {}
for call in ast.walk(restored):
    if not isinstance(call, ast.Call):
        continue
    name = ast.unparse(call.func)
    if name == "tl.load":
        ptr = ast.unparse(call.args[0])
        kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        assert kw == {"boundary_check": (1 if ptr == "p_k" else 0,), "padding_option": "zero"}
        load_axes[ptr] = kw
        call.keywords = []
    if name == "tl.store" and ast.unparse(call.args[0]) == "p_o":
        assert {k.arg: ast.literal_eval(k.value) for k in call.keywords} == {"boundary_check": (0,)}
        call.keywords = []
assert load_axes.keys() == {"p_q", "p_k", "p_v"}
assert dump(restored) == dump(ok), "candidate changed more than the exact repair delta"
casts = [n for n in ast.walk(nk) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "to" and ast.unparse(n.args[0]) != "tl.int64"]
assert len(casts) == 6
for n in ast.walk(new):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
        assert n.func.attr not in {"contiguous", "clone", "resolve_conj", "resolve_neg"}
public = lambda module: next(n for n in next(n for n in module.body if isinstance(n, ast.ClassDef)
                                           and n.name == "AttentionFunction").body if isinstance(n, ast.FunctionDef))
assert dump(public(old).args) == dump(public(new).args)
assert ast.unparse(public(new).body[-1].value) == "_forward_with_state(q, k, v, store, ifcond)[0]"

print(json.dumps({"status":"PASS", "cases":len(rows), "source_AST_delta":"PASS", "GPU_execution":False, "native_numerical_conformance":False}))
