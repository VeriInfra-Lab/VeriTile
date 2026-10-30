"""Run the actual repaired RMS wrappers/JIT ASTs on CPU storage.
Checks source indexing and host behavior; does not claim native GPU numerics.
"""
import ast
import contextlib
import copy
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rms_source_base", HERE.parent / "rbe_triton_transform/check_source.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Tensor(base.Tensor):
    def t(self):
        return Tensor(self.value.t(), device=self.device.index)

    def view(self, *shape):
        return Tensor(self.value.view(*shape), device=self.device.index)

    def to(self, dtype):
        return Tensor(self.value.to(dtype), device=self.device.index)


class Pointer:
    def __init__(self, array, offset, dtype, writes):
        self.array, self.offset, self.dtype, self.writes = array, offset, dtype, writes

    def __add__(self, offset):
        return Pointer(self.array, self.offset + offset, self.dtype, self.writes)


class Reinterpreted:
    def __init__(self, tensor, dtype):
        self.tensor, self.dtype = tensor, dtype


class TL(base.TL):
    float16, int8, float8e5 = np.float16, np.int8, "e5"
    math = SimpleNamespace(rsqrt=lambda x: np.float32(1) / np.sqrt(x))
    reshape = staticmethod(np.reshape)
    split = staticmethod(lambda x: (x[..., 0], x[..., 1]))
    join = staticmethod(lambda x, y: np.stack((x, y), axis=-1))
    dot = staticmethod(lambda x, y: np.matmul(x.astype(np.float32), y.astype(np.float32)))

    @staticmethod
    def cast(value, dtype, bitcast=False):
        if dtype == "e5":
            assert bitcast
            return torch.from_numpy(np.asarray(value, dtype=np.int8).copy()).view(torch.float8_e5m2).float().numpy()
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def load(ptr, mask, other=0):
        offsets = np.asarray(ptr.offset)
        mask = np.broadcast_to(mask, offsets.shape)
        indexes = offsets[mask]
        assert np.all((indexes >= 0) & (indexes < ptr.array.size))
        result = np.full(offsets.shape, other, dtype=ptr.array.dtype)
        result[mask] = ptr.array[indexes]
        return result

    @staticmethod
    def store(ptr, values, mask):
        offsets = np.asarray(ptr.offset)
        mask = np.broadcast_to(mask, offsets.shape)
        indexes = offsets[mask]
        assert np.all((indexes >= 0) & (indexes < ptr.array.size))
        selected = np.broadcast_to(values, offsets.shape)[mask]
        if ptr.dtype == "e5":
            selected = torch.from_numpy(selected.copy()).to(torch.float8_e5m2).view(torch.int8).numpy()
        ptr.array[indexes] = selected
        np.add.at(ptr.writes, indexes, 1)


class Casts(base.Casts):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            return ast.Call(func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="cast", ctx=ast.Load()),
                            args=[node.func.value, *node.args], keywords=node.keywords)
        return node


class Launch:
    def __init__(self, fn, tl, records):
        self.fn, self.tl, self.records = fn, tl, records

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            assert kwargs.pop("num_warps") == 4 and kwargs.pop("num_stages") == 4
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            tensors, counters = {}, {}
            for name, value in list(bound.arguments.items()):
                dtype = None
                if isinstance(value, Reinterpreted):
                    dtype, value = value.dtype, value.tensor
                if isinstance(value, Tensor):
                    raw = value.untyped_storage()
                    array = torch.empty(0, dtype=value.dtype).set_(raw, 0, (raw.nbytes() // value.element_size(),), (1,)).numpy()
                    counter = np.zeros(array.size, dtype=np.int64)
                    tensors[name], counters[name] = value.value, counter
                    bound.arguments[name] = Pointer(array, value.storage_offset(), dtype or array.dtype, counter)
                elif name in ("EPS", "THETA"):
                    bound.arguments[name] = np.float32(value)
            for batch in range(grid[0]):
                for program in range(grid[1]):
                    self.tl.ids = batch, program
                    self.fn(*bound.args, **bound.kwargs)
            for name, tensor in tensors.items():
                expected = np.zeros_like(counters[name])
                if name in ("out_ptr", "q_ptr", "k_ptr", "v_ptr"):
                    expected[tensor.storage_offset():tensor.storage_offset() + tensor.numel()] = 1
                assert np.array_equal(counters[name], expected), (name, tensor.shape)
            self.records.append(grid)
        return call


def environment(source, single=False):
    tl, records = TL(), []
    def empty(shape, dtype, device):
        return Tensor(torch.empty(shape, dtype=dtype), device=device.index)
    ns = {"math": math, "tl": tl, "triton": SimpleNamespace(cdiv=TL.cdiv, reinterpret=Reinterpreted),
          "torch": SimpleNamespace(Tensor=Tensor, float32=torch.float32, float16=torch.float16,
              int8=torch.int8, strided=torch.strided, empty=empty,
              cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))}
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        node = copy.deepcopy(node)
        if node.decorator_list:
            node.decorator_list = []
            node = Casts().visit(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), ns)
        if node.name == ("rms_matmul_rbe" if single else "rms_matmul_rbe_qkv"):
            ns[node.name] = Launch(ns[node.name], tl, records)
    return ns, records


def bytes_of(x):
    return torch.empty(0, dtype=torch.uint8).set_(x.untyped_storage(), 0, (x.untyped_storage().nbytes(),), (1,))


def decode(x):
    return x.view(torch.float8_e5m2).float() if x.dtype == torch.int8 else x.float()


def reference(x, w, rms, rotary, pos, head, dtype):
    rms = rms.to(w.dtype)
    y = (x * decode(rms)) @ decode(w).t()
    y *= torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    if rotary:
        pairs = torch.arange(0, w.shape[0], 2)
        phase = (torch.arange(x.shape[1], dtype=torch.int64) + pos).float()[:, None]
        phase = phase / (10000. ** ((pairs % head).float() / head))[None, :]
        cosine, sine = phase.cos(), phase.sin()
        real, imag = y[..., ::2].clone(), y[..., 1::2].clone()
        y[..., ::2] = real * cosine - imag * sine
        y[..., 1::2] = real * sine + imag * cosine
    return y.to(torch.float8_e5m2).view(torch.int8) if dtype == torch.int8 else y.to(dtype)


def wrapped(value, dtype=None):
    value = value if dtype is None else value.to(dtype)
    if dtype == torch.int8:
        value = (value.float() / 8).to(torch.float8_e5m2).view(torch.int8)
    raw = torch.empty(value.numel() + 9, dtype=value.dtype)
    raw.fill_(13)
    raw[3:3 + value.numel()].copy_(value.reshape(-1))
    return Tensor(raw[3:3 + value.numel()].view(value.shape))


def check(single=False):
    torch.manual_seed(73)
    source = HERE.parent / "rms_rbe_matmul/rms_rbe_matmul_repaired.py" if single else HERE / "rms_matmul_rbe_repaired.py"
    ns, records = environment(source, single)
    run = ns["rms_matmul_rbe_wrapper" if single else "rms_matmul_rbe_qkv_wrapper"]
    cases, rejected = 0, 0
    for dtype in (torch.float16, torch.int8):
        for B, M, K, H, D in ((0, 3, 1, 1, 2), (2, 0, 3, 2, 2), (1, 1, 1, 1, 2),
                              (2, 3, 3, 2, 6), (1, 17, 65, 2, 34), (2, 2, 127, 3, 22)):
            N = H * D
            x = wrapped(torch.randint(-4, 5, (B, M, K)).float() / 8)
            w = wrapped(torch.randint(-4, 5, (N, K)).float() / 8, dtype)
            rms = wrapped(torch.ones(K), dtype)
            if K == 1:
                x = Tensor(torch.as_strided(x.value, x.shape, (2**62, 2**62, 2**62)))
                rms = Tensor(torch.as_strided(rms.value, rms.shape, (2**62,)))
            for rotary in ((False, True) if single else (True,)):
                if single and dtype == torch.float16 and K > 1:
                    rms = wrapped(torch.full((K,), 1.125), torch.float32)
                reads = [x, w, rms]
                before = [bytes_of(t.value).clone() for t in reads]
                pos = 2**33 if M == 3 else -2
                if single:
                    out = run(x, w, rms, rotary, pos, H, D)
                    outputs = (out,)
                    refs = (reference(x.value, w.value, rms.value, rotary, pos, D, dtype),)
                else:
                    k = wrapped(torch.empty(B * M * N), dtype)
                    v = wrapped(torch.empty(B, M, H, D), dtype)
                    kbytes, vbytes = bytes_of(k.value).clone(), bytes_of(v.value).clone()
                    outputs = run(x, pos, w, w, w, rms, H, D, k, v)
                    assert outputs[1].data_ptr() == k.data_ptr() and outputs[2].data_ptr() == v.data_ptr()
                    refs = tuple(reference(x.value, w.value, rms.value, flag, pos, D, target)
                                 for flag, target in ((True, torch.float16), (True, dtype), (False, dtype)))
                    for t, initial in ((k, kbytes), (v, vbytes)):
                        now = bytes_of(t.value)
                        start, stop = t.storage_offset() * t.element_size(), (t.storage_offset() + t.numel()) * t.element_size()
                        assert torch.equal(now[:start], initial[:start]) and torch.equal(now[stop:], initial[stop:])
                for out, ref in zip(outputs, refs):
                    assert out.shape == (B, M, H, D) and out.dtype == ref.dtype
                    torch.testing.assert_close(decode(out.value), decode(ref).view(B, M, H, D), rtol=.002, atol=.002)
                for t, initial in zip(reads, before):
                    assert torch.equal(bytes_of(t.value), initial)
                cases += 1
    x, w, rms = Tensor(torch.ones(1, 2, 3)), Tensor(torch.ones(4, 3, dtype=torch.float16)), Tensor(torch.ones(3, dtype=torch.float16))
    if single:
        good = [x, w, rms, True, 0, 1, 4]
        variants = [(0, x.value), (0, Tensor(x.value.double())), (0, Tensor(x.value, cuda=False)),
                    (0, Tensor(x.value.transpose(1, 2))), (0, Tensor(torch._neg_view(x.value))),
                    (1, Tensor(w.value, device=1)), (2, Tensor(torch.ones(2))), (3, 1),
                    (4, 1.5), (4, 2**63 - 2), (5, True), (5, 0), (6, 3)]
    else:
        good = [x, 0, w, w, w, rms, 1, 4, Tensor(torch.zeros(8, dtype=torch.float16)), Tensor(torch.zeros(8, dtype=torch.float16))]
        variants = [(0, x.value), (0, Tensor(x.value.double())), (0, Tensor(x.value, cuda=False)),
                    (0, Tensor(x.value.transpose(1, 2))), (0, Tensor(torch._neg_view(x.value))),
                    (2, Tensor(w.value, device=1)), (3, Tensor(w.value.to(torch.int8))), (5, Tensor(torch.ones(2))),
                    (1, 1.5), (1, 2**63 - 2), (6, True), (6, 0), (7, 3), (8, Tensor(torch.zeros(7, dtype=torch.float16))),
                    (9, good[8]), (8, Tensor(w.value.reshape(-1)[:8]))]
    for index, bad in variants:
        args = good.copy()
        args[index] = bad
        try:
            run(*args)
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError(("admitted bad argument", index))
    # A float32 RMS alias of X is valid for single projection; a mixed-width
    # weight alias is rejected by both wrappers before launch.
    args = good.copy()
    shared = torch.ones(20, dtype=torch.float32)
    args[0] = Tensor(shared[:6].view(1, 2, 3))
    args[1 if single else 2] = Tensor(shared.view(torch.float16)[:12].view(4, 3))
    try:
        run(*args)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("mixed-dtype backing alias admitted")
    # Same dtype/base with different tensor offsets is admitted. DLPack can
    # expose a shifted backing base for the same bytes; that imported overlap
    # is outside the selected base-keyed representation scope.
    shared = torch.arange(16, dtype=torch.float16) / 16
    args = good.copy()
    if single:
        args[1], args[2] = Tensor(shared[:12].view(4, 3)), Tensor(shared[1:4])
        imported_index, imported = 2, torch.from_dlpack(shared[1:4])
    else:
        args[2], args[3], args[4] = (Tensor(shared[j:j+12].view(4, 3)) for j in (0, 1, 2))
        imported_index, imported = 3, torch.from_dlpack(shared[1:13].view(4, 3))
    saved = bytes_of(shared).clone()
    run(*args)
    assert torch.equal(bytes_of(shared), saved)
    cases += 1
    assert imported.untyped_storage().data_ptr() != shared.untyped_storage().data_ptr()
    assert imported.data_ptr() == args[imported_index].data_ptr()
    args[imported_index] = Tensor(imported)
    try:
        run(*args)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("overlapping imported backing bases admitted")
    if single:
        args = good.copy()
        args[2] = Tensor(x.value[0, 0])
        run(*args)
        cases += 1
        raw_cases = 0
        for B, M, K, pos in ((1, 1, 1, 0), (2, 3, 127, -2), (1, 3, 1025, 2**33), (1, 1, 3, 2**63-2)):
            x = torch.randn(B, M, K)
            original = x.numpy().reshape(-1).copy()
            output = np.full(x.numel()+7, 17, dtype=np.float32)
            writes = np.zeros(output.size, dtype=np.int64)
            input_writes = np.zeros(original.size, dtype=np.int64)
            for batch in range(B):
                for program in range(TL.cdiv(M, 2)*TL.cdiv(K, 1024)):
                    ns["tl"].ids = batch, program
                    ns["rbe_triton"](Pointer(original, 0, np.float32, input_writes),
                        Pointer(output, 3, np.float32, writes), M, K, M*K, K, 1, M*K, K, 1,
                        pos, np.float32(10000), 2, 1024)
            expected = np.zeros_like(writes)
            expected[3:3+x.numel()] = 1
            assert np.array_equal(writes, expected) and not input_writes.any()
            assert np.array_equal(original, x.numpy().reshape(-1))
            assert np.all(output[:3] == 17) and np.all(output[3+x.numel():] == 17)
            # Independent dense formula using the adapter's scalar pow ports.
            # At huge positions, a one-ulp Torch/NumPy pow difference can
            # change the phase substantially; this check concerns indexing.
            pairs = np.arange(0, K, 2, dtype=np.int64)
            frequency = np.power(np.float32(10000), (pairs % 128).astype(np.float32) / np.float32(128))
            angle = (np.arange(M, dtype=np.int64) + pos).astype(np.float32)[:, None] / frequency[None, :]
            real = x.numpy()[..., ::2]
            imag = np.zeros_like(real)
            imag[..., :K//2] = x.numpy()[..., 1::2]
            ref = np.empty_like(x.numpy())
            ref[..., ::2] = real*np.cos(angle)-imag*np.sin(angle)
            ref[..., 1::2] = (real*np.sin(angle)+imag*np.cos(angle))[..., :K//2]
            np.testing.assert_allclose(output[3:3+x.numel()].reshape(B, M, K), ref, rtol=5e-5, atol=4e-6)
            raw_cases += 1
        print(f"PASS: {raw_cases} actual in-file raw rotary helper cases, odd tails and int64 positions")
    print(f"PASS: {'single' if single else 'QKV'} {cases} actual-source cases, {rejected} rejected calls, complete stores and full backing frames")
    print("CPU AST/storage adapter; no native GPU numerical conformance claimed.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        check()
