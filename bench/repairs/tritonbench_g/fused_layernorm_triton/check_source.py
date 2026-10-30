"""Run the actual source AST on CPU arrays and check wrapper admission.
CUDA metadata is mocked for guard tests. This does not validate GPU numerics.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fused_layernorm_triton_repaired.py"


class Tensor:
    def __init__(self, value, cuda=True, device=0):
        self.value, self.is_cuda = value, cuda
        self.device = SimpleNamespace(index=device)

    def __getattr__(self, name):
        return getattr(self.value, name)


def guards(tree):
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_checked_inputs"))
    ns = {"torch": SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), ns)
    check = ns["_checked_inputs"]
    x, w, b = torch.randn(3, 5), torch.randn(5), torch.randn(5)
    for shape in ((0, 5), (1, 1), (3, 5)):
        a = torch.empty(shape)
        assert check(Tensor(torch.ones(shape[1])), Tensor(torch.ones(shape[1])), Tensor(a)) == shape
    # Shared input storage and noncanonical singleton strides remain admitted.
    assert check(Tensor(w), Tensor(w), Tensor(x)) == x.shape
    singleton = torch.as_strided(torch.ones(5), (1, 5), (999, 1))
    assert singleton.is_contiguous()
    assert check(Tensor(w), Tensor(b), Tensor(singleton)) == (1, 5)
    exceptional = torch.tensor([[float("inf"), float("-inf"), float("nan"), -0.0, 0.0]])
    assert check(Tensor(w), Tensor(b), Tensor(exceptional)) == (1, 5)
    bad = [
        (w, b, x),
        (Tensor(w.double()), Tensor(b), Tensor(x)),
        (Tensor(w, cuda=False), Tensor(b), Tensor(x)),
        (Tensor(w, device=1), Tensor(b), Tensor(x)),
        (Tensor(w), Tensor(b), Tensor(x.t())),
        (Tensor(torch.arange(6, dtype=torch.float32)[1:]), Tensor(b), Tensor(x)),
        (Tensor(torch._neg_view(w)), Tensor(b), Tensor(x)),
        (Tensor(w), Tensor(b), Tensor(torch.ones(3, 0))),
        (Tensor(w[:4]), Tensor(b), Tensor(x)),
    ]
    for byte in (1, 2, 3):
        storage = torch.empty(4 * 5 + byte, dtype=torch.uint8).untyped_storage()[byte:]
        misaligned = torch.empty(0, dtype=torch.float32).set_(storage, 0, (5,), (1,))
        assert misaligned.data_ptr() % 4 == byte
        bad.append((Tensor(misaligned), Tensor(b), Tensor(x)))
    for args in bad:
        try:
            check(*args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid request was admitted")
    print(f"PASS: six admitted guard cases and {len(bad)} rejected cases")


class Pointer:
    def __init__(self, array, offset=0, bf16=False, writes=None):
        self.array, self.offset, self.bf16, self.writes = array, offset, bf16, writes

    def __add__(self, offset):
        return Pointer(self.array, self.offset + offset, self.bf16, self.writes)


class TL:
    float32, int64 = np.float32, np.int64
    constexpr = int

    def __init__(self):
        self.row = 0

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)

    def program_id(self, axis):
        assert axis == 0
        return np.int64(self.row)

    @staticmethod
    def arange(a, b):
        return np.arange(a, b, dtype=np.int64)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype=dtype)

    where = staticmethod(np.where)
    broadcast_to = staticmethod(np.broadcast_to)
    reshape = staticmethod(np.reshape)
    debug_barrier = staticmethod(lambda: None)

    @staticmethod
    def load(ptr, mask, **kwargs):
        offsets = np.asarray(ptr.offset)
        mask = np.broadcast_to(mask, offsets.shape)
        indexes = offsets[mask]
        assert np.all((indexes >= 0) & (indexes < len(ptr.array)))
        value = np.full(offsets.shape, np.nan, dtype=np.float32)
        value[mask] = ptr.array[indexes]
        return value

    @staticmethod
    def store(ptr, values, mask):
        offsets = np.asarray(ptr.offset)
        mask = np.ones(offsets.shape, dtype=bool) if mask is None else np.broadcast_to(mask, offsets.shape)
        indexes = offsets[mask]
        assert np.all((indexes >= 0) & (indexes < len(ptr.array)))
        values = np.broadcast_to(values, offsets.shape)[mask]
        if ptr.bf16:
            values = torch.from_numpy(values.copy()).to(torch.bfloat16).float().numpy()
        ptr.array[indexes] = values
        if ptr.writes is not None:
            ptr.writes[indexes] += 1


class Helpers:
    @staticmethod
    def welford_reduce(x, mean, m2, weight, first):
        if first:
            return x, np.zeros_like(x), np.ones_like(x)
        delta = x - mean
        weight = weight + np.float32(1)
        mean = mean + delta / weight
        return mean, m2 + delta * (x - mean), weight

    @staticmethod
    def welford(mean, m2, weight, axis):
        assert axis == 1
        mean, m2, weight = mean[0].copy(), m2[0].copy(), weight[0].copy()
        while len(mean) > 1:
            left, right = mean[::2], mean[1::2]
            lw, rw = weight[::2], weight[1::2]
            delta = np.where(left == right, np.float32(0), right - left)
            weight = lw + rw
            ratio = np.where(weight == 0, np.float32(0), rw / weight)
            mean = left + delta * ratio
            m2 = m2[::2] + m2[1::2] + delta * delta * lw * ratio
        return mean, m2, weight


class Casts(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            return ast.Call(func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="cast", ctx=ast.Load()),
                            args=[node.func.value, *node.args], keywords=[])
        return node


def execute(tree, name="triton_red_fused_native_layer_norm_0"):
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node = Casts().visit(node)
    tl = TL()
    ns = {"tl": tl, "triton_helpers": Helpers,
          "libdevice": SimpleNamespace(rsqrt=lambda x: np.float32(1) / np.sqrt(x))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), ns)
    kernel = ns[node.name]
    rng = np.random.default_rng(11)
    cases = 0
    results = []
    for block in (1024, 2048):
        shapes = [(rows, width, False) for rows, width in
                  ((0, 3), (1, 1), (3, 3), (2, 1023), (2, 1024), (2, 1025), (2, 2049), (2, 4099))]
        shapes += [(2, 5, True), (2, 2051, True)]
        for rows, width, alias in shapes:
            x = rng.normal(size=(rows, width)).astype(np.float32)
            w, b = rng.normal(size=(2, width)).astype(np.float32)
            if alias:
                w = x.ravel()[:width]
                b = w
            saved = tuple(a.copy() for a in (x, w, b))
            y = np.full(x.size, np.nan, dtype=np.float32)
            mean, inv = np.full(rows, np.nan, np.float32), np.full(rows, np.nan, np.float32)
            writes = np.zeros(x.size, dtype=np.int64)
            for row in range(rows):
                tl.row = row
                kernel(Pointer(inv), Pointer(x.ravel()), Pointer(w), Pointer(b), Pointer(mean),
                       Pointer(y, bf16=True, writes=writes), np.int64(rows), np.int64(width), 1, block)
            assert np.all(writes == 1)
            for actual, before in zip((x, w, b), saved):
                np.testing.assert_array_equal(actual, before)
            if rows:
                tx, tw, tb = torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b)
                reference = torch.nn.functional.layer_norm(tx, (width,), tw, tb, 1e-5).to(torch.bfloat16).float()
                np.testing.assert_allclose(y.reshape(rows, width), reference.numpy(), rtol=0.008, atol=0.001)
                np.testing.assert_allclose(mean, x.astype(np.float64).mean(1), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(inv, 1 / np.sqrt(x.astype(np.float64).var(1) + 1e-5), rtol=2e-5)
            results.append(tuple(a.copy() for a in (mean, inv, y)))
            cases += 1
    # A concrete original int32 product wraps, while the repaired arithmetic fits.
    exact = 65537 * 65537
    assert np.int64(65537) * np.int64(65537) == exact
    assert np.asarray(exact, dtype=np.int64).astype(np.int32).item() != exact
    print(f"PASS: {name}: {cases} actual source AST cases, both blocks, tails/multiple chunks and single stores; original index wrap reproduced")
    return results


def main():
    tree = ast.parse(SOURCE.read_text())
    guards(tree)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        original = execute(tree)
        candidate = execute(tree, "triton_red_fused_native_layer_norm_hoisted")
        for before, after in zip(original, candidate):
            for lhs, rhs in zip(before, after):
                np.testing.assert_array_equal(lhs.view(np.uint32), rhs.view(np.uint32))
        print("PASS: both actual source ASTs agree bitwise on all Mean/Rstd/Y test outputs")
    print("CPU source checks only; CUDA metadata mocked; no GPU numerical conformance claimed.")


if __name__ == "__main__":
    main()
