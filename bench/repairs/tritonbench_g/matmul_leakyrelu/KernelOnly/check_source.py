"""CPU controls executing the canonical kernel AST, with no GPU/TF32 claim."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "bench/repairs/tritonbench_g/matmul_leakyrelu/matmul_leakyrelu_repaired.py"
ORIGINAL = ROOT / "bench/tritonbench_g/matmul_leakyrelu/matmul_leakyrelu.py"
EVIDENCE = ROOT / ".veritile-cache/kernel-evidence/matmul_leakyrelu"
EXPECTED = "ca9ec0b702afebc59e4fbed320a55e4784005a361856a0bba10d13e60e2db34a"


class Tile(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tile)


class Pointer:
    __array_priority__ = 1000

    def __init__(self, storage, offset=0):
        self.storage, self.offset = storage, np.asarray(offset, dtype=np.int64)

    def __add__(self, other):
        return Pointer(self.storage, self.offset + other)


class Language:
    constexpr = object
    float32, float16 = np.float32, np.float16
    pid = 0
    stores = []
    loads = 0

    def program_id(self, axis):
        assert axis == 0
        return self.pid

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def arange(a, b):
        return np.arange(a, b).view(Tile)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype).view(Tile)

    @staticmethod
    def dot(a, b):
        return (a @ b).view(Tile)

    @staticmethod
    def where(c, a, b):
        return np.where(c, a, b).view(Tile)

    def load(self, ptr, mask, other):
        mask = np.broadcast_to(mask, ptr.offset.shape)
        selected = ptr.offset[mask]
        assert np.all((selected >= 0) & (selected < len(ptr.storage)))
        out = np.full(ptr.offset.shape, other, dtype=np.float32)
        out[mask] = ptr.storage[selected]
        self.loads += len(selected)
        return out.view(Tile)

    def store(self, ptr, value, mask):
        selected = ptr.offset[mask]
        assert np.all((selected >= 0) & (selected < len(ptr.storage)))
        self.stores.extend(selected.tolist())
        ptr.storage[selected] = value[mask]


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in {"matmul_kernel", "leaky_relu"}}


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    canonical, original = functions(SOURCE), functions(ORIGINAL)
    assert {k: ast.dump(v) for k, v in canonical.items()} == {k: ast.dump(v) for k, v in original.items()}
    language = Language()
    for node in canonical.values():
        node.decorator_list = []
    module = ast.Module(body=list(canonical.values()), type_ignores=[])
    env = {"tl": language}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), env)
    cases = [(0, 0, 0, "dense"), (0, 3, 5, "dense"), (3, 0, 0, "dense"),
             (3, 0, 5, "dense"), (1, 1, 1, "dense"), (31, 31, 33, "transpose"),
             (32, 32, 32, "dense"), (33, 33, 31, "slice"), (129, 65, 67, "dense"),
             (129, 1, 33, "broadcast"), (33, 33, 33, "alias"), (2, 3, 4, "nonfinite")]
    results = []
    rng = np.random.default_rng(112358)
    with np.errstate(all="ignore"):
        for m, k, n, kind in cases:
            aoff, boff, coff = 3, 5, 7
            am, ak = max(k, 1), 1
            bk, bn = max(n, 1), 1
            if kind == "transpose":
                bk, bn = 1, max(k, 1)
            elif kind == "slice":
                bk, bn = 3 * max(n, 1) + 2, 3
            elif kind == "broadcast":
                bk, bn = 0, 0
            a = (np.arange(max(1, m * max(k, 1)) + aoff + 11) % 17 - 8).astype(np.float32) / 4
            bsize = max(1, max(k - 1, 0) * bk + max(n - 1, 0) * bn + 1)
            b = (np.arange(bsize + boff + 13) % 13 - 6).astype(np.float32) / 4
            if kind == "alias":
                b, boff = a, aoff
            if kind == "nonfinite":
                a[aoff:aoff+3] = [np.nan, np.inf, -np.inf]
            before_a, before_b = a.copy(), b.copy()
            A = a[aoff + np.arange(m)[:, None] * am + np.arange(k)[None, :] * ak]
            B = b[boff + np.arange(k)[:, None] * bk + np.arange(n)[None, :] * bn]
            real = A.astype(np.float64) @ B.astype(np.float64)
            count = ((m + 31) // 32) * ((n + 31) // 32)
            for activation in ["", "leaky_relu", "unrecognized"]:
                expected = real
                if activation == "leaky_relu":
                    expected = np.where(real >= 0, real, np.float32(0.01) * real)
                expected = expected.astype(np.float16)
                orders = [list(range(count)), list(reversed(range(count))), rng.permutation(count).tolist()]
                for order in orders:
                    c = np.full(coff + m*n + 9, np.float16(123))
                    language.stores, language.loads = [], 0
                    for pid in order:
                        language.pid = pid
                        env["matmul_kernel"](Pointer(a, aoff), Pointer(b, boff), Pointer(c, coff),
                            m, n, k, am, ak, bk, bn, max(n, 1), 1, 32, 32, 32, 4, activation)
                    np.testing.assert_array_equal(c[coff:coff+m*n].reshape(m, n), expected)
                    np.testing.assert_array_equal(a, before_a)
                    np.testing.assert_array_equal(b, before_b)
                    assert np.all(c[:coff] == 123) and np.all(c[coff+m*n:] == 123)
                    assert sorted(language.stores) == list(range(coff, coff+m*n))
                    if k == 0:
                        assert language.loads == 0
                    results.append({"shape": [m, k, n], "layout": kind, "activation": activation,
                                    "order": order, "stores": len(language.stores), "loads": language.loads})
    report = {"canonical_sha256": EXPECTED, "kernel_ast_equal_to_original": True,
              "complete_source_executions": len(results), "cases": results,
              "scope": "Canonical AST execution under NumPy CPU scalar/dot controls; no native TF32, compiler, or GPU equivalence claim."}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "source-controls.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
