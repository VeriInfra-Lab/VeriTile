"""Execute unmodified wrapper/guard ASTs with metadata API doubles.

No Tensor payload allocation, JIT execution, or GPU launch is performed. The
launch recorder exposes the actual wrapper grid after all Python guards pass.
"""
import ast
import contextlib
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "rms_rbe_matmul/rms_rbe_matmul_repaired.py"
NEW = HERE / "rms_rbe_matmul_launch_guard.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tensor:
    cursor = 4096

    def __init__(self, shape, dtype="fp32", storage=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.layout = len(self.shape), "strided"
        self.device = SimpleNamespace(type="cuda", index=0)
        if storage is None:
            base = (Tensor.cursor + 7) // 8 * 8
            size = self.numel() * self.element_size()
            self.storage = SimpleNamespace(data_ptr=lambda: base, nbytes=lambda: size)
            Tensor.cursor = base + size + 256
        else:
            self.storage = storage

    def numel(self): return math.prod(self.shape)
    def element_size(self): return {"fp32": 4, "fp16": 2, "int8": 1}[self.dtype]
    def data_ptr(self): return self.storage.data_ptr()
    def untyped_storage(self): return self.storage
    def is_contiguous(self): return True
    def is_neg(self): return False
    def to(self, dtype): return Tensor(self.shape, dtype)
    def t(self): return Tensor(self.shape[::-1], self.dtype, self.storage)
    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype, self.storage)


def environment(path):
    trace = {"allocations": [], "launches": []}

    class Launch:
        def __getitem__(self, grid):
            def call(**kwargs):
                trace["launches"].append(list(grid))
                assert kwargs["BLOCK_SIZE_M"] == 16 and kwargs["BLOCK_SIZE_N"] == 64
            return call

    def empty(shape, dtype, device):
        trace["allocations"].append(list(shape))
        return Tensor(shape, dtype)

    ns = {"math": math, "rms_matmul_rbe": Launch(),
          "triton": SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b, reinterpret=lambda t, _: t),
          "tl": SimpleNamespace(float8e5="e5", float16="fp16"),
          "torch": SimpleNamespace(Tensor=Tensor, strided="strided", float32="fp32",
              float16="fp16", int8="int8", empty=empty,
              cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["rms_matmul_rbe_wrapper"], trace


def invoke(path, B, M, N, *, dtype="fp16", rotary=False):
    run, trace = environment(path)
    args = [Tensor((B, M, 1)), Tensor((N, 1), dtype), Tensor((1,), dtype), rotary, 0, N//2, 2]
    try:
        out = run(*args)
        result = {"accepted": True, "shape": list(out.shape)}
    except ValueError as exc:
        result = {"accepted": False, "error": str(exc)}
    return {**result, **trace}


def main():
    old_ast, new_ast = ast.parse(OLD.read_text()), ast.parse(NEW.read_text())
    old_functions = {node.name: node for node in old_ast.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: node for node in new_ast.body if isinstance(node, ast.FunctionDef)}
    for name in old_functions.keys() - {"rms_matmul_rbe_wrapper"}:
        assert ast.dump(old_functions[name]) == ast.dump(new_functions[name]), name
    wrapper = new_functions["rms_matmul_rbe_wrapper"]
    added = [node for node in wrapper.body if isinstance(node, ast.If)
             and any(isinstance(child, ast.Constant) and child.value == "nonempty launch grid exceeds CUDA axis limits"
                     for child in ast.walk(node))]
    assert len(added) == 1
    wrapper.body.remove(added[0])
    assert ast.dump(wrapper) == ast.dump(old_functions["rms_matmul_rbe_wrapper"])
    witness_old = invoke(OLD, 1, 1048561, 2)
    witness_new = invoke(NEW, 1, 1048561, 2)
    assert witness_old["accepted"] and witness_old["launches"] == [[1, 65536]]
    assert not witness_new["accepted"] and not witness_new["allocations"] and not witness_new["launches"]
    cases = [
        ("y boundary", 1, 1048560, 2, True),
        ("y boundary plus one tile", 1, 1048561, 2, False),
        ("flattened M/N boundary", 1, 4112, 16320, True),
        ("flattened M/N overflow", 1, 4097, 16322, False),
        ("N-only y boundary", 1, 1, 4194240, True),
        ("N-only y overflow", 1, 1, 4194242, False),
        ("x boundary", 2147483647, 1, 2, True),
        ("x boundary plus one", 2147483648, 1, 2, False),
        ("axes legal despite large product", 2147483647, 1048560, 2, True),
        ("empty batch bypasses y bound", 0, 1048561, 2, True),
        ("empty rows bypass x bound", 2147483648, 0, 2, True),
    ]
    results = []
    for name, B, M, N, accepted in cases:
        for dtype in ("fp16", "int8"):
            result = invoke(NEW, B, M, N, dtype=dtype, rotary=True)
            assert result["accepted"] == accepted, (name, dtype, result)
            if not accepted:
                assert not result["allocations"] and not result["launches"]
            elif not B or not M:
                assert not result["launches"]
            results.append({"case": name, "B": B, "M": M, "N": N, "dtype": dtype, **result})
    report = {"old_source": {"path": str(OLD), "sha256": sha(OLD)},
              "candidate": {"path": str(NEW), "sha256": sha(NEW)},
              "scope": "actual wrapper AST with Tensor metadata doubles and a recorded launch; no GPU execution",
              "kernel_and_helper_ast_unchanged": True, "only_wrapper_change_is_axis_guard": True,
              "admitted_counterexample": {"B": 1, "M": 1048561, "K": 1, "N": 2,
                  "heads": 1, "head_dim": 2, "start_pos": 0, "use_rbe": False,
                  "old": witness_old, "candidate": witness_new},
              "boundary_cases": results}
    root = HERE.parents[3]
    dest = root / ".veritile-cache/rms-rbe-launch-guard-source.json"
    dest.write_text(json.dumps(report, indent=2)+"\n")
    print(f"PASS: concrete admitted counterexample, {len(results)} boundary cases, unchanged kernel/helper ASTs")
    print(dest)


if __name__ == "__main__":
    main()
