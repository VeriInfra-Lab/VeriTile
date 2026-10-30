"""Task-local source/host checks; CPU metadata stubs do not claim GPU execution."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "bgmv_shrink_kernel_repaired.py"
ORIGINAL = HERE.parent / "bgmv_shrink_kernel/bgmv_shrink_kernel_repaired.py"
EVIDENCE = ROOT / ".veritile-cache/bgmv-shrink-kernel-physical-evidence"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def check_source():
    # Check the pasted source and fixed caller ABI without normalizing casts,
    # assignment forms, tuple/list spellings or the source sentinel return.
    import sys
    sys.path.insert(0, str(ROOT))
    from bench.jit_correspondence import check, check_manifest
    before = function(ORIGINAL, "_bgmv_shrink_kernel")
    selected = function(SOURCE, "_bgmv_shrink_kernel")
    assert ast.dump(before) == ast.dump(selected), "original JIT body or signature changed"
    pairs = json.loads((HERE / "jit-correspondence.json").read_text())
    results = check_manifest(ROOT, HERE / "jit-correspondence.json")
    assert len(results) == 8 and all(row["status"] == "PASS" for row in results)
    row = pairs[0]
    lean = (ROOT / row["lean"]).read_text()
    mutations = [
        ("BLOCK_K * SPLIT_K", "BLOCK_K"),
        ("other=0.0", "other=1.0"),
        ("accumulator *= scaling", "accumulator += scaling"),
        ("tl.atomic_add(c_ptr", "tl.store(c_ptr"),
        ("current_k < K", "current_k <= K"),
        ("axis=1", "axis=0"),
        (".to(tl.int64)", ".to(tl.int32)"),
    ]
    for old, new in mutations:
        text = SOURCE.read_text().replace(old, new, 1)
        assert text != SOURCE.read_text()
        try:
            check(text, lean, row["python_function"], row["lean_definition"],
                  signature_definition=row["signature_definition"],
                  expected_signature=row["expected_signature"],
                  expected_call_bindings=row["expected_call_bindings"])
        except ValueError:
            pass
        else:
            raise AssertionError((old, new))
    pair_spec = importlib.util.spec_from_file_location("bgmv_shrink_exact_pair", HERE / "check_pair_source.py")
    pair = importlib.util.module_from_spec(pair_spec)
    pair_spec.loader.exec_module(pair)
    pair.check_pair()
    old_wrapper, wrapper = function(ORIGINAL, "_bgmv_shrink"), function(SOURCE, "_bgmv_shrink")
    launch = lambda f: next(n for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript))
    assert ast.dump(launch(old_wrapper)) == ast.dump(launch(wrapper)), "64-split launch changed"
    return {"full_ast_match": True, "jit_body_unchanged": True, "launch_unchanged": True,
            "raw_body_profiles": len(results), "source_mutants_rejected": len(mutations),
            "candidate_full_module": True}


def check_guards(candidate_text=None):
    spec = importlib.util.spec_from_file_location("shrink_physical_helpers", HERE.parent / "bgmv_expand_slice/check_contract.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    Tensor = helper.Tensor
    calls = []
    class Launch:
        def __getitem__(self, grid):
            return lambda *args, **kw: calls.append((grid, args, kw))
    fake_torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float16=torch.float16,
        bfloat16=torch.bfloat16, float32=torch.float32, int32=torch.int32, int64=torch.int64,
        all=torch.all, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    def host(path):
        node = function(path, "_bgmv_shrink") if path != SOURCE or candidate_text is None else next(
            n for n in ast.parse(candidate_text).body if isinstance(n, ast.FunctionDef) and n.name == "_bgmv_shrink")
        node.decorator_list = []
        env = {"torch": fake_torch, "triton": triton, "math": math, "_bgmv_shrink_kernel": Launch()}
        exec(compile(ast.fix_missing_locations(ast.Module([node], type_ignores=[])), str(path), "exec"), env)
        return env[node.name]
    original, candidate = host(ORIGINAL), host(SOURCE)
    results = []
    def case(b=1, n=1, k=257, dtype=torch.float16, itype=torch.int32, four=False):
        return [Tensor(torch.zeros((b, k), dtype=dtype)),
                Tensor(torch.zeros((1, 1, n, k) if four else (1, n, k), dtype=dtype)),
                Tensor(torch.ones((b, n), dtype=torch.float32)),
                Tensor(torch.zeros(b, dtype=itype)), 1.0]
    def run(label, args, admitted, launched):
        calls.clear()
        saved = [helper.raw(t.data).clone() for t in args[:4]]
        try:
            result = candidate(*args)
        except ValueError as error:
            assert not admitted, (label, str(error))
            assert not calls
            results.append({"case": label, "admitted": False, "reason": str(error)})
        else:
            assert admitted and result is None and len(calls) == int(launched), label
            if launched:
                grid, actual, kw = calls[0]
                assert grid == (64, args[3].numel()) and kw["BLOCK_K"] == 256 and kw["SPLIT_K"] == 64
                assert kw["BLOCK_N"] * 256 <= 2**20
                assert actual[2] is args[2] and actual[5] is args[3]
            results.append({"case": label, "admitted": True, "launched": bool(launched)})
        assert all(torch.equal(helper.raw(t.data), v) for t, v in zip(args[:4], saved)), label
    for dtype in (torch.float16, torch.bfloat16):
        for itype in (torch.int32, torch.int64):
            for four in (False, True):
                for side in (0, 1):
                    args = case(dtype=dtype, itype=itype, four=four)
                    raw = torch.zeros(1024, dtype=torch.uint8)
                    args[side] = Tensor(raw[:514].view(dtype).reshape(args[side].shape))
                    args[2] = Tensor(torch.from_dlpack(raw[512:516]).view(torch.float32).view(1, 1))
                    assert args[side].untyped_storage()._cdata != args[2].untyped_storage()._cdata
                    original(*args)  # Prove the exact old domain accepted this physical overlap.
                    run(f"dlpack-{dtype}-{itype}-rank{4 if four else 3}-read{side}", args, False, False)
    for itype in (torch.int32, torch.int64):
        args = case(itype=itype)
        raw = torch.zeros(16, dtype=torch.uint8)
        args[3] = Tensor(raw[:torch.empty((), dtype=itype).element_size()].view(itype))
        args[2] = Tensor(torch.from_dlpack(raw[:4]).view(torch.float32).view(1, 1))
        original(*args)
        run(f"metadata-overlap-{itype}", args, False, False)
    for n in (4095, 4096, 4097):
        run(f"tile-N{n}", case(n=n, k=1), n <= 4096, n <= 4096)
    for b, n in ((0, 4097), (1, 0)):
        args = case(b=b, n=n, k=1)
        # No metadata validation or static-tile cap occurs on the empty path.
        args[3].data.fill_(-2)
        run(f"empty-B{b}-N{n}", args, True, False)
    args = case(b=0, n=1, k=1)
    shared = torch.zeros(0, dtype=torch.float32)
    args[0] = Tensor(shared.view(torch.float16).view(0, 1))
    args[2] = Tensor(shared.view(0, 1))
    run("newly-admitted-same-storage-empty", args, True, False)
    args = case(k=256)
    args[1] = Tensor(args[0].data.view(1, 1, 256))
    run("read-read-alias", args, True, True)
    run("K0-still-64-atomic-launch", case(k=0), True, True)
    args = case(k=8)
    raw = torch.zeros(32, dtype=torch.uint8)
    args[0] = Tensor(torch.from_dlpack(raw[:16]).view(torch.float16).view(1, 8))
    args[2] = Tensor(torch.from_dlpack(raw[16:20]).view(torch.float32).view(1, 1))
    run("adjacent-distinct-storage-ranges", args, True, True)
    for scale in (0.0, -0.0, 1e-50, -1e-50, 1e300, -1e300):
        args = case(k=0)
        args[4] = scale
        run(f"finite-scale-{scale.hex()}", args, True, True)
        assert calls[0][1][6].hex() == scale.hex()
    return results


def check_guard_mutants():
    source = SOURCE.read_text()
    overlap = "output_start < output_end and start < end and output_start < end and start < output_end"
    mutants = {
        "storage-identity-only": source.replace(overlap, "output_storage._cdata == storage._cdata"),
        "missing-static-tile-check": source.replace("if 256 * BLOCK_N > 2**20:", "if False:"),
        "closed-interval-boundary": source.replace("output_start < end and start < output_end",
                                                   "output_start <= end and start <= output_end"),
        "static-tile-check-before-empty": source.replace("    if B == 0 or N == 0:\n",
            "    if 256 * BLOCK_N > 2**20:\n        raise ValueError('mutated empty policy')\n    if B == 0 or N == 0:\n"),
    }
    for label, mutant in mutants.items():
        assert mutant != source
        try:
            check_guards(mutant)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"host checker admitted mutant: {label}")
    return list(mutants)


if __name__ == "__main__":
    source = check_source()
    guards = check_guards()
    guard_mutants = check_guard_mutants()
    report = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "quote_sha256": hashlib.sha256((HERE / "Kernels.lean").read_bytes()).hexdigest(),
              "source": source, "guards": guards, "guard_mutants_rejected": guard_mutants, "gpu_execution": False,
              "scope": "Exact Python AST with CPU tensor storage and CUDA launch metadata stubs."}
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / "candidate-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source": source, "guard_cases": len(guards), "guard_mutants_rejected": len(guard_mutants),
                      "gpu_execution": False}))
