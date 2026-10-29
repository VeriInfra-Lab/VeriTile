#!/usr/bin/env python3
"""Offline native compilation of the unchanged decorated backward softmax JIT.

The original host tests and GPU launches are never executed. Compile failures
are retained as evidence, not converted into a narrower source program.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "ksoftmax_triton.py"
SOURCE_SHA256 = "8f392086c34ec0cd665c31e614d1534fa2899fba0beecc6435cc06e2095d0e66"
FLOATS = ("fp16", "bf16", "fp32", "fp64", "fp8e4nv", "fp8e5")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ast_digest(node: ast.AST) -> str:
    return digest(ast.dump(node, include_attributes=False).encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--matrix", choices=("homogeneous", "mixed", "all"), default="all")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    if digest(raw) != SOURCE_SHA256:
        raise RuntimeError("Pinned backward source hash mismatch")
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.artifacts / "cache"))

    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime.jit import JITFunction

    tree = ast.parse(raw)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    names = ("get_depth", "_softmax_backward", "softmax_backward")
    extracted = ast.Module(body=[
        ast.Import(names=[ast.alias(name="torch")]),
        ast.Import(names=[ast.alias(name="triton")]),
        ast.Import(names=[ast.alias(name="triton.language", asname="tl")]),
        *[functions[name] for name in names],
    ], type_ignores=[])
    path = args.artifacts / "original_backward.py"
    path.write_text(ast.unparse(extracted) + "\n")
    emitted = {node.name: node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.FunctionDef)}
    for name in names:
        if ast_digest(emitted[name]) != ast_digest(functions[name]):
            raise RuntimeError(f"AST changed during extraction: {name}")
    spec = importlib.util.spec_from_file_location("veritile_original_backward_native", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    decorated = module._softmax_backward
    configs = [config.num_warps for config in decorated.configs]
    if configs != [1, 2, 4, 8, 16, 32]:
        raise RuntimeError(f"Unexpected original launch choices: {configs}")
    fn = decorated
    heuristics = {}
    while not isinstance(fn, JITFunction):
        heuristics.update(getattr(fn, "values", {}))
        fn = fn.fn
    if set(heuristics) != {"DEPTH", "IS_FP16"}:
        raise RuntimeError(f"Unexpected original heuristics: {heuristics.keys()}")

    class Destination:
        def __init__(self, dtype):
            self.dtype = dtype

    # These are original source heuristic functions, evaluated on type-only
    # host stand-ins; no device tensor is allocated.
    profiles = []
    if args.matrix in ("homogeneous", "all"):
        for storage, log, causal, warps, count in itertools.product(
                FLOATS, (False, True), (False, True), configs, (3, 257)):
            profiles.append((storage, storage, storage, log, causal, warps, count))
    if args.matrix in ("mixed", "all"):
        for grad, saved, dest, log, causal in itertools.product(
                FLOATS, FLOATS, FLOATS, (False, True), (False, True)):
            profiles.append((grad, saved, dest, log, causal, 4, 3))
    profiles = list(dict.fromkeys(profiles))
    results = []
    receipt = {
        "source_sha256": SOURCE_SHA256,
        "source_ast_sha256": {name: ast_digest(functions[name]) for name in names},
        "checker_sha256": digest(Path(__file__).read_bytes()),
        "triton_version": triton.__version__, "torch_version": torch.__version__,
        "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
        "kernel_launched": False, "matrix": args.matrix,
        "original_num_warps": configs, "expected_profiles": len(profiles),
        "scope": "Unchanged complete backward JIT, its decorators and host wrapper ASTs. Homogeneous matrix covers six floating storage types, all LOG/CAUSAL flags, six original autotune warp choices and representative K=3/DEPTH=4 and K=257/DEPTH=512. Mixed matrix independently varies all three floating storage types at K=3/DEPTH=4 and four warps. Source IS_FP16 derives from destination dtype; runtime count and all strides use signed i32 ABI. Offline compile only: no GPU run, exhaustive shape/ABI admission, real/IEEE correctness, source-to-Lean correspondence or independent proof replay is claimed.",
        "results": results,
    }
    patterns = ("tt.load", "tt.store", "tt.reduce", "arith.extf", "arith.truncf",
                "arith.select", "arith.addf", "arith.subf", "arith.mulf", "math.exp")
    for index, (grad, saved, dest, log, causal, warps, count) in enumerate(profiles):
        nargs = {"K": count, "GradIn": Destination(torch.float16 if dest == "fp16" else object())}
        constants = {"LOG": log, "CAUSAL": causal,
                     "DEPTH": heuristics["DEPTH"](nargs),
                     "IS_FP16": heuristics["IS_FP16"](nargs)}
        signature = {"GradIn": "*" + dest, "GradOut": "*" + grad, "Out": "*" + saved,
                     **{name: "i32" for name in
                        ("stride_bm", "stride_bn", "stride_gm", "stride_gn",
                         "stride_om", "stride_on", "K")}}
        options = {"num_warps": warps, "enable_fp_fusion": True}
        name = f"g{grad}_o{saved}_d{dest}_l{int(log)}_c{int(causal)}_w{warps}_k{count}"
        row = {"name": name, "gradient": grad, "saved": saved, "destination": dest,
               "signature": signature, "constexprs": constants, "options": options}
        try:
            compiled = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                      target=GPUTarget("cuda", 90, 32), options=options)
            for kind in ("ttir", "ttgir", "llir", "ptx"):
                content = compiled.asm[kind]
                artifact = args.artifacts / f"{name}.{kind}"
                artifact.write_text(content)
                row[kind + "_sha256"] = digest(content.encode())
            ttir = compiled.asm["ttir"]
            row.update(status="compiled", precision_lines=[line.strip() for line in ttir.splitlines()
                       if any(pattern in line for pattern in patterns)])
            if sum("tt.store " in line for line in ttir.splitlines()) != 1:
                row.update(status="unexpected_lowering", error="Expected one original masked output store")
        except Exception as error:
            row.update(status="rejected", error_type=type(error).__name__, error=str(error))
        results.append(row)
        print(f"{index + 1}/{len(profiles)} {name}: {row['status']}", flush=True)
        receipt["status_counts"] = dict(Counter(item["status"] for item in results))
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return int(any(row["status"] != "compiled" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
