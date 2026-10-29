#!/usr/bin/env python3
"""Compile the pinned complete JIT offline, without running its CUDA tests.

These are native source-admission and lowering samples, not correctness or
equivalence proofs. Runtime scalars remain runtime scalars. Only the autotune
and heuristic decorators are removed, and their values are supplied explicitly.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1] / "layer_norm_fwd.py"
SOURCE_SHA256 = "27dfccf23671d5b9c6f4b749cef39ddf69079793c486a833cbae9c57fe3a4de4"
FLAGS = (
    "IS_RMS_NORM", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "HAS_BIAS",
    "HAS_DROPOUT", "STORE_DROPOUT_MASK", "HAS_ROWSCALE", "HAS_X1", "HAS_W1", "HAS_B1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ir-dir", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned source hash mismatch")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/veritile-layer-norm-fwd-native-cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_layer_norm_fwd_1pass_kernel")
    original_body = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
    node.decorator_list = [ast.Attribute(value=ast.Name(id="triton", ctx=ast.Load()),
                                         attr="jit", ctx=ast.Load())]
    assert original_body == ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
    profiles = [
        ("plain", {"HAS_BIAS"}, "fp32", "fp32"),
        ("rms", {"IS_RMS_NORM", "HAS_BIAS"}, "fp32", "fp32"),
        ("dual_dropout", {"HAS_X1", "HAS_W1", "HAS_B1", "HAS_DROPOUT", "STORE_DROPOUT_MASK",
                          "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_BIAS"}, "fp32", "fp32"),
        ("rowscale", {"HAS_ROWSCALE", "HAS_BIAS", "STORE_RESIDUAL_OUT"}, "fp16", "bf16"),
        ("all_layer", set(FLAGS) - {"IS_RMS_NORM"}, "fp32", "fp32"),
        ("all_rms", set(FLAGS), "fp32", "fp32"),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="veritile-layer-norm-source-") as directory:
        module_path = Path(directory) / "layer_norm_source.py"
        module_path.write_text("import triton\nimport triton.language as tl\n\n" + ast.unparse(node) + "\n")
        spec = importlib.util.spec_from_file_location("layer_norm_source", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        fn = module._layer_norm_fwd_1pass_kernel
        for name, enabled, storage, output in profiles:
            constants = {flag: flag in enabled for flag in FLAGS}
            constants["BLOCK_N"] = 128
            signature = {arg: "i32" for arg in fn.arg_names if arg not in constants}
            for arg in fn.arg_names[:15]:
                signature[arg] = "*" + storage
            signature.update(SEEDS="*i64", DROPOUT_MASK="*i1", Mean="*fp32", Rstd="*fp32",
                             Y="*" + output, Y1="*" + output, eps="fp32", dropout_p="fp32")
            row = {"name": name, "constexprs": constants, "signature": signature}
            try:
                compiled = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                          target=GPUTarget("cuda", 90, 32),
                                          options={"num_warps": 4, "num_stages": 1})
                ttir = compiled.asm["ttir"]
                row.update(status="compiled", ttir_sha256=hashlib.sha256(ttir.encode()).hexdigest(),
                           ptx_bytes=len(compiled.asm["ptx"]),
                           int64_seed_loads=sum("tt.load" in line and "i64" in line for line in ttir.splitlines()),
                           seed_narrowing=[line.strip() for line in ttir.splitlines()
                                           if "arith.trunci" in line and "i64 to i32" in line])
                if args.ir_dir:
                    args.ir_dir.mkdir(parents=True, exist_ok=True)
                    (args.ir_dir / (name + ".ttir")).write_text(ttir)
            except Exception as error:
                row.update(status="compile_failure", error_type=type(error).__name__, error=str(error))
            results.append(row)
            print(f"{name}: {row['status']}", flush=True)
    receipt = {
        "source_sha256": SOURCE_SHA256,
        "source_commit": "603e28a",
        "kernel": "_layer_norm_fwd_1pass_kernel",
        "triton_version": importlib.metadata.version("triton"),
        "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
        "scope": "Six source profiles at BLOCK_N=128; body unchanged, offline compilation only. All-flags samples include the JIT-only X1+rowscale combination. No GPU execution, mathematical proof, FP equivalence, exhaustive dtype or constexpr coverage is claimed.",
        "results": results,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return int(any(row["status"] != "compiled" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
