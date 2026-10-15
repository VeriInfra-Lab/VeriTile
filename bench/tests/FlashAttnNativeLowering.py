#!/usr/bin/env python3
"""Check native admission of the unchanged FlashAttention floating family.

Both causal modes use fp16 K/V, fp32 L, fp32 runtime scale, signed32 runtime
dimensions/strides and the wrapper's block sizes. This is an offline compiler
check at one CUDA target, not GPU execution or an IEEE correctness test.
The benchmark's module-level CUDA tests are never imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "tritonbench_g/flash_attn/flash_attn.py"
PROFILES = ("fp16", "bf16", "fp32", "fp64", "fp8e4nv", "fp8e5")
CONSTANTS = {"BLOCK_M": 128, "BLOCK_N": 64, "DIM": 64}


def compile_profile(dtype: str, output: str, causal: bool) -> dict:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    function = next(node for node in ast.parse(SOURCE.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_fwd_kernel")
    with tempfile.TemporaryDirectory(prefix="veritile-flash-source-") as directory:
        path = Path(directory) / "flash_source.py"
        path.write_text("import triton\nimport triton.language as tl\n\n" + ast.unparse(function) + "\n")
        spec = importlib.util.spec_from_file_location("flash_source", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        function = module._fwd_kernel
        constexprs = {**CONSTANTS, "IS_CAUSAL": causal}
        signature = {name: "i32" for name in function.arg_names if name not in constexprs}
        signature.update(Q="*" + dtype, K="*fp16", V="*fp16", L="*fp32",
                         O="*" + output, sm_scale="fp32")
        try:
            result = triton.compile(ASTSource(function, signature, constexprs=constexprs),
                                    target=GPUTarget("cuda", 90, 32),
                                    options={"num_warps": 4, "num_stages": 4})
            ttir = result.asm["ttir"]
            return {"storage": dtype, "output": output, "causal": causal, "status": "compiled",
                    "ptx_bytes": len(result.asm["ptx"]),
                    "ttir_sha256": hashlib.sha256(ttir.encode()).hexdigest(),
                    "fp64_multiply": any("arith.mulf" in line and "f64" in line
                                         for line in ttir.splitlines())}
        except Exception as error:
            return {"storage": dtype, "output": output, "causal": causal, "status": "compile_failure",
                    "error_type": type(error).__name__, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=PROFILES, help=argparse.SUPPRESS)
    parser.add_argument("--output-type", choices=PROFILES, default="fp16", help=argparse.SUPPRESS)
    parser.add_argument("--causal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(compile_profile(args.child, args.output_type, args.causal)))
        return 0
    if args.output is None:
        parser.error("--output is required")
    results = []
    for dtype in PROFILES:
        for output in (["fp16"] if dtype == "fp16" else ["fp16", dtype]):
            for causal in (False, True):
                command = [sys.executable, __file__, "--child", dtype, "--output-type", output]
                if causal:
                    command.append("--causal")
                run = subprocess.run(command, capture_output=True, text=True)
                if run.returncode:
                    raise RuntimeError(f"Probe process failed for {dtype}/{output}/{causal}: {run.stderr}")
                result = json.loads(run.stdout)
                result["expected"] = "compiled" if output == "fp16" else "block_store_dtype_rejection"
                result["matches_expected"] = (
                    result["status"] == "compiled" and
                    result["fp64_multiply"] == (dtype == "fp64")
                    if output == "fp16" else result["status"] == "compile_failure" and
                    "Block element type(" in result["error"] and "mismatch" in result["error"])
                results.append(result)
                print(f"Q={dtype}, O={output}, causal={causal}: {result['status']}", flush=True)
    import triton.language.semantic as semantic
    semantic_path = Path(semantic.__file__)
    receipt = {"source": str(SOURCE.relative_to(SOURCE.parents[3])),
               "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "triton_version": importlib.metadata.version("triton"),
               "semantic_sha256": hashlib.sha256(semantic_path.read_bytes()).hexdigest(),
               "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
               "constexprs": CONSTANTS, "options": {"num_warps": 4, "num_stages": 4},
               "scope": "Six Q labels with O fp16; non-fp16 Q=O rejected at the block store; K/V fp16, L fp32, scale fp32, runtime integers i32; offline compilation only.",
               "results": results}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return int(any(not result["matches_expected"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
