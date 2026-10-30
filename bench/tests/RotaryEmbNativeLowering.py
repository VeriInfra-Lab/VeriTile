#!/usr/bin/env python3
"""Probe the unchanged rotary source with Triton's offline CUDA compiler.

This records six homogeneous-storage samples at one concrete shape. It does
not run a GPU, validate mixed-storage profiles, or certify numerical results.
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


SOURCE = Path(__file__).resolve().parents[1] / "tritonbench_g/rotary_emb/rotary_emb.py"
PROFILES = ("fp32", "fp64", "fp16", "bf16", "fp8e4nv", "fp8e5")
CONSTANTS = {"BLOCK_HEAD": 4, "BLOCK_SEQ": 16, "BLOCK_DMODEL": 64}


def compile_profile(dtype: str) -> dict:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    function = next(
        node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_rotary_kernel"
    )
    with tempfile.TemporaryDirectory(prefix="veritile-rotary-source-") as directory:
        path = Path(directory) / "rotary_source.py"
        path.write_text("import triton\nimport triton.language as tl\n\n" + ast.unparse(function) + "\n")
        spec = importlib.util.spec_from_file_location("rotary_source", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        function = module._rotary_kernel
        signature = {
            name: "*" + dtype if index < 4 else "i32"
            for index, name in enumerate(function.arg_names[:17])
        }
        try:
            result = triton.compile(
                ASTSource(function, signature, constexprs=CONSTANTS),
                target=GPUTarget("cuda", 90, 32),
                options={"num_warps": 4, "num_stages": 1},
            )
            return {"storage": dtype, "status": "compiled", "ptx_bytes": len(result.asm["ptx"])}
        except Exception as error:
            return {"storage": dtype, "status": "compile_failure",
                    "error_type": type(error).__name__, "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=PROFILES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(compile_profile(args.child)))
        return 0
    if args.output is None:
        parser.error("--output is required")
    results = []
    for dtype in PROFILES:
        run = subprocess.run([sys.executable, __file__, "--child", dtype], capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"Probe process failed for {dtype}: {run.stderr}")
        result = json.loads(run.stdout)
        if result["status"] != "compiled":
            result["diagnostic_excerpt"] = "\n".join(run.stderr.splitlines()[:3])
        results.append(result)
    receipt = {
        "source": str(SOURCE.relative_to(SOURCE.parents[3])),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "triton_version": importlib.metadata.version("triton"),
        "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
        "constexprs": CONSTANTS,
        "options": {"num_warps": 4, "num_stages": 1},
        "runtime_scalar_signature": "i32",
        "scope": "Homogeneous Q/K/Cos/Sin storage only; compile only; no GPU execution or numerical checks.",
        "results": results,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    for result in results:
        print(f"{result['storage']}: {result['status']}")
    return int(any(result["status"] != "compiled" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
