"""Compile original warp choices and repaired edge branches offline for SM80."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime.jit import JITFunction
    path = Path(__file__).with_name("logsumexp_fwd_repaired.py")
    spec = importlib.util.spec_from_file_location("lse_lowering_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.logsumexp_fwd_kernel
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    report = []
    for warps in (1, 2, 4, 8, 16, 32):
        for d, b, scaled in ((1, 1, False), (17, 32, True), (65537, 65536, True)):
            name = f"w{warps}-d{d}-{'scale' if scaled else 'plain'}"
            signature = {"x": "*fp32", "z": "*fp32", "scale": "fp32" if scaled else "constexpr",
                         "D": "constexpr", "B": "constexpr", "HAS_SCALE": "constexpr"}
            constants = {"D": d, "B": b, "HAS_SCALE": scaled}
            if not scaled:
                constants["scale"] = None
            compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                                     target=GPUTarget("cuda", 80, 32), options={"num_warps": warps})
            for stage in ("ttir", "llir", "ptx"):
                (output / f"{name}.{stage}").write_text(compiled.asm[stage])
            ptx = compiled.asm["ptx"]
            memory = sorted(set(re.findall(r"\b(?:ld|st)\.global[^\s]+", ptx)))
            assert memory and all(op.endswith("b32") for op in memory), (name, memory)
            assert "ex2.approx.f32" in ptx or "ex2.approx.ftz.f32" in ptx, name
            assert "__nv_logf" in compiled.asm["llir"] or "lg2.approx" in ptx or "__nv_logf" in ptx, name
            if scaled:
                assert "mul.f32" in ptx and "setp.eq.f32" in ptx, name
            report.append({"case": name, "memory": memory})
            print(f"{name}: SM80 compilation passed")
    (output / "report.json").write_text(json.dumps({"triton": triton.__version__,
        "gpu_execution": False, "cases": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
