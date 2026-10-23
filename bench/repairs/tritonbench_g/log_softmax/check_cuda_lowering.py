"""Offline SM80 compilation of both kernels, every BM/warp branch and scalar one."""
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
    path = Path(__file__).with_name("log_softmax_repaired.py")
    spec = importlib.util.spec_from_file_location("log_softmax_lowering_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = []
    for direction in ("forward", "backward"):
        fn = module.log_softmax_kernel_repaired if direction == "forward" else module.log_softmax_backward_kernel_repaired
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        for bm in (1, 2, 4, 8):
            for n in (1, 17, 1025, 2049):
                pointers = ("output_ptr", "input_ptr") if direction == "forward" else ("out_ptr", "out_grad_ptr", "in_grad_ptr")
                signature = dict.fromkeys(pointers, "*fp32")
                signature.update(M="i32", N="i32", K="i32", BLOCK_M="constexpr", BLOCK_N="constexpr")
                constants = {"BLOCK_M": bm, "BLOCK_N": triton.next_power_of_2(n)}
                if n == 1:
                    signature.update(M="constexpr", N="constexpr", K="constexpr")
                    constants.update(M=1, N=1, K=1)
                warps = module.heur_num_warps({"N": n})
                name = f"{direction}-m{bm}-n{n}-w{warps}"
                compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                    target=GPUTarget("cuda", 80, 32), options={"num_warps": warps, "enable_fp_fusion": True})
                for stage in ("ttir", "llir", "ptx"):
                    (output / f"{name}.{stage}").write_text(compiled.asm[stage])
                ptx = compiled.asm["ptx"]
                memory = sorted(set(re.findall(r"\b(?:ld|st)\.global[^\s]+", ptx)))
                assert memory and all(op.endswith("b32") for op in memory), (name, memory)
                assert "ex2.approx" in ptx, name
                if direction == "backward":
                    assert "fma.rn.f32" in ptx, name
                if n > 1:
                    assert "i64" in compiled.asm["ttir"] and re.search(r"setp\.(?:lt|ge)\.s(?:32|64)", ptx), name
                report.append({"case": name, "memory": memory, "fma": "fma.rn.f32" in ptx})
                print(f"{name}: SM80 compilation passed")
    (output / "report.json").write_text(json.dumps({"triton": triton.__version__,
        "gpu_execution": False, "cases": report}, indent=2) + "\n")


if __name__ == "__main__":
    main()
