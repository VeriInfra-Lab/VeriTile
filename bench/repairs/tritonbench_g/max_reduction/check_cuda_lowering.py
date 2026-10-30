"""Compile the selected max source offline, including scalar-one specialization."""
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
    path = Path(__file__).with_name("max_reduction_repaired.py")
    spec = importlib.util.spec_from_file_location("max_reduction_lowering_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dim = module.max_kernel
    while not isinstance(dim, JITFunction):
        dim = dim.fn
    cases = []
    configs = [
        ("stage1-one", module.max_kernel_1,
         {"inp": "*fp32", "mid": "*fp32", "M": "constexpr", "BLOCK_SIZE": "constexpr"},
         {"M": 1, "BLOCK_SIZE": 1}),
        ("stage1-tail", module.max_kernel_1,
         {"inp": "*fp32", "mid": "*fp32", "M": "i32", "BLOCK_SIZE": "constexpr"},
         {"BLOCK_SIZE": 256}),
        ("stage2-one", module.max_kernel_2,
         {"mid": "*fp32", "out": "*fp32", "mid_size": "constexpr", "BLOCK_MID": "constexpr"},
         {"mid_size": 1, "BLOCK_MID": 1}),
        ("stage2-tail", module.max_kernel_2,
         {"mid": "*fp32", "out": "*fp32", "mid_size": "i32", "BLOCK_MID": "constexpr"},
         {"BLOCK_MID": 256}),
    ]
    for block_m in (8, 16, 32):
        for scalar_one in (False, True):
            signature = {"inp": "*fp32", "out_value": "*fp32", "out_index": "*i64",
                         "M": "i32", "N": "i32", "K": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"}
            constants = {"BLOCK_M": block_m, "BLOCK_N": 1 if scalar_one else 64}
            if scalar_one:
                signature.update(M="constexpr", N="constexpr", K="constexpr")
                constants.update(M=1, N=1, K=1)
            configs.append((f"dimension-{block_m}-{'one' if scalar_one else 'tail'}", dim, signature, constants))
    for name, fn, signature, constants in configs:
        compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                                 target=GPUTarget("cuda", 80, 32), options={"num_warps": 8})
        for stage in ("ttir", "llir", "ptx"):
            (output / f"{name}.{stage}").write_text(compiled.asm[stage])
        ptx = compiled.asm["ptx"]
        memory = sorted(set(re.findall(r"\b(?:ld|st)\.global[^\s]+", ptx)))
        assert memory, name
        if name.startswith("dimension"):
            assert any(op.startswith("st") and op.endswith("b64") for op in memory), (name, memory)
            if name.endswith("tail"):
                assert "setp.gt.f32" in ptx and "setp.eq.f32" in ptx, name
        else:
            assert all(op.endswith("b32") for op in memory), (name, memory)
        cases.append({"name": name, "constants": constants, "memory": memory})
        print(f"{name}: SM80 compilation passed")
    (output / "report.json").write_text(json.dumps({"triton": triton.__version__,
        "gpu_execution": False, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
