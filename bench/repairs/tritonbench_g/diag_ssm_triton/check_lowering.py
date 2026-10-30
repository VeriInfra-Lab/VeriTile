"""Inspect current task PTX without GPU execution, using the existing offline method.

This checks the documented scalar operation profile and scalar masked accesses.
It does not prove compiler correctness or validate floating instruction results.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import re

EXPECTED = {
    "diag_ssm_forward_kernel": ["fma.rn.f32"],
    "diag_ssm_backward_kernel": ["add.f32", "fma.rn.f32", "mul.f32"],
    "diag_ssm_forward_kernel_complex": [
        "mul.f32", "neg.f32", "fma.rn.f32", "add.f32",
        "mul.f32", "fma.rn.f32", "add.f32"],
    "diag_ssm_backward_kernel_complex": [
        "add.f32", "add.f32", "mul.f32", "neg.f32", "fma.rn.f32", "add.f32",
        "mul.f32", "fma.rn.f32", "add.f32", "mul.f32", "neg.f32", "fma.rn.f32",
        "mul.f32", "fma.rn.f32", "sub.f32", "sub.f32", "sub.f32", "sub.f32"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = Path(__file__).resolve().parent / "diag_ssm_triton_repaired.py"
    spec = importlib.util.spec_from_file_location("diag_ssm_lowering", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, expected in EXPECTED.items():
        function = getattr(module, name)
        for integer in ("i32", "u32", "i64"):
            signature = {arg: ("*fp32" if arg.endswith("_ptr") else
                "constexpr" if arg == "BLOCK_SIZE" else integer) for arg in function.arg_names}
            kernel = triton.compile(ASTSource(function, signature=signature,
                constexprs={"BLOCK_SIZE": 128}), target=GPUTarget("cuda", 80, 32),
                options={"num_warps": 4, "num_stages": 3})
            for stage in ("ttir", "ttgir", "llir", "ptx"):
                (output / f"{name}_{integer}.{stage}").write_text(kernel.asm[stage])
            ptx = kernel.asm["ptx"]
            operations = re.findall(r"^\s*((?:add|sub|mul|fma|neg)\.[^\s]*f32)\s+", ptx, re.M)
            assert operations == expected, (name, integer, operations)
            memory = re.findall(r"^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)", ptx, re.M)
            assert memory and all(predicate and instruction in ("ld.global.b32", "st.global.b32")
                for predicate, instruction in memory), memory
            assert ".address_size 64" in ptx
            print(f"PASS: {name}, {integer}, SM80: scalar graph and predicated b32 accesses")
        # Triton may specialize scalar arguments equal to one as constexprs.
        # This route must accept the same explicit int64 casts as dynamic sizes.
        signature = {arg: ("*fp32" if arg.endswith("_ptr") else "constexpr")
                     for arg in function.arg_names}
        constants = {arg: (128 if arg == "BLOCK_SIZE" else 1)
                     for arg in function.arg_names if signature[arg] == "constexpr"}
        specialized = triton.compile(ASTSource(function, signature=signature,
            constexprs=constants), target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4, "num_stages": 3})
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (output / f"{name}_scalar_one.{stage}").write_text(specialized.asm[stage])
        print(f"PASS: {name}, scalar-one specialization, SM80")
    print(f"Triton {triton.__version__}; 16 offline cases; no GPU execution.")


if __name__ == "__main__":
    main()
