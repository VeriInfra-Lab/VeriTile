"""Compile source profiles offline; no GPU or numerical-conformance claim."""
import argparse
import importlib.util
import os
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    out = p.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(out / "cache")
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source = Path(__file__).with_name("layer_norm_liger_repaired.py")
    spec = importlib.util.spec_from_file_location("liger_lowering", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    count = 0
    for direction in ("forward", "backward"):
        fn = getattr(mod, f"_layer_norm_{direction}_kernel")
        for block in (1, 128, 2048, 8192, 32768, 65536):
            # Source backward does not pass num_warps, and retains Triton's default 4.
            warps = mod.calculate_settings(block)[1] if direction == "forward" else 4
            for integer in ("i32", "u32", "i64") if block == 128 else ("i32",):
                constants = {"BLOCK_SIZE": block}
                if direction == "backward":
                    constants.update(rows_per_program=3, dtype=tl.float32)
                sig = {name: "constexpr" if name in constants else "*fp32" if name.endswith("_ptr")
                       else "fp32" if name == "eps" else integer for name in fn.arg_names}
                compiled = triton.compile(ASTSource(fn, signature=sig, constexprs=constants),
                    target=GPUTarget("cuda", 80, 32), options={"num_warps": warps})
                stem = f"{direction}_{block}_{integer}"
                for stage in ("ttir", "ttgir", "llir", "ptx"):
                    (out / f"{stem}.{stage}").write_text(compiled.asm[stage])
                ttir, ptx = compiled.asm["ttir"], compiled.asm["ptx"]
                assert re.search(r"arith\.muli[^\n]*i64", ttir), stem
                if direction == "backward":
                    assert re.search(r"scf\.for[^\n]*i64", ttir), stem
                memory = re.findall(r"(?:ld|st)\.global[^\s;]+", ptx)
                assert memory and all(op in ("ld.global.b32", "st.global.b32") for op in memory), memory
                print(f"PASS: {stem}, num_warps={warps}, int64 indices and scalar FP32 storage")
                count += 1
        constants = {name: (tl.float32 if name == "dtype" else 1) for name in fn.arg_names
                     if not name.endswith("_ptr") and name != "eps"}
        sig = {name: "constexpr" if name in constants else "*fp32" if name.endswith("_ptr") else "fp32"
               for name in fn.arg_names}
        compiled = triton.compile(ASTSource(fn, signature=sig, constexprs=constants),
            target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
        (out / f"{direction}_scalar_one.ptx").write_text(compiled.asm["ptx"])
        print(f"PASS: {direction}, all integer scalars specialized to one")
        count += 1
    print(f"Triton {triton.__version__}; {count} offline SM80 profiles; no GPU execution.")


if __name__ == "__main__":
    main()
