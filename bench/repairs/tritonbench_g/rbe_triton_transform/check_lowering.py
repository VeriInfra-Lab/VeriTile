"""Offline SM80 compiler profiles for the unchanged 2x1024 source tile."""
import argparse
import importlib.util
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    out = parser.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(out / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    spec = importlib.util.spec_from_file_location("rbe_lowering", Path(__file__).with_name("rbe_triton_transform_repaired.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.rbe_triton
    profiles = [(integer, False) for integer in ("i32", "u32", "i64")] + [("i64", True)]
    for integer, singleton in profiles:
        constants = {"THETA": 10000., "BLOCK_SIZE_M": 2, "BLOCK_SIZE_K": 1024}
        if singleton:
            constants.update({name: 1 for name in fn.arg_names if name not in constants and name not in ("x_ptr", "out_ptr")})
        sig = {name: "constexpr" if name in constants else "*fp32" if name in ("x_ptr", "out_ptr") else integer
               for name in fn.arg_names}
        code = triton.compile(ASTSource(fn, signature=sig, constexprs=constants), target=GPUTarget("cuda", 80, 32))
        stem = integer + ("_all_one" if singleton else "")
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (out / f"{stem}.{stage}").write_text(code.asm[stage])
        if not singleton:
            assert re.search(r"arith\.muli[^\n]*i64", code.asm["ttir"])
        memory = re.findall(r"(?:ld|st)\.global[^\s;]+", code.asm["ptx"])
        assert memory and all(op in ("ld.global.b32", "ld.global.nc.b32", "st.global.b32") for op in memory), memory
        assert "fma.rn.f32" in code.asm["ptx"]
        print(f"PASS: {stem}; source 2x1024 tile, default settings", flush=True)
    print(f"Triton {triton.__version__}; four offline SM80 profiles; no GPU execution.")


if __name__ == "__main__":
    main()
