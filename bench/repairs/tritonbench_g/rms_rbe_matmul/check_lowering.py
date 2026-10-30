"""Compile both single-projection rotary branches offline for SM80."""
import argparse
import importlib.util
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("rms_profiles", here.parent / "rms_matmul_rbe/check_lowering.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.check(here / "rms_rbe_matmul_repaired.py", args.output_dir.resolve(), single=True)
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source_spec = importlib.util.spec_from_file_location("single_raw_rotary", here / "rms_rbe_matmul_repaired.py")
    source = importlib.util.module_from_spec(source_spec)
    source_spec.loader.exec_module(source)
    fn = source.rbe_triton
    output = args.output_dir.resolve()
    for integer, singleton in [(x, False) for x in ("i32", "u32", "i64")] + [("i64", True)]:
        constants = dict(THETA=10000., BLOCK_SIZE_M=2, BLOCK_SIZE_K=1024)
        if singleton:
            constants.update({name: 1 for name in fn.arg_names if name not in constants and not name.endswith("_ptr")})
        signature = {name: "constexpr" if name in constants else "*fp32" if name.endswith("_ptr") else integer
                     for name in fn.arg_names}
        code = triton.compile(ASTSource(fn, signature=signature, constexprs=constants), target=GPUTarget("cuda", 80, 32))
        stem = "raw_rotary_" + integer + ("_ones" if singleton else "")
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (output / f"{stem}.{stage}").write_text(code.asm[stage])
        assert "ld.global" in code.asm["ptx"] and "st.global" in code.asm["ptx"]
        assert "fma.rn.f32" in code.asm["ptx"]
        print(f"PASS: in-file {stem}; 2x1024, default settings", flush=True)
    print("PASS: four additional in-file raw rotary helper profiles; no GPU execution.")


if __name__ == "__main__":
    main()
