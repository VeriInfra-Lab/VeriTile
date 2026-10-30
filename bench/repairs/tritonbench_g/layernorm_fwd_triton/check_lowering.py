"""Offline source compilation, including empty and constexpr scalar inputs."""
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
    spec = importlib.util.spec_from_file_location("layernorm_lowering", Path(__file__).with_name("layernorm_fwd_triton_repaired.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod._layer_norm_fwd_kernel
    profiles = [(kind, {"BLOCK_SIZE": 128}) for kind in ("i32", "u32", "i64")]
    profiles += [("fp32", {name: 128 if name == "BLOCK_SIZE" else value for name in fn.arg_names
                           if name not in ("X", "W", "Y", "eps")}) for value in (0, 1)]
    for index, (kind, constants) in enumerate(profiles):
        sig = {name: "constexpr" if name in constants else "*fp32" if name in ("X", "W", "Y")
               else "fp32" if name == "eps" else kind for name in fn.arg_names}
        result = triton.compile(ASTSource(fn, signature=sig, constexprs=constants), target=GPUTarget("cuda", 80, 32))
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (out / f"profile_{index}.{stage}").write_text(result.asm[stage])
        ttir, ptx = result.asm["ttir"], result.asm["ptx"]
        if index < 3:
            assert re.search(r"arith\.muli[^\n]*i64", ttir)
            assert re.search(r"scf\.for[^\n]*i64", ttir)
        memory = re.findall(r"(?:ld|st)\.global[^\s;]+", ptx)
        assert all(op in ("ld.global.b32", "st.global.b32") for op in memory), memory
        if index != 3:
            assert memory
        print(f"PASS: profile {index}, integer ABI {kind}, constants {constants}")
    print(f"Triton {triton.__version__}; {len(profiles)} offline SM80 profiles; no GPU execution.")


if __name__ == "__main__":
    main()
