"""Offline SM80 compiler profiles for all three repaired kernels; no GPU run."""
import argparse
import importlib.util
import hashlib
import json
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
    spec = importlib.util.spec_from_file_location("fast_layernorm_lowering", Path(__file__).with_name("fast_layernorm_repaired.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pointers = {"Y", "X", "W", "b", "r", "mu", "dX", "dY", "dW", "db"}
    profiles = []
    for name in ("layernorm_forward", "layernorm_backward"):
        for block in (1, 128, 2048, 8192, 32768, 65536):
            for integer in ("i32", "u32", "i64") if block == 128 else ("i32",):
                constants = {"BLOCK_SIZE": block}
                if name == "layernorm_forward":
                    constants["HAS_BIAS"] = True
                profiles.append((name, integer, constants, mod.calculate_settings(block)[1], False))
    profiles.append(("layernorm_forward", "i64", {"BLOCK_SIZE": 128, "HAS_BIAS": False}, 4, False))
    for integer in ("i32", "u32", "i64"):
        for bias in (False, True):
            profiles.append(("layernorm_affine_backward", integer, {"BLOCK_SIZE": 128, "HAS_BIAS": bias}, 4, False))
    for name in ("layernorm_forward", "layernorm_backward", "layernorm_affine_backward"):
        fn = getattr(mod, name)
        constants = {arg: True if arg == "HAS_BIAS" else 1 for arg in fn.arg_names if arg not in pointers | {"eps"}}
        profiles.append((name, "i64", constants, 4, True))
    records = []
    for index, (name, integer, constants, warps, singleton) in enumerate(profiles):
        fn = getattr(mod, name)
        sig = {arg: "constexpr" if arg in constants else "*fp32" if arg in pointers
               else "fp32" if arg == "eps" else integer for arg in fn.arg_names}
        compiled = triton.compile(ASTSource(fn, signature=sig, constexprs=constants),
            target=GPUTarget("cuda", 80, 32), options={"num_warps": warps})
        stem = f"{index}_{name}_{integer}"
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (out / f"{stem}.{stage}").write_text(compiled.asm[stage])
        (out / f"{stem}.cubin").write_bytes(compiled.asm["cubin"])
        ttir, ptx = compiled.asm["ttir"], compiled.asm["ptx"]
        if not singleton:
            assert re.search(r"arith\.muli[^\n]*i64", ttir), stem
            if name == "layernorm_affine_backward":
                assert re.search(r"scf\.(?:for|while)[^\n]*i64", ttir), stem
        memory = re.findall(r"(?:ld|st)\.global[^\s;]+", ptx)
        assert memory and all(op in ("ld.global.b32", "st.global.b32") for op in memory), (stem, memory)
        arithmetic = re.findall(r"\b(?:add|sub|mul|div|fma|rsqrt)\.[^\s;]+", ptx)
        assert not any(re.search(r"\.(?:f16|f64|bf16)(?:\.|$)", op) for op in arithmetic), (stem, arithmetic)
        if name == "layernorm_forward":
            assert "rsqrt.approx.ftz.f32" in arithmetic, stem
        files = {str(out / f"{stem}.{stage}"): hashlib.sha256((out / f"{stem}.{stage}").read_bytes()).hexdigest()
                 for stage in ("ttir", "ttgir", "llir", "ptx", "cubin")}
        records.append({"kernel": name, "integer_abi": integer, "constants": constants,
                        "warps": warps, "signature": sig, "arithmetic": arithmetic,
                        "files": files, "compiler_hash": compiled.hash})
        print(f"PASS: {stem}, warps={warps}, constants={constants}", flush=True)
    report = {"triton": triton.__version__, "target": "SM80", "profiles": records,
              "source_sha256": hashlib.sha256(Path(__file__).with_name("fast_layernorm_repaired.py").read_bytes()).hexdigest(),
              "scope": "Fresh offline compiler evidence; no GPU execution or generic decoder instantiation."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Triton {triton.__version__}; {len(profiles)} offline SM80 profiles; no GPU execution.")


if __name__ == "__main__":
    main()
