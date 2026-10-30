"""Offline SM80 compilation of all four actual FP32 kernels and autotune configurations."""
import argparse
import importlib.util
import os
from pathlib import Path


def check(output):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source = Path(__file__).with_name("chunk_retention_ops_repaired.py")
    spec = importlib.util.spec_from_file_location("retention_ops_lowering", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ["chunk_retention_fwd_kernel_h", "chunk_retention_fwd_kernel_o",
             "chunk_retention_bwd_kernel_dh", "chunk_retention_bwd_kernel_dqkv"]
    count = 0
    for K, V in [(1, 1), (17, 65), (65, 17)]:
        H, T, BT = 2, 65, 64
        BK, BV = max(16, min(64, triton.next_power_of_2(K))), max(16, min(64, triton.next_power_of_2(V)))
        NT = triton.cdiv(T, BT)
        for name in names:
            fn = getattr(module, name).fn
            for warps in [1, 2, 4]:
                constants = {"H": H, "T": T, "K": K, "V": V, "BT": BT, "BK": BK, "BV": BV}
                if "NT" in fn.arg_names: constants["NT"] = NT
                for flag in ["USE_INITIAL_STATE", "STORE_FINAL_STATE", "USE_FINAL_GRAD", "STORE_INITIAL_GRAD"]:
                    if flag in fn.arg_names: constants[flag] = True
                for stride in ["s_qk_d", "s_vo_d"]: constants[stride] = 1
                signature = {}
                for arg in fn.arg_names:
                    signature[arg] = ("constexpr" if arg in constants else "fp32" if arg == "scale"
                                      else "i32" if arg.startswith("s_") else "*fp32")
                code = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                    target=GPUTarget("cuda", 80, 32), options={"num_warps": warps})
                stem = f"{name}_k{K}_v{V}_w{warps}"
                for stage in ["ttir", "ttgir", "llir", "ptx"]:
                    (output / f"{stem}.{stage}").write_text(code.asm[stage])
                assert "mma.sync" not in code.asm["ptx"]
                assert "fma.rn.f32" in code.asm["ptx"]
                assert "ld.global" in code.asm["ptx"] and "st.global" in code.asm["ptx"]
                count += 1
                print(f"PASS: {stem}; BT64/BK{BK}/BV{BV}, scalar FP32 FMA, TF32 disabled", flush=True)
    print(f"Triton {triton.__version__}; {count} SM80 profiles; no GPU execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    check(parser.parse_args().output_dir.resolve())
