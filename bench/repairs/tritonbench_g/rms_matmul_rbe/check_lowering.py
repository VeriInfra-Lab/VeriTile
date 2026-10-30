"""Compile source RMS kernels offline for SM80; no GPU execution."""
import argparse
import importlib.util
import os
from pathlib import Path
import re


def check(source, output, single=False):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    spec = importlib.util.spec_from_file_location("rms_lowering", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.rms_matmul_rbe if single else mod.rms_matmul_rbe_qkv
    count = 0
    for fp8 in (False, True):
        for rotary in ((False, True) if single else (True,)):
            for integer, singleton in [(x, False) for x in ("i32", "u32", "i64")] + [("i64", True)]:
                constants = dict(USE_FP8=fp8, THETA=10000., EPS=1e-6, HEAD_DIM=64,
                                 BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)
                if single:
                    constants["RBE_EPILOGUE"] = rotary
                if singleton:
                    constants.update({name: 1 for name in fn.arg_names
                                      if name not in constants and not name.endswith("_ptr")})
                signature = {}
                for name in fn.arg_names:
                    if name in constants:
                        signature[name] = "constexpr"
                    elif not name.endswith("_ptr"):
                        signature[name] = integer
                    elif name == "x_ptr":
                        signature[name] = "*fp32"
                    elif name == "q_ptr":
                        signature[name] = "*fp16"
                    elif name in ("out_ptr", "k_ptr", "v_ptr"):
                        signature[name] = "*fp8e5" if fp8 else "*fp16"
                    else:
                        signature[name] = "*i8" if fp8 else "*fp16"
                code = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                                      target=GPUTarget("cuda", 80, 32),
                                      options={"num_warps": 4, "num_stages": 4})
                stem = f"{'e5' if fp8 else 'half'}_{'rotary' if rotary else 'plain'}_{integer}" + ("_ones" if singleton else "")
                for stage in ("ttir", "ttgir", "llir", "ptx"):
                    (output / f"{stem}.{stage}").write_text(code.asm[stage])
                ptx, ttir = code.asm["ptx"], code.asm["ttir"]
                assert "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32" in ptx
                if not singleton:
                    assert re.search(r"arith\.muli[^\n]*i64", ttir)
                    assert re.search(r"scf\.for[^\n]*: i64", ttir)
                assert "st.global" in ptx and "ld.global" in ptx
                if rotary:
                    assert "fma.rn.f32" in ptx and "math.cos" in ttir and "math.sin" in ttir
                count += 1
                print(f"PASS: {source.stem} {stem}; 16x64x64, four warps, four stages", flush=True)
    print(f"Triton {triton.__version__}; {count} offline SM80 profiles; no GPU execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check(Path(__file__).with_name("rms_matmul_rbe_repaired.py"), args.output_dir.resolve())


if __name__ == "__main__":
    main()
