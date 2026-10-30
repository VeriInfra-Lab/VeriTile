"""Offline SM80 FP32-memory/FP16-compute compilation; no GPU execution."""
from pathlib import Path
import importlib.util
import os
import json
import hashlib

out = Path("/tmp/veritile-mixed-sparse-lowering")
out.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(out / "cache")
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

path = Path(__file__).with_name("mixed_sparse_attention_repaired.py")
spec = importlib.util.spec_from_file_location("mixed_sparse_lowering", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod._triton_mixed_sparse_attn_fwd_kernel
records = []
for dim in [16, 32, 64, 128]:
    for ones in [False, True]:
        tag = f"D{dim}-ones{int(ones)}"
        constants = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_DMODEL": dim, "dtype": tl.float16}
        sig = {n: "i32" for n in fn.arg_names if n not in constants}
        for n in ["Q", "K", "V", "Out"]:
            sig[n] = "*fp32"
        for n in ["seqlens", "block_count", "block_offset", "column_count", "column_index"]:
            sig[n] = "*i32"
        sig["sm_scale"] = "fp32"
        if ones:
            for n in ["stride_qk", "stride_kk", "stride_vk", "stride_ok", "Z", "H"]:
                constants[n] = 1
        compiled = triton.compile(ASTSource(fn, signature=sig, constexprs=constants),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4, "num_stages": 2, "enable_fp_fusion": True})
        ptx = compiled.asm["ptx"]
        assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
        assert "cvt.rn.f16.f32" in ptx and "cvt.f32.f16" in ptx
        assert "ex2.approx.ftz.f32" in ptx and "st.global" in ptx
        (out / (tag + ".ptx")).write_text(ptx)
        (out / (tag + ".ttgir")).write_text(compiled.asm["ttgir"])
        records.append({"tag": tag, "pass": True})
        print(tag, flush=True)
(out / "report.json").write_text(json.dumps({"triton": triton.__version__, "target": "SM80",
    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": records}, indent=2))
