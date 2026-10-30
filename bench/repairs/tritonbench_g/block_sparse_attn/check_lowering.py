"""Selected B16 source compilation; this performs no GPU execution."""
from pathlib import Path
import importlib.util
import os
import json
import hashlib

out = Path("/tmp/veritile-block-sparse-lowering")
out.mkdir(exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(out / "cache")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

path = Path(__file__).with_name("block_sparse_attn_repaired.py")
spec = importlib.util.spec_from_file_location("block_sparse_lowering", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.block_sparse_attention_kernel
records = []
for blocks in [1, 2, 3, 4]:
    for even in [False, True]:
        tag = f"blocks{blocks}-even{int(even)}"
        constants = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_D": 16,
                     "NUM_D_BLOCKS": blocks, "EVEN_M": even, "EVEN_N": even}
        sig = {n: "i32" for n in fn.arg_names if n not in constants}
        for n in ["out", "Q", "K", "V"]:
            sig[n] = "*fp32"
        for n in ["layout_csr_row_indices", "layout_csr_col_indices"]:
            sig[n] = "*i32"
        sig["softmax_scale"] = "fp32"
        compiled = triton.compile(ASTSource(fn, signature=sig, constexprs=constants),
            target=GPUTarget("cuda", 90, 32),
            options={"num_warps": 4, "num_stages": 3, "enable_fp_fusion": True})
        ptx = compiled.asm["ptx"]
        assert "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32" in ptx
        assert "wgmma" not in ptx
        assert "ex2.approx" in ptx and "fma.rn.f32" in ptx and "st.global" in ptx
        (out / (tag + ".ptx")).write_text(ptx)
        (out / (tag + ".ttgir")).write_text(compiled.asm["ttgir"])
        records.append({"tag": tag, "pass": True})
        print(tag, flush=True)
(out / "report.json").write_text(json.dumps({"triton": triton.__version__, "target": "SM90",
    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": records}, indent=2))
