"""Compile the twenty existing FP32 masked_select configurations without a GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
output = parser.parse_args().output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

path = Path(__file__).with_name("masked_select_repaired.py")
spec = importlib.util.spec_from_file_location("masked_select_checked", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
kernel = module.masked_select_kernel.fn
profiles = []
for config in module.cfggen():
    width = config.kwargs["BLOCK_SIZE"]
    compiled = triton.compile(
        ASTSource(kernel,
                  signature=dict(inp_ptr="*fp32", select_mask_ptr="*i1",
                                 prefix_sum_ptr="*i64", out_ptr="*fp32",
                                 n_elements="i32", BLOCK_SIZE="constexpr"),
                  constexprs=dict(BLOCK_SIZE=width), attrs={}),
        target=GPUTarget("cuda", 90, 32), options=dict(num_warps=config.num_warps))
    ptx = compiled.asm["ptx"]
    ttir = compiled.asm["ttir"]
    assert ".address_size 64" in ptx
    assert "ld.global.b64" in ptx or "ld.global.u64" in ptx
    assert "ld.global.b32" in ptx or "ld.global.u32" in ptx
    assert "st.global.b32" in ptx or "st.global.u32" in ptx
    assert "setp.lt.s32" in ptx
    assert "tensor<" in ttir and "xi64>" in ttir
    name = f"block-{width}-warps-{config.num_warps}"
    for stage in ("ttir", "ptx"):
        (output / f"{name}.{stage}").write_text(compiled.asm[stage])
    profiles.append(dict(block_size=width, num_warps=config.num_warps,
                         ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest()))
report = dict(triton=triton.__version__, sm=90, gpu_execution=False,
              input="FP32", mask="Bool", prefix="Int64", output="FP32",
              profiles=profiles)
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(dict(offline_profiles=len(profiles), sm=90, gpu_execution=False))
