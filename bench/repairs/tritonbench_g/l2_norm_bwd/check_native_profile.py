"""Offline compiler-profile checks; neither GPU execution nor FP experiments."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("L2_BWD_PROFILE_OUT", "/tmp/l2-norm-bwd-kernel-profile"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(OUT / "cache"))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

source = HERE / "l2_norm_bwd_repaired.py"
spec = importlib.util.spec_from_file_location("l2_bwd_profile_source", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fn = module._l2_norm_bwd_kernel_repaired
signature = {name: ("*fp32" if name in {"X", "DY", "DX"} else "fp32" if name == "eps" else "i64")
             for name in fn.arg_names if name != "BLOCK_N"}
assert triton.__version__ == "3.7.1", triton.__version__
rows = []
for exponent in range(15):
    block = 2 ** exponent
    kernel = triton.compile(ASTSource(fn, signature=signature, constexprs={"BLOCK_N": block}),
                           target=GPUTarget("cuda", 90, 32),
                           options={"num_warps": 4, "num_ctas": 1, "num_stages": 3,
                                    "enable_fp_fusion": True})
    ptx = kernel.asm["ptx"]
    assert "sqrt.approx.ftz.f32" in ptx
    assert "div.full.f32" in ptx
    assert "fma.rn.f32" in ptx
    assert "neg.f32" in ptx
    assert "ld.global.b32" in ptx and "st.global.b32" in ptx
    assert not re.search(r"(?:ld|st)\.global\.v[24]", ptx)
    assert "rsqrt." not in ptx
    assert re.search(r"setp\.(?:lt|gt|le|ge)\.s64", ptx)
    assert ".param .f32" in ptx
    artifacts = {}
    for kind in ("ttir", "ttgir", "ptx"):
        path = OUT / f"block-{block}.{kind}"
        path.write_text(kernel.asm[kind])
        artifacts[kind] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    counts = {op: ptx.count(op) for op in ("add.f32", "mul.f32", "fma.rn.f32", "sqrt.approx.ftz.f32",
                                         "div.full.f32", "neg.f32", "shfl.sync.bfly.b32")}
    rows.append({"block": block, "artifacts": artifacts, "instruction_counts": counts})
    print(f"block={block}: scalar FP32 memory, signed64 mask, sqrt.approx.ftz/div.full/FMA/neg retained", flush=True)
report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "compiler": triton.__version__, "target": "cuda-sm90", "num_warps": 4,
          "num_ctas": 1, "num_stages": 3, "enable_fp_fusion": True,
          "runtime_specialization": False, "runtime_alignment_specialization": False,
          "gpu_execution": False, "numerical_experiment": False,
          "scope": "Offline profile and instruction checks; compiler binding remains an explicit model boundary.",
          "rows": rows}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
