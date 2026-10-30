"""Compile the six FP8 source profiles offline; no GPU execution."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
output = parser.parse_args().output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton

source = Path(__file__).with_name("f8_conversion_utils_repaired.py")
spec = importlib.util.spec_from_file_location("f8_native", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = []
for fp8 in ["fp8e5", "fp8e4nv"]:
    for src, dst, kernel in [(fp8,"fp16",module.kernel_f8_to_f16),
                             ("fp16",fp8,module.kernel_f16_to_f8),
                             ("fp32",fp8,module.kernel_f16_to_f8)]:
        compiled = triton.compile(ASTSource(kernel,
            signature={"Y":"*"+dst,"X":"*"+src,"N":"i32","BLOCK_SIZE":"constexpr"},
            constexprs={"BLOCK_SIZE":1024},attrs={}), target=GPUTarget("cuda",90,32))
        name = src+"-to-"+dst
        for stage in ["ttir","ptx"]:
            (output/(name+"."+stage)).write_text(compiled.asm[stage])
        ptx=compiled.asm["ptx"]
        conversions=[line.strip() for line in ptx.splitlines() if "cvt." in line]
        assert ".address_size 64" in ptx and "setp.lt.s32" in ptx
        if dst.startswith("fp8"):
            assert any("satfinite" in line for line in conversions)
        cases.append(dict(source=src,destination=dst,sm=90,conversions=conversions))
report=dict(triton=triton.__version__,gpu_execution=False,numerical_conformance=False,cases=cases)
(output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print({"offline_compilations":len(cases),"sm":90,"gpu_execution":False})
