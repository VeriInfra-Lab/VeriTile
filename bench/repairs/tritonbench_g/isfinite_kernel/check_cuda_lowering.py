"""Compile both isfinite precision branches and stride ABIs; no GPU execution."""
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

source = Path(__file__).with_name("isfinite_kernel_repaired.py")
spec = importlib.util.spec_from_file_location("isfinite_native", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = []
for kernel_name in ["isfinite_strided_chunk", "isfinite_strided_chunk_index_commuted"]:
    kernel = getattr(module, kernel_name)
    for precision in ["fp32", "fp64"]:
        for stride_abi in ["i32", "u32", "i64", "constexpr"]:
            signature = dict(in0_ptr="*"+precision, out0_ptr="*i1", in0_stride=stride_abi,
                             out0_stride=stride_abi, n="i32", BLOCK_SIZE="constexpr")
            constants = dict(BLOCK_SIZE=1024)
            if stride_abi == "constexpr":
                constants.update(in0_stride=1, out0_stride=1)
            compiled = triton.compile(ASTSource(kernel,
                signature=signature, constexprs=constants, attrs={}), target=GPUTarget("cuda",90,32))
            name = kernel_name+"-"+precision+"-"+stride_abi
            for stage in ["ttir", "ptx"]:
                (output/(name+"."+stage)).write_text(compiled.asm[stage])
            ptx = compiled.asm["ptx"]
            instructions = [line.strip() for line in ptx.splitlines()
                            if any(op in line for op in ["testp.", "cvt.", "mul.", "mad.", "setp."])]
            assert ".address_size 64" in ptx
            if precision == "fp32":
                assert "setp.ne.f32" in ptx and "0f7F800000" in ptx
            else:
                assert "setp.ne.b32" in ptx and "2146435072" in ptx
            assert "setp.lt.s32" in ptx
            assert "cvt.rn.f32.f64" not in ptx
            cases.append(dict(kernel=kernel_name, precision=precision, stride_abi=stride_abi, instructions=instructions))
report = dict(triton=triton.__version__, sm=90, gpu_execution=False,
              numerical_conformance=False, cases=cases)
(output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
print(dict(offline_compilations=len(cases), sm=90, gpu_execution=False))
