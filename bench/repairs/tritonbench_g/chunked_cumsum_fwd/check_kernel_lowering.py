"""Compile the canonical kernel offline to inspect its runtime clamp and stores."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(out / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = Path(__file__).with_name("chunked_cumsum_fwd_repaired.py")
    spec = importlib.util.spec_from_file_location("chunked_runtime_lowering", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module._chunk_cumsum_fwd_kernel.fn
    profiles = [(block, 4, bias, softplus) for block in (1, 2, 4, 8, 16, 32, 64)
                for bias in (False, True) for softplus in (False, True)]
    profiles += [(block, width, True, True) for block in (1, 2, 4, 8, 16, 32, 64)
                 for width in (1, 64)]
    results = []
    for index, (block, width, bias, softplus) in enumerate(profiles):
        constants = {"BLOCK_SIZE_H": block, "BLOCK_SIZE_CHUNK": width,
                     "HAS_DT_BIAS": bias, "DT_SOFTPLUS": softplus}
        signature = {name: "constexpr" if name in constants else
                     "*fp32" if name.endswith("_ptr") else
                     "fp32" if name in ("dt_min", "dt_max") else "i64"
                     for name in fn.arg_names}
        compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                                  target=GPUTarget("cuda", 80, 32))
        prefix = f"profile_{index:02d}_h{block}_c{width}_bias{int(bias)}_softplus{int(softplus)}"
        artifacts = {}
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            path = out / f"{prefix}.{stage}"
            path.write_text(compiled.asm[stage])
            artifacts[stage] = {"path": str(path), "sha256": sha256(path)}
        ttir, ptx = compiled.asm["ttir"], compiled.asm["ptx"]
        assert re.search(r"%dt_min: f32", ttir) and re.search(r"%dt_max: f32", ttir)
        assert "arith.maxnumf" in ttir and "arith.minnumf" in ttir
        assert re.search(r"\bmax\.f32\b", ptx) and re.search(r"\bmin\.f32\b", ptx)
        assert re.search(r"arith\.muli[^\n]*i64", ttir)
        assert re.search(r"arith\.cmpi ult", ttir)
        assert "tt.scan" in ttir or width == 1
        memory = re.findall(r"(?:ld|st)\.global[^\s;]+", ptx)
        assert memory and all(op in ("ld.global.b32", "st.global.b32") for op in memory), memory
        record = {"block_heads": block, "block_chunk": width, "bias": bias, "softplus": softplus,
                  "integer_abi": "i64", "clamp_abi": ["fp32", "fp32"],
                  "clamp_nan_policy": "PropagateNan.NONE / maxnumf / minnumf",
                  "fma_sites": len(re.findall(r"fma\.[^\s;]+", ptx)), "artifacts": artifacts}
        results.append(record)
        print(f"PASS: {prefix}; runtime fp32 bounds, maxnum/minnum, uint64 masks, FP32 memory.", flush=True)
    driver = Path(triton.__file__).parent / "backends/nvidia/driver.c"
    driver_text = driver.read_text()
    match = re.search(r"bool extractFP32\(void \*ptr, PyObject \*obj\) \{[^}]+\}", driver_text)
    assert match and "PyFloat_AsDouble(obj)" in match[0] and "(float)temp_double" in match[0]
    report = {"source": str(source.resolve()), "source_sha256": sha256(source),
              "triton": triton.__version__, "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
              "profiles": results, "driver_source": str(driver), "driver_sha256": sha256(driver),
              "extract_fp32": match[0],
              "scope": "Offline source compilation only; no GPU execution, scalar conformance, or complete scan-tree binding"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Triton {triton.__version__}; {len(results)} offline SM80 profiles; no GPU execution.")


if __name__ == "__main__":
    main()
