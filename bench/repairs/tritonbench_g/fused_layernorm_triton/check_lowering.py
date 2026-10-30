"""Offline lowering of both source configurations and all integer ABI routes.
This checks address/index types and instruction storage widths, not GPU numerics.
"""
import argparse
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    source = Path(__file__).with_name("fused_layernorm_triton_repaired.py")
    spec = importlib.util.spec_from_file_location("fused_norm_lowering", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.triton_red_fused_native_layer_norm_0.fn
    reports = []
    for block in (1024, 2048):
        for integer in ("i32", "u32", "i64"):
            signature = {name: ("constexpr" if name in ("XBLOCK", "RBLOCK") else
                integer if name in ("xnumel", "rnumel") else
                "*bf16" if name == "out_ptr1" else "*fp32") for name in function.arg_names}
            compiled = triton.compile(ASTSource(function, signature=signature,
                constexprs={"XBLOCK": 1, "RBLOCK": block}),
                target=GPUTarget("cuda", 80, 32), options={"num_warps": 8, "num_stages": 1})
            stem = f"fused_{block}_{integer}"
            for stage in ("ttir", "ttgir", "llir", "ptx"):
                (output / f"{stem}.{stage}").write_text(compiled.asm[stage])
            ttir, ptx = compiled.asm["ttir"], compiled.asm["ptx"]
            # Explicit loop counters and row/column offsets remain 64-bit.
            assert "iter_args" in ttir and "i64" in ttir
            assert re.search(r"arith\.muli[^\n]*i64", ttir), stem
            assert re.search(r"arith\.addi[^\n]*i64", ttir), stem
            assert "st.global.b16" in ptx and "st.global.b32" in ptx, stem
            assert re.search(r"ld\.global[^\s]*\.b32", ptx) and ".address_size 64" in ptx, stem
            assert "cvt.rn.bf16.f32" in ptx, stem
            memory = re.findall(r"(?:ld|st)\.global[^\s;]+", ptx)
            assert all(((op.startswith("ld.global") and op.endswith(".b32")) or op in ("st.global.b32", "st.global.b16")) for op in memory), memory
            # Bind each observed contraction to its actual source location.
            location, instructions = None, []
            for line in ptx.splitlines():
                match = re.search(r"\.loc\s+(\d+)\s+(\d+)\s+(\d+)", line)
                if match:
                    location = tuple(map(int, match.groups()))
                if re.search(r"(?:fma\.rn|add|mul|sub|div\.full|setp\.eq)\.f32", line):
                    instructions.append({"location": location, "instruction": line.strip()})
            fused = [i for i in instructions if i["instruction"].startswith("fma.rn.f32")]
            helper_sites = {(i["location"][1], i["location"][2]) for i in fused if i["location"][0] == 2}
            # Pin helper source sites rather than assuming every mul/add stays split.
            assert {(295, 22), (307, 17), (308, 22)} <= helper_sites, helper_sites
            assert any(i["location"][0] == 1 and i["location"][1] == 106 for i in fused)
            assert "bar.sync" in ptx and "setp.eq.f32" in ptx
            reports.append({"block": block, "integer_abi": integer, "architecture": "SM80",
                "fma_sites": sorted({tuple(i["location"]) for i in fused}),
                "ordered_scalar_instructions": instructions,
                "files": {stage: {"path": str(output / f"{stem}.{stage}"),
                    "sha256": hashlib.sha256(compiled.asm[stage].encode()).hexdigest()}
                    for stage in ("ttir", "ttgir", "llir", "ptx")}})
            print(f"PASS: RBLOCK={block}, {integer}, SM80; int64 indexing, typed storage, and four source FMA sites")
        signature = {name: ("constexpr" if name in ("XBLOCK", "RBLOCK", "xnumel", "rnumel")
            else "*bf16" if name == "out_ptr1" else "*fp32") for name in function.arg_names}
        compiled = triton.compile(ASTSource(function, signature=signature,
            constexprs={"XBLOCK": 1, "RBLOCK": block, "xnumel": 1, "rnumel": 1}),
            target=GPUTarget("cuda", 80, 32), options={"num_warps": 8, "num_stages": 1})
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (output / f"fused_{block}_scalar_one.{stage}").write_text(compiled.asm[stage])
        print(f"PASS: RBLOCK={block}, scalar-one specialization, SM80")
    helper = Path(module.triton_helpers.__file__)
    report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "helper": str(helper), "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "triton": triton.__version__, "cases": reports, "offline_cases": 8,
        "native_execution": False, "numeric_transformations": [],
        "boundary": "Shape-selected helper tree, genuine FMA contraction, and per-node add/mul operand order; scalar instruction semantics are model boundaries."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Triton {triton.__version__}; eight offline cases; no GPU execution.")


if __name__ == "__main__":
    main()
