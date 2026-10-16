#!/usr/bin/env python3
"""Compile the original uniform JIT offline; never launch a CUDA kernel.

The extracted module retains the original try/except helper selection, launch
heuristics, and complete decorated kernel AST. The CUDA host wrapper/tests are
not executed. An additional, separately labelled module probes the original
fallback helper body; it does not impersonate the installed-library selection.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "uniform_sampling.py"
SOURCE_SHA256 = "d2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ast_digest(node: ast.AST) -> str:
    return digest(ast.dump(node, include_attributes=False).encode())


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_jit(fn):
    from triton.runtime.jit import JITFunction
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    return fn


def helper_record(helper) -> dict:
    pyfn = helper.fn
    path = Path(inspect.getsourcefile(pyfn))
    return {
        "python_module": pyfn.__module__,
        "python_name": pyfn.__qualname__,
        "source_path": str(path),
        "source_file_sha256": digest(path.read_bytes()),
        "jit_source": helper.src,
        "jit_source_sha256": digest(helper.src.encode()),
    }


def evidence(ttir: str, ptx: str) -> dict:
    lines = ttir.splitlines()
    patterns = ("arith.addi", "arith.andi", "arith.shrsi", "arith.shrui",
                "arith.trunci", "arith.extsi", "arith.extui", "arith.cmpi",
                "arith.sitofp", "arith.uitofp", "arith.mulf", "arith.addf",
                "arith.subf", "arith.truncf", "arith.extf", "tt.store",
                "tt.get_program_id", "tt.make_range")
    return {
        "store_count": sum("tt.store " in line for line in lines),
        "entry_signature": next((line.strip() for line in lines if "tt.func public" in line), None),
        "integer_and_precision_lines": [line.strip() for line in lines
                                        if any(p in line for p in patterns)],
        "ptx_floating_instructions": [line.strip() for line in ptx.splitlines()
                                      if any(p in line for p in ("fma.", "mul.f", "add.f", "sub.f",
                                                                 "mul.rn.f", "add.rn.f", "cvt.rn.f"))],
        "ir_note": "TTIR integers are signless. Source u32 promotion is separately recorded from the actual compiler semantic function; a signless i32 add alone does not prove unsigned source provenance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    if digest(raw) != SOURCE_SHA256:
        raise RuntimeError("Pinned uniform_sampling source hash mismatch")
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.artifacts / "cache"))

    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.language.semantic import TritonSemantic

    tree = ast.parse(raw)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    kernel = functions["uniform_kernel"]
    helper_try = next(n for n in tree.body if isinstance(n, ast.Try))
    fallback = next(n for n in helper_try.handlers[0].body if isinstance(n, ast.FunctionDef))
    imports = [ast.Import(names=[ast.alias(name="triton")]),
               ast.Import(names=[ast.alias(name="triton.language", asname="tl")])]
    suffix = [functions["heur_block"], functions["heur_num_warps"], kernel]
    modules = {}
    for label, helper_node in (("selected", helper_try), ("fallback", fallback)):
        module_tree = ast.Module(body=imports + [helper_node] + suffix, type_ignores=[])
        path = args.artifacts / f"uniform_source_{label}.py"
        path.write_text(ast.unparse(module_tree) + "\n")
        reparsed = ast.parse(path.read_text())
        emitted_kernel = next(n for n in reparsed.body if isinstance(n, ast.FunctionDef)
                              and n.name == "uniform_kernel")
        if ast_digest(emitted_kernel) != ast_digest(kernel):
            raise RuntimeError("Kernel AST changed during extraction")
        if label == "selected":
            emitted_helper = next(n for n in reparsed.body if isinstance(n, ast.Try))
        else:
            emitted_helper = next(n for n in reparsed.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "uint_to_uniform_float")
        if ast_digest(emitted_helper) != ast_digest(helper_node):
            raise RuntimeError("Helper AST changed during extraction")
        modules[label] = load_module(path, f"veritile_uniform_source_{label}")

    # This method is pure dtype promotion; it does not need a GPU or IR builder.
    semantic = object.__new__(TritonSemantic)
    promotion = semantic.integer_promote_impl(tl.uint32, tl.int32)
    if promotion != tl.uint32:
        raise RuntimeError(f"Unexpected uint32 + int32 source promotion: {promotion}")
    semantic_path = Path(inspect.getsourcefile(TritonSemantic))
    results = []
    for helper_mode, module in modules.items():
        # The primary matrix covers every source wrapper block/warp choice.
        # Fallback compatibility uses one representative original launch profile.
        launches = [(512, 4), (1024, 8), (1024, 16)] if helper_mode == "selected" else [(512, 4)]
        for storage in ("fp16", "bf16", "fp32", "fp64", "fp8e4nv", "fp8e5"):
            for block, warps in launches:
                name = f"{helper_mode}_{storage}_b{block}_w{warps}"
                signature = {"out_ptr": "*" + storage, "N": "i32", "philox_seed": "i64",
                             "philox_offset": "i64", "from_": "fp32", "to": "fp32"}
                constants = {"BLOCK": block}
                options = {"num_warps": warps, "num_stages": 1, "enable_fp_fusion": True}
                row = {"name": name, "helper_mode": helper_mode, "signature": signature,
                       "constexprs": constants, "options": options}
                try:
                    compiled = triton.compile(ASTSource(unwrap_jit(module.uniform_kernel), signature,
                                                       constexprs=constants),
                                              target=GPUTarget("cuda", 90, 32), options=options)
                    for suffix_name in ("ttir", "ttgir", "llir", "ptx"):
                        content = compiled.asm[suffix_name]
                        path = args.artifacts / f"{name}.{suffix_name}"
                        path.write_text(content)
                        row[suffix_name + "_sha256"] = digest(content.encode())
                    row.update(status="compiled", evidence=evidence(compiled.asm["ttir"], compiled.asm["ptx"]))
                    if row["evidence"]["store_count"] != 4:
                        row.update(status="unexpected_lowering", error="Expected all four original masked stores")
                except Exception as error:
                    row.update(status="compile_failure", error_type=type(error).__name__, error=str(error))
                results.append(row)
                print(f"{name}: {row['status']}", flush=True)

    receipt = {
        "source_commit": "603e28a", "source_sha256": SOURCE_SHA256,
        "kernel_ast_sha256": ast_digest(kernel), "try_helper_ast_sha256": ast_digest(helper_try),
        "fallback_helper_ast_sha256": ast_digest(fallback),
        "checker_sha256": digest(Path(__file__).read_bytes()),
        "triton_version": importlib.metadata.version("triton"),
        "target": {"backend": "cuda", "arch": 90, "warp_size": 32},
        "kernel_launched": False,
        "helper_selection": {"selected_is_installed_tl_helper":
                                 modules["selected"].uint_to_uniform_float is getattr(tl, "uint_to_uniform_float", None),
                             "selected": helper_record(modules["selected"].uint_to_uniform_float),
                             "fallback": helper_record(modules["fallback"].uint_to_uniform_float)},
        "source_promotion": {"u32_plus_i32": str(promotion),
                             "semantic_file": str(semantic_path),
                             "semantic_file_sha256": digest(semantic_path.read_bytes()),
                             "integer_promotion_function": inspect.getsource(TritonSemantic.integer_promote_impl),
                             "binary_type_function": inspect.getsource(TritonSemantic.binary_op_type_checking_impl)},
        "scope": "Offline unchanged decorated original JIT: all 3 wrapper B/warp choices and 6 floating destination types, runtime N:i32/seed:i64/offset:i64/from:fp32/to:fp32. Separately labelled unchanged fallback helper tested at B512/w4. No GPU execution, exact host ABI inference, exhaustive scalar/dtype profile admission, Lean source correspondence, mathematical correctness or hardware conformance is claimed. FP fusion is explicitly enabled and its resulting instructions retained.",
        "results": results,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return int(any(r["status"] != "compiled" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
