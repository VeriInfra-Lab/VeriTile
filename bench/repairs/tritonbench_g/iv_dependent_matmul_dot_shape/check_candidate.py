"""Explicit dot-shape repair; actual metadata guards and offline compiler checks.

The original selected Python is imported only for its JIT function. No tensor
allocation, floating-point execution or GPU launch occurs in this audit.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parent / "iv_dependent_matmul_dot_shape.py"
SOURCE_SHA = "9256084fd67ed7b828363cf6b36b9e1ee4d03b4888c5026110faddc3df0ab57b"
MODES = ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters",
         "post_load_three_iters"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_guard():
    tree = ast.parse(SOURCE.read_text())
    fn = copy.deepcopy(next(node for node in tree.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == "iv_dependent_matmul_wrapper_i64"))
    stop = next(index for index, node in enumerate(fn.body)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "torch.rand")
    fn.body = fn.body[:stop] + [ast.Return(ast.Constant("accepted"))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {
        "torch": SimpleNamespace(float32="fp32", get_default_dtype=lambda: "fp32",
                                 device=lambda name: SimpleNamespace(type=name)),
        "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
    }
    exec(compile(module, "<actual-wrapper-preallocation-guards>", "exec"), namespace)
    return namespace[fn.name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert sha(SOURCE) == SOURCE_SHA
    # The JIT AST and all original wrapper statements are preserved exactly.
    old = SOURCE.parent.parent / "iv_dependent_matmul" / "iv_dependent_matmul_i64.py"
    current_tree, old_tree = ast.parse(SOURCE.read_text()), ast.parse(old.read_text())
    current_fn = next(x for x in current_tree.body if isinstance(x, ast.FunctionDef)
                      and x.name == "iv_dependent_matmul_wrapper_i64")
    added = current_fn.body.pop(next(i for i, node in enumerate(current_fn.body)
                                     if isinstance(node, ast.If) and "BLOCK_SIZE_K < 16" in ast.unparse(node.test)))
    assert isinstance(added, ast.If) and "BLOCK_SIZE_K < 16" in ast.unparse(added.test)
    assert ast.dump(current_tree) == ast.dump(old_tree)
    guard = metadata_guard()
    guard_cases = []
    for mode in MODES:
        for m, k, n in [(1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)]:
            for block_inner in [1, 2, 4, 8, 16, 32]:
                accepted = m == 0 or n == 0 or block_inner >= 16
                try:
                    result = guard(m, k, n, 1, 1, block_inner, type=mode, device="cuda")
                except ValueError as error:
                    assert not accepted and "launched FP32 dot" in str(error)
                    result = "rejected_dot_shape"
                else:
                    assert accepted and result == "accepted"
                guard_cases.append({"shape_M_K_N": [m, k, n],
                                    "blocks_BM_BN_BK": [1, 1, block_inner],
                                    "schedule": mode, "outcome": result})
    os.environ["TRITON_CACHE_DIR"] = str(output / "compiler-cache")
    os.environ.pop("TRITON_INTERPRET", None)
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend, min_dot_size
    from triton.compiler import ASTSource
    from triton._C.libtriton import native_specialize_impl

    spec = importlib.util.spec_from_file_location("iv_selected_block_defect", SOURCE)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    fn = selected.iv_dependent_matmul_kernel_i64
    target = GPUTarget("cuda", 80, 32)
    backend = CUDABackend(target)
    minimum = min_dot_size(target)(tl.block_type(tl.float32, [1, 1]),
                                   tl.block_type(tl.float32, [1, 1]))
    assert minimum == (1, 1, 16), minimum
    compile_cases = []
    fixtures = [(mode, 1, 1, 16, k) for mode in MODES for k in (0, 1)]
    for mode, bm, bn, bk, inner in fixtures:
        should_compile = True
        constants = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn,
                     "BLOCK_SIZE_K": bk, "type": mode}
        scalars = {name: 1 for name in ["M", "N", "K", "stride_am", "stride_ak",
                                      "stride_bk", "stride_bn", "stride_cm", "stride_cn"]}
        scalars["K"] = inner
        signature, attributes, specialization = {}, {}, {}
        for index, name in enumerate(fn.arg_names):
            if name in ("a_ptr", "b_ptr", "c_ptr"):
                signature[name] = "*fp32"
                attributes[(index,)] = backend.parse_attr("D")
            elif name in constants:
                signature[name] = "constexpr"
            else:
                kind, attribute = native_specialize_impl(backend, scalars[name],
                                                        False, True, True)
                specialization[name] = [kind, attribute]
                signature[name] = kind
                if kind == "constexpr":
                    constants[name] = attribute
                elif attribute:
                    attributes[(index,)] = backend.parse_attr(attribute)
        options = {"num_warps": 4, "num_stages": 4 if mode == MODES[-1] else 3}
        record = {"shape_M_K_N": [1, inner, 1], "blocks_BM_BN_BK": [bm, bn, bk],
                  "schedule": mode, "signature": signature,
                  "specialization": specialization, "options": options}
        try:
            compiled = triton.compile(ASTSource(fn, signature, constexprs=constants,
                                               attrs=attributes),
                                      target=target, options=options)
        except Exception as error:
            assert not should_compile, str(error)
            message = str(error)
            cause = str(error.__cause__)
            assert "Input shapes should have M >= 1, N >= 1 and K >= 16" in message + cause
            assert "tl.dot(a, b)" in message
            log = output / f"{mode}-bk{bk}-rejection.txt"
            log.write_text(f"{type(error).__name__}: {message}\nCause: {cause}\n")
            record.update({"outcome": "compile_rejected", "error_type": type(error).__name__,
                           "message": message, "cause": cause,
                           "evidence": {"path": str(log), "sha256": sha(log)}})
        else:
            assert should_compile, "Invalid K tile unexpectedly compiled"
            artifacts = {}
            for extension in ["ttir", "ttgir", "llir", "ptx"]:
                path = output / f"{mode}-K{inner}-bk{bk}.{extension}"
                path.write_text(compiled.asm[extension])
                artifacts[extension] = {"path": str(path), "sha256": sha(path)}
            record.update({"outcome": "compiled", "artifacts": artifacts})
        compile_cases.append(record)
    import triton.language.semantic as semantic
    import triton.backends.nvidia.compiler as cuda_compiler
    report = {
        "classification": "explicit_static_shape_repair", "whole_current_domain_accepted": False,
        "source": {"path": str(SOURCE), "sha256": sha(SOURCE)},
        "change": "Only conditional nonempty-output BK>=16 admission check; JIT and other wrapper AST unchanged.",
        "old_source": {"path": str(old), "sha256": sha(old)},
        "source_AST_preservation": True,
        "triton_version": triton.__version__, "target": "sm80",
        "minimum_fp32_dot_tile_M_N_K": minimum,
        "guard_audit": "Actual wrapper AST prefix through every preallocation check with fake integer metadata",
        "guard_cases": guard_cases, "compile_cases": compile_cases,
        "compiler_source": {str(Path(module.__file__)): sha(module.__file__)
                            for module in [semantic, cuda_compiler]},
        "script_sha256": sha(__file__), "tensor_allocations": 0,
        "numerical_experiments": 0, "gpu_runs": 0,
        "scope": "Offline compilation checks shape legality only; no numerical or hardware execution claim",
    }
    (output / "source-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"guard_cases": len(guard_cases), "compile_rejections": 0,
                      "valid_compile_controls": 10, "gpu_runs": 0}))


if __name__ == "__main__":
    main()
