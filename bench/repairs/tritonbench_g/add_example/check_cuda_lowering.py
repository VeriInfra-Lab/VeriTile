"""Offline evidence for the scalar CUDA memory model; no GPU execution.

This deliberately checks the observed lowering of this kernel. A compiler change
that fails a check requires inspection, not an automatic change of input scope.
"""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace


def inspect_ptx(ptx, block):
    memory = re.findall(r"^\s*(@%p\d+)?\s*((?:ld|st)\.global[^\s]*)\s+([^;]+);",
                        ptx, re.MULTILINE)
    assert memory and all(predicate for predicate, _, _ in memory), memory
    assert all(op in ("ld.global.b32", "st.global.b32") for _, op, _ in memory), memory
    assert sum(op.startswith("ld") for _, op, _ in memory) == 2 * sum(op.startswith("st") for _, op, _ in memory)
    assert re.search(r"setp\.lt\.s32", ptx)
    assert re.search(r"mul\.wide\.s32\s+%rd\d+,\s*%r\d+,\s*4;", ptx)
    assert re.search(r"add\.f32", ptx) and not re.search(r"add\.ftz\.f32", ptx)
    if block == 4:
        assert [op for _, op, _ in memory] == ["ld.global.b32", "ld.global.b32", "st.global.b32"], memory
        active = re.search(r"setp\.lt\.s32\s+(%p\d+),", ptx).group(1)
        assert [predicate for predicate, op, _ in memory if op.startswith("ld")] == ["@" + active] * 2
        store_predicate = memory[-1][0][1:]
        store_guard = re.search(r"and\.pred\s+" + re.escape(store_predicate)
                                + r",\s*(%p\d+),\s*(%p\d+);", ptx)
        assert store_guard and active in store_guard.groups(), "store must retain the source mask"
    assert ".address_size 64" in ptx
    return [op for _, op, _ in memory]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for compiler evidence and its local cache")
    parser.add_argument("--task", choices=("add_example", "vector_addition", "vector_addition_custom"),
                        default="add_example")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache")

    import triton
    from triton.backends.compiler import BaseBackend, GPUTarget
    from triton.compiler import ASTSource

    here = Path(__file__).resolve().parent
    task = args.task
    source = here.parent / task / (task + "_repaired.py")
    original = here.parents[2] / "tritonbench_g" / task / (task + ".py")
    names, arguments, block = {
        "add_example": (("add_kernel", "add_commuted_kernel"),
                        ("in_ptr0", "in_ptr1", "out_ptr", "n_elements", "BLOCK_SIZE"), 4),
        "vector_addition": (("add_kernel", "add_commuted_kernel"),
                            ("x_ptr", "y_ptr", "output_ptr", "n_elements", "BLOCK_SIZE"), 1024),
        "vector_addition_custom": (("_add_kernel", "_add_commuted_kernel"),
                                   ("A", "B", "C", "size", "BLOCK"), 16),
    }[task]

    def kernel_ast(path):
        return next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == names[0])

    assert ast.dump(kernel_ast(source)) == ast.dump(kernel_ast(original))
    spec = importlib.util.spec_from_file_location("add_example_lowering_review", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The 16-byte specialization is an optimization hint, not a 4-byte guard.
    for remainder in range(16):
        tensor = SimpleNamespace(data_ptr=lambda r=remainder: 0x1000 + r)
        hint = BaseBackend.get_tensor_specialization(tensor, align=True)
        assert hint == ("D" if remainder == 0 else "")

    cases = []
    specializations = (("no_alignment_hint", ()), ("pointers_align16", (0, 1, 2)),
                       ("pointers_and_extent_align16", (0, 1, 2, 3))) if block == 4 else (("no_alignment_hint", ()),)
    for kernel_name in names:
        for sm in (80, 90):
            for label, aligned_args in specializations:
                attrs = {(i,): BaseBackend.parse_attr("D") for i in aligned_args}
                signature = dict(zip(arguments, ("*fp32", "*fp32", "*fp32", "i32", "constexpr")))
                ast_source = ASTSource(getattr(module, kernel_name), signature=signature,
                    constexprs={arguments[-1]: block}, attrs=attrs)
                kernel = triton.compile(ast_source, target=GPUTarget("cuda", sm, 32))
                name = f"{kernel_name}_sm{sm}_{label}"
                for stage in ("ttir", "ttgir", "llir", "ptx"):
                    (output / f"{name}.{stage}").write_text(kernel.asm[stage])
                operations = inspect_ptx(kernel.asm["ptx"], block)
                cases.append({"kernel": kernel_name, "target": f"sm_{sm}", "specialization": label,
                              "memory_instructions": operations, "fp32_add": True, "explicit_ftz": False})
                print(f"{name}: predicated b32 accesses, signed byte offsets and FP32 addition passed")
    (output / "report.json").write_text(json.dumps({
        "task": task, "triton_version": triton.__version__, "block_size": block,
        "gpu_execution": False, "cases": cases,
    }, indent=2) + "\n")
    print(f"{len(cases)} offline compilation cases passed; no GPU execution.")


if __name__ == "__main__":
    main()
