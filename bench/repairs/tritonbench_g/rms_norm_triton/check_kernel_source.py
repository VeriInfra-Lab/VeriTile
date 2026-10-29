"""Audit the selected JIT, exact guards, and every shared DSL statement.

The one explicit epsilon ABI conversion is a reported model boundary. This
source check performs no native arithmetic experiment or GPU execution.
"""
import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from bench import audit_source
from check_contract import source_checks

ORIGINAL = ROOT / "bench/tritonbench_g/rms_norm_triton/rms_norm_triton.py"
SELECTED = HERE / "rms_norm_triton_repaired.py"
DSL = ROOT / "bench/tritonbench_g/rms_norm_triton/Kernels.lean"


def jit(path):
    return next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == "rms_norm_kernel")


class FP32Output(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if ast.unparse(node) == "Y.dtype.element_ty":
            return ast.copy_location(ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()), attr="float32", ctx=ast.Load()), node)
        return self.generic_visit(node)


def audit():
    source_checks()
    original, selected = jit(ORIGINAL), jit(SELECTED)
    assert ast.dump(original) == ast.dump(selected), "Selected JIT changed"
    source = audit_source.select_lean_audit_source(DSL.read_text())
    body = audit_source.strip_region_declarations(audit_source.lean_first_triton_body(source))
    epsilon = "$((eps : ℝ)).to(tl.float32)"
    assert body.count(epsilon) == 1, "Expected one runtime epsilon ABI annotation"
    body = body.replace(epsilon, "eps")
    body = re.sub(r"\$\((\w+)\)", r"\1", body)
    lean_ast = ast.parse(textwrap.dedent(body).strip())
    python_ast = FP32Output().visit(ast.Module(body=selected.body, type_ignores=[]))
    assert ast.dump(lean_ast) == ast.dump(python_ast), "DSL/source statement mismatch"
    return {
        "jit_ast_identical": True,
        "exact_existing_guard_checks": "pass",
        "fixed_shift_block_checks": "pass",
        "all_dsl_statements_ast_equal": True,
        "statement_count": len(selected.body),
        "normalizations": ["Lean antiquotation syntax", "FP32 region dtype for Y.dtype.element_ty",
                           "one explicit runtime-epsilon FP32 ABI model annotation"],
        "primitive_and_compiler_binding": "model boundary; no new native certificate",
        "numerical_experiments": [],
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [ORIGINAL, SELECTED, DSL]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
