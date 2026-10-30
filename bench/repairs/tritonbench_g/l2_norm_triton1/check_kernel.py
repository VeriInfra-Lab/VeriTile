"""Focused actual-source CPU replay for the two related L2 kernel tasks.

This verifies source/ABI binding and the concrete integer-only candidate. It
does not experimentally validate primitive IEEE operations or claim GPU runs.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load_functions(path):
    parsed = ast.parse(path.read_text())
    body = [node for node in parsed.body if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.FunctionDef) and "kernel" in node.name]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    functions = {node.name: scope[node.name] for node in body if isinstance(node, ast.FunctionDef)}
    return functions


def verify_candidate(original, candidate):
    original_functions = [node for node in ast.parse(original.read_text()).body
                          if isinstance(node, ast.FunctionDef) and "kernel" in node.name]
    candidates = {node.name: node for node in ast.parse(candidate.read_text()).body
                  if isinstance(node, ast.FunctionDef)}
    for node in original_functions:
        node.name = node.name.removesuffix("_repaired") + "_index_commuted"
        for statement in node.body:
            if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                assert statement.target.id in {"X", "Y", "DY", "DX"}
                assert isinstance(statement.value.op, ast.Mult)
                assert statement.value.left.id == "row" and statement.value.right.id == "N"
                statement.value.left, statement.value.right = statement.value.right, statement.value.left
        assert ast.dump(node, include_attributes=False) == ast.dump(candidates[node.name], include_attributes=False)


def main():
    torch.manual_seed(2718)
    cases = [(2, n, "finite", 1e-6) for n in (1, 3, 16, 32, 129, 1025, 16384)]
    cases += [(1, 3, "zero", 0.0), (1, 3, "exceptional", 1e-6),
              (2, 3, "finite", float("nan")), (1, 3, "finite", -1.0)]
    for task in ("l2_norm_triton1", "l2_norm_triton2"):
        folder = HERE.parent / task
        source = folder / (task + "_repaired.py")
        candidate = folder / (task + "_candidate.py")
        verify_candidate(source, candidate)
        original_functions, candidate_functions = load_functions(source), load_functions(candidate)
        observations = []
        for rows, n, profile, eps in cases:
            values = torch.randn((rows, n), dtype=torch.float32)
            if profile == "zero":
                values.zero_()
            elif profile == "exceptional":
                values[0] = torch.tensor([float("nan"), float("inf"), -0.0])
            gradient = torch.randn_like(values)
            input_before = values.view(torch.uint8).clone()
            gradient_before = gradient.view(torch.uint8).clone()
            block = triton.next_power_of_2(n)
            for name, kernel in original_functions.items():
                reverse = "bwd" in name
                changed = candidate_functions[name.removesuffix("_repaired") + "_index_commuted"]
                outputs = []
                for fn in (kernel, changed):
                    storage = torch.full((rows*n+8,), -91.0, dtype=torch.float32)
                    output = storage[4:4+rows*n].view(rows, n)
                    arguments = (values, gradient, output, n, eps) if reverse else (values, output, n, eps)
                    fn[(rows,)](*arguments, BLOCK_N=block)
                    assert torch.equal(storage[:4], torch.full((4,), -91.0))
                    assert torch.equal(storage[-4:], torch.full((4,), -91.0))
                    outputs.append(output.clone())
                assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8)), (task, name, n, profile)
                assert torch.equal(values.view(torch.uint8), input_before)
                assert torch.equal(gradient.view(torch.uint8), gradient_before)
                reference_checked = profile == "finite" and eps == 1e-6
                if reference_checked:
                    squared = (values*values).sum(-1, keepdim=True)
                    rstd = torch.rsqrt(squared+eps)
                    reference = (gradient*rstd - (gradient*values).sum(-1, keepdim=True)
                                 / (squared+eps)*rstd*values) if reverse else values*rstd
                    torch.testing.assert_close(outputs[0], reference, rtol=2e-5, atol=2e-6)
                observations.append({"kernel": name, "rows": rows, "columns": n,
                                     "block": block, "profile": profile,
                                     "epsilon": "NaN" if eps != eps else eps,
                                     "candidate_bit_identical": True,
                                     "input_bytes_unchanged": True,
                                     "output_guard_elements_unchanged": True,
                                     "independent_formula_control": reference_checked})
        receipt = {
            "task": task, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution": "actual unmodified original/candidate JIT function ASTs under Triton CPU interpreter",
            "triton_version": triton.__version__, "torch_version": torch.__version__,
            "sources": [{"path": str(path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (source, candidate)],
            "candidate_change": "Only X/Y/DY/DX integer row-base products row*N commute to N*row.",
            "kernel_pairs": len(observations), "actual_kernel_executions": 2*len(observations),
            "observations": observations, "numerical_deferrals": [],
            "gpu_executed": False, "independent_proof_replay": False,
        }
        target = folder / "kernel-validation.json"
        target.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"task": task, "kernel_pairs": len(observations), "receipt": str(target)}))


if __name__ == "__main__":
    main()
