"""Reproduce the dimension argmax defect with the actual Triton interpreter.

Only imports and the selected, unmodified JIT function are loaded from each
source, avoiding the original file's automatic CUDA benchmark. No Triton
operation, reduction, mask, or store is replaced. This is CPU source execution,
not GPU execution, a wrapper proof, or a numerical-assumption experiment.
"""

import argparse
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
SOURCES = (
    ("original", ROOT / "bench/tritonbench_g/triton_argmax/triton_argmax.py",
     "argmax_kernel"),
    ("selected_repair", HERE / "triton_argmax_repaired.py",
     "argmax_kernel_repaired"),
)


def load_kernel(path, name):
    parsed = ast.parse(path.read_text())
    function = next(node for node in parsed.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    body = [node for node in parsed.body
            if isinstance(node, (ast.Import, ast.ImportFrom)) or node is function]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope[name], {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kernel": name,
        "kernel_line": function.lineno,
        "kernel_ast_sha256": hashlib.sha256(
            ast.dump(function, include_attributes=False).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "source-defect-validation.json")
    args = parser.parse_args()
    cases = (
        ("nan_at_one", [1.0, float("nan")], 0, 1, True),
        ("nan_between_finite_values", [-3.0, float("nan"), 2.0], 0, 1, True),
        ("finite_maximum", [-3.0, 1.0, 2.0], 2, 2, False),
        ("least_tie_index", [-3.0, 2.0, 2.0], 1, 1, False),
        ("negative_infinity_tie", [-float("inf"), -float("inf")], 0, 0, False),
    )
    observations, sources = [], []
    for label, path, name in SOURCES:
        kernel, source = load_kernel(path, name)
        sources.append({"label": label, **source})
        for case, values, expected_source, expected_reference, is_defect in cases:
            inp = torch.tensor(values, dtype=torch.float32)
            original_bytes = inp.view(torch.uint8).clone()
            out_storage = torch.full((3,), -99, dtype=torch.int64)
            out = out_storage[1:2]
            parameters = (inp, out, 1, len(values), 1)
            if label == "selected_repair":
                parameters += (0,)
            kernel[(1,)](*parameters, BLOCK_M=128, BLOCK_N=128, INT64_INDEX=False)
            observed = out.item()
            reference = torch.argmax(inp).item()
            assert observed == expected_source, (label, case, observed)
            assert reference == expected_reference, (case, reference)
            assert (observed != reference) == is_defect
            assert torch.equal(inp.view(torch.uint8), original_bytes)
            assert out_storage[0].item() == out_storage[2].item() == -99
            observations.append({
                "source": label,
                "case": case,
                "input_fp32_words": [f"0x{word:08x}" for word in inp.view(torch.uint32).tolist()],
                "input_shape": list(inp.shape),
                "input_contiguous": inp.is_contiguous(),
                "M": 1, "N": len(values), "K": 1,
                "BLOCK_M": 128, "BLOCK_N": 128,
                "INT64_INDEX": False,
                "observed_output": observed,
                "torch_argmax_reference": reference,
                "violates_argmax_target": is_defect,
                "input_bytes_unchanged": True,
                "output_guard_elements_unchanged": True,
            })
    receipt = {
        "task": "triton_argmax",
        "status": "source_defect_pending_repair",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "unmodified JIT function ASTs under the actual Triton CPU interpreter",
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
        "sources": sources,
        "runs": len(observations),
        "defect_observations": sum(item["violates_argmax_target"] for item in observations),
        "control_observations": sum(not item["violates_argmax_target"] for item in observations),
        "observations": observations,
        "gpu_executed": False,
        "wrapper_executed": False,
        "numerical_deferrals": [],
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runs": receipt["runs"],
                      "defect_observations": receipt["defect_observations"],
                      "control_observations": receipt["control_observations"]}))


if __name__ == "__main__":
    main()
