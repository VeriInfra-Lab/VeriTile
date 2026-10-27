"""Check complete pasted bodies, external call profiles and original launch.

This is syntax/ABI evidence. Lean execution, safety, Real correctness and
FP equivalence are checked separately; no compiler/GPU result is implied.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest

SOURCE = HERE / "bgmv_expand_slice_physical_tile_checked.py"
MANIFEST = HERE / "jit-correspondence.json"


def check_source():
    pairs = check_manifest(ROOT, MANIFEST)
    assert len(pairs) == 4
    assert all(p["mode"] == "pasted_body" and p["statements"] == 13 for p in pairs)
    module = ast.parse(SOURCE.read_text())
    wrapper = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_bgmv_expand_slice")
    launches = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript)
                and ast.unparse(n.func.value) == "_bgmv_expand_slice_kernel"]
    assert len(launches) == 1
    launch = launches[0]
    assert ast.unparse(launch.func.slice) == "(N, B)"
    assert [ast.unparse(a) for a in launch.args] == [
        "inputs", "weights", "output_tensor", "N", "K", "lora_indices_tensor",
        "K", "1", "N * K", "K", "1", "C", "1", "slice_offset"]
    assert {k.arg: ast.unparse(k.value) for k in launch.keywords} == {
        "BLOCK_N": "256", "BLOCK_K": "BLOCK_K", "SPLIT_N": "N",
        "EVEN_K": "K % BLOCK_K == 0", "ADD_INPUTS": "add_inputs", "CAST_TYPE": "True"}
    return {"status": "PASS", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "complete_original_and_candidate_bodies": pairs,
            "wrapper_launch_checked": True, "index_widths": [32, 64],
            "GPU_execution": False, "native_numerical_conformance": False}


if __name__ == "__main__":
    print(json.dumps(check_source(), indent=2))
