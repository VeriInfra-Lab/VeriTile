"""Check the minimal kernel/host edit and pin the current source/proof package."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from bench.jit_correspondence import quotation
from bench.repairs.tritonbench_g.triton_conv2d_fwd.check_body_source import check_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL = ROOT / "bench/tritonbench_g/triton_conv2d_fwd/triton_conv2d_fwd.py"
CANDIDATE = HERE / "triton_conv2d_fwd_repaired.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def strip_doc(node):
    node = copy.deepcopy(node)
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body.pop(0)
    return node


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    original, candidate = ast.parse(ORIGINAL.read_text()), ast.parse(CANDIDATE.read_text())
    old_kernel = strip_doc(function(original, "conv2d_forward_kernel"))
    kernel = strip_doc(function(candidate, "conv2d_forward_kernel"))
    restored = copy.deepcopy(kernel)
    edits = []
    for node in ast.walk(restored):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.load":
            fill = [keyword for keyword in node.keywords if keyword.arg == "other"]
            assert len(fill) == 1 and ast.literal_eval(fill[0].value) == 0
            edits.append(ast.unparse(node))
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "other"]
    assert len(edits) == 2 and dump(restored) == dump(old_kernel)
    old_host = function(original, "conv2d_forward")
    host = function(candidate, "conv2d_forward")
    restored_host = copy.deepcopy(host)
    restored_host.body = [node for node in restored_host.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_validate_inputs") and not (
        isinstance(node, ast.If) and ast.unparse(node.test) == "batch_dim == 0")]
    assert dump(restored_host) == dump(old_host)
    assert dump(host.args) == dump(old_host.args)
    assert sha(CANDIDATE) == "37ad52bccd9a329291c6509200adfb8a3125975225685c9e2c4f300578a8561b"
    # Tracked source hashes replace the unavailable private snapshot dependency.
    # This checks preservation from the reviewed pre-migration public contract;
    # it does not claim to rerun the historical initial-snapshot comparison.
    baseline = {'Contract.lean': 'b90c5a99e90b960b94c9fa11bb52683d08df829e13fca36faeab6cec71a5e994', 'Proofs/Contract.lean': 'b99a4f6a6db1e407048f7303ad944c09160591806ee1b10d449bd449b7c4471c'}
    domain_hashes = {}
    for relative, expected in baseline.items():
        assert sha(HERE / relative) == expected, (relative, "admitted domain changed")
        domain_hashes[relative] = {"baseline_commit": "c369f8e2da7dba42d02878022b4ace716afbea0c",
                                  "baseline_sha256": expected, "current_sha256": sha(HERE / relative),
                                  "scope": "Byte-identical reviewed public contract before raw-body migration."}
    bodies = check_sources(ROOT)
    rows = json.loads((HERE / "jit-correspondence.json").read_text())
    quotations = []
    for row in rows:
        _, body = quotation((ROOT / row["lean"]).read_text(), row["lean_definition"])
        quotations.append({"lean": row["lean"], "definition": row["lean_definition"],
                           "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                           "fp16": row["expected_specialized_booleans"]["fp16"],
                           "explicit_zero_loads": body.count("other=0.0"),
                           "source_operand_half_casts": body.count(".to(tl.float16)")})
    kernels = (HERE / "Kernels.lean").read_text()
    assert "if fp16 then KernelHalf.source p else KernelDefault.source p" in kernels
    lean = {str(path.relative_to(ROOT)): sha(path) for path in sorted(HERE.rglob("*.lean"))}
    report = {
        "status": "PASS", "original_sha256": sha(ORIGINAL), "candidate_sha256": sha(CANDIDATE),
        "kernel_ast_sha256": hashlib.sha256(dump(kernel).encode()).hexdigest(),
        "kernel_edits": edits, "kernel_docstring_correction": "grouped weights have C/groups channels",
        "host_edits": ["explicit checked metadata/storage/signed32/direct-grid domain", "empty batch returns fresh output without launch"],
        "preserved": ["original direct three-dimensional grid including redundant grouped feature programs",
                      "128x32x32 tiles", "h/w/channel loop order and all source pointer expressions",
                      "FP32 inputs and output", "fp16/tf32 flags and defaults with fp16 precedence",
                      "default compiler options", "original fresh torch.empty and returned tensor", "no copies or layout normalization"],
        "contract": ["resolved rank4 positive-stride FP32 CUDA tensors on the same current device",
                     "actual data pointers/storage offsets; input and weight may alias",
                     "N may be zero; other dimensions/kernel/stride/group positive; padding nonnegative",
                     "divisible channels/features, exact grouped weight shape, positive output dimensions",
                     "actual storage span bounds; checked signed32 envelope for all padded formed expressions",
                     "nonempty launch grid Y and Z at most 65535"],
        "formal_route": "same repaired shared typed DSL, complete original grid, independent initial-array Real/native references, actual Tensor readback, per-store concrete bytes and full frames",
        "readable_DSL": quotations,
        "raw_body_correspondence": bodies,
        "domain_unchanged": domain_hashes,
        "source_binding_scope": "Exact original/candidate bodies for both fp16 specializations, public invocation aliases and historical Python repair delta; not a verified Python-to-DSL translator",
        "compiler_carry_lowering": "source dot remains pending until outer add; compiler interpretation supplies the old carry to the first native instruction without asserting rounded split/fused equality",
        "compiler_scope": "pinned concrete Triton 3.7.1 SM90 profiles and symbolic PTX binding; not universal compiler correctness",
        "numerical_boundary": ["scalar fma.rn.f32 equality to IEEE RN-even software profile",
                               "scalar FP32-to-FP16 RN conversion",
                               "one fixed output lane of literal WGMMA m64n32k16 f32.f16.f16",
                               "one fixed output lane of literal WGMMA m64n32k8 f32.tf32.tf32"],
        "real_target": "independent grouped cross-correlation on original caller values",
        "GPU_execution": False, "native_numerical_conformance": False,
        "lean_sources": lean,
    }
    (HERE / "source-binding.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: exact four raw bodies; two zero-fill kernel edits; host and domain preservation; {len(lean)} Lean sources pinned")


if __name__ == "__main__":
    main()
