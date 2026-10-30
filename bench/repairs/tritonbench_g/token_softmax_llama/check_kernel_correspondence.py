"""Exact source/domain/candidate transfer from the frozen identical Bloom task.

This checker executes no Torch, Triton, or floating-point numerical experiment.
It checks entire Python files, the unique candidate AST rewrite, the original
metadata-stride defect, and hashes of the reused independent review evidence.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BLOOM = HERE.parent / "token_softmax_bloom"
EVIDENCE = ROOT / ".veritile-cache/kernel-evidence/token_softmax_llama"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def main():
    canonical = HERE / "token_softmax_llama_repaired.py"
    shared = BLOOM / "token_softmax_bloom_repaired.py"
    candidate = HERE / "token_softmax_llama_index_commuted.py"
    shared_candidate = BLOOM / "token_softmax_bloom_index_commuted.py"
    original = ROOT / "bench/tritonbench_g/token_softmax_llama/token_softmax_llama.py"
    assert sha(canonical) == "3a7a843d8532cc306d9ea665674859dbf91e88933c413074287757d43a86eab4"
    assert sha(original) == "0fa02ee678899b4524478c77876bd07c331494faf2320e76f5f7669d458a1f41"
    assert canonical.read_bytes() == shared.read_bytes()
    assert ast.dump(function(candidate, "_fwd_kernel_token_softmax")) == ast.dump(
        function(shared_candidate, "_fwd_kernel_token_softmax"))

    class Commute(ast.NodeTransformer):
        count = 0

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name)
                    and node.left.id == "cur_batch" and isinstance(node.right, ast.Name)
                    and node.right.id == "stride_length"):
                node.left, node.right = node.right, node.left
                self.count += 1
            return node

    rewrite = Commute()
    transformed = rewrite.visit(copy.deepcopy(ast.parse(canonical.read_text())))
    assert rewrite.count == 1
    assert ast.dump(transformed) == ast.dump(ast.parse(candidate.read_text()))
    # The original's b=1 metadata loads use physical offset 1, not logical
    # offset 2, for ordinary stride-2 views. All intended segments are legal.
    old_kernel = function(original, "_fwd_kernel_token_softmax")
    for target, pointer in [("cur_batch_seq_len", "B_Seqlen"),
                            ("cur_batch_in_all_start_index", "B_Start_Loc")]:
        statement = next(node for node in old_kernel.body if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id == target)
        assert ast.dump(statement.value.args[0]) == ast.dump(
            ast.parse(pointer + " + cur_batch", mode="eval").body)
    starts_storage, lengths_storage = [0, 1, 4], [2, 1, 2]
    intended = list(zip(starts_storage[::2], lengths_storage[::2]))
    original_loaded = list(zip(starts_storage[:2], lengths_storage[:2]))
    assert intended == [(0, 2), (4, 2)]
    assert original_loaded == [(0, 2), (1, 1)]
    intended_writes = {i for start, count in intended for i in range(start, start + count)}
    original_writes = {i for start, count in original_loaded for i in range(start, start + count)}
    assert intended_writes - original_writes == {4, 5}

    packet_path = Path("/tmp/token-softmax-bloom-kernel-delivery.json")
    assert sha(packet_path) == "8143d76982577e8a9505c81e1560603f2abfbfb34651214e4c7623f4f9a5d4c1"
    packet = json.loads(packet_path.read_text())
    reused = {}
    for name in ["contract-check.log", "source-review.log", "source-review.json",
                 "review_source.py", "candidate-check.log", "source-preservation.json"]:
        path = ROOT / ".veritile-cache/kernel-evidence/token_softmax_bloom" / name
        relative = str(path.relative_to(ROOT))
        assert sha(path) == packet["evidence_sha256"][relative]
        reused[relative] = sha(path)
    report = {
        "canonical_source": {"path": str(canonical.relative_to(ROOT)), "sha256": sha(canonical)},
        "original_source": {"path": str(original.relative_to(ROOT)), "sha256": sha(original)},
        "entire_canonical_python_including_guards_and_launch_is_identical_to_bloom": True,
        "candidate_source": {"path": str(candidate.relative_to(ROOT)), "sha256": sha(candidate)},
        "complete_candidate_kernel_ast_is_identical_to_bloom": True,
        "candidate_retains_entire_unchanged_canonical_wrapper": True,
        "candidate_ast_change": "Exactly one cur_batch * stride_length becomes stride_length * cur_batch",
        "other_kernel_and_wrapper_ast_unchanged": True,
        "original_metadata_stride_counterexample": {
            "heads": 1, "tokens": 6, "maximum": 2, "metadata_stride": 2,
            "starts_storage": starts_storage, "lengths_storage": lengths_storage,
            "intended_segments": intended, "original_segments": original_loaded,
            "canonical_required_but_original_untouched_outputs": [4, 5],
            "concrete_payload_and_before": "Six zero logits and six output cells initially -7",
            "canonical_real_values_at_4_and_5": "1/2 each; original leaves -7",
            "scope": "Defect belongs to the preserved original, not ledger.contract_revision.source"
        },
        "independent_canonical_review": {
            "formula": "Single padded global maximum, exp(x-max), full segment sum, division",
            "domain": "All canonical guards unchanged; no added assumptions or dtype variants",
            "precision": "FP32 payload/output; signed int32 metadata and int64 indices; 4/8/16 warps at 2048/4096",
            "corner_cases": ["No active input for zero axes", "Zero maximum validates zero lengths",
                "Unsorted, empty and disjoint segments with gaps", "Broadcast input/metadata and singleton output strides",
                "Read allocations may alias across dtypes", "Nonunit strides and offsets on all four tensors",
                "All masked lane indices bounded", "Output storage disjoint from read allocations",
                "Global maximum avoids online recurrence failure on negative infinity prefix",
                "Native reference admits every raw FP32 word without finite-data assumptions"],
            "remaining_canonical_counterexample": None
        },
        "reused_evidence": reused,
        "new_numerical_experiments": [],
        "gpu_execution": "not_run",
        "python_to_dsl_boundary": "Audited source/AST correspondence, not a verified Python compiler"
    }
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE / "source-correspondence.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: entire canonical file and complete candidate kernel AST equal frozen Bloom; exact one-product rewrite.")
    print("PASS: preserved-original stride defect confirmed analytically; canonical source/domain review complete.")
    print("PASS: six reused evidence files match frozen hashes; no new numerical experiments.")


if __name__ == "__main__":
    main()
