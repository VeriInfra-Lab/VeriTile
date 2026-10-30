"""Audit the source delta and execute only wrapper guards on integer metadata.

This does not import Torch/Triton, allocate tensors, execute floating-point
operations, or launch a kernel. It is a source/guard audit, not GPU validation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


TASK = Path(__file__).resolve().parent
ORIGINAL = TASK.parent / "rbe_triton_transform/rbe_triton_transform_repaired.py"
CANDIDATE = TASK / "rbe_triton_transform_checked_launch.py"
ORIGINAL_SHA256 = "9095bec159a1bceccea0b6415d198eb1efed6fd30947e79d378ff6f70f17c592"
INSERTION = b'''    if x.numel() and batch > (1 << 31) - 1:
        raise ValueError("batch dimension exceeds CUDA grid-x limit")
    if x.numel() and triton.cdiv(M, 2) * triton.cdiv(K, 1024) > 65535:
        raise ValueError("launch grid-y exceeds CUDA grid-y limit")
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Metadata:
    def __init__(self, shape=(1, 1, 1), *, layout="strided", dtype="float32",
                 device="cuda", contiguous=True, negative=False, pointer=4096):
        self.shape = shape
        self.ndim = len(shape)
        self.layout = layout
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.contiguous = contiguous
        self.negative = negative
        self.pointer = pointer

    def numel(self):
        value = 1
        for size in self.shape:
            value *= size
        return value

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def data_ptr(self):
        return self.pointer


def guard_prefix(source):
    """Compile the actual wrapper prefix, stopping before its CUDA context."""
    module = ast.parse(source)
    fn = copy.deepcopy(next(node for node in module.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == "rbe_triton_wrapper"))
    stop = next(index for index, node in enumerate(fn.body)
                if isinstance(node, ast.With))
    fn.body = fn.body[:stop] + [ast.Return(ast.Constant("accepted"))]
    prefix = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {
        "torch": SimpleNamespace(Tensor=Metadata, strided="strided", float32="float32"),
        "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
    }
    exec(compile(prefix, "<actual-wrapper-guard-prefix>", "exec"), namespace)
    return namespace[fn.name]


def outcome(guard, metadata, position):
    try:
        return guard(metadata, position)
    except (TypeError, ValueError) as error:
        return f"{type(error).__name__}: {error}"


def audit():
    original = ORIGINAL.read_bytes()
    candidate = CANDIDATE.read_bytes()
    assert sha(ORIGINAL) == ORIGINAL_SHA256
    assert candidate.count(INSERTION) == 1
    restored = candidate.replace(INSERTION, b"", 1)
    assert restored == original, "The candidate has an unrelated source edit"
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(original))
    old_guard = guard_prefix(original)
    new_guard = guard_prefix(candidate)
    grid_error = "ValueError: launch grid-y exceeds CUDA grid-y limit"
    batch_error = "ValueError: batch dimension exceeds CUDA grid-x limit"
    padded_error = "ValueError: padded row positions must fit signed int64"
    position_error = "ValueError: position must be a signed int64 Python integer"
    rank_error = "ValueError: input must be a resolved contiguous rank-three Tensor"
    alignment_error = "ValueError: FP32 data pointer must be four-byte aligned"
    type_error = "TypeError: expected a strided FP32 Tensor"
    accepted = "accepted"
    cases = [
        ("grid_y_limit", Metadata((1, 131070, 1)), 0, accepted, accepted),
        ("original_counterexample", Metadata((1, 131071, 1)), 0, accepted, grid_error),
        ("grid_x_limit", Metadata(((1 << 31) - 1, 1, 1)), 0, accepted, accepted),
        ("grid_x_overflow", Metadata((1 << 31, 1, 1)), 0, accepted, batch_error),
        ("empty_rows_large_batch", Metadata((1 << 31, 0, 1)), (1 << 63) - 1,
         accepted, accepted),
        ("empty_columns_large_batch_and_grid", Metadata((1 << 31, 131071, 0)),
         (1 << 63) - 1, accepted, accepted),
        ("empty_batch_large_grid", Metadata((0, 131071, 1)), (1 << 63) - 1,
         accepted, accepted),
        ("coupled_grid_limit", Metadata((1, 509, 263168)), 0, accepted, accepted),
        ("coupled_grid_overflow", Metadata((1, 509, 263169)), 0, accepted, grid_error),
        ("batch_guard_priority", Metadata((1 << 31, 131071, 1)), 0,
         accepted, batch_error),
        ("python_bool_position", Metadata(), True, accepted, accepted),
        ("negative_position_odd_tail", Metadata((1, 3, 3)), -19, accepted, accepted),
        ("int64_min_position", Metadata(), -(1 << 63), accepted, accepted),
        ("padded_position_overflow", Metadata(), (1 << 63) - 1,
         padded_error, padded_error),
        ("old_position_guard_priority", Metadata((1 << 31, 131071, 1)),
         (1 << 63) - 1, padded_error, padded_error),
        ("int64_position_overflow", Metadata(), 1 << 63, position_error, position_error),
        ("int64_position_underflow", Metadata(), -(1 << 63) - 1,
         position_error, position_error),
        ("noninteger_position", Metadata(), "0", position_error, position_error),
        ("non_tensor", object(), 0, "TypeError: expected a Tensor",
         "TypeError: expected a Tensor"),
        ("non_strided", Metadata(layout="sparse"), 0, type_error, type_error),
        ("non_fp32", Metadata(dtype="float16"), 0, type_error, type_error),
        ("non_cuda", Metadata(device="cpu"), 0, "ValueError: input must be on CUDA",
         "ValueError: input must be on CUDA"),
        ("wrong_rank", Metadata((1, 1)), 0, rank_error, rank_error),
        ("noncontiguous", Metadata(contiguous=False), 0, rank_error, rank_error),
        ("lazy_negative", Metadata(negative=True), 0, rank_error, rank_error),
        ("unaligned_nonempty", Metadata(pointer=4097), 0,
         alignment_error, alignment_error),
        ("unaligned_empty", Metadata((1, 0, 1), pointer=4097), 0,
         accepted, accepted),
        ("lazy_negative_empty", Metadata((1, 0, 1), negative=True), 0,
         rank_error, rank_error),
    ]
    results = []
    for name, metadata, position, expected_old, expected_new in cases:
        old_result = outcome(old_guard, metadata, position)
        new_result = outcome(new_guard, metadata, position)
        assert (old_result, new_result) == (expected_old, expected_new), (
            name, old_result, new_result, expected_old, expected_new)
        entry = {"case": name, "position": position, "original": old_result,
                 "candidate": new_result, "pass": True}
        if isinstance(metadata, Metadata):
            entry["shape"] = metadata.shape
            if len(metadata.shape) == 3:
                b, m, k = metadata.shape
                entry["nonempty"] = metadata.numel() != 0
                entry["grid"] = [b, ((m + 1) // 2) * ((k + 1023) // 1024)]
        results.append(entry)
    return {
        "original_source": str(ORIGINAL), "original_sha256": sha(ORIGINAL),
        "candidate_source": str(CANDIDATE), "candidate_sha256": sha(CANDIDATE),
        "source_delta": "Exactly four added lines implement two nonempty launch guards before CUDA context",
        "byte_identity_after_removing_guards": True,
        "full_ast_identity_after_removing_guards": True,
        "helper_kernel_allocation_grid_and_launch_unchanged": True,
        "audit_kind": "Actual Python wrapper guard prefix with fake integer-only metadata",
        "cases_passed": len(results), "cases": results,
        "torch_imports": 0, "triton_imports": 0, "tensor_allocations": 0,
        "numerical_experiments": 0, "gpu_runs": 0,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
