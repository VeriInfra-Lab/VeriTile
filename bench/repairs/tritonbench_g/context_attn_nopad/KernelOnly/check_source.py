"""Focused actual-source checks; CUDA metadata is an explicit CPU facade."""
from pathlib import Path
import hashlib
import json
import runpy

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[3]
OUT = ROOT / ".veritile-cache/kernel-evidence/context_attn_nopad"
SOURCE = TASK / "context_attn_nopad_repaired.py"
PIN = "cf0939e3b862aa0e05b117cb1d66e828ffa7e65e870b62764454b782df79fed1"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == PIN
base = runpy.run_path(str(TASK / "check_contract.py"))
torch, repair = base["torch"], base["repair"]
Tensor, Device, Launch = base["Tensor"], base["Device"], base["Launch"]


def backing(tensor):
    return torch.empty(0, dtype=torch.uint8).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))


records = []


def check(width, length, metadata_alias=False, exceptional=False):
    rows, heads = length + 3, 1
    payload = [torch.randn(rows + 4, heads, width) * 0.1 for _ in range(3)]
    q, k, v = [t[2:-2] for t in payload]
    out_backing = torch.full((rows + 4, heads, width), 71.0)
    out = out_backing[2:-2]
    start_backing = torch.tensor([912, 1, 914], dtype=torch.int32)
    length_backing = torch.tensor([923, length, 925], dtype=torch.int32)
    start, lengths = start_backing[1:2], length_backing[1:2]
    if metadata_alias:
        integers = q.view(torch.int32).flatten()
        integers[0], integers[1] = 1, length
        start, lengths = integers[:1], integers[1:2]
        k = q
    if exceptional:
        q[1, 0, 0] = float("nan")
        k[1, 0, 1] = float("inf")
        v[1, 0, 2] = -0.0
    inputs = (q, k, v, start, lengths)
    snapshots = [backing(t).clone() for t in inputs]
    before = backing(out).clone()
    count = Launch.count
    assert repair.context_attention_fwd(*[Tensor(t) for t in (q, k, v, out, start, lengths)], length + 7) is None
    assert Launch.count - count == int(length > 0)
    assert Device.current == torch.device("cuda:1")
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(backing(tensor), snapshot)
    touched = torch.zeros_like(before, dtype=torch.bool)
    byte_begin = (out.storage_offset() + width) * 4
    touched[byte_begin:byte_begin + length * width * 4] = True
    assert torch.equal(backing(out)[~touched], before[~touched])
    if not exceptional:
        for row in range(length):
            scores = k[1:row + 2, 0] @ q[row + 1, 0] / (width ** 0.5)
            expected = torch.softmax(scores, 0) @ v[1:row + 2, 0]
            torch.testing.assert_close(out[row + 1, 0], expected, rtol=4e-4, atol=1e-5)
    records.append({"dimension": width, "length": length,
                    "payload_metadata_read_alias": metadata_alias,
                    "NaN_Inf_signed_zero_admitted": exceptional,
                    "complete_input_backing_and_output_complement": True})


torch.manual_seed(11)
for width, length in ((16, 1), (32, 127), (64, 128), (128, 129), (16, 257), (16, 0)):
    check(width, length)
check(16, 3, metadata_alias=True)
check(16, 3, exceptional=True)
receipt = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": PIN,
           "historical_actual_source_and_guard_controls": "passed",
           "additional_complete_source_cases": records,
           "additional_complete_source_executions": len(records),
           "total_nonempty_launches_in_this_process": Launch.count,
           "GPU_execution": False, "native_instruction_numerical_conformance": False}
(OUT / "source-controls.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"additional_complete_source_executions": len(records),
                  "complete_backing_frames": "passed", "source_unchanged": True}))
