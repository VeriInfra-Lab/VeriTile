"""Actual admission checks for unused untyped-storage tail bytes; no kernel run."""
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import sys

sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / ".veritile-cache/task-deliveries/triton_conv2d_fwd"
SOURCE = HERE / "triton_conv2d_fwd_repaired.py"


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tensor(capacity):
    storage = torch.UntypedStorage(capacity)
    result = torch.empty(0, dtype=torch.float32).set_(
        storage, 0, (1, 1, 1, 1), (1, 1, 1, 1))
    assert result.untyped_storage().nbytes() == capacity
    assert result.numel() == capacity // 4 == 1
    return result


spec = importlib.util.spec_from_file_location("conv_storage_tail_metadata", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
old_cuda = torch.Tensor.is_cuda
old_current = torch.cuda.current_device
torch.Tensor.is_cuda = property(lambda _: True)
torch.cuda.current_device = lambda: None
rows = []
try:
    fixtures = [(tensor(a), tensor(b), False) for a, b in itertools.product((5, 6, 7), repeat=2)]
    fixtures += [(x, x, True) for x in (tensor(n) for n in (5, 6, 7))]
    for x, w, alias in fixtures:
        module._validate_inputs(x, w, 1, 1, 1, 1, 0, 0, 1, False, False)
        rows.append({"input_bytes": x.untyped_storage().nbytes(),
                     "weight_bytes": w.untyped_storage().nbytes(),
                     "complete_fp32_elements": 1, "same_object": alias,
                     "actual_validator_accepts": True})
finally:
    torch.Tensor.is_cuda = old_cuda
    torch.cuda.current_device = old_current

report = {"status": "PASS", "source": record(SOURCE), "script": record(Path(__file__).resolve()),
          "torch_version": torch.__version__, "cases": rows,
          "scope": "Actual CPU Tensor storage API and source admission function; only CUDA residency/current-device observations adapted.",
          "kernel_execution": False, "GPU_execution": False, "native_numerical_conformance": False}
(OUT / "storage-tail-metadata.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"PASS: {len(rows)} actual validator cases with 5/6/7-byte backing storage, including read-only aliasing")
