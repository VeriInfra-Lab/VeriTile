"""Whole selected source CPU controls; no GPU or instruction conformance claim."""
import os
os.environ['TRITON_INTERPRET'] = '1'
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'Whole001/attention_fwd_triton2_repaired.py'
spec = importlib.util.spec_from_file_location('attention_fwd2_cpu', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CudaFacade(torch.Tensor):
    @property
    def device(self):
        return torch.device('cuda:0')


allocations = []


def empty_like(q, dtype):
    tensor = torch.full_like(q, 7, dtype=dtype)
    allocations.append(tensor)
    return tensor


def empty(shape, device, dtype):
    assert device == torch.device('cuda:0')
    tensor = torch.full(shape, 37, dtype=dtype).as_subclass(CudaFacade)
    allocations.append(tensor)
    return tensor


module.torch = SimpleNamespace(Tensor=CudaFacade, strided=torch.strided,
    float32=torch.float32, bfloat16=torch.bfloat16, empty=empty, empty_like=empty_like,
    cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
records = []
for B, H, N, alias in [(1, 1, 1, True), (1, 2, 65, False), (2, 3, 129, False),
                        (1, 1, 127, False), (0, 2, 64, False), (2, 0, 64, False), (2, 3, 0, False)]:
    shapes = [(B, H, N, 96)] * 3 + [(B, H, (N + 127) // 128), (B, H, (N + 63) // 64)]
    tensors, backings = [], []
    for shape, value in zip(shapes, [0, 0, 2, 1, 1]):
        count = 1
        for dim in shape:
            count *= dim
        backing = torch.full((5 + count + 17,), -13, dtype=torch.float32)
        view = backing[5:5 + count].reshape(shape)
        view.fill_(value)
        tensors.append(view.as_subclass(CudaFacade))
        backings.append(backing)
    if alias:
        tensors[1] = tensors[0]
        tensors[4] = tensors[3]
    before = [b.clone() for b in backings]
    allocations.clear()
    result = module.forward(*tensors)
    assert result is allocations[0] and len(allocations) == 2
    assert result.dtype == torch.bfloat16 and tuple(result.shape) == shapes[0]
    assert torch.equal(result.as_subclass(torch.Tensor), torch.full(shapes[0], 2, dtype=torch.bfloat16))
    assert all(torch.equal(a, b) for a, b in zip(before, backings))
    assert torch.equal(allocations[1].as_subclass(torch.Tensor), torch.full((B, H, N), 37, dtype=torch.float32))
    row = {'shape': list(shapes[0]), 'empty': not B * H * N, 'readonly_alias': alias,
           'actual_returned_tensor': True, 'output_dtype': 'bfloat16',
           'all_output_words': 'exact BF16 2', 'all_input_backing_frames': True,
           'input_prefix_elements': 5, 'input_suffix_elements': 17, 'unused_M_frame': True}
    records.append(row)
    print('PASS', row, flush=True)
    report = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'method': 'Actual source and wrapper in Triton CPU interpreter on real Tensor storage; exact-class/device/context/allocation facade',
              'GPU_execution': False, 'instruction_conformance_experiment': False,
              'reference': 'Q=K=0, QS=KS=1, V=2; complete output is exactly 2',
              'cases': records, 'complete_nonempty_cases': sum(not x['empty'] for x in records),
              'empty_cases': sum(x['empty'] for x in records)}
    (OUT / 'interpreter-report.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS', len(records), 'complete-source and empty controls', flush=True)
