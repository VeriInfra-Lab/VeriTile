"""Unchanged complete source on CPU; CUDA metadata facade, no GPU claim."""
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
SOURCE = OUT.parent / 'attention_kernel_repaired.py'
spec = importlib.util.spec_from_file_location('attention_cpu_complete', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CudaFacade(torch.Tensor):
    @property
    def device(self):
        return torch.device('cuda:0')


module.torch = SimpleNamespace(Tensor=CudaFacade, float16=torch.float16,
    bfloat16=torch.bfloat16, strided=torch.strided,
    cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
records = []
for dtype in [torch.float16, torch.bfloat16]:
    for B, H, N, D, BM, BN in [(1, 1, 1, 16, 16, 16), (1, 2, 17, 16, 16, 16),
                              (2, 2, 65, 32, 32, 64), (1, 2, 129, 16, 16, 32),
                              (0, 2, 16, 16, 16, 16), (2, 0, 16, 16, 16, 16),
                              (2, 2, 0, 16, 16, 16)]:
        shape = (B, H, N, D)
        buffers, tensors, layouts = [], [], []
        for j, initial in enumerate([0.0, 0.0, 2.0, 7.0]):
            head = N*D+16*(j+1)
            batch = H*head+16*(j+1)
            strides = (batch, head, D, 1)
            end = 0 if not B*H*N else (B-1)*batch+(H-1)*head+(N-1)*D+D
            backing = torch.full((5+end+17,), initial, dtype=dtype)
            tensors.append(torch.as_strided(backing, shape, strides, storage_offset=5).as_subclass(CudaFacade))
            buffers.append(backing)
            layouts.append(strides)
        bias = torch.zeros((B, H, N, 128), dtype=dtype).as_subclass(CudaFacade)
        originals = [x.clone() for x in buffers]
        original_bias = bias.as_subclass(torch.Tensor).clone()
        result = module._attention_rel_h_rel_w_kernel_aligned_device(
            *tensors[:3], bias, 0.125, tensors[3], BM, BN, 4, 1)
        assert result is None
        expected = originals[3].clone()
        torch.as_strided(expected, shape, layouts[3], storage_offset=5).fill_(2)
        assert torch.equal(buffers[3], expected), (dtype, shape)
        assert all(torch.equal(buffers[j], originals[j]) for j in range(3))
        assert torch.equal(bias.as_subclass(torch.Tensor), original_bias)
        records.append({'dtype': str(dtype), 'shape': list(shape), 'blocks': [BM, BN],
                        'all_logical_outputs': 'exact constant 2', 'full_backing_frame': True,
                        'input_frame': True, 'empty': not B*H*N, 'result_is_None': True})
report = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'backend': 'actual Triton CPU interpreter; exact-class CUDA metadata facade',
          'GPU_execution': False, 'runtime_profile_conformance_claim': False,
          'reference': 'Q=K=bias=0, V=2 gives every admitted output exactly 2; entire gapped output backing and all read backings checked',
          'cases': records, 'case_count': len(records)}
(OUT / 'interpreter-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, sort_keys=True))
