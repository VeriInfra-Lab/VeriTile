"""Actual forward JIT edge replays and concrete branch/FMA witnesses; no GPU."""
import ctypes
import importlib.util
import json
from pathlib import Path
import struct
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rope_transform_fixtures', HERE / 'check_contract.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
    kernel, ids = f.f.cpu_kernel(f.SOURCE, '_triton_rope')
    cases = [(2, 1, 2, 3, 2), (3, 1, 3, 1, 6), (2, 3, 1, 5, 34), (1, 5, 7, 3, 10)]
    for batch, seq, hq, hk, dim in cases:
        half = dim // 2
        shapes = [(batch, seq, hq, dim), (batch, seq, hk, dim), (seq, half), (seq, half)]
        values, backing, old = [], [], []
        for shape in shapes:
            count = torch.empty(shape).numel()
            raw = torch.arange(count + 15, dtype=torch.float32) % 17 - 8
            values.append(raw[5:5 + count].reshape(shape))
            backing.append(raw)
            old.append(raw.clone())
        q, k, c, s = values
        expected = []
        for value in (q, k):
            a, b = value[..., :half], value[..., half:]
            cc, ss = c.view(1, seq, 1, half), s.view(1, seq, 1, half)
            expected.append(torch.cat((a * cc - b * ss, b * cc + a * ss), -1))
        pad = lambda n: 1 << (n - 1).bit_length()
        # Reverse completion order exercises disjoint row ownership.
        for row in reversed(range(batch * seq)):
            ids[0] = row
            kernel(f.Pointer(q), hq * dim, f.Pointer(k), hk * dim, f.Pointer(c), half,
                   f.Pointer(s), half, seq, batch, hq, hk, dim, pad(hq), pad(hk), pad(dim),
                   max(pad(hq), pad(hk)), False)
        for value, want in zip((q, k), expected):
            assert torch.equal(value.view(torch.int32), want.view(torch.int32))
        for value, raw, initial in zip(values, backing, old):
            assert torch.equal(raw[:5].view(torch.int32), initial[:5].view(torch.int32))
            assert torch.equal(raw[5 + value.numel():].view(torch.int32), initial[5 + value.numel():].view(torch.int32))
        assert all(torch.equal(raw.view(torch.int32), initial.view(torch.int32)) for raw, initial in zip(backing[2:], old[2:]))
    branch = {}
    for backward in (False, True):
        q, k = torch.tensor([1., 2.]), torch.tensor([1., 2.])
        c, s = torch.tensor([3.]), torch.tensor([4.])
        ids[0] = 0
        kernel(f.Pointer(q), 2, f.Pointer(k), 2, f.Pointer(c), 1, f.Pointer(s), 1,
               1, 1, 1, 1, 2, 1, 1, 2, 1, backward)
        assert torch.equal(q, k)
        branch['backward' if backward else 'forward'] = q.tolist()
    assert branch == {'forward': [-5., 10.], 'backward': [11., 2.]}
    libm = ctypes.CDLL('libm.so.6')
    libm.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    libm.fmaf.restype = ctypes.c_float
    a = c = 1 + 2**-23
    b, s = 1., 1 + 2**-22
    low = libm.fmaf(c, a, -fp32(s * b))
    low_split = fp32(fp32(c * a) - fp32(s * b))
    a = s = 1 + 2**-23
    b, c = 1., -(1 + 2**-22)
    high = libm.fmaf(s, a, fp32(c * b))
    high_split = fp32(fp32(c * b) + fp32(s * a))
    assert bits(low) == bits(high) == 0x28800000 and bits(low_split) == bits(high_split) == 0
    print(json.dumps({
        'source_replay': 'actual extracted forward JIT with CPU storage adapter',
        'layout_cases': cases, 'layout_case_count': len(cases),
        'full_q_k_outputs_coefficients_and_backing_frame': True,
        'branch_counterexample': branch,
        'nonunit_not_inverse': {'input': [1, 2], 'coefficients': [2, 0], 'roundtrip': [4, 8]},
        'formed_i32_counterexample': {'rows': 2**30 + 1, 'dimension': 2, 'last_offset': 2**31 + 1},
        'fma_counterexamples': {
            'low': {'a': '0x3f800001', 'b': '0x3f800000', 'c': '0x3f800001', 's': '0x3f800002',
                    'fused': hex(bits(low)), 'split': hex(bits(low_split))},
            'high': {'a': '0x3f800001', 'b': '0x3f800000', 'c': '0xbf800002', 's': '0x3f800001',
                     'fused': hex(bits(high)), 'split': hex(bits(high_split))}},
        'gpu_execution': False, 'native_universal_equivalence_test_claim': False,
    }, indent=2))


if __name__ == '__main__':
    main()
