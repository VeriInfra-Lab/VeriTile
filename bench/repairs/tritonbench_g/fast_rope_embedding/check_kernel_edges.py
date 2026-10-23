"""Kernel edge regressions and counterexamples; CPU/source replay, no GPU claim."""
import ctypes
import importlib.util
import json
from pathlib import Path
import struct
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fast_rope_fixtures', HERE / 'check_contract.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
    kernel, ids = f.f.f.f.f.cpu_kernel(f.SOURCE, '_rope_embedding')
    # These are actual Tensor arguments at the kernel boundary. No assertion
    # about which noncanonical strides a particular host reshape preserves.
    strides = (0, 245, 2**31 + 17, 2**62, 2**63 - 1)
    for stride in strides:
        for backward in (False, True):
            raw = torch.arange(253, dtype=torch.float32)
            q = torch.as_strided(raw, (1, 245), (stride, 1), storage_offset=3)
            assert q.is_contiguous() and q.stride(0) == stride
            old = raw.clone()
            initial = q.clone().view(1, 7, 35)
            cosine = torch.ones(1, 17) * 2
            sine = torch.ones(1, 17) * 3
            expected = initial.clone()
            a, b = initial[..., :17], initial[..., 17:34]
            expected[..., :17] = a * 2 + b * 3 if backward else a * 2 - b * 3
            expected[..., 17:34] = b * 2 - a * 3 if backward else b * 2 + a * 3
            ids[0] = 0
            for group in (1, 0):
                ids[1] = group
                kernel(f.Pointer(q), q.stride(0), f.Pointer(cosine), 17,
                       f.Pointer(sine), 17, 1, 35, 7, backward, 32)
            assert torch.equal(q.view(1, 7, 35), expected)
            assert torch.equal(q.view(1, 7, 35)[..., -1], initial[..., -1])
            assert torch.equal(raw[:3].view(torch.int32), old[:3].view(torch.int32))
            assert torch.equal(raw[248:].view(torch.int32), old[248:].view(torch.int32))
    ns, _ = f.host()
    # Metadata individually fits i32, but formed offsets do not.
    try:
        ns['_geometry'](1, 1, 2**30 + 1, 2)
    except ValueError as error:
        assert 'formed padded offsets' in str(error)
    else:
        raise AssertionError('overflowing source address admitted')
    assert 2 * (2**30) + 1 == 2**31 + 1
    forward = (2., 4.)  # (a,b)=(1,2), (c,s)=(2,0)
    backward = (forward[0] * 2, forward[1] * 2)
    assert backward == (4., 8.) and backward != (1., 2.)
    # A concrete obstruction to replacing genuine FMA with mul then sub.
    libm = ctypes.CDLL('libm.so.6')
    libm.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    libm.fmaf.restype = ctypes.c_float
    a = c = 1 + 2**-23
    b, s = 1., 1 + 2**-22
    fused = libm.fmaf(c, a, -fp32(s * b))
    unfused = fp32(fp32(c * a) - fp32(s * b))
    assert bits(fused) == 0x28800000 and bits(unfused) == 0
    assert bits(0. - 0.) == 0 and bits(-0.) == 0x80000000
    report = {
        'source_replay': 'actual extracted fast JIT with CPU memory adapter',
        'singleton_stride_forward_backward_cases': len(strides) * 2,
        'strides': strides, 'odd_tail_and_backing_frame': True,
        'overflow_guard_counterexample': {'row': 2**30, 'offset': 2**31 + 1},
        'nonunit_not_inverse': {'input': [1, 2], 'coefficients': [2, 0], 'roundtrip': backward},
        'fma_counterexample': {'a': hex(bits(a)), 'b': hex(bits(b)), 'c': hex(bits(c)), 's': hex(bits(s)),
                               'fused': hex(bits(fused)), 'unfused': hex(bits(unfused))},
        'source_sub_zero_vs_sign_neg': [hex(bits(0. - 0.)), hex(bits(-0.))],
        'gpu_execution': False, 'native_universal_equivalence_test_claim': False,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
