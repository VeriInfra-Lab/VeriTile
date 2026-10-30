"""Selected backward JIT layout regressions; CPU adapter and scalar witnesses."""
import ctypes
import importlib.util
import json
from pathlib import Path
import struct
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('backward_fixtures', HERE / 'check_contract.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def main():
    kernel, ids = f.f.f.cpu_kernel(f.SOURCE, '_triton_rope')
    layouts = [((11, 2), (13, 3)), ((0, 2), (13, 0)),
               ((11, 0), (0, 3)), ((0, 0), (0, 0)),
               ((2, 9), (3, 11)), ((11, 2), (11, 2))]
    cases = []
    for layout_index, (cs, ss) in enumerate(layouts):
        for hq, hk in ((3, 1), (0, 3), (3, 0)):
            batch, sequence, dim, half = 2, 3, 7, 3
            gradients, raws, entries = [], [], []
            for heads in (hq, hk):
                count = batch * sequence * heads * dim
                raw = torch.arange(count + 13, dtype=torch.float32) - 37
                gradients.append(raw[4:4 + count].reshape(batch, sequence, heads, dim))
                raws.append(raw)
                entries.append(raw.clone())
            tables = []
            for strides in (cs, ss):
                raw = torch.arange(151, dtype=torch.float32) % 7 - 3
                tables.append(torch.as_strided(raw, (sequence + 2, half), strides, storage_offset=5))
            if layout_index == len(layouts) - 1:
                tables[1] = tables[0]
            old_tables = [f.storage(t).clone() for t in tables]
            expected = []
            for q in gradients:
                value = q.clone()
                a, b = q[..., :half], q[..., half:2 * half]
                c, s = (t[:sequence].view(1, sequence, 1, half) for t in tables)
                value[..., :half] = a * c + b * s
                value[..., half:2 * half] = b * c - a * s
                expected.append(value)
            for row in reversed(range(batch * sequence)):
                ids[0] = row
                kernel(f.Pointer(gradients[0]), hq * dim, f.Pointer(gradients[1]), hk * dim,
                       f.Pointer(tables[0]), *cs, f.Pointer(tables[1]), *ss,
                       sequence, batch, hq, hk, dim,
                       1 << (max(hq, 1) - 1).bit_length(),
                       1 << (max(hk, 1) - 1).bit_length(), 8, 4, True)
            for value, want, raw, old in zip(gradients, expected, raws, entries):
                assert torch.equal(value.view(torch.int32), want.view(torch.int32))
                assert torch.equal(raw[:4].view(torch.int32), old[:4].view(torch.int32))
                assert torch.equal(raw[4 + value.numel():].view(torch.int32),
                                   old[4 + value.numel():].view(torch.int32))
            for t, old in zip(tables, old_tables):
                assert torch.equal(f.storage(t).view(torch.int32), old.view(torch.int32))
            cases.append({'cos_strides': cs, 'sin_strides': ss, 'q_heads': hq, 'k_heads': hk,
                          'tables_alias': layout_index == len(layouts) - 1})

    branch = {}
    for backward in (False, True):
        q, k = torch.tensor([1., 2.]), torch.tensor([1., 2.])
        c, s = torch.tensor([3.]), torch.tensor([4.])
        ids[0] = 0
        kernel(f.Pointer(q), 2, f.Pointer(k), 2, f.Pointer(c), 1, 1,
               f.Pointer(s), 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, backward)
        assert torch.equal(q, k)
        branch['backward' if backward else 'forward'] = q.tolist()
    assert branch == {'forward': [-5., 10.], 'backward': [11., 2.]}

    # This scalar libc witness only distinguishes two graphs; it is not a
    # premise of the native Lean theorem or a GPU conformance experiment.
    libm = ctypes.CDLL('libm.so.6')
    libm.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    libm.fmaf.restype = ctypes.c_float
    a, b, c, s = 1., 1 + 2**-23, 1 + 2**-23, 1 + 2**-22
    fused = libm.fmaf(c, b, -fp32(s * a))
    unfused = fp32(fp32(c * b) - fp32(s * a))
    assert bits(fused) == 0x28800000 and bits(unfused) == 0
    print(json.dumps({
        'source_replay': 'actual extracted backward JIT with CPU memory adapter',
        'layout_cases': cases, 'layout_case_count': len(cases),
        'odd_tail_all_outputs_and_full_backing_frame': True,
        'branch_counterexample': branch,
        'nonunit_not_inverse': {'input': [1, 2], 'coefficients': [2, 0], 'roundtrip': [4, 8]},
        'formed_i32_counterexample': {'rows': 2**30 + 1, 'head_dim': 2, 'last_offset': 2**31 + 1},
        'high_fma_counterexample': {'a': hex(bits(a)), 'b': hex(bits(b)), 'c': hex(bits(c)),
                                    's': hex(bits(s)), 'fused': hex(bits(fused)), 'unfused': hex(bits(unfused))},
        'gpu_execution': False, 'native_universal_equivalence_test_claim': False,
    }, indent=2))


if __name__ == '__main__':
    main()
