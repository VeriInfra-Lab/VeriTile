"""Small source-precision controls; CPU only, not native GPU conformance."""
import json
import math
import struct
from pathlib import Path

import torch

from check_contract import Context, Tensor, environment


def bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def main():
    torch.set_num_threads(1)
    ns, events, _ = environment()
    x = Tensor(torch.tensor([[-0.0]], dtype=torch.float32))
    w = Tensor(torch.ones(1, dtype=torch.float32))
    b = Tensor(torch.zeros(1, dtype=torch.float32))
    ctx = Context()
    ns['LayerNorm'].forward(ctx, x, 1, w, b, 1e-5)
    saved_mean = ctx.saved[3].data.item()
    # For width one, deleting source `0 + a` leaves a negative-zero leaf.
    source_mean = bits(saved_mean)
    omitted_initial_add = bits(x.data.item())
    assert source_mean == 0x00000000
    assert omitted_initial_add == 0x80000000
    # A positive finite binary64 epsilon is admitted even when its conversion
    # underflows to FP32 zero. No native nonzero-denominator premise is valid.
    tiny = float.fromhex('0x0.0000000000001p-1022')
    assert tiny > 0 and math.isfinite(tiny)
    assert bits(tiny) == 0
    ctx = Context()
    out = ns['LayerNorm'].forward(ctx, x, 1, w, b, tiny)
    assert math.isinf(ctx.saved[4].data.item())
    assert math.isnan(out.data.item())
    result = {
        'kind': 'CPU_source_precision_controls',
        'source_initial_zero_mean_bits': f'{source_mean:08x}',
        'omitted_initial_add_mean_bits': f'{omitted_initial_add:08x}',
        'epsilon_binary64_bits': struct.pack('>d', tiny).hex(),
        'epsilon_fp32_bits': f'{bits(tiny):08x}',
        'tiny_epsilon_admitted': True,
        'singleton_output_is_nan': True,
        'gpu_execution': 'not_run',
        'scope': 'CPU source replay and representation counterexamples, not native lowering conformance',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
