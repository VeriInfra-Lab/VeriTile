"""Finite binary64 host sqrt model; read-only comparison with installed Python."""
from pathlib import Path
import ctypes
import hashlib
import json
import math
import platform
import random
import struct
import sys
import triton

PACKAGE = Path(__file__).resolve().parent

def rounded_quotient(a, d):
    q, r = divmod(a, d)
    return q + int(2 * r > d or (2 * r == d and q % 2 == 1))

def binary64_integer(n):
    shift = max(n.bit_length() - 1 - 52, 0)
    return rounded_quotient(n, 1 << shift) << shift

def sqrt_significand(n):
    if n == 0:
        return 0, 0
    represented = binary64_integer(n)
    exponent = (represented.bit_length() - 1) // 2
    shift = 52 - exponent
    scaled_square = represented << (2 * shift)
    lower = math.isqrt(scaled_square)
    midpoint_square = (2 * lower + 1) ** 2
    value = lower + int(4 * scaled_square > midpoint_square or
                        (4 * scaled_square == midpoint_square and lower % 2 == 1))
    return value, exponent

def model(n):
    mantissa, exponent = sqrt_significand(n)
    if n == 0:
        return {'sqrt_bits': 0, 'ceil_sqrt': 0, 'block': 0}
    denominator = 1 << (52 - exponent)
    ceil = (mantissa + denominator - 1) // denominator
    block = 1 << (ceil - 1).bit_length()
    if mantissa == 1 << 53:
        mantissa >>= 1
        exponent += 1
    bits = ((exponent + 1023) << 52) | (mantissa - (1 << 52))
    return {'sqrt_bits': bits, 'ceil_sqrt': ceil, 'block': block}

def actual(n):
    value = math.sqrt(n)
    ceil = math.ceil(value)
    return {'sqrt_bits': int.from_bytes(struct.pack('>d', value), 'big'),
            'ceil_sqrt': ceil, 'block': triton.next_power_of_2(ceil)}

def threshold(k):
    square = 1 << (2 * k)
    if k <= 25:
        return square
    if k == 26:
        return square + 1
    return square + 3 * (1 << (2 * k - 53)) - 1

rng = random.Random(20261016)
inputs = {0, 1, (1 << 63) - 1}
for k in range(32):
    for center in (1 << (2 * k), threshold(k)):
        inputs.update(n for n in range(max(0, center - 2050), center + 2051) if n < 1 << 63)
for k in range(1, 64):
    inputs.update(n for n in range(max(0, (1 << k) - 4), min(1 << 63, (1 << k) + 5)))
inputs.update(rng.randrange(1 << 63) for _ in range(100000))
for n in sorted(inputs):
    assert binary64_integer(n) == int(float(n)), ('integer conversion', n)
    assert model(n) == actual(n), ('sqrt/ceil/block', n, model(n), actual(n))
    assert actual(n)['block'] <= 1 << 20 if n <= 1 << 40 else actual(n)['block'] > 1 << 20
boundaries = []
for k in range(32):
    limit = threshold(k)
    assert actual(limit)['block'] == 1 << k
    assert actual(limit + 1)['block'] == 1 << (k + 1)
    boundaries.append({'k': k, 'maximum_n_for_block': limit,
                       'at': actual(limit), 'after': actual(limit + 1)})
path = PACKAGE / 'evidence/host-math-probe.json'
assert not path.exists()
result = {'kind': 'finite_host_math_model_native_comparison_not_acceptance',
          'status': 'passed', 'inputs_compared': len(inputs), 'boundaries': boundaries,
          'python_version': sys.version, 'platform': platform.platform(),
          'math_extension': math.__file__,
          'math_extension_sha256': hashlib.sha256(Path(math.__file__).read_bytes()).hexdigest(),
          'rounding_mode': ctypes.CDLL(None).fegetround(),
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'limitations': 'Finite sample/native check, not a proof of CPython or libc refinement. Pure model uses nearest-even binary64 conversion and sqrt; input domain is nonnegative signed64 numel. Zero has block0 and the original subsequent cdiv(0,0) raises.'}
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'inputs_compared': len(inputs),
                  'receipt': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
