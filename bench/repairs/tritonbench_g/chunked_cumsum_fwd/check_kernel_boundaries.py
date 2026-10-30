"""Kernel-revision boundary evidence; CPU AST replay, not GPU conformance."""

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import struct
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chunked_contract_helpers", HERE / "check_contract.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
BLOCKS = (1, 2, 4, 8, 16, 32, 64)


def bits64(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def rounded_shift(value, shift):
    if shift <= 0:
        return value << -shift
    quotient, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + int(remainder > half or (remainder == half and quotient % 2 == 1))


def narrow64(word):
    """Integer-only RN-even oracle, including the x86 quiet-NaN payload rule."""
    sign = (word >> 63) << 31
    exponent, fraction = (word >> 52) & 2047, word & ((1 << 52) - 1)
    if exponent == 2047:
        if not fraction:
            return sign | 0x7F800000
        return sign | 0x7FC00000 | ((fraction >> 29) & 0x3FFFFF)
    significand = fraction if exponent == 0 else (1 << 52) + fraction
    scale = -1074 if exponent == 0 else exponent - 1075
    if not significand:
        return sign
    power = significand.bit_length() - 1 + scale
    if power < -126:
        return sign | rounded_shift(significand, -149 - scale)
    rounded = rounded_shift(significand, significand.bit_length() - 24)
    if rounded == 1 << 24:
        power += 1
        rounded >>= 1
    if power > 127:
        return sign | 0x7F800000
    return sign | ((power + 127) << 23) | (rounded - (1 << 23))


def abi_checks():
    values = [0.0, -0.0, float("inf"), -float("inf"),
              2.0 ** -150, -2.0 ** -150, 3 * 2.0 ** -150, -3 * 2.0 ** -150,
              2.0 ** -126, 2.0 ** -126 - 2.0 ** -150,
              1 + 2.0 ** -24, 1 + 3 * 2.0 ** -24,
              2.0 ** 128 - 2.0 ** 103, 2.0 ** 128 - 2.0 ** 104]
    words = [bits64(v) for v in values]
    words += [0x7FF0000000000001, 0x7FF8000000000001, 0x7FF7FFFFFFFFFFFF,
              0x7FFFFFFFFFFFFFFF, 0xFFF0000000000001, 0xFFF8123456789ABC]
    rng = random.Random(20261020)
    words += [rng.getrandbits(64) for _ in range(4096)]
    for word in words:
        value = struct.unpack("<d", struct.pack("<Q", word))[0]
        actual = bits32(ctypes.c_float(value).value)
        assert actual == narrow64(word), (hex(word), hex(actual), hex(narrow64(word)))
    return {"cases": len(words), "seed": 20261020,
            "scope": "integer RN-even oracle versus local C double-to-float cast",
            "input_words_sha256": hashlib.sha256(b"".join(struct.pack("<Q", w) for w in words)).hexdigest(),
            "fixtures": [{"input_binary64": f"{w:016x}", "output_fp32": f"{narrow64(w):08x}"}
                         for w in words[:20]]}


def scalar_reference(dt, rate, chunk, bias, softplus, limits):
    """Logical-coordinate preparation and an inclusive scalar prefix."""
    batches, sequence, heads = dt.shape
    chunks = -(-sequence // chunk)
    prepared = torch.zeros((batches, heads, chunks, chunk), dtype=torch.float32)
    total = torch.empty_like(prepared)
    lower, upper = (torch.tensor(x, dtype=torch.float32) for x in limits)
    for b in range(batches):
        for h in range(heads):
            for k in range(chunks):
                prefix = None
                for j in range(chunk):
                    time = k * chunk + j
                    if time < sequence:
                        value = dt[b, time, h].clone()
                        if bias is not None:
                            value = value + bias[h]
                        if softplus and bool(value <= 20):
                            value = torch.log(1 + torch.exp(value))
                        value = torch.fmin(torch.fmax(value, lower), upper)
                        prepared[b, h, k, j] = value
                    product = prepared[b, h, k, j] * rate[h]
                    prefix = product if prefix is None else prefix + product
                    total[b, h, k, j] = prefix
    return total, prepared


def fixtures():
    dt = torch.tensor([[[1., -2., 3.], [4., 5., -6.]]])
    rate = torch.tensor([1., -1., 2.])
    for name, limits in [
        ("lower_nan", (float("nan"), 1.)),
        ("upper_nan", (0., float("nan"))),
        ("reversed", (2., -1.)),
        ("positive_infinity", (float("inf"), float("inf"))),
        ("negative_infinity", (0., -float("inf"))),
        ("ties_even_and_odd", (1 + 2.0 ** -24, 1 + 3 * 2.0 ** -24)),
        ("negative_zero_upper", (-1., -0.)),
        ("underflow_bounds", (-2.0 ** -150, 2.0 ** -150)),
    ]:
        yield name, dt, rate, 3, None, False, limits
    yield "infinite_rate_padding", torch.ones(1, 1, 3), torch.tensor([float("inf"), -float("inf"), float("nan")]), 3, None, False, (0., float("inf"))
    yield "nan_bias_padding", torch.ones(1, 1, 1), torch.ones(1), 3, torch.tensor([float("nan")]), True, (float("nan"), float("nan"))
    yield "eager_softplus", torch.tensor([[[1000.], [-1000.], [20.], [21.], [float("nan")]]]), torch.ones(1), 3, None, True, (0., float("inf"))
    yield "special_dt_bias_softplus", torch.tensor([[[float("inf"), -float("inf"), float("nan")], [1., 2., 3.]]]), rate, 3, torch.tensor([1., -1., 2.]), True, (-float("inf"), float("inf"))
    backing = torch.arange(2 * 7 * 6, dtype=torch.float32).reshape(2, 7, 6)
    yield "aliased_strided_inputs", backing[:, :, ::2], backing[0, 0, ::2], 4, backing[1, 1, ::2], False, (0., 256.)
    yield "zero_stride_inputs", torch.ones(1, 7, 1).expand(2, 7, 3), torch.ones(1).expand(3), 4, torch.ones(1).expand(3), True, (0., float("inf"))
    yield "boolean_chunk", dt, rate, True, None, False, (0., float("inf"))
    for shape in [(0, 7, 3), (2, 0, 3), (1, 4, 0)]:
        yield "empty_" + "_".join(map(str, shape)), torch.empty(shape), torch.ones(shape[2]), 3, None, False, (float("nan"), -float("inf"))


def replay(case, block):
    name, dt, rate, chunk, bias, softplus, limits = case
    module = helpers.load(helpers.SOURCE, "chunked_boundary_source")
    kernel, pid = helpers.interpreter()
    launches, allocations, stores, exponentials = [], [], [0], []
    # Triton 3.7.1 defaults to PropagateNan.NONE. Offline SM80 evidence uses
    # maxnumf/minnumf, whereas the older finite-only helper used propagating
    # torch.maximum/minimum. This is a CPU proxy, not a GPU NaN-payload claim.
    kernel.__globals__["tl"].minimum = lambda x, y: torch.fmin(x, torch.tensor(y, dtype=torch.float32))
    kernel.__globals__["tl"].maximum = lambda x, y: torch.fmax(x, torch.tensor(y, dtype=torch.float32))
    original_exp = kernel.__globals__["tl"].exp

    def observed_exp(values):
        result = original_exp(values)
        exponentials.append(bool(torch.isinf(result).any()))
        return result

    kernel.__globals__["tl"].exp = observed_exp

    def empty(*shape, device, dtype):
        count = math.prod(shape)
        storage = torch.full((count + 16,), -12345., dtype=dtype)
        tensor = helpers.Tensor(storage[8:8 + count].reshape(shape), str(device))
        allocations.append((tensor, storage))
        return tensor

    x, a = helpers.Tensor(dt), helpers.Tensor(rate)
    b = None if bias is None else helpers.Tensor(bias)
    inputs = [dt, rate] + ([] if bias is None else [bias])
    before = [helpers.raw_storage(t).view(torch.uint8).clone() for t in inputs]

    class Launch:
        def __getitem__(self, grid):
            def run(actual_dt, actual_rate, actual_bias, out, total, *args, **kwargs):
                launches.append((actual_dt, actual_rate, actual_bias, out, total))
                assert len(allocations) == 2
                assert (actual_dt is x and actual_rate is a and actual_bias is b)
                assert bits64(args[4]) == bits64(limits[0])
                assert bits64(args[5]) == bits64(limits[1])
                assert type(args[3]) is int and args[3] == int(chunk)
                config = dict(kwargs, BLOCK_SIZE_H=block)
                dims = grid(config)
                assert dims == (dt.shape[0], -(-dt.shape[1] // int(chunk)), -(-dt.shape[2] // block))
                for batch in reversed(range(dims[0])):
                    for part in reversed(range(dims[1])):
                        for head in reversed(range(dims[2])):
                            pid[:] = [batch, part, head]
                            kernel(helpers.Pointer(dt), helpers.Pointer(rate), None if bias is None else helpers.Pointer(bias),
                                   helpers.Pointer(out.data), helpers.Pointer(total.data), *args, **config)
                            stores[0] += 1
            return run

    module.torch = SimpleNamespace(Tensor=helpers.Tensor, strided=torch.strided, float32=torch.float32,
                                  empty=empty, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module._chunk_cumsum_fwd_kernel = Launch()
    total, out = module._chunk_cumsum_fwd(x, a, chunk, dt_bias=b, dt_softplus=softplus, dt_limit=limits)
    assert len(launches) == 1
    assert allocations[0][0] is out and allocations[1][0] is total
    assert launches[0][3] is out and launches[0][4] is total
    expected_total, expected_out = scalar_reference(dt, rate, int(chunk), bias, softplus, limits)
    torch.testing.assert_close(out.data, expected_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(total.data, expected_total, rtol=2e-6, atol=5e-7, equal_nan=True)
    for tensor, snapshot in zip(inputs, before):
        assert torch.equal(helpers.raw_storage(tensor).view(torch.uint8), snapshot)
    for _, storage in allocations:
        assert bool((storage[:8] == -12345).all() and (storage[-8:] == -12345).all())
    if name == "infinite_rate_padding":
        assert bool((out.data[..., 1:] == 0).all())
        assert bool(torch.isnan(total.data[..., 1:]).all())
        assert bool(torch.isinf(total.data[0, :2, 0, 0]).all())
    if name == "nan_bias_padding":
        assert bool(torch.isnan(out.data[..., 0]).all()) and bool((out.data[..., 1:] == 0).all())
    if name == "eager_softplus":
        assert any(exponentials) and out.data[0, 0, 0, 0].item() == 1000.
    if name == "reversed":
        assert bool((out.data[..., :2] == -1).all()) and bool((out.data[..., 2] == 0).all())
    return {"name": name, "block_heads": block, "programs": stores[0],
            "lower_binary64": f"{bits64(limits[0]):016x}", "upper_binary64": f"{bits64(limits[1]):016x}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    helpers.source_checks()
    abi = abi_checks()
    cases = list(fixtures())
    results = [replay(case, block) for case in cases for block in BLOCKS]
    value = torch.tensor(1.)
    upper = torch.tensor(float("nan"))
    assert torch.fmin(value, upper).item() == 1.
    assert torch.isnan(torch.where(value < upper, value, upper))
    report = {"source_sha256": hashlib.sha256(helpers.SOURCE.read_bytes()).hexdigest(),
              "scope": "CPU execution of actual source AST using fmax/fmin proxies for observed PropagateNan.NONE, and an independent scalar reference; no GPU/FP32 native scan or NaN-payload claim",
              "abi": abi, "fixtures": len(cases), "launches": len(results),
              "programs": sum(item["programs"] for item in results), "results": results,
              "counterexamples": ["ordered compare/where selects a NaN upper operand instead of the finite minimum", "zero padded prepared values times infinite rate produce NaN", "softplus eagerly computes overflowing exp before selecting finite large input"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(cases)} fixtures, {len(results)} independent launches, {report['programs']} programs, all seven head blocks.")
    print(f"PASS: {abi['cases']} binary64-to-FP32 C ABI cases, including ties, underflow, overflow, signed zero and NaN payloads.")
    print(report["scope"])


if __name__ == "__main__":
    main()
