"""CPU source/storage/Philox controls with mocked CUDA; no native conformance claim."""
import ast
import importlib.util
import struct
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "multinomial_sampling_repaired.py"
ORIGINAL = HERE.parents[2] / "review-checkpoints/2026-10-14/next-value-repair/multinomial-sampling/native-original/multinomial_sampling.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load(HERE.parent / "add_example/check_contract.py", "multinomial_storage_helpers")
function = shared.function


def raw_storage(tensor):
    return torch.empty(0, dtype=tensor.dtype).set_(
        tensor.untyped_storage(), 0,
        (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,))


def source_checks():
    class RemovePointerClamp(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if ast.unparse(node.func) == "tl.minimum":
                assert ast.unparse(node.args[0]) == "valid_pos"
                assert ast.unparse(node.args[1]) == "num_tokens - 1"
                return node.args[0]
            return node
    assert ast.dump(RemovePointerClamp().visit(function(SOURCE, "_multinomial_sampling_kernel"))) == ast.dump(
        function(ORIGINAL, "_multinomial_sampling_kernel"))
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    assert "FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert "Correct" not in (HERE / "FPEquiv.lean").read_text()
    for name in ("Host", "Native", "Numerics", "Contract", "Guarded", "Bits"):
        text = (HERE / f"Proofs/{name}.lean").read_text()
        assert "import bench.repairs.tritonbench_g.multinomial_sampling.Correct" not in text
        assert "import bench.repairs.tritonbench_g.multinomial_sampling.FPEquiv" not in text


class Tensor(shared.TensorFacade):
    clones = []

    def __getitem__(self, index):
        return Tensor(self.data[index], str(self.device))

    def dim(self):
        return self.data.dim()

    def data_ptr(self):
        return self.data.data_ptr()

    def size(self):
        return self.data.size()

    def stride(self, dim=None):
        return self.data.stride() if dim is None else self.data.stride(dim)

    def detach(self):
        return self

    def cpu(self):
        return self.data

    def is_conj(self):
        return self.data.is_conj()

    def expand_as(self, other):
        return Tensor(self.data.expand_as(other.data), str(self.device))

    def clone(self):
        backing = torch.full((self.numel() + 8,), -123456789, dtype=self.dtype)
        result = Tensor(backing[4:4 + self.numel()].reshape(self.shape), str(self.device))
        result.data.copy_(self.data)
        Tensor.clones.append(backing)
        return result


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


def philox(seed, offset):
    mask = 2**32 - 1
    c, key = [offset & mask, 0, 0, 0], [seed & mask, (seed >> 32) & mask]
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [((p1 >> 32) ^ c[1] ^ key[0]) & mask, p1 & mask,
             ((p0 >> 32) ^ c[3] ^ key[1]) & mask, p0 & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return c[0]


SCALE = struct.unpack("<f", struct.pack("<I", 0x2fffffff))[0]


def sample(seed, offset):
    word = philox(seed, offset)
    folded = (2**32 - 1) - word if word >= 2**31 else word
    return float(torch.tensor(folded, dtype=torch.float32) * SCALE)


def checks():
    module = load(SOURCE, "multinomial_contract_review")
    pid, programs, launches, rng_calls = [0], [0], [], []
    forced = [None]
    current_device, launch_devices, fail_launch = [0], [], [False]
    class DeviceContext:
        def __init__(self, device): self.device = torch.device(device)
        def __enter__(self):
            self.previous = current_device[0]
            current_device[0] = self.device.index
        def __exit__(self, *exc): current_device[0] = self.previous

    def read(ptr, mask, other=0):
        offsets, masks = torch.broadcast_tensors(torch.as_tensor(ptr.offset), mask)
        result = torch.full(offsets.shape, other, dtype=ptr.tensor.dtype)
        active = offsets[masks]
        assert bool((active >= 0).all()) and bool((active < ptr.tensor.numel()).all())
        result[masks] = raw_storage(ptr.tensor)[ptr.tensor.storage_offset() + active]
        return result

    def write(ptr, values, mask):
        active = ptr.offset[mask]
        assert bool((active >= 0).all()) and bool((active < ptr.tensor.numel()).all())
        raw_storage(ptr.tensor)[ptr.tensor.storage_offset() + active] = values[mask]

    def rand(seeds, offsets):
        assert seeds.dtype == torch.int64 and offsets.dtype == torch.int32
        rng_calls.extend(zip(seeds.tolist(), offsets.tolist()))
        return torch.tensor([sample(s, o) if forced[0] is None else forced[0]
                             for s, o in zip(seeds.tolist(), offsets.tolist())], dtype=torch.float32)

    node = function(SOURCE, "_multinomial_sampling_kernel")
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    namespace = {"tl": SimpleNamespace(
        int32=torch.int32, float32=torch.float32,
        program_id=lambda _: pid[0], arange=torch.arange, load=read, store=write,
        rand=rand, zeros=torch.zeros, cumsum=torch.cumsum,
        minimum=lambda x, y: torch.minimum(x, torch.tensor(y, dtype=torch.int64)),
        max=lambda x, dim: torch.max(x, dim).values, sum=torch.sum,
        argmax=torch.argmax, where=torch.where)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)

    class Launch:
        def __getitem__(self, grid):
            def run(scores, seeds, offsets, indices, output, **config):
                assert current_device[0] == scores.device.index
                launch_devices.append(current_device[0])
                if fail_launch[0]: raise RuntimeError("mock submission error")
                launches.append((grid, config))
                assert config.pop("num_warps") == 8
                assert config["BLOCK"] == 8 and config["BLOCK_N"] == 128
                for program in reversed(range(grid[0])):
                    pid[0] = program
                    programs[0] += 1
                    namespace["_multinomial_sampling_kernel"](
                        *(Pointer(x.data) for x in (scores, seeds, offsets, indices, output)), **config)
            return run

    module._multinomial_sampling_kernel = Launch()
    module.torch = SimpleNamespace(
        Tensor=Tensor, float32=torch.float32, int64=torch.int64, strided=torch.strided,
        cuda=SimpleNamespace(device=DeviceContext),
        arange=lambda n, device: Tensor(torch.arange(n, dtype=torch.int64), str(device)))
    admitted = 0
    cases = [(0, 1, False), (3, 1, False), (3, 1, True), (0, 2, False),
             (1, 2, False), (7, 3, True), (8, 127, False), (9, 128, True),
             (17, 129, False), (9, 255, True), (2, 256, True), (3, 257, False)]
    for batches, tokens, supplied in cases:
        data = torch.zeros((batches, tokens), dtype=torch.float32)
        for r in range(batches):
            if tokens == 1:
                data[r, 0] = 1
            else:
                data[r, (r * 17) % tokens] += .5
                data[r, tokens - 1] += .5
        scores = Tensor(data)
        labels = Tensor(torch.arange(batches * tokens, dtype=torch.int64).reshape(batches, tokens) * -7 - 13)
        seeds = Tensor(torch.tensor([[-2**63, -1, 0, 1, 2**63 - 1][r % 5] for r in range(batches)], dtype=torch.int64))
        offsets = Tensor(torch.tensor([[0, 2**31, 2**32 + 7, -1, -2**63][r % 5] for r in range(batches)], dtype=torch.int64))
        if tokens == 129 or (tokens == 1 and supplied):
            for tensor in (scores, labels, seeds, offsets): tensor.device = torch.device("cuda:1")
        before = [x.data.clone() for x in (scores, labels, seeds, offsets)]
        launches.clear()
        Tensor.clones.clear()
        result = module.multinomial_sampling(scores, seeds if tokens != 1 else object(),
                                             offsets if tokens != 1 else object(), labels if supplied else None)
        assert result.shape == ((batches, 1) if tokens == 1 else (batches,))
        expected = []
        for r in range(batches):
            u = Fraction(sample(int(seeds.data[r]), int(offsets.data[r])))
            chosen, cumulative = 0, Fraction(0)
            for t in range(tokens):
                previous = cumulative
                cumulative += Fraction(float(data[r, t]))
                if previous < u <= cumulative:
                    chosen = t
            expected.append(int(labels.data[r, chosen]) if supplied else chosen)
        assert result.data.flatten().tolist() == expected
        for x, old in zip((scores, labels, seeds, offsets), before):
            torch.testing.assert_close(x.data, old, rtol=0, atol=0)
        assert all(bool((x[:4] == -123456789).all()) and bool((x[-4:] == -123456789).all()) for x in Tensor.clones)
        assert len(launches) == (0 if tokens == 1 else 1)
        assert current_device[0] == 0
        admitted += 1

    # Isolated source branch controls at zero and an exact upper interval edge.
    for value, expected in [(0., 31), (.5, 31), (.75, 99)]:
        forced[0] = value
        result = module.multinomial_sampling(Tensor(torch.tensor([[.5, .5]])),
            Tensor(torch.tensor([0])), Tensor(torch.tensor([0])), Tensor(torch.tensor([[31, 99]])))
        assert result.data.tolist() == [expected]
        assert current_device[0] == 0
        admitted += 1
    forced[0] = None

    # Exact normalized FP32 input with a canonical (unforced) Philox sample.
    # The second rounded prefix crosses u, while its exact Real value is below u.
    u = sample(0, 0)
    assert u == 0.7980929017066956
    weights = torch.tensor([[u - 2.**-24, 3 * 2.**-26,
                             1 - u + 2.**-26]], dtype=torch.float32)
    assert sum(Fraction(float(x)) for x in weights[0]) == 1
    assert Fraction(float(weights[0, 0])) + Fraction(float(weights[0, 1])) < Fraction(u)
    result = module.multinomial_sampling(Tensor(weights), Tensor(torch.tensor([0])),
        Tensor(torch.tensor([0])), Tensor(torch.tensor([[11, 22, 33]])))
    assert result.data.tolist() == [22]
    assert Fraction(u) <= sum(Fraction(float(x)) for x in weights[0])
    # Independent exact inverse CDF selects position 2 (label 33).
    prefix, exact_position = Fraction(0), None
    for position, value in enumerate(weights[0]):
        before, prefix = prefix, prefix + Fraction(float(value))
        if before < Fraction(u) <= prefix:
            exact_position = position
    assert exact_position == 2
    admitted += 1

    # Zero is a distinguished discrete boundary: the source retains its first
    # label even when that label has zero mathematical probability mass.
    forced[0] = 0.
    result = module.multinomial_sampling(Tensor(torch.tensor([[0., 1.]])),
        Tensor(torch.tensor([0])), Tensor(torch.tensor([0])), Tensor(torch.tensor([[31, 99]])))
    assert result.data.tolist() == [31]
    forced[0] = None
    admitted += 1

    good = Tensor(torch.tensor([[.5, .5]]))
    seed = Tensor(torch.tensor([1]))
    offset = Tensor(torch.tensor([0]))
    labels = Tensor(torch.tensor([[4, 5]]))
    bad = [([], seed, offset, None), (Tensor(torch.tensor([[.5, .5]], dtype=torch.float64)), seed, offset, None),
           (Tensor(good.data, "cpu", False), seed, offset, None),
           (Tensor(torch.ones(2)), seed, offset, None), (Tensor(torch.empty(1, 0)), seed, offset, None),
           (Tensor(torch.full((2, 2), .5).t()), seed, offset, None),
           (Tensor(torch._neg_view(torch.tensor([[-.5, -.5]]))), seed, offset, None),
           (Tensor(torch.tensor([[float("nan"), 1.]])), seed, offset, None),
           (Tensor(torch.tensor([[float("inf"), 0.]])), seed, offset, None),
           (Tensor(torch.tensor([[-.5, 1.5]])), seed, offset, None),
           (Tensor(torch.tensor([[.25, .5]])), seed, offset, None),
           (Tensor(torch.tensor([[.1, .2, .7]])), seed, offset, None),
           (good, Tensor(torch.tensor([1.], dtype=torch.float32)), offset, None),
           (good, Tensor(torch.tensor([1, 2])), offset, None),
           (good, Tensor(torch.tensor([1]), "cuda:1"), offset, None),
           (good, seed, Tensor(torch.tensor([[0]])), None),
           (good, seed, offset, Tensor(torch.tensor([[1, 2]], dtype=torch.int32))),
           (good, seed, offset, Tensor(torch.tensor([[1]]))),
           (good, seed, offset, Tensor(labels.data, "cuda:1")),
           (good, seed, offset, Tensor(torch.tensor([[1, 2], [3, 4]]).t()))]

    class Misaligned(Tensor):
        def data_ptr(self):
            return super().data_ptr() + 1

    class OverflowPointer(Tensor):
        def data_ptr(self):
            return 2**64 - 8

    class HugeStride(Tensor):
        def stride(self, dim=None):
            strides = (2**30, 1)
            return strides if dim is None else strides[dim]

    bad.extend([(Misaligned(good.data), seed, offset, None),
                (good, Misaligned(seed.data), offset, None),
                (OverflowPointer(good.data), seed, offset, None),
                (HugeStride(good.data), seed, offset, None)])
    rejected = 0
    for args in bad:
        launches.clear()
        try:
            module.multinomial_sampling(*args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("unsupported input admitted")
        assert launches == []
        rejected += 1
    assert philox(0, 0) == 0x6627E8D5
    assert any(o < 0 for _, o in rng_calls)
    assert float(torch.tensor([.1, .2, .7]).sum()) == 1.
    assert sum(Fraction(float(x)) for x in torch.tensor([.1, .2, .7])) != 1
    exact_normalized = torch.tensor([[2.**-25, .5, .5-2.**-25]], dtype=torch.float32)
    module._probability_rows(Tensor(exact_normalized))
    assert Fraction(float(exact_normalized[0, 0] + exact_normalized[0, 1])) != (
        Fraction(float(exact_normalized[0, 0])) + Fraction(float(exact_normalized[0, 1])))
    mismatch = [Tensor(torch.tensor([[.5, .5]]), "cuda:1"),
                Tensor(torch.tensor([1]), "cuda:1"), Tensor(torch.tensor([0]), "cuda:1")]
    fail_launch[0] = True
    try: module.multinomial_sampling(*mismatch)
    except RuntimeError: pass
    else: raise AssertionError("expected mock submission failure")
    finally: fail_launch[0] = False
    assert 1 in launch_devices and current_device[0] == 0
    return rejected, admitted, programs[0]


if __name__ == "__main__":
    source_checks()
    rejected, admitted, programs = checks()
    print(f"Passed source binding, {rejected} guard rejections, {admitted} admissions, {programs} program executions.")
    print("CPU storage/Philox replay with mocked CUDA; normalized rounding counterexample and zero boundary passed. No native compiler/device conformance claim.")
