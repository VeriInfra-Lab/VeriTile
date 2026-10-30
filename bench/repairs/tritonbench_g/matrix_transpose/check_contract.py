"""Targeted contracts and both actual kernel variants; no CUDA execution."""

import ast
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[2] / "tritonbench_g/matrix_transpose/matrix_transpose.py"
SOURCE = HERE / "matrix_transpose_repaired.py"


def function(path, name):
    return next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def compile_function(node, namespace):
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module([node], [])), str(SOURCE), "exec"), namespace)


class Tensor:
    def __init__(self, shape, data=None):
        self.shape = tuple(shape)
        self.dtype = torch.float16
        self.device = "cuda"
        self.data = data

    def stride(self):
        return (max(self.shape[1], 1), 1)


class Pointer:
    def __init__(self, tensor, offsets=0):
        self.tensor = tensor
        self.offsets = offsets

    def __add__(self, offsets):
        return Pointer(self.tensor, self.offsets + offsets)


class Language:
    tensor = int32 = constexpr = object

    @staticmethod
    def arange(start, stop):
        return torch.arange(start, stop, dtype=torch.int64)

    @staticmethod
    def load(pointer):
        return pointer.tensor.data.flatten()[pointer.offsets]

    @staticmethod
    def trans(values):
        return values.t()

    @staticmethod
    def store(pointer, values):
        pointer.tensor.data.flatten()[pointer.offsets] = values


class Runtime:
    float16 = torch.float16

    def __init__(self, body, execute=False, fail=None):
        self.body = body
        self.execute = execute
        self.fail = fail
        self.events = []

    def randn(self, shape, dtype, device):
        self.events.append("randn")
        assert dtype == self.float16 and device == "cuda"
        if self.fail == "randn":
            raise RuntimeError("randn")
        data = None
        if self.execute:
            # The full 256x256 case covers every FP16 bit pattern exactly once.
            bits = torch.arange(shape[0] * shape[1], dtype=torch.int32).to(torch.int16)
            data = bits.view(torch.float16).reshape(shape)
        self.matrix = Tensor(shape, data)
        return self.matrix

    def zeros(self, shape, dtype, device):
        self.events.append("zeros")
        assert dtype == self.float16 and device == "cuda"
        if self.fail == "zeros":
            raise RuntimeError("zeros")
        self.output = Tensor(shape, torch.zeros(shape, dtype=dtype) if self.execute else None)
        return self.output

    def __getitem__(self, grid):
        assert grid == (1,)
        return self.launch

    def launch(self, matrix, output, sx, sy, ox, oy, m, d):
        self.events.append("launch")
        assert matrix is self.matrix and output is self.output
        assert matrix is not output
        assert matrix.shape == (m, d) and output.shape == (d, m)
        assert (sx, sy, ox, oy) == (d, 1, m, 1)
        if self.fail == "launch":
            raise RuntimeError("launch")
        if self.execute:
            self.body(Pointer(matrix), Pointer(output), sx, sy, ox, oy, m, d)


def instantiate(body, execute=False, fail=None):
    runtime = Runtime(body, execute, fail)
    namespace = {"torch": runtime, "kernel": runtime}
    compile_function(function(SOURCE, "wrapper"), namespace)
    return namespace["wrapper"], runtime


def main():
    original = function(ORIGINAL, "kernel")
    candidate = function(SOURCE, "kernel")
    assert ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False)
    # The candidate is exactly the existing Lean index-commuted variant.
    commuted = function(SOURCE, "kernel_index_commuted")
    expected_commuted = function(ORIGINAL, "kernel")
    expected_commuted.name = "kernel_index_commuted"
    product = expected_commuted.body[2].value.left.right
    assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
    product.left, product.right = product.right, product.left
    commuted.body = commuted.body[1:]  # Remove its explanatory docstring.
    assert ast.dump(expected_commuted, include_attributes=False) == ast.dump(commuted, include_attributes=False)
    wrapper = function(SOURCE, "wrapper")
    guard_expressions = [
        "type(size_m) is not int or type(d_head) is not int",
        "size_m <= 0 or d_head <= 0",
        "size_m & (size_m - 1) or d_head & (d_head - 1)",
        "size_m * d_head > 2**20",
    ]
    assert len([node for node in wrapper.body if isinstance(node, ast.If)]) == 4
    for node, expression in zip(wrapper.body, guard_expressions):
        assert isinstance(node, ast.If)
        assert ast.dump(node.test) == ast.dump(ast.parse(expression, mode="eval").body)
        assert len(node.body) == 1 and isinstance(node.body[0], ast.Raise)
    old_body = function(ORIGINAL, "wrapper").body
    assert [ast.dump(node) for node in wrapper.body[4:]] == [ast.dump(node) for node in old_body]

    kernel_namespace = {"tl": Language}
    compile_function(candidate, kernel_namespace)
    compile_function(commuted, kernel_namespace)
    body = kernel_namespace["kernel"]
    commuted_body = kernel_namespace["kernel_index_commuted"]
    admitted = [(2**a, 2**b) for a in range(21) for b in range(21) if a + b <= 20]
    assert len(admitted) == 231
    for shape in admitted:
        call, runtime = instantiate(body)
        result = call(*shape)
        assert result is runtime.output
        assert result.shape == shape[::-1]
        assert runtime.events == ["randn", "zeros", "launch"]

    class IntegerSubclass(int):
        pass

    rejected = [
        (True, 1), (1, False), (1.0, 1), (1, "2"), (IntegerSubclass(1), 1),
        (None, 1), (0, 1), (1, 0), (-1, 4), (4, -2), (3, 4), (4, 3),
        (1, 2**21), (2**21, 1), (2**10, 2**11), (2**1000, 1),
    ]
    for shape in rejected:
        call, runtime = instantiate(body)
        expected = TypeError if any(type(n) is not int for n in shape) else ValueError
        try:
            call(*shape)
        except expected:
            pass
        else:
            raise AssertionError(f"accepted unsupported shape {shape}")
        assert runtime.events == []

    executed = [(1, 1), (1, 64), (64, 1), (2, 4), (16, 16), (32, 64), (256, 256)]
    for shape in executed:
        results = []
        for variant in (body, commuted_body):
            call, runtime = instantiate(variant, execute=True)
            result = call(*shape)
            expected = runtime.matrix.data.view(torch.int16).t().contiguous()
            assert torch.equal(result.data.view(torch.int16), expected)
            assert result is runtime.output
            results.append(result.data.view(torch.int16))
        assert torch.equal(*results)
    for failure, events in [
        ("randn", ["randn"]),
        ("zeros", ["randn", "zeros"]),
        ("launch", ["randn", "zeros", "launch"]),
    ]:
        call, runtime = instantiate(body, fail=failure)
        try:
            call(2, 4)
        except RuntimeError as exc:
            assert str(exc) == failure
        else:
            raise AssertionError("primitive failure was swallowed")
        assert runtime.events == events

    # Reader entries are independent; the shared source imports no proof entry.
    kernels = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in kernels and ".Correct" not in kernels and ".FPEquiv" not in kernels
    assert ".FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert ".Correct" not in (HERE / "FPEquiv.lean").read_text()
    for entry in ("Correct.lean", "FPEquiv.lean"):
        text = (HERE / entry).read_text()
        assert "runWrapper" not in text and "wrapper_correct" not in text
        assert ".Proofs.Host" not in text and ".HostCorrect" not in text
    real = (HERE / "Proofs/KernelReal.lean").read_text()
    fp = (HERE / "Proofs/KernelFP.lean").read_text()
    assert ".Proofs.FP" not in real and ".HostCorrect" not in real and ".FullScope" not in real
    assert ".HostCorrect" not in fp and ".FullScope" not in fp
    print("PASS: unchanged kernel and admitted wrapper AST; 231 supported shapes;")
    print("16 pre-allocation rejections; both kernel variants at 7 shapes covering all FP16 words;")
    print("3 propagated primitive failures; independent kernel-only proof entries. No GPU execution.")


if __name__ == "__main__":
    main()
