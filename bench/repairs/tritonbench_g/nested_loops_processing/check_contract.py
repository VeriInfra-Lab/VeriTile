"""Integer-only source/storage replay with mocked CUDA allocation and printing."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "nested_loops_processing_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/nested_loops_processing/nested_loops_processing.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load(HERE.parent / "add_example/check_contract.py", "nested_storage_helpers")


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "a_ptrs":
            node.id = "a_offsets"
        if node.id == "c_ptrs":
            node.id = "c_offsets"
        return node

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("stride_m", "stride_n", "n_elements", "unique"):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "tl.debug_barrier":
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            return node.func.value
        name = ast.unparse(node.func)
        if name == "tl.minimum":
            return node.args[0]
        if name in ("tl.load", "tl.store"):
            node.keywords = []
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Name) and node.left.id in ("in_ptr", "out_ptr"):
            return node.right
        return node


def source_checks():
    old = Normalize().visit(shared.function(ORIGINAL, "nested3"))
    new = Normalize().visit(shared.function(SOURCE, "nested3_repaired"))
    new.name = old.name
    new.args.args.pop()
    assert ast.dump(old) == ast.dump(new)
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    assert "FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert "Correct" not in (HERE / "FPEquiv.lean").read_text()


class Tensor(shared.TensorFacade):
    def reshape(self, shape):
        result = Tensor(self.data.reshape(shape), str(self.device))
        if "data_ptr" in self.__dict__:
            result.data_ptr = self.data_ptr
        return result

    def stride(self, dimension):
        return self.data.stride(dimension)


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


def raw(tensor):
    return torch.empty(0, dtype=torch.int32).set_(tensor.untyped_storage(), 0,
        (tensor.untyped_storage().nbytes() // 4,), (1,))


def oracle(rows, cols):
    size = rows * cols
    result = [0] * size
    for t in range(24):
        outer, middle, inner = t // 12, t % 12 // 6, t % 6 // 3
        shift = 14 * outer if t % 3 == 0 else (
            14 * outer + 6 * middle + 2 if t % 3 == 1 else
            14 * outer + 6 * middle + 2 * inner + 4)
        for r in range(2):
            for c in range(2):
                dst, src = r * cols + c + 2 * t, r * cols + c + shift
                if dst < size:
                    result[dst] = ((src + 2**31) % 2**32) - 2**31 if src < size else 0
    return result


def checks():
    module = load(SOURCE, "nested_contract_review")
    allocations, printed, events = [], [], []
    programs = [0]
    pointer_fault = [None]

    def allocate(shape, dtype, values=None):
        size = 1
        for dimension in shape:
            size *= dimension
        backing = torch.full((size + 8,), -123456789, dtype=dtype)
        view = Tensor(backing[4:4 + size].reshape(shape))
        view.data.copy_(torch.zeros(shape, dtype=dtype) if values is None else values.reshape(shape))
        allocations.append((backing, view))
        if pointer_fault[0] is not None:
            view.data_ptr = lambda: pointer_fault[0]
        return view

    def read(ptr, mask, other):
        assert bool((ptr.offset >= 0).all()) and bool((ptr.offset < ptr.tensor.numel()).all())
        result = torch.full(ptr.offset.shape, other, dtype=torch.int32)
        result[mask] = raw(ptr.tensor)[ptr.tensor.storage_offset() + ptr.offset[mask]]
        events.append("load")
        return result

    def write(ptr, values, mask):
        assert bool((ptr.offset >= 0).all()) and bool((ptr.offset < ptr.tensor.numel()).all())
        indices = ptr.offset[mask].flatten().tolist()
        payloads = values[mask].flatten().tolist()
        assert len(indices) == len(set(indices)), "one source vector store races internally"
        # Reverse lane order independently in every source store.
        for address, value in reversed(list(zip(indices, payloads))):
            raw(ptr.tensor)[ptr.tensor.storage_offset() + address] = value
        events.append("store")

    node = shared.function(SOURCE, "nested3_repaired")
    node.decorator_list = []
    namespace = {"tl": SimpleNamespace(
        int64=torch.int64, cast=lambda x, dtype: x,
        arange=lambda start, end: torch.arange(start, end, dtype=torch.int64),
        minimum=lambda x, y: torch.minimum(x, torch.tensor(y, dtype=torch.int64)),
        load=read, store=write, debug_barrier=lambda: events.append("barrier"))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(SOURCE), "exec"), namespace)

    class Launch:
        def __getitem__(self, grid):
            assert grid == (1,)

            def run(x, y, sm, sn, count):
                programs[0] += 1
                namespace["nested3_repaired"](Pointer(x.data), Pointer(y.data), sm, sn, count)
            return run

    module.nested3_repaired = Launch()
    module.torch = SimpleNamespace(
        int32=torch.int32,
        arange=lambda start, end, device, dtype: allocate((end,), dtype, torch.arange(start, end, dtype=dtype)),
        zeros=lambda shape, device, dtype: allocate(shape, dtype))
    module.print = lambda output: printed.append(output)
    admitted = 0
    for rows, cols in [(0, 0), (0, 7), (8, 0), (0, 2**32), (1, 1), (2, 1), (49, 1),
                       (1, 2), (2, 2), (3, 3), (4, 4), (8, 8), (16, 16),
                       (1, 47), (1, 48), (1, 49), (2, 47), (2, 48), (2, 49), (17, 129)]:
        allocations.clear()
        printed.clear()
        events.clear()
        returned = module.wrapper_nested3_repaired(rows, cols)
        assert returned is None and len(printed) == 1
        assert printed[0] is allocations[1][1]
        assert printed[0].shape == (rows, cols) and printed[0].dtype == torch.int32
        assert printed[0].data.flatten().tolist() == oracle(rows, cols)
        assert allocations[0][1].data.flatten().tolist() == list(range(rows * cols))
        assert all(bool((b[:4] == -123456789).all()) and bool((b[-4:] == -123456789).all()) for b, _ in allocations)
        if rows * cols:
            assert events.count("load") == 14 and events.count("store") == events.count("barrier") == 24
            assert all(events[i + 1] == "barrier" for i, event in enumerate(events) if event == "store")
        else:
            assert events == []
        admitted += 1
    rejected = 0
    for rows, cols in [(True, 1), (1, False), (1., 1), (1, torch.tensor(1)), (-1, 8),
                       (8, -1), (2**63, 0), (0, 2**63), (2**32, 2**32)]:
        allocations.clear()
        printed.clear()
        try:
            module.wrapper_nested3_repaired(rows, cols)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("unsupported dimensions admitted")
        assert allocations == [] and printed == []
        rejected += 1
    for address in [1, -4, 2**64 - 4]:
        pointer_fault[0] = address
        printed.clear()
        events.clear()
        try:
            module.wrapper_nested3_repaired(2, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generated buffer admitted")
        assert printed == [] and events == []
        rejected += 1
    return rejected, admitted, programs[0]


if __name__ == "__main__":
    source_checks()
    rejected, admitted, programs = checks()
    print(f"Passed source binding, {rejected} guard rejections, {admitted} admissions, {programs} program executions.")
    print("CPU int32 storage replay with mocked CUDA and print; no native execution claim.")
