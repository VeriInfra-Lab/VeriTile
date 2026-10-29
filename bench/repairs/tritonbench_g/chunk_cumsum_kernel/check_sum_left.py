"""Paired actual-source CPU controls for the carry ADD-COMMUTE rewrite."""
import ast
import copy
import importlib.util
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import numpy as np
import torch
from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

HERE = Path(__file__).resolve().parent


def bind_pair(path, original, candidate):
    functions = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    old, new = copy.deepcopy(functions[original]), copy.deepcopy(functions[candidate])
    class Swap(ast.NodeTransformer):
        changes = 0
        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "b_z":
                assert isinstance(node.op, ast.Add)
                self.changes += 1
                return ast.Assign(targets=[node.target], value=ast.BinOp(
                    left=node.value, op=ast.Add(), right=ast.Name(id="b_z", ctx=ast.Load())))
            return node
    swap = Swap()
    old = swap.visit(old)
    assert swap.changes == 1
    old.name = new.name = "paired"
    old.decorator_list = new.decorator_list = []
    assert ast.dump(old) == ast.dump(new), "candidate changes more than the one carry addition"
    spec = importlib.util.spec_from_file_location("actual_pair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, original).fn, getattr(module, candidate)


def payloads(tensor):
    return tensor.view(torch.int32)


def check_pair(original, candidate, x, launch):
    before = x.clone()
    n = x.numel()
    a = torch.full((n+11,), -173.25, dtype=torch.float32)
    b = a.clone()
    za, zb = a[3:3+n].reshape(x.shape), b[3:3+n].reshape(x.shape)
    launch(original, x, za)
    launch(candidate, x, zb)
    assert torch.equal(payloads(a), payloads(b)), "full output backing differs"
    assert torch.equal(payloads(x), payloads(before)), "input changed"
    assert torch.all(a[:3] == -173.25) and torch.all(a[3+n:] == -173.25), "exterior changed"


CONFIGS = [(16,2), (32,4), (32,2), (64,8), (64,4)]


def main():
    original, candidate = bind_pair(HERE/"chunk_cumsum_kernel_repaired.py",
        "chunk_global_cumsum_scalar_kernel_repaired", "chunk_global_cumsum_scalar_kernel_sum_left")
    count = 0
    for bt, warps in CONFIGS:
        for t in (1, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129):
            storage = (torch.arange(6*t+11, dtype=torch.float32)%37-18)/8
            x = storage[3:3+6*t].reshape(2,3,t)
            check_pair(original, candidate, x,
                lambda k,s,z: k[(6,)](s,z,T=t,BT=bt,num_warps=warps))
            count += 1
    old_load = InterpreterBuilder.create_masked_load
    poisoned = [0]
    def poison(self, ptr, mask, other, *args, **kwargs):
        if other is None:
            poisoned[0] += 1
            other = TensorHandle(np.full(ptr.data.shape, np.nan, dtype=np.float32), ptr.dtype.element_ty)
        return old_load(self, ptr, mask, other, *args, **kwargs)
    InterpreterBuilder.create_masked_load = poison
    try:
        for bt, warps in CONFIGS:
            t = 2*bt+1
            x = torch.arange(t,dtype=torch.float32).reshape(1,1,t)
            check_pair(original, candidate, x,
                lambda k,s,z: k[(1,)](s,z,T=t,BT=bt,num_warps=warps))
        assert poisoned[0] > 0
    finally:
        InterpreterBuilder.create_masked_load = old_load
    print(f"PASS: only carry swap in actual candidate AST; {count} paired CPU controls")
    print("PASS: five NaN-poison final-tail pairs; exact full backing equality, input and exterior frames")
    print(f"torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; no GPU execution")


if __name__ == "__main__": main()
