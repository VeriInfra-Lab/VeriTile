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


CONFIGS = [(bt,w) for bt in (16,32,64) for w in (2,4,8)]


def launch(kernel, x, z, bt, warps):
    b,h,t,f = x.shape
    kernel[((f+31)//32,b*h)](x,z,H=h,T=t,S=f,BS=32,BT=bt,num_warps=warps,
        s_s_b=x.stride(0),s_s_h=x.stride(1),s_s_t=x.stride(2),s_s_d=x.stride(3),
        s_z_b=z.stride(0),s_z_h=z.stride(1),s_z_t=z.stride(2),s_z_d=z.stride(3))


def main():
    original, candidate = bind_pair(HERE/"chunk_cumsum_vector_repaired.py",
        "chunk_global_cumsum_vector_kernel", "chunk_global_cumsum_vector_kernel_sum_left")
    count = 0
    for bt, warps in CONFIGS:
        for t,f in [(1,1),(4,5),(bt-1,31),(bt,32),(bt+1,33),(2*bt+1,65)]:
            storage = (torch.arange(2*t*f+11,dtype=torch.float32)%37-18)/8
            x = storage[3:3+2*t*f].reshape(1,2,t,f)
            check_pair(original,candidate,x,lambda k,s,z: launch(k,s,z,bt,warps))
            count += 1
    print(f"PASS: only carry swap in actual candidate AST; {count} paired CPU controls across all nine configs")
    print("PASS: time/feature tails, exact full backing equality, input and exterior frames")
    print(f"torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; no GPU execution")


if __name__ == "__main__": main()
