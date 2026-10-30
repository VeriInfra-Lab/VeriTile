"""Execute both complete JIT sources in Triton's actual CPU interpreter."""
import ast
import copy
import importlib.util
import os
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
PATH = HERE / "embedding_triton_kernel_repaired.py"
tree = ast.parse(PATH.read_text())
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
original = functions["embedding_kernel_repaired"]
candidate = copy.deepcopy(functions["embedding_kernel_index_commuted"])
assert ast.unparse(original.body[0]) == "start_n = tl.program_id(0).to(tl.int64) * BLOCK_N"
assert ast.unparse(candidate.body[0]) == "start_n = BLOCK_N * tl.program_id(0).to(tl.int64)"
candidate.name = original.name
candidate.body[0] = copy.deepcopy(original.body[0])
assert ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False)

spec = importlib.util.spec_from_file_location("embedding_repair_interpreter", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
kernels = (module.embedding_kernel_repaired, module.embedding_kernel_index_commuted)
words = torch.tensor([0, -2147483648, 1, 2139095040, -8388608,
                      2143289345, 2141266757, -1], dtype=torch.int32)
launches = 0
cases = 0


def case(ids, weight, first, last):
    global launches, cases
    cases += 1
    n, hidden = len(ids), weight.shape[1]
    expected = torch.zeros(n, hidden, dtype=torch.float32)
    for row, token in enumerate(ids.tolist()):
        if first <= token < last:
            expected[row] = weight[token-first]
    before_ids = ids.clone()
    before_weight = weight.view(torch.int32).clone()
    results = []
    for kernel in kernels:
        backing = torch.full((n+2, hidden), 37., dtype=torch.float32)
        output = backing[1:-1]
        kernel[((n+63)//64, 1, 1)](
            weight, ids, output, first, last, hidden, hidden, n,
            hiden_size=hidden, BLOCK_DMODEL=1 << (hidden-1).bit_length(),
            BLOCK_N=64, BLOCK_NN=1, num_warps=1, num_stages=1)
        launches += 1
        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
        assert torch.all(backing[[0, -1]] == 37.)
        assert torch.equal(ids, before_ids)
        assert torch.equal(weight.view(torch.int32), before_weight)
        results.append(output.view(torch.int32).clone())
    assert torch.equal(*results)


for hidden in (1, 3, 8, 33):
    backing = words.repeat((5*hidden+len(words)-1)//len(words))[:5*hidden].view(torch.float32).reshape(5, hidden)
    for n in (1, 65):
        ids = (torch.arange(n+2, dtype=torch.int32) % 7 - 3)[1:-1]
        case(ids, backing[1:4], -2, 1)
case(torch.tensor([-2147483648, 2147483647], dtype=torch.int32),
     words[:3].view(torch.float32).reshape(1, 3), 2147483647, 2**31)
case(torch.tensor([-2147483648, 0, 2147483647], dtype=torch.int32),
     torch.empty(0, 3), 0, 0)
print(f"Embedding exact candidate AST check passed; {cases} cases, {launches} actual CPU Triton interpreter launches.")
print("Both full kernels preserve payload bits, input storage and exterior output frames; no GPU execution.")
