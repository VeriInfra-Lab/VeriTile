"""Exact triangular-comparison binding and paired CPU storage execution."""
import ast
import copy
import torch
import check_contract as contract

NAME = 'chunk_global_reversed_cumsum_vector_kernel'
CANDIDATE = NAME + '_compare_right'

def source_binding():
    original = contract.function(contract.SOURCE, NAME)
    candidate = contract.function(contract.SOURCE, CANDIDATE)
    normalized = copy.deepcopy(candidate)
    normalized.name = original.name
    normalized.decorator_list = copy.deepcopy(original.decorator_list)
    before = next(n for n in original.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'm_s')
    after = next(n for n in normalized.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'm_s')
    assert ast.unparse(after.value) == 'tl.where(o_i[None, :] >= o_i[:, None], 1.0, 0.0)'
    assert ast.dump(after.value) != ast.dump(before.value)
    after.value = copy.deepcopy(before.value)
    assert ast.dump(normalized) == ast.dump(original)
    lean = (contract.HERE / 'Kernels.lean').read_text()
    assert 'def compareRight' in lean and 'o_i[None, :] >= o_i[:, None]' in lean


def main():
    torch.set_num_threads(2)
    source_binding()
    original, pid_original = contract.interpreter()
    candidate, pid_candidate = contract.interpreter(CANDIDATE)
    pairs = 0
    for bt in [16, 32, 64]:
        for warps in [2, 4, 8]:
            for t, features in [(1, 1), (17, 33), (33, 31), (65, 33)]:
                size = 2*t*features
                values = torch.tensor([0., -0., 1., -2., 1e20, 1e-30, float('inf'), float('-inf'), float('nan')])
                backing = values.repeat((size+20)//len(values)+1)
                x = backing[3:3+size].reshape(2, 1, t, features)
                before = backing.view(torch.int32).clone()
                outputs = []
                for kernel, pid in [(original, pid_original), (candidate, pid_candidate)]:
                    storage = torch.full((size+8,), -321., dtype=torch.float32)
                    z = storage[4:-4].reshape(2, 1, t, features)
                    for row in [0, 1]:
                        for block in range((features+31)//32):
                            pid[:] = [block, row]
                            kernel(contract.Pointer(x), contract.Pointer(z), t*features, features, 1, t, features, bt, 32)
                    assert bool((storage[:4] == -321).all()) and bool((storage[-4:] == -321).all())
                    outputs.append(storage.view(torch.int32))
                assert torch.equal(outputs[0], outputs[1])
                assert torch.equal(backing.view(torch.int32), before)
                pairs += 1
    print(f'PASS: exact source-pair binding and {pairs} bitwise FP storage pairs, including both tails and all nine configurations.')
    print('CPU AST execution with Torch dot only; no native-to-IEEE conformance assertion or numerical experiment obligation.')

if __name__ == '__main__':
    main()
