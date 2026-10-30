"""Bind the intended source pair and compare its actual CPU Triton executions."""
import ast
import copy
import importlib.util
import torch
import check_contract as contract

NAME = 'chunk_global_reversed_cumsum_scalar_kernel'
CANDIDATE = NAME + '_index_commuted'

def source_binding():
    tree = ast.parse(contract.SOURCE.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME)
    candidate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == CANDIDATE)
    normalized = copy.deepcopy(candidate)
    normalized.name = original.name
    normalized.decorator_list = copy.deepcopy(original.decorator_list)
    original_loop = next(n for n in original.body if isinstance(n, ast.For))
    candidate_loop = next(n for n in normalized.body if isinstance(n, ast.For))
    before = next(n for n in original_loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'p_s')
    after = next(n for n in candidate_loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'p_s')
    assert ast.unparse(after.value) == 's + stride_sb * (i_bh // H) + stride_sh * (i_bh % H) + times * stride_st'
    assert ast.dump(after.value) != ast.dump(before.value)
    after.value = copy.deepcopy(before.value)
    assert ast.dump(normalized) == ast.dump(original)
    lean = (contract.HERE / 'Kernels.lean').read_text()
    assert 'def indexCommuted' in lean
    assert '$(h.inputStrides.batch) * (i_bh // $(h.H))' in lean
    assert '$(h.inputStrides.head) * (i_bh % $(h.H))' in lean


def main():
    source_binding()
    spec = importlib.util.spec_from_file_location('scalar_variant_check', contract.SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = getattr(module, NAME).fn
    candidate = getattr(module, CANDIDATE)
    controls = contract.interpreter_checks(candidate)
    pairs = 0
    for bt, warps in [(16, 2), (32, 4), (32, 2), (64, 8), (64, 4)]:
        for t in [17, 65]:
            # Exceptional values belong to the accepted input contract; the rewrite
            # does not depend on cancellation, finite operands or reduction laws.
            values = torch.tensor([0., -0., 1., -2., 1e20, 1e-30, float('inf'), float('-inf'), float('nan')])
            backing = values.repeat((6*t+20)//len(values)+1)
            x = backing[3:3+6*t].reshape(2, 3, t)
            before = backing.view(torch.int32).clone()
            outputs = []
            for kernel in [original, candidate]:
                storage = torch.full((6*t+8,), -321., dtype=torch.float32)
                z = storage[4:-4].reshape(2, 3, t)
                kernel[(6,)](x, z, T=t, H=3, stride_sb=x.stride(0), stride_sh=x.stride(1), stride_st=1,
                    stride_ob=z.stride(0), stride_oh=z.stride(1), stride_ot=1, BT=bt, num_warps=warps)
                assert bool((storage[:4] == -321).all()) and bool((storage[-4:] == -321).all())
                outputs.append(storage.view(torch.int32))
            assert torch.equal(outputs[0], outputs[1])
            assert torch.equal(backing.view(torch.int32), before)
            pairs += 1
    print(f'PASS: exact source-pair binding, {controls} candidate suffix/tail controls and {pairs} bitwise FP execution pairs over all five configurations.')
    print('CPU Triton interpreter only; no native-to-IEEE conformance assertion or numerical experiment obligation.')

if __name__ == '__main__':
    main()
