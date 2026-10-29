"""Exact index-candidate AST relation and CPU complete-storage comparisons."""
import ast
import copy
from pathlib import Path

import torch

import check_contract as check

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "triton_softmax_index_commuted.py"


def main():
    source = check.function(check.SOURCE, "softmax_kernel")
    candidate = check.function(CANDIDATE, "softmax_kernel")

    class Commute(ast.NodeTransformer):
        count = 0

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name)
                    and node.left.id == "row_idx" and isinstance(node.right, ast.Name)
                    and node.right.id == "input_row_stride"):
                node.left, node.right = node.right, node.left
                self.count += 1
            return node

    transform = Commute()
    assert ast.dump(transform.visit(copy.deepcopy(source))) == ast.dump(candidate)
    assert transform.count == 1
    original_path = check.SOURCE
    source_kernel = check.Kernel()
    check.SOURCE = CANDIDATE
    candidate_kernel = check.Kernel()
    check.SOURCE = original_path
    comparisons = 0
    for width in (1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2048, 2049, 32769):
        for exceptional in (False, True):
            backing = torch.full((2 * width + 9,), 47., dtype=torch.float32)
            x = backing[3:3 + 2 * width].reshape(2, width)
            if exceptional:
                words = torch.tensor([0xff800000, 0, 0x80000000, 0x7f800000,
                                      0x7fc00001, 0x7fa00003, 1, 0x7f7fffff], dtype=torch.int64)
                x.view(torch.int32).copy_(words.to(torch.int32).repeat((2 * width + 7) // 8)[:2 * width].reshape(x.shape))
            else:
                x.copy_(torch.linspace(-5, 5, 2 * width).reshape(x.shape))
            before = bytes(backing.untyped_storage())
            outputs = []
            for kernel in (source_kernel, candidate_kernel):
                out_backing = torch.full((2 * width + 11,), 73., dtype=torch.float32)
                y = out_backing[4:4 + 2 * width].reshape(x.shape)
                raw_before = out_backing.view(torch.int32).clone()
                for kernel.pid in range(2):
                    kernel.body(check.Pointer(y), check.Pointer(x), width, 1, width, 1,
                                width, check.next_power(width))
                assert torch.equal(out_backing.view(torch.int32)[:4], raw_before[:4])
                assert torch.equal(out_backing.view(torch.int32)[4 + 2 * width:], raw_before[4 + 2 * width:])
                assert bytes(backing.untyped_storage()) == before
                outputs.append(out_backing.view(torch.int32).clone())
                if not exceptional:
                    torch.testing.assert_close(y, torch.softmax(x, dim=1), rtol=3e-6, atol=2e-7)
            assert torch.equal(*outputs)
            comparisons += 1
    # A global row maximum must handle these admitted, well-defined -inf/finite mixtures.
    for width in (65, 1025, 32769):
        x = torch.zeros(1, width)
        x[:, :width - 1] = -float("inf")
        y = torch.full_like(x, 73.)
        source_kernel.pid = 0
        source_kernel.body(check.Pointer(y), check.Pointer(x), width, 1, width, 1,
                           width, check.next_power(width))
        expected = torch.zeros_like(x)
        expected[0, -1] = 1
        assert torch.equal(y, expected)
    print(f"PASS: exact one-product AST rewrite; {comparisons} complete-storage bit comparisons.")
    print("Finite/exceptional words, offsets, tail widths, input/exterior frame and three -inf/finite controls; CPU only.")


if __name__ == "__main__":
    main()
