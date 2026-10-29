"""Bind the real candidate to the four integer rewrites and replay both kernels.
This CPU check supplies source/coverage evidence, not GPU arithmetic conformance.
"""
import ast
import importlib.util
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("softmax_optimize_cpu", HERE / "check_contract.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def check_ast():
    source = ast.parse((HERE / "softmax_optimize_repaired.py").read_text())
    candidate = ast.parse((HERE / "softmax_optimize_index_commuted.py").read_text())
    source_fn = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                     and n.name == "softmax_kernel_online_v2_repaired")
    candidate_fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                        and n.name == "softmax_kernel_online_v2_index_commuted")
    candidate_fn.name = source_fn.name

    class Commute(ast.NodeTransformer):
        count = 0

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Name)
                    and node.left.id == "pid_m" and isinstance(node.right, ast.Name)
                    and node.right.id == "N"):
                node.left, node.right = node.right, node.left
                self.count += 1
            return node

    transform = Commute()
    assert ast.dump(transform.visit(source_fn)) == ast.dump(candidate_fn)
    assert transform.count == 4
    source_helper = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                         and n.name == "prev_multiple_of")
    candidate_helper = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                            and n.name == "prev_multiple_of")
    assert ast.dump(source_helper) == ast.dump(candidate_helper)


def main():
    check_ast()
    source, _, _ = h.load(HERE / "softmax_optimize_repaired.py")
    candidate, _, _ = h.load(HERE / "softmax_optimize_index_commuted.py")
    words = torch.tensor([0, -2147483648, 1065353216, -1082130432,
                          2139095040, -8388608, 2143289345, -4194302], dtype=torch.int32)
    cases = 0
    for rows, cols in [(1, 1), (2, 3), (3, 7), (2, 33), (1, 4095),
                       (2, 4096), (2, 4097), (1, 8193)]:
        for exceptional in (False, True):
            count = rows * cols
            raw = (words.repeat((count + 7) // 8)[:count].clone() if exceptional
                   else torch.linspace(-4, 4, count).view(torch.int32))
            input_storage = torch.full((count + 8,), 1065353216, dtype=torch.int32)
            input_storage[3:count + 3] = raw
            before = input_storage.clone()
            inp = h.Tensor(input_storage.view(torch.float32)[3:count + 3].view(rows, cols))
            stores = []
            tile = min(4096, 1 << (cols.bit_length() - 1))
            for ns, name in [(source, "softmax_kernel_online_v2_repaired"),
                             (candidate, "softmax_kernel_online_v2_index_commuted")]:
                output = torch.full((count + 10,), 305419896, dtype=torch.int32)
                view = h.Tensor(output.view(torch.float32)[4:count + 4].view(rows, cols))
                ns[name][(rows, 1, 1)](view, inp, rows, cols, tile,
                                      enable_fp_fusion=True, num_warps=4, num_stages=3)
                assert torch.equal(output[:4], torch.full((4,), 305419896, dtype=torch.int32))
                assert torch.equal(output[count + 4:], torch.full((6,), 305419896, dtype=torch.int32))
                stores.append(output)
            assert torch.equal(stores[0], stores[1])
            assert torch.equal(input_storage, before)
            cases += 1
    print(f"passed: exact four-product AST rewrite; {cases} source/candidate full-storage bit comparisons")
    print("finite/exceptional words, both output loops, 4096 boundary, input/output offsets and exterior frame; CPU only")


if __name__ == "__main__":
    main()
