"""Check the complete candidate AST without importing Triton or running a device."""
import ast
from pathlib import Path


def check():
    here = Path(__file__).resolve().parent
    def function(name, symbol):
        return next(n for n in ast.parse((here / name).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == symbol)
    original = function("chunk_delta_fwd_static_dot.py", "chunk_delta_rule_fwd_kernel_h")
    candidate = function("candidate_kernels.py", "chunk_delta_rule_fwd_kernel_h_address_fold")
    candidate.name = original.name

    class FoldAddress(ast.NodeTransformer):
        count = 0
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.BinOp)
                    and isinstance(node.left.op, ast.Mult)
                    and isinstance(node.left.left, ast.Name) and node.left.left.id in ("i_bh", "i_t")
                    and isinstance(node.left.right, ast.Name) and node.left.right.id == "K"
                    and isinstance(node.right, ast.Name) and node.right.id == "V"):
                self.count += 1
                return ast.BinOp(left=node.left.left, op=ast.Mult(),
                                 right=ast.BinOp(left=node.left.right, op=ast.Mult(), right=node.right))
            return node
    rewrite = FoldAddress()
    assert ast.dump(rewrite.visit(original)) == ast.dump(candidate)
    assert rewrite.count == 3
    print("PASS: complete function AST differs only at three state/history integer address products.")


if __name__ == "__main__":
    check()
