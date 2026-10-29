"""Check the full persistent source pair without importing Torch or Triton."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check_pair(original_text=None, candidate_text=None):
    original = ast.parse((HERE / 'matmul_persistent_triton_scalar_cast.py').read_text()
                         if original_text is None else original_text)
    candidate = ast.parse((HERE / 'matmul_persistent_triton_reordered.py').read_text()
                          if candidate_text is None else candidate_text)
    before = ast.parse('start_m = pid_m * BLOCK_SIZE_M').body[0]
    after = ast.parse('start_m = BLOCK_SIZE_M * pid_m').body[0]
    dump = lambda node: ast.dump(node, include_attributes=False)

    class Restore(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if dump(node) == dump(after):
                self.count += 1
                return before
            return self.generic_visit(node)

    restore = Restore()
    restored = restore.visit(candidate)
    if restore.count != 1 or dump(restored) != dump(original):
        raise ValueError('candidate must change exactly the row-index product operand order')
    return {'status': 'PASS', 'scope': 'Complete module AST, one exact integer multiplication commutation.'}


if __name__ == '__main__':
    print(check_pair())
