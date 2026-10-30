"""Check the complete row-offset candidate module without importing Torch."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = 'triton_red_fused_native_layer_norm_no_welford'
CANDIDATE = ORIGINAL + '_row_offset_reused'


def check_pair(original_text=None, candidate_text=None):
    original = ast.parse((HERE / 'layer_norm_welfold_repaired.py').read_text()
                         if original_text is None else original_text)
    candidate = ast.parse((HERE / 'layer_norm_welfold_row_offset_reused.py').read_text()
                          if candidate_text is None else candidate_text)
    assignment = ast.parse('row_offset = rnumel * x0').body[0]
    product = assignment.value
    dump = lambda node: ast.dump(node, include_attributes=False)

    entries = [node for node in candidate.body
               if isinstance(node, ast.FunctionDef) and node.name == CANDIDATE]
    if len(entries) != 1 or not entries[0].body or not isinstance(entries[0].body[-1], ast.For):
        raise ValueError('candidate must retain its final normalization loop')
    final_loop = entries[0].body[-1]
    if len(final_loop.body) < 4 or dump(final_loop.body[3]) != dump(assignment):
        raise ValueError('row product must be assigned after r1 and before its first use')

    class Restore(ast.NodeTransformer):
        assignments = 0
        uses = 0

        def visit_FunctionDef(self, node):
            if node.name == CANDIDATE:
                node.name = ORIGINAL
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if dump(node) == dump(assignment):
                self.assignments += 1
                return None
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == CANDIDATE:
                node.id = ORIGINAL
            if node.id == 'row_offset' and isinstance(node.ctx, ast.Load):
                self.uses += 1
                return product
            return node

    restore = Restore()
    restored = restore.visit(candidate)
    if restore.assignments != 1 or restore.uses != 2 or dump(restored) != dump(original):
        raise ValueError('candidate must only reuse the row product in the final loop load/store')
    return {'status': 'PASS', 'scope': 'Full module AST; one row-product assignment, two uses and JIT entry/call names only.'}


if __name__ == '__main__':
    print(check_pair())
