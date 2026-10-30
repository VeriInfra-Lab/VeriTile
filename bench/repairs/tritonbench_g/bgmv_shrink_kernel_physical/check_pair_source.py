"""Exact complete-module check for the existing scalar-left multiplication pair."""
import ast
import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check_pair(original_text=None, candidate_text=None):
    original = ast.parse((HERE / 'bgmv_shrink_kernel_repaired.py').read_text()
                         if original_text is None else original_text)
    candidate = ast.parse((HERE / 'bgmv_shrink_scalar_left.py').read_text()
                          if candidate_text is None else candidate_text)
    name = '_bgmv_shrink_kernel'
    before = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
    after = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == name + '_scalar_left')
    positions = [i for i, n in enumerate(before.body) if isinstance(n, ast.AugAssign)
                 and isinstance(n.target, ast.Name) and n.target.id == 'accumulator'
                 and isinstance(n.op, ast.Mult) and isinstance(n.value, ast.Name)
                 and n.value.id == 'scaling']
    if len(positions) != 1 or len(after.body) != len(before.body):
        raise ValueError('expected exactly the original final multiplication')
    index = positions[0]
    expected = ast.parse('accumulator = scaling * accumulator').body[0]
    if ast.dump(after.body[index], include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('candidate changes more than multiplication operand order')
    after.body[index] = copy.deepcopy(before.body[index])

    class RestoreNames(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == name + '_scalar_left':
                node.name = name
            return node

        def visit_Name(self, node):
            if node.id == name + '_scalar_left':
                node.id = name
            return node

    restored = RestoreNames().visit(candidate)
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError('complete candidate module differs outside the declared expression and JIT names')
    return {'status': 'PASS', 'scope': 'Complete module AST, one final multiplication commutation and JIT name/call replacement only.'}


if __name__ == '__main__':
    print(check_pair())
