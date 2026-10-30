"""Check the complete candidate differs only by the justified split specialization."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "lora_expand_gemv_byte_separated.py"
CANDIDATE = ROOT / "lora_expand_gemv_unit_split.py"

class Restore(ast.NodeTransformer):
    def __init__(self):
        self.specializations = 0
        self.renames = 0

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == "_bgmv_expand_unit_split_kernel":
            node.name = "_bgmv_expand_kernel"
            self.renames += 1
        return node

    def visit_Name(self, node):
        if node.id == "_bgmv_expand_unit_split_kernel":
            node.id = "_bgmv_expand_kernel"
            self.renames += 1
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == "split_n_length" for t in node.targets):
            assert ast.dump(node.value) == ast.dump(ast.Constant(value=1))
            node.value = ast.parse("tl.cdiv(N, SPLIT_N)", mode="eval").body
            self.specializations += 1
        return node

original = ast.parse(ORIGINAL.read_text())
candidate = ast.parse(CANDIDATE.read_text())
compile(original, str(ORIGINAL), "exec")
compile(candidate, str(CANDIDATE), "exec")
restore = Restore()
restored = restore.visit(candidate)
assert restore.specializations == 1
assert restore.renames == 2
assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
print(json.dumps({
    "status": "PASS",
    "full_ast_preserved_after_restoring": "one cdiv(N,SPLIT_N) -> 1 assignment and kernel name",
    "unchanged": "all wrapper checks, byte interval admission, empty returns, scalar bindings, grid (N,B), FP instructions, masks, branches, loop and stores",
    "original_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
    "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
}, indent=2))
