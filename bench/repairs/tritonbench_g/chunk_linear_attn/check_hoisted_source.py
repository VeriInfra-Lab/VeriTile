"""Check the complete Python candidate delta and its ordinary Lean quotation."""
from pathlib import Path
import ast
import hashlib
import json
import re

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[3]

def antiquote(text):
    while "$(" in text:
        start = text.index("$(")
        depth, end = 1, start + 2
        while depth:
            if text[end] == "(": depth += 1
            if text[end] == ")": depth -= 1
            end += 1
        text = text[:start] + text[start + 2:end - 1] + text[end:]
    return text


def quotation_python(name, text):
    section = text.split("def " + name + " ", 1)[1].split("triton {", 1)[1]
    lines, indent = [], 0
    for raw in section.splitlines():
        line = raw.strip()
        if not line or line.startswith("tl.region "):
            continue
        if line == "}":
            if indent == 0:
                break
            indent -= 1
            continue
        opens = line.endswith("{")
        if opens:
            line = line[:-1].rstrip() + ":"
        line = antiquote(line).replace(" : List Nat", "").replace(" : ℝ", "")
        line = re.sub(r"\bdo_\b", "do", line)
        lines.append("    " * indent + line)
        if opens:
            indent += 1
    assert indent == 0 and lines, name
    return ast.parse("\n".join(lines)).body


class Normalize(ast.NodeTransformer):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            assert isinstance(node.value, ast.Tuple)
            assert len(node.targets[0].elts) == len(node.value.elts)
            assert all(isinstance(v, ast.Call) and ast.unparse(v.func) == "tl.program_id" for v in node.value.elts)
            return [ast.Assign(targets=[t], value=v) for t, v in zip(node.targets[0].elts, node.value.elts)]
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        spelling = ast.unparse(node.func)
        if spelling == "tl.make_block_ptr":
            fields = ["base", "shape", "strides", "offsets", "block_shape", "order"]
            values = dict(zip(fields, node.args))
            values.update({kw.arg: kw.value for kw in node.keywords})
            assert set(values) == set(fields)
            node.args = [values[key] for key in fields]
            node.keywords = []
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            assert len(node.args) == 1
            original = ast.unparse(node.args[0])
            assert original == "tl.float32" or original.endswith(".dtype") or original.endswith(".dtype.element_ty")
            node.args = [ast.parse("tl.float32", mode="eval").body]
        for kw in node.keywords:
            if kw.arg == "boundary_check":
                assert isinstance(kw.value, (ast.Tuple, ast.List))
                kw.value = ast.List(elts=kw.value.elts, ctx=ast.Load())
        return node


original_path = TASK / "chunk_linear_attn_repaired.py"
candidate_path = TASK / "chunk_linear_attn_hoisted.py"
original = ast.parse(original_path.read_text())
candidate = ast.parse(candidate_path.read_text())
fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_linear_attn_fwd_kernel_o")
loop_index = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.For))
loop = fn.body[loop_index]

class Hoist(ast.NodeTransformer):
    count = 0
    def visit_BinOp(self, node):
        if ast.dump(node) == ast.dump(ast.parse("i_bh * s_qk_h", mode="eval").body):
            self.count += 1
            return ast.Name(id="qk_head", ctx=ast.Load())
        return self.generic_visit(node)

hoist = Hoist()
hoist.visit(loop)
assert hoist.count == 2
fn.body.insert(loop_index, ast.parse("qk_head = i_bh * s_qk_h").body[0])
assert ast.dump(original) == ast.dump(candidate), "unexpected complete Python module change"

candidate_fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
expected = Normalize().visit(ast.Module(body=candidate_fn.body, type_ignores=[]))
actual = Normalize().visit(ast.Module(body=quotation_python("forwardOHoisted", (TASK / "Candidate.lean").read_text()), type_ignores=[]))
assert ast.dump(expected) == ast.dump(actual), "candidate quotation does not match complete Python output kernel"

original_lean = (TASK / "Kernels.lean").read_text().split("def forwardO ", 1)[1].split("\ndef backwardH ", 1)[0]
expected_lean = "def forwardOHoisted " + original_lean
expected_lean = expected_lean.replace("        for i_k in range", "        qk_head = i_bh * $(s_qk_h)\n        for i_k in range", 1)
expected_lean = expected_lean.replace("base=q + (i_bh * $(s_qk_h))", "base=q + qk_head").replace("base=k + (i_bh * $(s_qk_h))", "base=k + qk_head")
actual_lean = "def forwardOHoisted " + (TASK / "Candidate.lean").read_text().split("def forwardOHoisted ", 1)[1].split("\nend VeriTile.", 1)[0]
assert expected_lean.strip() == actual_lean.strip(), "unexpected complete Lean output-kernel change"

paths = [original_path, candidate_path, TASK / "Kernels.lean", TASK / "Candidate.lean", Path(__file__)]
print(json.dumps({"complete_python_module_delta": "one hoisted shared head multiplication; two loop uses",
    "candidate_complete_quotation_match": True, "lean_source_delta_match": True,
    "unchanged_host_and_other_kernels": True,
    "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2))
