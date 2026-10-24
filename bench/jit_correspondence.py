"""Compare complete Python JIT bodies with their Lean source quotations.

Only syntax adaptations are allowed: braces/indentation, parameter antiquotes,
and tuple/list shape spelling in tl.full. Casts, calls, branches, operands and
statement order are compared as ASTs. Unsupported syntax fails closed.
This checks source correspondence; it does not prove compiler correctness.
"""
from __future__ import annotations

import ast
import copy
import difflib
import json
from pathlib import Path
import re

from bench.audit_source import lean_code_mask, strip_lean_comments


def quotation(text: str, definition: str) -> tuple[str, str]:
    mask = lean_code_mask(text)
    start = re.search(r"\bdef\s+" + re.escape(definition) + r"\b", mask)
    if not start:
        raise ValueError(f"missing Lean definition: {definition}")
    next_def = re.search(r"\n(?:def|theorem|specification|end)\b", mask[start.end():])
    end = start.end() + next_def.start() if next_def else len(text)
    selected, code = text[start.start():end], mask[start.start():end]
    begin = re.search(r"\btriton\b", code)
    if not begin:
        raise ValueError(f"definition has no direct quotation: {definition}")
    opening = code.find("{", begin.end())
    if opening < 0:
        raise ValueError("missing quotation body")
    depth = 1
    for i in range(opening + 1, len(code)):
        depth += (code[i] == "{") - (code[i] == "}")
        if depth == 0:
            if code[i + 1:].strip():
                raise ValueError("unexpected code after quotation")
            return selected[:opening], strip_lean_comments(selected[opening + 1:i])
    raise ValueError("unterminated quotation")


def source_statements(body: str, constants: set[str]) -> tuple[list[ast.stmt], dict[str, str]]:
    parameters: dict[str, str] = {}
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("tl.param "):
            m = re.fullmatch(r"tl\.param (\w+)\s*:\s*(tl\.\w+)", stripped)
            if not m or m[1] in parameters:
                raise ValueError("invalid or repeated runtime parameter declaration")
            parameters[m[1]] = m[2]
        elif stripped.startswith("tl.region "):
            if not re.fullmatch(r"tl\.region \w+\s*=\s*tl\.\w+(?:\s*,\s*\w+\s*=\s*tl\.\w+)*", stripped):
                raise ValueError("unsupported region declaration")
        else:
            lines.append(line)
    body = "\n".join(lines)

    def antiquote(match):
        name = match[1]
        if name not in constants and name != "0":
            raise ValueError(f"antiquote is not a constexpr parameter: {name}")
        return name

    def static_comparison(match):
        expression = ast.parse(match[1], mode="eval").body
        if not isinstance(expression, ast.Compare) or len(expression.ops) != 1 or not isinstance(expression.ops[0], ast.Eq):
            raise ValueError("unsupported constexpr comparison")
        if not (isinstance(expression.left, ast.Name) and expression.left.id in constants
                and isinstance(expression.comparators[0], ast.Constant)
                and type(expression.comparators[0].value) is int):
            raise ValueError("constexpr comparison must use a template parameter and integer literal")
        return "(" + match[1] + ")"

    body = re.sub(r"\$\(\(\((\w+\s*==\s*\d+)\)\s*:\s*Bool\)\)", static_comparison, body)
    body = re.sub(r"\$\((\w+)\)", antiquote, body)
    if "$" in body:
        raise ValueError("unsupported antiquotation; cannot certify source correspondence")

    output, pending, indent, delimiters = [], [], 0, []
    quote = None
    escaped = False

    def emit(suffix=""):
        text = "".join(pending).strip()
        pending.clear()
        if text:
            output.append("    " * indent + text + suffix)

    for char in body:
        if quote:
            pending.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            pending.append(char)
        elif char in "([":
            delimiters.append(char)
            pending.append(char)
        elif char in ")]":
            if not delimiters or "(["[")]".index(char)] != delimiters.pop():
                raise ValueError("unbalanced source expression")
            pending.append(char)
        elif char == "{" and not delimiters:
            if not re.match(r"\s*(if|elif|else|for)\b", "".join(pending)):
                raise ValueError("unsupported source block")
            emit(":")
            indent += 1
        elif char == "}" and not delimiters:
            emit()
            indent -= 1
            if indent < 0:
                raise ValueError("unbalanced source block")
        elif char == "\n" and not delimiters:
            emit()
        else:
            pending.append(char if char != "\n" else " ")
    emit()
    if quote or delimiters or indent:
        raise ValueError("incomplete source body")
    return ast.parse("\n".join(output)).body, parameters


class ShapeSpelling(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.func) == "tl.full":
            if node.args and isinstance(node.args[0], ast.Tuple):
                node.args[0] = ast.List(node.args[0].elts, ast.Load())
            if len(node.args) == 3:
                if any(k.arg == "dtype" for k in node.keywords):
                    raise ValueError("duplicate full dtype")
                node.keywords.append(ast.keyword("dtype", node.args.pop()))
        return node



def call_signature(text: str, name: str) -> dict:
    code = strip_lean_comments(text)
    match = re.search(r"\bdef\s+" + re.escape(name) + r"\s*:\s*CallSignature\s+where\s*(.*?)(?=\ndef |\nnoncomputable |\nend\b|\Z)", code, re.S)
    if not match:
        raise ValueError("missing declarative call signature")
    fields = {}
    for field in ("pointers", "scalars", "constexpr"):
        value = re.search(r"\b" + field + r"\s*:=\s*(\[[^\]]*\])", match[1])
        if not value:
            raise ValueError(f"missing call signature field: {field}")
        source = re.sub(r"\.(fp16|bf16|fp32|fp64|i8|i16|i32|i64|u8|u16|u32|u64|boolean|integer)\b", r'"\1"', value[1])
        fields[field] = json.loads(json.dumps(ast.literal_eval(source)))
    names = [n for n, _ in fields["pointers"] + fields["scalars"]] + fields["constexpr"]
    if len(names) != len(set(names)):
        raise ValueError("duplicate call signature parameter")
    return fields

def check(python_text: str, lean_text: str, python_function: str, lean_definition: str,
          signature_definition: str | None = None, expected_signature: dict | None = None) -> dict:
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d) == "triton.jit" for d in functions[0].decorator_list):
        raise ValueError("expected one explicitly selected @triton.jit function")
    function = functions[0]
    constants = {a.arg for a in function.args.args
                 if a.annotation is not None and ast.unparse(a.annotation) == "tl.constexpr"}
    runtime = {a.arg: ast.unparse(a.annotation) for a in function.args.args
               if a.annotation is not None and a.arg not in constants}
    signature, body = quotation(lean_text, lean_definition)
    profile = None
    if signature_definition is not None:
        if re.search(r"\btl\.(region|param)\b|\$", body):
            raise ValueError("pasted body must not contain metadata directives or Lean antiquotes")
        if not re.search(r"\bwith_triton_signature\s+" + re.escape(signature_definition) + r"\s+in\b", signature):
            raise ValueError("quotation does not use the declared call signature")
        profile = call_signature(lean_text, signature_definition)
        if expected_signature is not None and profile != expected_signature:
            raise ValueError("call signature differs from the reviewed launch types")
        names = {n for n, _ in profile["pointers"] + profile["scalars"]} | set(profile["constexpr"])
        if names != {a.arg for a in function.args.args} or set(profile["constexpr"]) != constants:
            raise ValueError("Python parameters and call signature differ")
        scalar_types = {"fp32": "tl.float32", "fp64": "tl.float64", "integer": "tl.int64"}
        supplied = {n: scalar_types[t] for n, t in profile["scalars"]}
        if any(supplied.get(n) != t for n, t in runtime.items()):
            raise ValueError("runtime scalar annotation differs from call signature")
    statements, parameters = source_statements(body, constants)
    if parameters != runtime:
        raise ValueError(f"runtime signature mismatch: Python {runtime}; Lean {parameters}")
    for name in constants:
        if not re.search(r"\b" + re.escape(name) + r"\b", signature):
            raise ValueError(f"missing constexpr binder: {name}")
    left = ShapeSpelling().visit(ast.Module(copy.deepcopy(function.body), []))
    right = ShapeSpelling().visit(ast.Module(statements, []))
    if ast.dump(left) != ast.dump(right):
        diff = "\n".join(difflib.unified_diff(ast.unparse(left).splitlines(),
                        ast.unparse(right).splitlines(), fromfile="Python JIT", tofile="Lean JIT", lineterm=""))
        raise ValueError("statement correspondence mismatch:\n" + diff)
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "statements": len(function.body),
            "runtime_parameters": dict(profile["scalars"]) if profile is not None else runtime, "constexpr_parameters": sorted(constants)}


def check_manifest(root: Path, manifest: Path) -> list[dict]:
    records = json.loads(manifest.read_text())
    return [check((root / r["python"]).read_text(), (root / r["lean"]).read_text(),
                  r["python_function"], r["lean_definition"],
                  r.get("signature_definition"), r.get("expected_signature")) for r in records]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(check_manifest(args.root, args.manifest), indent=2))
