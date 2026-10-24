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
import io
from pathlib import Path
import re
import textwrap
import tokenize

from bench.audit_source import lean_code_mask, strip_lean_comments


def quotation(text: str, definition: str) -> tuple[str, str]:
    mask = lean_code_mask(text)
    start = re.search(r"\bdef\s+" + re.escape(definition) + r"\b", mask)
    if not start:
        raise ValueError(f"missing Lean definition: {definition}")
    next_def = re.search(r"\n(?:def|theorem|specification|abbrev|structure|inductive|namespace|end|noncomputable\s+def)\b", mask[start.end():])
    boundary = start.end() + next_def.start() if next_def else len(text)
    selected, code = text[start.start():], mask[start.start():]
    begin = re.search(r"\btriton\b", code)
    if not begin:
        raise ValueError(f"definition has no direct quotation: {definition}")
    opening = code.find("{", begin.end())
    if opening < 0:
        raise ValueError("missing quotation body")
    if start.start() + opening >= boundary:
        raise ValueError(f"definition has no direct quotation: {definition}")
    native = bool(re.search(r"\bwith_triton_(?:signature|call)\b", code[:opening]))
    if native:
        fragment = selected[opening:]
        offsets = [0]
        for line in fragment.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        depth = 0
        try:
            for token in tokenize.generate_tokens(io.StringIO(fragment).readline):
                if token.type != tokenize.OP:
                    continue
                depth += (token.string == "{") - (token.string == "}")
                if token.string == "}" and depth == 0:
                    closing = opening + offsets[token.start[0]-1] + token.start[1]
                    remainder = mask[start.start()+closing+1:]
                    end = re.search(r"\n(?:def|theorem|specification|abbrev|structure|inductive|namespace|end|noncomputable\s+def)\b", remainder)
                    suffix = remainder[:end.start()] if end else remainder
                    if suffix.strip():
                        raise ValueError("unexpected code after quotation")
                    return selected[:opening], selected[opening+1:closing]
        except (tokenize.TokenError, IndentationError) as error:
            raise ValueError(f"invalid Python body quotation: {error}") from error
        raise ValueError("unterminated Python body quotation")
    selected, code = selected[:boundary-start.start()], code[:boundary-start.start()]
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
    for field in ("readPorts", "writePorts"):
        value = re.search(r"\b" + field + r"\s*:=\s*(\[[^\]]*\])", match[1])
        if value:
            fields[field] = ast.literal_eval(value[1])
            if (not isinstance(fields[field], list) or len(set(fields[field])) != len(fields[field])
                or not set(fields[field]) <= {n for n, _ in fields["pointers"]}):
                raise ValueError("invalid call memory ports")
    names = [n for n, _ in fields["pointers"] + fields["scalars"]] + fields["constexpr"]
    if len(names) != len(set(names)):
        raise ValueError("duplicate call signature parameter")
    return fields

def check(python_text: str, lean_text: str, python_function: str, lean_definition: str,
          signature_definition: str | None = None, expected_signature: dict | None = None,
          expected_call_bindings: dict | None = None,
          expected_pointer_offsets: dict | None = None) -> dict:
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d) == "triton.jit" for d in functions[0].decorator_list):
        raise ValueError("expected one explicitly selected @triton.jit function")
    function = functions[0]
    def annotation(argument):
        value = argument.annotation
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        return ast.unparse(value) if value is not None else None

    if function.args.posonlyargs or function.args.kwonlyargs or function.args.vararg or function.args.kwarg:
        raise ValueError("unsupported Python call signature")
    constants = {a.arg for a in function.args.args if annotation(a) == "tl.constexpr"}
    runtime = {a.arg: ast.unparse(a.annotation) for a in function.args.args
               if a.annotation is not None and a.arg not in constants}
    signature, body = quotation(lean_text, lean_definition)
    profile = None
    call_bindings = None
    pointer_offsets = None
    if expected_pointer_offsets is not None and signature_definition is None:
        raise ValueError("pointer offsets require a checked call signature")
    if signature_definition is not None:
        if expected_signature is None:
            raise ValueError("pasted body requires an independently reviewed call signature")
        masked = list(body)
        offsets = [0]
        for line in body.splitlines(keepends=True): offsets.append(offsets[-1] + len(line))
        try:
            for token in tokenize.generate_tokens(io.StringIO(body).readline):
                if token.type in (tokenize.STRING, tokenize.COMMENT):
                    a, b = (offsets[pos[0]-1] + pos[1] for pos in (token.start, token.end))
                    for index in range(a, b):
                        if masked[index] != "\n": masked[index] = " "
        except (tokenize.TokenError, IndentationError):
            pass  # The strict AST parser below rejects malformed Python.
        if re.search(r"\btl\.(region|param)\b|\$", "".join(masked)):
            raise ValueError("pasted body must not contain metadata directives or Lean antiquotes")
        if not re.search(r"\bwith_triton_signature\s+" + re.escape(signature_definition) + r"\s+in\b", signature):
            call = re.search(r"\bwith_triton_call\s+" + re.escape(signature_definition)
                             + r"\s+call_args\s*\[([^\]]*)\](?:\s+pointer_offsets\s*\[([^\]]*)\])?\s+in\b", signature)
            if not call:
                raise ValueError("quotation does not use the declared call signature")
            call_bindings = {}
            for binding in call[1].split(",") if call[1].strip() else []:
                match = re.fullmatch(r"\s*(\w+)\s*:=\s*([\w.]+)\s*", binding)
                if not match or match[1] in call_bindings:
                    raise ValueError("unsupported or duplicate per-call scalar binding")
                call_bindings[match[1]] = match[2]
            if expected_call_bindings is None or call_bindings != expected_call_bindings:
                raise ValueError("per-call scalar bindings differ from the reviewed invocation")
            if call[2] is not None:
                pointer_offsets = {}
                for binding in call[2].split(",") if call[2].strip() else []:
                    match = re.fullmatch(r"\s*(\w+)\s*:=\s*([\w.]+)\s*", binding)
                    if not match or match[1] in pointer_offsets:
                        raise ValueError("unsupported or duplicate pointer offset binding")
                    pointer_offsets[match[1]] = match[2]
        if pointer_offsets != expected_pointer_offsets:
            raise ValueError("pointer offsets differ from the reviewed invocation")
        profile = call_signature(lean_text, signature_definition)
        if expected_signature is not None and profile != expected_signature:
            raise ValueError("call signature differs from the reviewed launch types")
        names = {n for n, _ in profile["pointers"] + profile["scalars"]} | set(profile["constexpr"])
        if names != {a.arg for a in function.args.args} or set(profile["constexpr"]) != constants:
            raise ValueError("Python parameters and call signature differ")
        # `integer` is intentionally unbounded; it cannot certify an explicit
        # int32/int64 ABI annotation. Pointer annotations are not scalar args.
        scalar_types = {"fp32": "tl.float32", "fp64": "tl.float64", "i32": "tl.int32", "i64": "tl.int64"}
        supplied = {n: scalar_types.get(t) for n, t in profile["scalars"]}
        pointer_names = {n for n, _ in profile["pointers"]}
        if pointer_offsets is not None and not set(pointer_offsets) <= pointer_names:
            raise ValueError("offset binding is not a pointer parameter")
        if call_bindings is not None:
            integer_names = {n for n, t in profile["scalars"] if t in ("i32", "i64", "integer")}
            fp32_names = {n for n, t in profile["scalars"] if t == "fp32"}
            if not integer_names <= set(call_bindings) <= integer_names | fp32_names:
                raise ValueError("per-call bindings must cover every integer scalar and only supported scalar kinds")
        for arg in function.args.args:
            t = annotation(arg)
            if t is None or arg.arg in constants:
                continue
            if arg.arg in pointer_names and t == "tl.tensor":
                continue
            if supplied.get(arg.arg) != t:
                raise ValueError("runtime scalar annotation differs from call signature")
    if profile is not None:
        # Parse the pasted body as Python directly: braces, Lean assignment
        # syntax and other legacy spelling changes cannot become a copy pass.
        # A wrapper preserves indentation *inside* triple-quoted strings.
        # textwrap.dedent mistakes unindented docstring text for source code.
        wrapped = ast.parse("def __pasted_body__():\n" + body)
        statements, parameters = wrapped.body[0].body, {}
    else:
        statements, parameters = source_statements(body, constants)
    if profile is None and parameters != runtime:
        raise ValueError(f"runtime signature mismatch: Python {runtime}; Lean {parameters}")
    for name in constants:
        if not re.search(r"\b" + re.escape(name) + r"\b", signature):
            raise ValueError(f"missing constexpr binder: {name}")
    left = ast.Module(copy.deepcopy(function.body), [])
    right = ast.Module(statements, [])
    # The legacy mode permits the documented shape spelling adaptations.
    # Body-copy mode preserves the actual Python expressions unchanged.
    if profile is None:
        left, right = ShapeSpelling().visit(left), ShapeSpelling().visit(right)
    if ast.dump(left) != ast.dump(right):
        diff = "\n".join(difflib.unified_diff(ast.unparse(left).splitlines(),
                        ast.unparse(right).splitlines(), fromfile="Python JIT", tofile="Lean JIT", lineterm=""))
        raise ValueError("statement correspondence mismatch:\n" + diff)
    result = {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body" if profile is not None else "legacy_ast",
            "argument_mode": ("typed_per_call_specialization" if profile is not None and call_bindings is not None
                              and any(n in call_bindings and t == "fp32" for n,t in profile["scalars"]) else
                              "mixed_integer_calls_and_runtime_registers" if profile is not None and call_bindings is not None
                              and len(call_bindings) < len(profile["scalars"]) else
                              "integer_per_call_specialization" if call_bindings is not None else "runtime_registers"),
            "call_bindings": call_bindings,
            "statements": len(function.body),
            "runtime_parameters": dict(profile["scalars"]) if profile is not None else runtime, "constexpr_parameters": sorted(constants)}
    if pointer_offsets is not None:
        result["pointer_offsets"] = pointer_offsets
    return result


def validate_manifest_records(records: list[dict]) -> None:
    """Allow explicit distinct ABI profiles of one Python function.

    A profile tag cannot excuse a duplicate mapping: both a distinct Lean
    definition and a distinct checked ABI are required. Ordinary duplicate
    records continue to fail closed.
    """
    lean_keys = [(r["lean"], r["lean_definition"]) for r in records]
    if len(lean_keys) != len(set(lean_keys)):
        raise ValueError("reused Python or Lean correspondence entry")
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["python"], record["python_function"]), []).append(record)
    for group in groups.values():
        if len(group) == 1:
            continue
        tags = [record.get("profile_variant") for record in group]
        if not all(isinstance(tag, str) and tag.strip() for tag in tags) or len(tags) != len(set(tags)):
            raise ValueError("reused Python correspondence requires distinct explicit profile variants")
        abis = []
        for record in group:
            signature = record.get("expected_signature")
            if not record.get("signature_definition") or not isinstance(signature, dict):
                raise ValueError("reused Python correspondence requires checked ABI profiles")
            abis.append(json.dumps({key: signature.get(key) for key in
                                   ("pointers", "scalars", "constexpr")}, sort_keys=True))
        if len(abis) != len(set(abis)):
            raise ValueError("reused Python correspondence has duplicate ABI profiles")


def check_manifest(root: Path, manifest: Path) -> list[dict]:
    records = json.loads(manifest.read_text())
    validate_manifest_records(records)
    results = []
    for record in records:
        result = check((root / record["python"]).read_text(), (root / record["lean"]).read_text(),
                       record["python_function"], record["lean_definition"],
                       record.get("signature_definition"), record.get("expected_signature"),
                       record.get("expected_call_bindings"), record.get("expected_pointer_offsets"))
        if "profile_variant" in record:
            result["profile_variant"] = record["profile_variant"]
        results.append(result)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(check_manifest(args.root, args.manifest), indent=2))
