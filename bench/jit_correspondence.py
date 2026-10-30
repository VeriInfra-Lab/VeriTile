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


def scalar_binding_term(text: str) -> str:
    """Compare caller arithmetic syntactically, without evaluating Lean as Python.

    Preserve the existing identifier/projection/literal spellings. A bounded
    arithmetic AST additionally admits readable canonical stride products;
    function calls, indexing and statement-like syntax remain unsupported.
    """
    if not isinstance(text, str):
        raise ValueError("unsupported per-call scalar binding")
    text = text.strip()
    typed = re.fullmatch(r"\((.+)\s*:\s*(Nat|Int)\s*\)", text, re.S)
    if typed:
        return "(" + scalar_binding_term(typed[1]) + " : " + typed[2] + ")"
    if re.fullmatch(r"[\w.]+", text):
        return text
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError("unsupported per-call scalar binding") from error
    allowed = (ast.Expression, ast.Name, ast.Attribute, ast.Load, ast.Constant,
               ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) or
           (isinstance(node, ast.Constant) and type(node.value) is not int)
           for node in ast.walk(tree)):
        raise ValueError("unsupported per-call scalar binding")
    return ast.unparse(tree.body)


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
    native = bool(re.search(r"\b(?:with_triton_(?:signature|call)|triton_function)\b", code[:opening]))
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
    match = re.search(r"\bdef\s+" + re.escape(name) + r"(?P<parameters>(?:\s*\([^()\n]*\))*)\s*:\s*CallSignature\s+where\s*(?P<body>.*?)(?=\ndef |\nnoncomputable |\nend\b|\Z)", code, re.S)
    if not match:
        raise ValueError("missing declarative call signature")
    fields = {}
    parameter_text = match["parameters"].strip()
    if parameter_text:
        parameters = re.findall(r"\(\s*(\w+)\s*:\s*([^()]+?)\s*\)", parameter_text)
        if (len(parameters) != parameter_text.count("(") or
                len({n for n, _ in parameters}) != len(parameters)):
            raise ValueError("signature parameters must be distinct typed names")
        fields["parameters"] = [[name, " ".join(dtype.split())] for name, dtype in parameters]
    ordered = re.search(r"\borderedRowSums\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\borderedRowSums\s*:=", match["body"]) and not ordered:
        raise ValueError("ordered row sums must be a literal metadata list")
    if ordered:
        pattern = r'\(\s*"([A-Za-z_]\w*)"\s*,\s*⟨\s*([A-Za-z_]\w*|[0-9]+)\s*,\s*([A-Za-z_]\w*)\s*⟩\s*\)'
        entries = re.findall(pattern, ordered[1])
        remainder = re.sub(pattern, "", ordered[1])
        if (re.sub(r"[\s,\[\]]", "", remainder) or not entries or
                len({n for n, _, _ in entries}) != len(entries)):
            raise ValueError("ordered row sums require distinct operands and named width/tree metadata")
        fields["orderedRowSums"] = [list(entry) for entry in entries]
    for field in ("pointers", "scalars", "constexpr"):
        value = re.search(r"\b" + field + r"\s*:=\s*(\[[^\]]*\])", match["body"])
        if not value:
            raise ValueError(f"missing call signature field: {field}")
        source = re.sub(r"\.(fp16|bf16|fp32|fp64|f8e4|f8e5|i8|i16|i32|i64|u8|u16|u32|u64|boolean|integer|inputFloat|outputFloat)\b", r'"\1"', value[1])
        fields[field] = json.loads(json.dumps(ast.literal_eval(source)))
    dtype_pointers = re.search(r"\bsourceDTypePointers\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\bsourceDTypePointers\s*:=", match["body"]) and not dtype_pointers:
        raise ValueError("source dtype pointers must be a literal list")
    if dtype_pointers:
        values = ast.literal_eval(dtype_pointers[1])
        if (not isinstance(values, list) or not all(isinstance(n, str) and re.fullmatch(r"[A-Za-z_]\w*", n) for n in values)
                or len(values) != len(set(values))):
            raise ValueError("source dtype pointers must be distinct parameter names")
        fields["sourceDTypePointers"] = values
    specialized = re.search(r"\bspecializeBooleans\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\bspecializeBooleans\s*:=", match["body"]) and not specialized:
        raise ValueError("Boolean specializations must be a literal list")
    if specialized:
        values = ast.literal_eval(specialized[1])
        if (not isinstance(values, list) or not all(isinstance(n, str) for n in values)
                or len(values) != len(set(values)) or not set(values) <= set(fields["constexpr"])):
            raise ValueError("Boolean specialization names must be distinct constexpr parameters")
        fields["specializeBooleans"] = values
    pointer_names = {n for n, _ in fields["pointers"]} | set(fields.get("sourceDTypePointers", []))
    optional = re.search(r"\boptionalPointers\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\boptionalPointers\s*:=", match["body"]) and not optional:
        raise ValueError("optional pointers must be a literal list")
    if optional:
        values = ast.literal_eval(optional[1])
        if (not isinstance(values, list) or not all(isinstance(n, str) for n in values)
                or len(values) != len(set(values)) or not set(values) <= pointer_names):
            raise ValueError("optional pointers must be distinct declared pointer names")
        fields["optionalPointers"] = values
    for field in ("readPorts", "writePorts"):
        value = re.search(r"\b" + field + r"\s*:=\s*(\[[^\]]*\])", match["body"])
        if value:
            fields[field] = ast.literal_eval(value[1])
            if (not isinstance(fields[field], list) or len(set(fields[field])) != len(fields[field])
                or not set(fields[field]) <= pointer_names):
                raise ValueError("invalid call memory ports")
    storage = re.search(r"\btypedFP32Storage\s*:=\s*(?:Bool\.)?(true|false)\b", match["body"])
    if re.search(r"\btypedFP32Storage\s*:=", match["body"]) and not storage:
        raise ValueError("typed FP32 storage selection must be a literal Boolean")
    if storage:
        fields["typedFP32Storage"] = storage[1] == "true"
    selected_storage = re.search(r"\btypedFP32Pointers\s*:=\s*(\[[^\]]*\])(?=\s*(?:[A-Za-z_]\w*\s*:=|$))", match["body"], re.S)
    if re.search(r"\btypedFP32Pointers\s*:=", match["body"]) and not selected_storage:
        raise ValueError("typed FP32 pointers must be a literal list")
    if selected_storage:
        try:
            selected = ast.literal_eval(selected_storage[1])
        except (ValueError, SyntaxError) as exc:
            raise ValueError("typed FP32 pointers must be a literal list") from exc
        if (not isinstance(selected, list) or not all(isinstance(n, str) for n in selected)
                or len(selected) != len(set(selected))
                or not set(selected) <= {n for n, dtype in fields["pointers"] if dtype == "fp32"}):
            raise ValueError("typed FP32 pointers must be distinct FP32 pointer parameter names")
        fields["typedFP32Pointers"] = selected
    natural = re.search(r"\bnaturalIntegerPointers\s*:=\s*(\[[^\]]*\])(?=\s*(?:[A-Za-z_]\w*\s*:=|$))", match["body"], re.S)
    if re.search(r"\bnaturalIntegerPointers\s*:=", match["body"]) and not natural:
        raise ValueError("natural integer pointers must be a literal list")
    if natural:
        try:
            selected = ast.literal_eval(natural[1])
        except (ValueError, SyntaxError) as exc:
            raise ValueError("natural integer pointers must be a literal list") from exc
        integers = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}
        if (not isinstance(selected, list) or not all(isinstance(n, str) for n in selected)
                or len(selected) != len(set(selected))
                or not set(selected) <= {n for n, dtype in fields["pointers"] if dtype in integers}):
            raise ValueError("natural integer pointers must be distinct integer pointer parameter names")
        fields["naturalIntegerPointers"] = selected
    functions = re.search(r"\bfunctions\s*:=\s*\[([^\]]*)\]", match["body"])
    if re.search(r"\bfunctions\s*:=", match["body"]) and not functions:
        raise ValueError("source functions must be a literal list of declarations")
    if functions:
        names = [name.strip() for name in functions[1].split(",") if name.strip()]
        if any(not re.fullmatch(r"\w+(?:\.\w+)*", name) for name in names) or len(names) != len(set(names)):
            raise ValueError("source functions must be distinct named declarations")
        fields["functions"] = names
    imports = re.search(r"\bfiniteImports\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\bfiniteImports\s*:=", match["body"]) and not imports:
        raise ValueError("finite imports must be a literal list of source bindings")
    if imports:
        bindings = ast.literal_eval(imports[1])
        targets = {"triton.language.extra.cuda.libdevice.finitef", "triton.language.extra.cuda.libdevice.isfinited"}
        if (not isinstance(bindings, list) or any(not isinstance(p, tuple) or len(p) != 2
                or not isinstance(p[0], str) or not re.fullmatch(r"[A-Za-z_]\w*", p[0])
                or p[1] not in targets for p in bindings)
                or len({p[0] for p in bindings}) != len(bindings)):
            raise ValueError("invalid finite source import binding")
        fields["finiteImports"] = [list(p) for p in bindings]
    imports = re.search(r"\bunaryImports\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\bunaryImports\s*:=", match["body"]) and not imports:
        raise ValueError("unary imports must be a literal list of source bindings")
    if imports:
        bindings = ast.literal_eval(imports[1])
        targets = {"triton.language.extra.libdevice.tanh", "triton.language.extra.cuda.libdevice.tanh"}
        if (not isinstance(bindings, list) or any(not isinstance(p, tuple) or len(p) != 2
                or not isinstance(p[0], str) or not re.fullmatch(r"[A-Za-z_]\w*", p[0])
                or p[0] in {"tl", "triton"} or p[1] not in targets for p in bindings)
                or len({p[0] for p in bindings}) != len(bindings)):
            raise ValueError("invalid unary source import binding")
        fields["unaryImports"] = [list(p) for p in bindings]
    imports = re.search(r"\bbinaryImports\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\bbinaryImports\s*:=", match["body"]) and not imports:
        raise ValueError("binary imports must be a literal list of source bindings")
    if imports:
        bindings = ast.literal_eval(imports[1])
        targets = {"triton.language.extra.cuda.libdevice.pow"}
        if (not isinstance(bindings, list) or any(not isinstance(p, tuple) or len(p) != 2
                or not isinstance(p[0], str) or not re.fullmatch(r"[A-Za-z_]\w*", p[0])
                or p[0] in {"tl", "triton"} or p[1] not in targets for p in bindings)
                or len({p[0] for p in bindings}) != len(bindings)):
            raise ValueError("invalid binary source import binding")
        fields["binaryImports"] = [list(p) for p in bindings]
    globals_match = re.search(r"\bsourceConstants\s*:=\s*(\[[^\]]*\])", match["body"])
    if re.search(r"\bsourceConstants\s*:=", match["body"]) and not globals_match:
        raise ValueError("source constants must be a literal list of bindings")
    if globals_match:
        entries = ast.literal_eval(globals_match[1])
        if (not isinstance(entries, list) or any(not isinstance(entry, tuple) or len(entry) != 2
                or not isinstance(entry[0], str) or not re.fullmatch(r"[A-Za-z_]\w*", entry[0])
                or not isinstance(entry[1], str) for entry in entries)
                or len({entry[0] for entry in entries}) != len(entries)):
            raise ValueError("invalid source constant bindings")
        fields["sourceConstants"] = [list(entry) for entry in entries]
    names = [n for n, _ in fields["pointers"] + fields["scalars"]] + fields["constexpr"] + fields.get("sourceDTypePointers", [])
    external = [n for n, _ in fields.get("sourceConstants", [])]
    reserved = names + fields.get("functions", []) + [n for key in ("finiteImports", "unaryImports", "binaryImports") for n, _ in fields.get(key, [])] + ["tl", "triton", "libdevice", "float", "True", "False", "None"]
    if any(n in reserved for n in external):
        raise ValueError("source constants shadow parameters, helpers or imports")
    if len(names) != len(set(names)):
        raise ValueError("duplicate call signature parameter")
    return fields


def check_source_constants(python_text: str, function: ast.FunctionDef, expected: dict[str, str]) -> None:
    """Bind unique literal module constexpr declarations, without executing Python.

    Preserve literal spelling as well as Python's value. Any reassignment,
    shadowing, imported alias, deletion or computed initializer fails closed.
    """
    module = ast.parse(python_text)
    initializers = {}
    for node in module.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "tl.constexpr"):
            initializers.setdefault(node.targets[0].id, []).append(node)
    loaded = {n.id for statement in function.body for n in ast.walk(statement)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if loaded & set(initializers) != set(expected):
        raise ValueError("source constant bindings differ from referenced Python globals")
    def numeric(node):
        return ((isinstance(node, ast.Constant) and type(node.value) in (int, float))
                or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and numeric(node.operand)))
    for name, literal in expected.items():
        assignments = initializers.get(name, [])
        if len(assignments) != 1:
            raise ValueError("source constant requires one module declaration")
        declaration = assignments[0];call = declaration.value
        if len(call.args) != 1 or call.keywords or not numeric(call.args[0]):
            raise ValueError("source constant requires a numeric literal")
        actual = ast.get_source_segment(python_text, call.args[0])
        if actual != literal:
            raise ValueError("source constant literal differs from Python")
        for node in ast.walk(module):
            if (isinstance(node, ast.Name) and node.id == name
                    and isinstance(node.ctx, (ast.Store, ast.Del)) and node is not declaration.targets[0]):
                raise ValueError("source constant name is rebound")
            if isinstance(node, ast.arg) and node.arg == name:
                raise ValueError("source constant name is shadowed")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                raise ValueError("source constant name is rebound")
            if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == name:
                raise ValueError("source constant name is rebound")


def check_unary_imports(python_text: str, expected: dict[str, str]) -> None:
    """Resolve imports and a direct imported-module attribute alias, without execution.

    `from ... import tanh as f` and `from ... import libdevice; f = libdevice.tanh`
    bind the same function object. Calls, computed attributes and rebinding a
    checked alias are rejected. An earlier reassignment invalidates a module
    import before an attribute alias can use it.
    """
    imported, bound = {}, {}

    def target(expr):
        if isinstance(expr, ast.Name):
            return imported.get(expr.id)
        if isinstance(expr, ast.Attribute):
            parent = target(expr.value)
            return parent + "." + expr.attr if parent else None
        return None

    def bind(name, value):
        if name in bound:
            raise ValueError("unary source import name is rebound")
        imported[name] = value
        if name in expected:
            bound[name] = value

    for node in ast.parse(python_text).body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level:
                raise ValueError("relative import makes unary source binding ambiguous")
            if any(a.name == "*" for a in node.names):
                raise ValueError("wildcard import makes unary source binding ambiguous")
            for alias in node.names:
                bind(alias.asname or alias.name, node.module + "." + alias.name)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name in expected:
                    raise ValueError("unary source import name is rebound")
                bind(name, alias.name if alias.asname else name)
            continue
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in expected):
            name = node.targets[0].id
            value = target(node.value)
            if value != expected[name]:
                raise ValueError("unary source import differs from the original Python binding")
            bind(name, value)
            continue
        for child in ast.walk(node):
            if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(child.ctx, (ast.Store, ast.Del)):
                root = child.value
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name):
                    if root.id in expected:
                        raise ValueError("unary source import name is rebound")
                    imported.pop(root.id, None)
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                if child.id in expected:
                    raise ValueError("unary source import name is rebound")
                imported.pop(child.id, None)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child.name in expected:
                    raise ValueError("unary source import name is rebound")
                imported.pop(child.name, None)
    if bound != expected:
        raise ValueError("unary source import differs from the original Python binding")


def check_binary_imports(python_text: str, expected: dict[str, str]) -> dict[str, list[str]]:
    """Check the caller-selected primary import and record every fallback.

    ImportError fallback paths are provenance, not asserted intrinsic identities.
    Only a direct import or a pure nested ImportError/import ladder is accepted;
    arbitrary code, rebinding and shadowing fail closed.
    """
    module = ast.parse(python_text)
    result = {}
    allowed = set()
    targets = {"triton.language.extra.cuda.libdevice.pow",
               "triton.language.math.pow", "triton.language.libdevice.pow"}
    def ladder(nodes, name):
        if len(nodes) != 1:
            raise ValueError("binary source import branch must contain one import")
        node = nodes[0]
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module is None or len(node.names) != 1:
                raise ValueError("ambiguous binary source import")
            alias = node.names[0]
            target = node.module + "." + alias.name
            if (alias.asname or alias.name) != name or target not in targets:
                raise ValueError("unsupported binary source import branch")
            allowed.add(id(alias))
            return [target]
        if (isinstance(node, ast.Try) and not node.orelse and not node.finalbody
                and len(node.handlers) == 1 and node.handlers[0].name is None
                and isinstance(node.handlers[0].type, ast.Name)
                and node.handlers[0].type.id == "ImportError"):
            return ladder(node.body, name) + ladder(node.handlers[0].body, name)
        raise ValueError("binary source import needs a pure ImportError fallback")
    for name, target in expected.items():
        declarations = [n for n in module.body if any(
            isinstance(a, ast.alias) and (a.asname or a.name) == name for a in ast.walk(n))]
        if len(declarations) != 1:
            raise ValueError("binary source import requires one declaration")
        alternatives = ladder(declarations, name)
        if alternatives[0] != target or len(alternatives) != len(set(alternatives)):
            raise ValueError("binary source primary binding differs from the caller")
        result[name] = alternatives
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            raise ValueError("wildcard import makes binary source binding ambiguous")
        if isinstance(node, ast.ExceptHandler) and node.name in expected:
            raise ValueError("binary source import name is shadowed")
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in expected:
            raise ValueError("binary source import name is shadowed")
        if isinstance(node, ast.MatchMapping) and node.rest in expected:
            raise ValueError("binary source import name is shadowed")
        if isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) in expected:
            if id(node) not in allowed:
                raise ValueError("binary source import name is rebound")
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in expected:
            raise ValueError("binary source import name is rebound")
        if isinstance(node, ast.arg) and node.arg in expected:
            raise ValueError("binary source import name is shadowed")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in expected:
            raise ValueError("binary source import name is rebound")
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)): root = root.value
            if isinstance(root, ast.Name) and root.id in expected:
                raise ValueError("binary source import object is mutated")
    return result


def check(python_text: str, lean_text: str, python_function: str, lean_definition: str,
          signature_definition: str | None = None, expected_signature: dict | None = None,
          expected_call_bindings: dict | None = None,
          expected_pointer_offsets: dict | None = None,
          expected_source_dtype: dict | None = None,
          expected_pointer_presence: dict | None = None,
          expected_specialized_booleans: dict | None = None,
          expected_constexpr_bindings: dict | None = None,
          expected_signature_arguments: list[str] | None = None,
          expected_dtype_bindings: dict[str, str] | None = None) -> dict:
    native_ones: set[str] = set()
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1:
        raise ValueError("expected one explicitly selected @triton.jit function")
    function = functions[0]
    decorators = [d for d in function.decorator_list
                  if ast.unparse(d.func if isinstance(d, ast.Call) else d) == "triton.jit"]
    if len(decorators) != 1:
        raise ValueError("expected one explicitly selected @triton.jit function")
    jit_options = None
    if isinstance(decorators[0], ast.Call):
        decorator = decorators[0]
        if decorator.args:
            raise ValueError("unsupported positional triton.jit decorator arguments")
        jit_options = {}
        parameters = [a.arg for a in function.args.args]
        for kw in decorator.keywords:
            if kw.arg == "launch_metadata":
                if kw.arg in jit_options:
                    raise ValueError("duplicate triton.jit decorator option")
                if not isinstance(kw.value, ast.Name):
                    raise ValueError("launch_metadata requires a named top-level callback")
                callbacks = [n for n in ast.parse(python_text).body
                             if isinstance(n, ast.FunctionDef) and n.name == kw.value.id]
                if len(callbacks) != 1 or callbacks[0].decorator_list:
                    raise ValueError("launch_metadata requires one undecorated top-level callback")
                # Profiling metadata does not enter the pasted JIT body. Record
                # its source spelling; no host callback execution/proof is claimed.
                jit_options[kw.arg] = kw.value.id
                continue
            if kw.arg not in ("do_not_specialize", "do_not_specialize_on_alignment"):
                raise ValueError("unsupported triton.jit decorator option")
            if kw.arg in jit_options:
                raise ValueError("duplicate triton.jit decorator option")
            try:
                values = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                raise ValueError("JIT specialization options require literal argument lists") from None
            if values is not None and (not isinstance(values, (list, tuple)) or any(
                    not ((isinstance(v, str) and v in parameters)
                         or (type(v) is int and 0 <= v < len(parameters))) for v in values)):
                raise ValueError("JIT specialization option refers to an unknown argument")
            # Record source spelling rather than interpreting a compiler option
            # as a constexpr annotation. Native compilation remains separate.
            jit_options[kw.arg] = ast.unparse(kw.value)
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
    source_dtype = None
    pointer_presence = None
    presence_syntax = list(re.finditer(r"\bpointer_presence\s*\[([^\]]*)\]\s+(?=in\b)", signature))
    if presence_syntax:
        if len(presence_syntax) != 1 or signature_definition is None:
            raise ValueError("pointer presence requires one checked call signature")
        pointer_presence = {}
        for binding in presence_syntax[0][1].split(","):
            entry = re.fullmatch(r"\s*(\w+)\s*:=\s*([\w.]+)\s*", binding)
            if not entry or entry[1] in pointer_presence:
                raise ValueError("unsupported or duplicate pointer presence binding")
            pointer_presence[entry[1]] = entry[2]
        signature = signature[:presence_syntax[0].start()] + signature[presence_syntax[0].end():]
    if pointer_presence != expected_pointer_presence:
        raise ValueError("pointer presence differs from the reviewed invocation")
    if expected_source_dtype is not None and signature_definition is None:
        raise ValueError("source dtype binding requires a checked call signature")
    if expected_pointer_offsets is not None and signature_definition is None:
        raise ValueError("pointer offsets require a checked call signature")
    helper = re.search(r"\btriton_function\s+(\w+)\s*\(([^)]*)\)(?:\s+constexpr\s*\[([^\]]*)\])?\s+in\s+triton\s*$", signature)
    if helper:
        if (signature_definition is not None or expected_signature is not None
                or expected_call_bindings is not None or expected_constexpr_bindings is not None):
            raise ValueError("source helpers use inferred argument values, not a kernel ABI")
        arguments = [name.strip() for name in helper[2].split(",") if name.strip()]
        if helper[1] != python_function or arguments != [a.arg for a in function.args.args]:
            raise ValueError("source helper name or parameters differ from Python")
        declared_constants = [name.strip() for name in (helper[3] or "").split(",") if name.strip()]
        if (len(declared_constants) != len(set(declared_constants))
                or set(declared_constants) != constants):
            raise ValueError("source helper constexpr parameters differ from Python")
        if runtime or len(arguments) != len(set(arguments)):
            raise ValueError("unsupported source helper parameter annotations")
        profile = {"pointers": [], "scalars": [], "constexpr": declared_constants}
    elif signature_definition is not None:
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
        signature_pattern = re.escape(signature_definition)
        if expected_signature_arguments is not None:
            if (not expected_signature_arguments or not all(isinstance(a, str) and
                    re.fullmatch(r"[A-Za-z_]\w*", a) for a in expected_signature_arguments)):
                raise ValueError("signature arguments must be reviewed caller names")
            signature_pattern = (r"\(\s*" + signature_pattern + r"\s+" +
                                 r"\s+".join(map(re.escape, expected_signature_arguments)) + r"\s*\)")
        dtype_call = re.search(r"\bwith_triton_signature\s+" + signature_pattern
                               + r"\s+source_dtype\s+(\w+)\s*:=\s*([\w.]+)\s+in\b", signature)
        if dtype_call:
            source_dtype = {"pointer": dtype_call[1], "value": dtype_call[2]}
            if source_dtype != expected_source_dtype:
                raise ValueError("source dtype binding differs from the reviewed invocation")
        elif not re.search(r"\bwith_triton_signature\s+" + signature_pattern + r"\s+in\b", signature):
            call = re.search(r"\bwith_triton_call\s+" + signature_pattern
                             + r"\s+call_args\s*\[([^\]]*)\](?:\s+pointer_offsets\s*\[([^\]]*)\])?\s+in\b", signature)
            if not call:
                raise ValueError("quotation does not use the declared call signature")
            call_bindings = {}
            for binding in call[1].split(",") if call[1].strip() else []:
                match = re.fullmatch(r"\s*(\w+)\s*:=\s*(.+?)\s*", binding, re.S)
                if not match or match[1] in call_bindings:
                    raise ValueError("unsupported or duplicate per-call scalar binding")
                call_bindings[match[1]] = scalar_binding_term(match[2])
            reviewed_bindings = ({k: scalar_binding_term(v) for k, v in expected_call_bindings.items()}
                                 if expected_call_bindings is not None else None)
            if reviewed_bindings is None or call_bindings != reviewed_bindings:
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
        if source_dtype != expected_source_dtype:
            raise ValueError("source dtype binding differs from the reviewed invocation")
        profile = call_signature(lean_text, signature_definition)
        if len(profile.get("parameters", [])) != len(expected_signature_arguments or []):
            raise ValueError("signature constructor argument count differs from its declaration")
        if set(profile.get("optionalPointers", [])) != set(pointer_presence or {}):
            raise ValueError("optional pointer declarations and caller presence bindings differ")
        if profile.get("sourceDTypePointers", []) != ([source_dtype["pointer"]] if source_dtype else []):
            raise ValueError("source dtype pointer declaration and invocation differ")
        if expected_signature is not None and profile != expected_signature:
            raise ValueError("call signature differs from the reviewed launch types")
        dtype_bindings = dict(re.findall(
            r"\blet\s+(\w+)\s*:\s*FloatDType\s*:=\s*\.(\w+)\b", signature))
        dtype_bindings = {n: v for n, v in dtype_bindings.items() if n in profile["constexpr"]}
        if dtype_bindings != (expected_dtype_bindings or {}):
            raise ValueError("source dtype constexpr differs from the reviewed caller binding")
        if dtype_bindings and (dtype_bindings != {"dtype": "fp32"} or
                               not profile.get("typedFP32Storage", False)):
            raise ValueError("only the checked typed-FP32 dtype constexpr is supported")
        for name, value in (expected_constexpr_bindings or {}).items():
            if name not in profile["constexpr"]:
                raise ValueError("constexpr binding requires a declared natural-valued parameter")
            if isinstance(value, dict) and set(value) == {"parameter"} and isinstance(value["parameter"], str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value["parameter"]):
                values = re.findall(r"\blet\s+" + re.escape(name)
                                    + r"\s*:\s*Nat\s*:=\s*([A-Za-z_][A-Za-z_0-9]*)\s*(?=\n)", signature)
                if values != [value["parameter"]]:
                    raise ValueError("constexpr caller alias differs from the reviewed invocation")
            else:
                if type(value) is not int or value < 0:
                    raise ValueError("constexpr binding requires a declared natural-valued parameter")
                values = re.findall(r"\blet\s+" + re.escape(name)
                                    + r"\s*:\s*Nat\s*:=\s*(\d+)\s*(?=\n)", signature)
                if len(values) != 1 or int(values[0]) != value:
                    raise ValueError("constexpr caller value differs from the reviewed invocation")
        specialized = profile.get("specializeBooleans", [])
        if set(specialized) != set(expected_specialized_booleans or {}):
            raise ValueError("Boolean specializations differ from the reviewed invocation")
        for name in specialized:
            values = re.findall(r"\blet\s+" + re.escape(name) + r"\s*:\s*Bool\s*:=\s*Bool\.(true|false)\b", signature)
            expected = expected_specialized_booleans[name]
            if type(expected) is not bool or len(values) != 1 or (values[0] == "true") != expected:
                raise ValueError("specialized Boolean caller value differs from the reviewed invocation")
        imported = {(a.asname or a.name, node.module + "." + a.name)
                    for node in ast.walk(ast.parse(python_text))
                    if isinstance(node, ast.ImportFrom) and node.module is not None
                    for a in node.names}
        for alias, target in profile.get("finiteImports", []):
            if (alias, target) not in imported:
                raise ValueError("finite source import differs from the original Python binding")
        check_source_constants(python_text, function, dict(profile.get("sourceConstants", [])))
        binary_imports = dict(profile.get("binaryImports", []))
        if binary_imports:
            check_binary_imports(python_text, binary_imports)
        unary_imports = dict(profile.get("unaryImports", []))
        if unary_imports:
            check_unary_imports(python_text, unary_imports)
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tanh"
               for n in ast.walk(function)) and "tanh" not in unary_imports:
            raise ValueError("bare tanh requires its original source import binding")
        names = ({n for n, _ in profile["pointers"] + profile["scalars"]} | set(profile["constexpr"])
                 | set(profile.get("sourceDTypePointers", [])))
        # The native scalar specializer treats an unannotated raw integer 1
        # as constexpr unless do_not_specialize excludes that parameter. The
        # reviewed Nat-valued caller binding above must pin exactly that case;
        # arbitrary runtime scalars cannot be relabeled as constexpr.
        native_ones = set(profile["constexpr"]) - constants
        disabled = []
        if isinstance(decorators[0], ast.Call):
            for kw in decorators[0].keywords:
                if kw.arg == "do_not_specialize":
                    disabled = ast.literal_eval(kw.value) or []
        arguments_by_name = {a.arg: (i, a) for i, a in enumerate(function.args.args)}
        for name in native_ones:
            entry = arguments_by_name.get(name)
            if (entry is None or annotation(entry[1]) is not None
                    or (expected_constexpr_bindings or {}).get(name) != 1
                    or name in disabled or entry[0] in disabled):
                raise ValueError("Python parameters and call signature differ: runtime constexpr specialization requires enabled unannotated raw integer one")
        if names != set(arguments_by_name) or set(profile["constexpr"]) != constants | native_ones:
            raise ValueError("Python parameters and call signature differ")
        # `integer` is intentionally unbounded; it cannot certify an explicit
        # int32/int64 ABI annotation. Pointer annotations are not scalar args.
        scalar_types = {"fp32": "tl.float32", "fp64": "tl.float64", "i32": "tl.int32", "i64": "tl.int64", "u32": "tl.uint32", "u64": "tl.uint64"}
        supplied = {n: scalar_types.get(t) for n, t in profile["scalars"]}
        pointer_names = {n for n, _ in profile["pointers"]} | set(profile.get("sourceDTypePointers", []))
        if pointer_offsets is not None and not set(pointer_offsets) <= pointer_names:
            raise ValueError("offset binding is not a pointer parameter")
        if call_bindings is not None:
            integer_names = {n for n, t in profile["scalars"] if t in ("i32", "i64", "u32", "u64", "integer")}
            fp32_names = {n for n, t in profile["scalars"] if t == "fp32"}
            if not integer_names <= set(call_bindings) <= integer_names | fp32_names:
                raise ValueError("per-call bindings must cover every integer scalar and only supported scalar kinds")
        for arg in function.args.args:
            t = annotation(arg)
            if t is None or arg.arg in constants:
                continue
            if arg.arg in pointer_names and t == "tl.tensor":
                continue
            if t == "int" and dict(profile["scalars"]).get(arg.arg) == "integer":
                # Python int is not a fixed native-width Triton annotation.
                # Keep the existing exact caller value and separate width proof.
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
    if profile is not None and native_ones:
        result["native_integer_one_specializations"] = sorted(native_ones)
    if helper:
        result["body_kind"] = "helper"
        result["helper_parameters"] = arguments
        result["argument_mode"] = "inferred_helper_values"
    if profile is not None and "functions" in profile:
        result["source_functions"] = profile["functions"]
    if pointer_offsets is not None:
        result["pointer_offsets"] = pointer_offsets
    if pointer_presence is not None:
        result["pointer_presence"] = pointer_presence
    if source_dtype is not None:
        result["source_dtype"] = source_dtype
    if expected_dtype_bindings is not None:
        result["dtype_bindings"] = dtype_bindings
    if jit_options is not None:
        result["jit_options"] = jit_options
    if profile is not None and profile.get("binaryImports"):
        result["source_binary_import_choices"] = check_binary_imports(python_text, dict(profile["binaryImports"]))
    return result


def validate_manifest_records(records: list[dict]) -> None:
    """Allow explicit distinct ABI profiles of one Python function.

    A profile tag cannot excuse a duplicate mapping. Besides distinct Lean
    definitions, require distinct checked ABIs, checked constexpr values or sets of scalar
    arguments specialized at the call. The latter checks both literal and
    runtime-register interfaces of the same source, without treating argument
    alias renaming or an empty binding list as a different interface.
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
            bindings = record.get("expected_call_bindings") or {}
            if not isinstance(bindings, dict):
                raise ValueError("reused Python correspondence requires checked call bindings")
            abis.append(json.dumps({
                "abi": {key: signature.get(key) for key in ("pointers", "scalars", "constexpr", "optionalPointers")},
                "specialized_scalars": sorted(bindings),
                "constexpr_bindings": record.get("expected_constexpr_bindings") or {},
                "specialized_booleans": record.get("expected_specialized_booleans") or {},
            }, sort_keys=True))
        if len(abis) != len(set(abis)):
            raise ValueError("reused Python correspondence has duplicate ABI profiles and scalar call modes")


def check_manifest(root: Path, manifest: Path) -> list[dict]:
    records = json.loads(manifest.read_text())
    validate_manifest_records(records)
    results = []
    for record in records:
        result = check((root / record["python"]).read_text(), (root / record["lean"]).read_text(),
                       record["python_function"], record["lean_definition"],
                       record.get("signature_definition"), record.get("expected_signature"),
                       record.get("expected_call_bindings"), record.get("expected_pointer_offsets"),
                       record.get("expected_source_dtype"), record.get("expected_pointer_presence"),
                       record.get("expected_specialized_booleans"), record.get("expected_constexpr_bindings"),
                       record.get("expected_signature_arguments"), record.get("expected_dtype_bindings"))
        if "profile_variant" in record:
            result["profile_variant"] = record["profile_variant"]
        results.append(result)
    for record, result in zip(records, results):
        for helper in result.get("source_functions", []):
            if not any(other["lean"] == record["lean"] and other["lean_definition"] == helper
                       and checked.get("body_kind") == "helper"
                       for other, checked in zip(records, results)):
                raise ValueError("registered source helper lacks a checked original body")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(check_manifest(args.root, args.manifest), indent=2))
