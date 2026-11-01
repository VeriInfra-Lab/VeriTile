"""Compare complete Python JIT bodies with their Lean source quotations.

Only syntax adaptations are allowed: braces/indentation, parameter antiquotes,
and tuple/list shape spelling in tl.full/tl.zeros. Casts, calls, branches, operands and
statement order are compared as ASTs. Unsupported syntax fails closed.
This checks source correspondence; it does not prove compiler correctness.
"""
from __future__ import annotations

# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import ast
import copy
import difflib
import io
import tokenize
import json
from pathlib import Path
import re
import textwrap

from scripts.bench.audit_source import lean_code_mask, strip_lean_comments


def quotation(text: str, definition: str, *, raw_python: bool = False) -> tuple[str, str]:
    mask = lean_code_mask(text)
    matches = list(re.finditer(r"(?m)^\s*(?:(?:noncomputable|private|protected)\s+)*def\s+" + re.escape(definition) + r"\b", mask))
    if not matches:
        raise ValueError(f"missing Lean definition: {definition}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous Lean definition: {definition}")
    start = matches[0]
    next_def = re.search(r"\n(?:(?:noncomputable|private|protected)\s+)*(?:def|abbrev|theorem|specification|inductive|structure|end|set_option)\b", mask[start.end():])
    end = start.end() + next_def.start() if next_def else len(text)
    selected, code = text[start.start():end], mask[start.start():end]
    begin = re.search(r"\b(?:triton|triton_function)\b", code)
    if not begin:
        raise ValueError(f"definition has no direct quotation: {definition}")
    opening = code.find("{", begin.end())
    if opening < 0:
        raise ValueError("missing quotation body")
    if raw_python:
        # The literal body uses Python comments and strings. Tokenization
        # keeps braces in either out of the quotation delimiter count.
        fragment = selected[opening:]
        offsets = [0]
        for line in fragment.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        depth = 0
        try:
            for token in tokenize.generate_tokens(io.StringIO(fragment).readline):
                if token.type != tokenize.OP:
                    continue
                if token.string == "{":
                    depth += 1
                elif token.string == "}":
                    depth -= 1
                    if depth == 0:
                        stop = opening + offsets[token.end[0] - 1] + token.end[1]
                        if code[stop:].strip():
                            raise ValueError("unexpected code after quotation")
                        return selected[:opening], selected[opening + 1:stop - 1]
        except (tokenize.TokenError, IndentationError) as error:
            raise ValueError("invalid Python quotation delimiters") from error
        raise ValueError("unterminated quotation")
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
        if ast.unparse(node.func) in {"tl.full", "tl.zeros"}:
            if node.args and isinstance(node.args[0], ast.Tuple):
                node.args[0] = ast.List(node.args[0].elts, ast.Load())
            if ast.unparse(node.func) == "tl.full" and len(node.args) == 3:
                if any(k.arg == "dtype" for k in node.keywords):
                    raise ValueError("duplicate full dtype")
                node.keywords.append(ast.keyword("dtype", node.args.pop()))
        return node



def call_signature(text: str, name: str) -> dict:
    code = strip_lean_comments(text)
    matches = list(re.finditer(r"(?m)^\s*def\s+" + re.escape(name) +
                              r"\s*((?:\([^\n()]*\)\s*)*):\s*CallSignature\s+where\s*"
                              r"(.*?)(?=\ndef |\nnoncomputable |\nend\b|\Z)", code, re.S))
    if len(matches) != 1:
        raise ValueError("missing declarative call signature")
    match = matches[0]
    dtype_parameters = []
    parameter_types = {}
    for group in re.findall(r"\(([^()]*)\)", match[1]):
        parameter = re.fullmatch(r"\s*([A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*)\s*:\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*", group)
        if not parameter:
            raise ValueError("unsupported call signature parameter")
        if parameter[2] in {"Nat", "Int", "Real", "String", "Type", "Prop"}:
            raise ValueError("call signature dtype parameters require finite enum types")
        for name in parameter[1].split():
            parameter_types[name] = parameter[2]
            dtype_parameters.append(name)
    if len(dtype_parameters) != len(set(dtype_parameters)):
        raise ValueError("duplicate call signature dtype parameter")
    fields = {}
    for field in ("pointers", "scalars", "constexpr"):
        value = re.search(r"\b" + field + r"\s*:=\s*(\[[^\]]*\])", match[2])
        if not value:
            if field == "pointers" and dtype_parameters:
                fields[field] = []
                continue
            raise ValueError(f"missing call signature field: {field}")
        source = value[1]
        if field == "pointers":
            source_types = {"fp16", "bf16", "fp32", "fp64", "f8e4", "f8e5", "i8", "i16", "i32", "i64",
                            "u8", "u16", "u32", "u64", "boolean"}
            selections = {}
            marker = "__veritile_profile_selection_"
            while marker in source:
                marker += "_"
            def finite_selection(match):
                parameter, branches = match[1], match[2]
                if parameter not in parameter_types:
                    raise ValueError("finite pointer selection uses an undeclared dtype parameter")
                cases = re.findall(r"\|\s*\.([A-Za-z_]\w*)\s*=>\s*\.([A-Za-z_]\w*)", branches)
                if (not cases or len({constructor for constructor, _ in cases}) != len(cases)
                        or any(dtype not in source_types for _, dtype in cases)):
                    raise ValueError("invalid finite pointer dtype selection")
                name = marker + str(len(selections))
                selections[name] = {"on": parameter, "cases": [list(pair) for pair in cases]}
                return name
            source = re.sub(r"\bmatch\s+([A-Za-z_]\w*)\s+with"
                            r"((?:\s*\|\s*\.[A-Za-z_]\w*\s*=>\s*\.[A-Za-z_]\w*)+)",
                            finite_selection, source)
            source = re.sub(r"\.(fp16|bf16|fp32|fp64|f8e4|f8e5|i8|i16|i32|i64|u8|u16|u32|u64|boolean)\b", r'"\1"', source)
            tree = ast.parse(source, mode="eval").body
            if not isinstance(tree, ast.List):
                raise ValueError("unsupported pointer signature")
            pairs = []
            for pair in tree.elts:
                if not (isinstance(pair, ast.Tuple) and len(pair.elts) == 2
                        and isinstance(pair.elts[0], ast.Constant) and type(pair.elts[0].value) is str):
                    raise ValueError("pointer signature requires named dtype pairs")
                dtype = pair.elts[1]
                if isinstance(dtype, ast.Name) and dtype.id in selections:
                    selected = selections[dtype.id]
                elif isinstance(dtype, ast.Name):
                    if parameter_types.get(dtype.id) != "SourceDType":
                        raise ValueError("pointer dtypes must use declared SourceDType parameters")
                    selected = dtype.id
                elif isinstance(dtype, ast.Constant) and dtype.value in source_types:
                    selected = dtype.value
                else:
                    raise ValueError("unsupported source pointer dtype")
                pairs.append([pair.elts[0].value, selected])
            fields[field] = pairs
        else:
            source = re.sub(r"\.(fp16|bf16|fp32|fp64|f8e4|f8e5|i8|i16|i32|i64|u8|u16|u32|u64|boolean|integer)\b", r'"\1"', source)
            fields[field] = json.loads(json.dumps(ast.literal_eval(source)))
    for field in ("readPorts", "writePorts", "sourceConstants", "unaryImports"):
        value = re.search(r"\b" + field + r"\s*:=\s*(\[[^\]]*\])", match[2])
        if value:
            fields[field] = json.loads(json.dumps(ast.literal_eval(value[1])))
            if field in {"sourceConstants", "unaryImports"} and any(
                    not isinstance(pair, list) or len(pair) != 2 or
                    not all(isinstance(item, str) for item in pair) for pair in fields[field]):
                raise ValueError("module bindings require pairs of source names and literal strings")
    floating = re.search(r"\bfloatingPointers\s*:=\s*(\[[^\]]*\])", match[2])
    floating_parameters = [name for name in dtype_parameters if parameter_types[name] == "FloatDType"]
    if floating_parameters or floating:
        if not floating:
            raise ValueError("missing polymorphic pointer signature")
        tree = ast.parse(floating[1], mode="eval").body
        if not isinstance(tree, ast.List):
            raise ValueError("unsupported polymorphic pointer signature")
        pairs = []
        for pair in tree.elts:
            if not (isinstance(pair, ast.Tuple) and len(pair.elts) == 2
                    and isinstance(pair.elts[0], ast.Constant) and type(pair.elts[0].value) is str
                    and isinstance(pair.elts[1], ast.Name) and pair.elts[1].id in floating_parameters):
                raise ValueError("polymorphic pointers must use declared dtype parameters")
            pairs.append([pair.elts[0].value, pair.elts[1].id])
        fields["floatingPointers"] = pairs
    if dtype_parameters:
        fields["dtype_parameters"] = dtype_parameters
        if any(parameter_types[name] != "FloatDType" for name in dtype_parameters):
            fields["dtype_parameter_types"] = parameter_types
    names = [n for n, _ in fields["pointers"] + fields.get("floatingPointers", []) + fields["scalars"]] + fields["constexpr"]
    if len(names) != len(set(names)):
        raise ValueError("duplicate call signature parameter")
    return fields

def check(python_text: str, lean_text: str, python_function: str, lean_definition: str,
          signature_definition: str | None = None, expected_signature: dict | None = None,
          expected_call_bindings: dict[str, str] | None = None,
          expected_pointer_offsets: dict[str, str] | None = None,
          expected_pointer_cells: dict[str, list[str]] | None = None,
          expected_optional_pointers: dict[str, str] | None = None,
          expected_static_strides: dict[str, str] | None = None,
          expected_pure_functions: dict[str, str] | None = None,
          expected_integer_args: dict[str, str] | None = None,
          expected_arange_args: dict[str, str] | None = None) -> dict:
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d.func if isinstance(d, ast.Call) else d) == "triton.jit"
                                      for d in functions[0].decorator_list):
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
    signature, body = quotation(lean_text, lean_definition, raw_python=signature_definition is not None)
    profile = None
    if signature_definition is not None:
        if expected_signature is None:
            raise ValueError("pasted body requires an independently reviewed call signature")
        if re.search(r"\btl\.(region|param|call_signature)\b|\$", body):
            raise ValueError("pasted body must not contain metadata directives or Lean antiquotes")
        profile = call_signature(lean_text, signature_definition)
        dtype_arguments = profile.get("dtype_parameters", [])
        selected_signature = (r"\(\s*" + re.escape(signature_definition) + r"\s+" +
                              r"\s+".join(map(re.escape, dtype_arguments)) + r"\s*\)"
                              if dtype_arguments else re.escape(signature_definition))
        offset_clause = (r"\s+pointer_offsets\s*\[([^\]]*)\]"
                         if expected_pointer_offsets is not None else "")
        if expected_pointer_cells is not None:
            if expected_pointer_offsets is not None or expected_call_bindings is not None:
                raise ValueError("pointer cell views require a signature without scalar bindings or pointer offsets")
            offset_clause = r"\s+pointer_cells\s*\[([^\]]*)\]"
        if expected_optional_pointers is not None:
            if expected_pointer_offsets is not None or expected_pointer_cells is not None:
                raise ValueError("optional pointer metadata cannot be mixed with coordinate bindings")
            offset_clause = r"\s+optional_pointers\s*\[([^\]]*)\]"
        if expected_static_strides is not None:
            if (expected_call_bindings is None or expected_pointer_offsets is not None
                    or expected_pointer_cells is not None or expected_optional_pointers is not None):
                raise ValueError("static strides require the reviewed scalar-only invocation")
            offset_clause = r"\s+static_strides"
        if expected_pure_functions is not None:
            if (expected_call_bindings is None or expected_pointer_offsets is not None
                    or expected_pointer_cells is not None or expected_optional_pointers is not None
                    or expected_static_strides is not None or not expected_pure_functions):
                raise ValueError("pure JIT helpers require a reviewed scalar-only call")
            offset_clause = r"\s+pure_functions\s*\[([^\]]*)\]"
        if expected_integer_args is not None or expected_arange_args is not None:
            if (expected_integer_args is None or expected_arange_args is None
                    or expected_call_bindings is None or expected_pointer_offsets is not None
                    or expected_pointer_cells is not None or expected_optional_pointers is not None
                    or expected_static_strides is not None or expected_pure_functions is not None):
                raise ValueError("integer/arange arguments require a reviewed scalar-only call")
            offset_clause = (r"\s+integer_args\s*\[([^\]]*)\]"
                             r"\s+arange_args\s*\[([^\]]*)\]")
        if expected_call_bindings is None:
            invocation = r"\bwith_triton_signature\s+" + selected_signature + offset_clause + r"\s+in\b"
            calls = list(re.finditer(invocation, signature))
            if len(calls) != 1:
                raise ValueError("quotation does not use the declared call signature")
        else:
            invocation = (r"\bwith_triton_call\s+" + selected_signature +
                          r"\s+call_args\s*\[([^\]]*)\]" + offset_clause + r"\s+in\b")
            calls = list(re.finditer(invocation, signature))
            if len(calls) != 1:
                raise ValueError("quotation must use the reviewed runtime call bindings")
            bindings = {}
            for entry in calls[0][1].split(","):
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|[0-9]+)\s*", entry)
                if not pair or pair[1] in bindings:
                    raise ValueError("runtime call bindings require distinct named caller arguments")
                bindings[pair[1]] = pair[2]
            if bindings != expected_call_bindings:
                raise ValueError("runtime call bindings differ from the reviewed invocation")
            integer_args = {n for n, t in profile["scalars"] if t in {"integer", "i32", "i64"}}
            fp32_args = {n for n, t in profile["scalars"] if t == "fp32"}
            handled = set(expected_integer_args or {})
            if (not handled <= integer_args or not (integer_args-handled) <= set(bindings)
                    or set(bindings) & handled or not set(bindings) <= integer_args | fp32_args):
                raise ValueError("runtime call bindings must include all integer parameters and only supported scalar parameters")
        if expected_integer_args is not None:
            for group, reviewed, label in [(2, expected_integer_args, "integer"),
                                             (3, expected_arange_args, "arange")]:
                values = {}
                text = calls[0][group].strip()
                for entry in text.split(",") if text else []:
                    pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*)\s*", entry)
                    if not pair or pair[1] in values:
                        raise ValueError(f"{label} arguments require distinct named caller bindings")
                    values[pair[1]] = pair[2]
                if values != reviewed:
                    raise ValueError(f"{label} arguments differ from the reviewed call")
            for caller in expected_integer_args.values():
                if not re.search(r"\([^()\n]*\b"+re.escape(caller)+
                                 r"\s*:\s*(?:VeriTile\.Triton\.)?IntegerArgument\s*\)", signature):
                    raise ValueError("integer ABI values require an explicit IntegerArgument caller")
            for size, proof in expected_arange_args.items():
                if size not in profile["constexpr"] or not re.search(
                        r"\([^()\n]*\b"+re.escape(proof)+
                        r"\s*:\s*(?:VeriTile\.Triton\.)?Random\.ArangeSizeDomain\s+"+
                        re.escape(size)+r"\s*\)", signature):
                    raise ValueError("arange arguments require their checked constexpr-size proof")
        if expected_pure_functions is not None:
            bindings = {}
            for item in calls[0][2].split(","):
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*)\s*", item)
                if not pair or pair[1] in bindings:
                    raise ValueError("pure JIT functions require distinct named source definitions")
                bindings[pair[1]] = pair[2]
            if bindings != expected_pure_functions:
                raise ValueError("pure JIT function bindings differ from the reviewed call")
            actual = {n.func.id for n in ast.walk(function) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id not in {"range", "float", "int"}}
            if actual != set(bindings):
                raise ValueError("pure JIT binding names differ from the Python callees")
        if expected_static_strides is not None:
            strides = {}
            definitions = {}
            for index, statement in enumerate(function.body):
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and isinstance(statement.value, ast.Call)
                        and ast.unparse(statement.value.func) == "tl.full"):
                    call = statement.value
                    if (len(call.args) == 3 and not call.keywords
                            and isinstance(call.args[0], ast.Tuple) and not call.args[0].elts
                            and isinstance(call.args[1], ast.Name) and call.args[1].id in constants
                            and ast.unparse(call.args[2]) == "tl.int64"):
                        name = statement.targets[0].id
                        if name in strides:
                            raise ValueError("duplicate static signed stride definition")
                        strides[name] = call.args[1].id
                        definitions[name] = index
            if not strides or strides != expected_static_strides:
                raise ValueError("static signed strides differ from the reviewed constructors")
            stores = [n.id for statement in function.body for n in ast.walk(statement)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))]
            if (any(stores.count(name) != 1 for name in strides)
                    or any(stores.count(name) != 0 for name in strides.values())):
                raise ValueError("static strides and their constexpr values must be immutable")
            for index, statement in enumerate(function.body):
                for node in ast.walk(statement):
                    if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                            and ast.unparse(node.iter.func) == "range" and len(node.iter.args) == 3
                            and isinstance(node.iter.args[2], ast.Name)):
                        name = node.iter.args[2].id
                        if name in definitions and index <= definitions[name]:
                            raise ValueError("static signed stride definition must dominate its range")
        if expected_pointer_offsets is not None:
            offsets = {}
            group = 1 if expected_call_bindings is None else 2
            offset_source = calls[0][group].strip()
            for entry in offset_source.split(",") if offset_source else []:
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|[0-9]+)\s*", entry)
                if not pair or pair[1] in offsets:
                    raise ValueError("pointer offset bindings require distinct named caller arguments")
                offsets[pair[1]] = pair[2]
            pointers = {n for n, _ in profile['pointers'] + profile.get('floatingPointers', [])}
            if offsets != expected_pointer_offsets or not set(offsets) <= pointers:
                raise ValueError("pointer offset bindings differ from the reviewed invocation")
        if expected_optional_pointers is not None:
            presence = {}
            group = 1 if expected_call_bindings is None else 2
            source = calls[0][group].strip()
            for entry in source.split(",") if source else []:
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*", entry)
                if not pair or pair[1] in presence:
                    raise ValueError("optional pointer bindings require distinct named caller presence values")
                presence[pair[1]] = pair[2]
            pointers = {n for n, _ in profile['pointers'] + profile.get('floatingPointers', [])}
            if presence != expected_optional_pointers or not set(presence) <= pointers:
                raise ValueError("optional pointer bindings differ from the reviewed invocation")
        if expected_pointer_cells is not None:
            cells = {}
            source = calls[0][1].strip()
            caller = r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|[0-9]+)"
            entry = re.compile(r"\s*([A-Za-z_]\w*)\s*:=\s*\(\s*" + caller +
                               r"\s*,\s*" + caller + r"\s*\)\s*(?:,|$)")
            while source:
                pair = entry.match(source)
                if not pair or pair[1] in cells:
                    raise ValueError("pointer cell bindings require distinct named caller base/width pairs")
                cells[pair[1]] = [pair[2], pair[3]]
                source = source[pair.end():].strip()
            pointers = {n for n, _ in profile['pointers'] + profile.get('floatingPointers', [])}
            if cells != expected_pointer_cells or not set(cells) <= pointers:
                raise ValueError("pointer cell bindings differ from the reviewed invocation")
        if expected_signature is not None and profile != expected_signature:
            raise ValueError("call signature differs from the reviewed launch types")
        module = ast.parse(python_text)
        class ModuleWrites(ast.NodeVisitor):
            def __init__(self):
                self.names = []
                self.wildcard_import = False
            def visit_Name(self, node):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    self.names.append(node.id)
            def visit_ImportFrom(self, node):
                self.wildcard_import |= any(alias.name == "*" for alias in node.names)
                self.names.extend(alias.asname or alias.name for alias in node.names)
            def visit_Import(self, node):
                self.names.extend(alias.asname or alias.name.split('.')[0] for alias in node.names)
            def visit_FunctionDef(self, node):
                self.names.append(node.name)
            visit_AsyncFunctionDef = visit_FunctionDef
            def visit_ClassDef(self, node):
                self.names.append(node.name)
        writes = ModuleWrites()
        writes.visit(module)
        module_names = [pair[0] for field in ("sourceConstants", "unaryImports")
                        for pair in profile.get(field, [])]
        if module_names and writes.wildcard_import:
            raise ValueError("wildcard imports may rebind source module bindings")
        if len(module_names) != len(set(module_names)):
            raise ValueError("duplicate source module binding")
        if any(writes.names.count(name) != 1 for name in module_names):
            raise ValueError("source module binding must have one static definition and no rebinding")
        for name, literal in profile.get("sourceConstants", []):
            assignments = [node for node in module.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == name
                                   for target in node.targets)]
            if len(assignments) != 1 or len(assignments[0].targets) != 1:
                raise ValueError("source constant needs one direct module definition")
            value = assignments[0].value
            if not (isinstance(value, ast.Call) and ast.unparse(value.func) == "tl.constexpr"
                    and len(value.args) == 1 and not value.keywords):
                raise ValueError("source constant must be an original tl.constexpr literal")
            try:
                original_value, declared_value = ast.literal_eval(value.args[0]), ast.literal_eval(literal)
            except (ValueError, SyntaxError) as error:
                raise ValueError("source constant must contain a literal integer") from error
            if type(original_value) is not int or type(declared_value) is not int or original_value != declared_value:
                raise ValueError("source constant differs from the Python module literal")
        for name, target in profile.get("unaryImports", []):
            imports = [node.module + "." + alias.name for node in module.body
                       if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                       for alias in node.names if (alias.asname or alias.name) == name]
            # Preserve the original module spelling, e.g. the FastCE source's
            # `from ...extra import libdevice; triton_tanh = libdevice.tanh`.
            # Only one direct, preceding import and one attribute assignment
            # are admissible; expressions and mutable module aliases fail closed.
            assignments = [node for node in module.body if isinstance(node, ast.Assign)
                           and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                           and node.targets[0].id == name]
            if not imports and len(assignments) == 1:
                assignment = assignments[0]
                value = assignment.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    base = value.value.id
                    roots = [node.module + "." + alias.name for node in module.body
                             if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                             and node.lineno < assignment.lineno
                             for alias in node.names if (alias.asname or alias.name) == base]
                    mutated = any(isinstance(node, ast.Attribute)
                                  and isinstance(node.ctx, (ast.Store, ast.Del))
                                  and isinstance(node.value, ast.Name) and node.value.id == base
                                  for node in ast.walk(module))
                    if len(roots) == 1 and writes.names.count(base) == 1 and not mutated:
                        imports = [roots[0] + "." + value.attr]
            if imports != [target]:
                raise ValueError("source intrinsic import differs from the Python module")
        pointer_bindings = profile["pointers"] + profile.get("floatingPointers", [])
        names = {n for n, _ in pointer_bindings + profile["scalars"]} | set(profile["constexpr"])
        if set(module_names) & names:
            raise ValueError("module bindings capture a function parameter")
        if names != {a.arg for a in function.args.args} or set(profile["constexpr"]) != constants:
            raise ValueError("Python parameters and call signature differ")
        # `integer` is intentionally unbounded; it cannot certify an explicit
        # int32/int64 ABI annotation. Pointer annotations are not scalar args.
        scalar_types = {"fp32": "tl.float32", "fp64": "tl.float64",
                        "i32": "tl.int32", "i64": "tl.int64"}
        supplied = {n: scalar_types.get(t) for n, t in profile["scalars"]}
        pointer_names = {n for n, _ in pointer_bindings}
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
        statements, parameters = ast.parse(textwrap.dedent(body)).body, {}
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
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body" if profile is not None else "legacy_ast",
            "statements": len(function.body),
            "runtime_parameters": dict(profile["scalars"]) if profile is not None else runtime, "constexpr_parameters": sorted(constants)}


def check_pure_function(python_text: str, lean_text: str, python_function: str,
                        lean_definition: str) -> dict:
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d) == "triton.jit" for d in functions[0].decorator_list):
        raise ValueError("expected one explicitly selected @triton.jit helper")
    fn = functions[0]
    if (len(fn.args.args) != 1 or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg
            or fn.args.kwarg or fn.args.defaults or fn.args.kw_defaults
            or fn.args.args[0].annotation or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return)):
        raise ValueError("pure JIT helper requires one unannotated parameter and one return")
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    argument = fn.args.args[0].arg
    if not re.search(r"\btriton_function\s*\(\s*" + re.escape(argument) + r"\s*\)\s*$", header):
        raise ValueError("pure JIT quotation parameter differs from Python")
    if "$" in body or re.search(r"\btl\.(region|param|source_helper|call_signature)\b", body):
        raise ValueError("pure JIT body must contain only pasted Python")
    if ast.dump(ast.Module(fn.body, [])) != ast.dump(ast.parse(textwrap.dedent(body))):
        raise ValueError("pure JIT return body differs from Python")
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body", "kind": "pure_jit_function",
            "statements": 1, "runtime_parameters": {argument: "inferred"}, "constexpr_parameters": []}


def check_manifest(root: Path, manifest: Path) -> list[dict]:
    records = json.loads(manifest.read_text())
    results = []
    for r in records:
        py, lean = (root / r["python"]).read_text(), (root / r["lean"]).read_text()
        if r.get("kind") == "pure_jit_function":
            if set(r) != {"kind", "python", "lean", "python_function", "lean_definition"}:
                raise ValueError("pure JIT helper record cannot override kernel call metadata")
            results.append(check_pure_function(py, lean, r["python_function"], r["lean_definition"]))
            continue
        results.append(check(py, lean, r["python_function"], r["lean_definition"],
                             r.get("signature_definition"), r.get("expected_signature"),
                             r.get("expected_call_bindings"), r.get("expected_pointer_offsets"),
                             r.get("expected_pointer_cells"), r.get("expected_optional_pointers"),
                             r.get("expected_static_strides"), r.get("expected_pure_functions"),
                             r.get("expected_integer_args"), r.get("expected_arange_args")))
        for name, definition in r.get("expected_pure_functions", {}).items():
            helpers = [h for h in records if h.get("kind") == "pure_jit_function"
                       and h["lean"] == r["lean"] and h["lean_definition"] == definition]
            if len(helpers) != 1:
                raise ValueError("pure JIT call needs a separately checked helper body")
            helper = helpers[0]
            module = ast.parse(py)
            bindings = []
            class Writes(ast.NodeVisitor):
                wildcard = False
                def visit_Name(self, node):
                    if isinstance(node.ctx, (ast.Store, ast.Del)): bindings.append(node.id)
                def visit_FunctionDef(self, node): bindings.append(node.name)
                visit_AsyncFunctionDef = visit_FunctionDef
                def visit_ClassDef(self, node): bindings.append(node.name)
                def visit_ImportFrom(self, node):
                    self.wildcard |= any(a.name == "*" for a in node.names)
                    bindings.extend(a.asname or a.name for a in node.names)
                def visit_Import(self, node):
                    bindings.extend(a.asname or a.name.split('.')[0] for a in node.names)
            writes = Writes(); writes.visit(module)
            if writes.wildcard or bindings.count(name) != 1:
                raise ValueError("pure JIT callee must have one static binding without rebinding")
            if helper["python"] == r["python"]:
                if helper["python_function"] != name:
                    raise ValueError("local pure JIT binding differs from the selected helper")
            else:
                selected = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                                and n.name == r["python_function"])
                imports = [(n.module, a.name) for n in module.body if isinstance(n, ast.ImportFrom)
                           and n.level == 0 and n.lineno < selected.lineno for a in n.names
                           if (a.asname or a.name) == name]
                if imports != [(Path(helper["python"]).stem, helper["python_function"])]:
                    raise ValueError("pure JIT import differs from the separately checked helper source")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(check_manifest(args.root, args.manifest), indent=2))
