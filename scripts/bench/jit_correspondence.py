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
import math
from pathlib import Path
import re
import textwrap
import inspect

from scripts.bench.audit_source import lean_code_mask, strip_lean_comments


def pasted_code_mask(body: str) -> str:
    """Exclude literal/comment text while rejecting Lean-only pasted syntax.

    PTX placeholders such as $0 are source string data, not Lean antiquotes.
    The full strings still participate unchanged in the subsequent AST comparison.
    """
    body = textwrap.dedent(body)
    lines = body.splitlines(keepends=True)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    masked = list(body)
    try:
        for token in tokenize.generate_tokens(io.StringIO(body).readline):
            if token.type not in (tokenize.STRING, tokenize.COMMENT):
                continue
            first = starts[token.start[0] - 1] + token.start[1]
            last = starts[token.end[0] - 1] + token.end[1]
            for index in range(first, last):
                if masked[index] != "\n":
                    masked[index] = " "
    except (tokenize.TokenError, IndentationError) as error:
        raise ValueError("invalid pasted body tokenization") from error
    return "".join(masked)


def source_scope(text: str, definition: str) -> tuple[str, str]:
    """Resolve a fully named source declaration inside a combined kernel module."""
    if '::' not in definition:
        return text, definition
    namespace, name = definition.split('::', 1)
    code = lean_code_mask(text)
    pattern = rf'(?m)^namespace[ \t]+{re.escape(namespace)}[ \t]*$'
    starts = list(re.finditer(pattern, code))
    if len(starts) != 1:
        raise ValueError(f'expected one source namespace: {namespace}')
    start = starts[0].end()
    end = re.search(rf'(?m)^end[ \t]+{re.escape(namespace)}[ \t]*$', code[start:])
    if end is None:
        raise ValueError(f'missing end of source namespace: {namespace}')
    return text[start:start + end.start()], name


def helper_header(header: str) -> str:
    """Read the common block header through its existing typed helper checks."""
    if not re.search(r"\bwith_triton_params\b", header):
        return header
    helper_type = re.search(
        r":\s*(PureJitFunction|BinaryPureJitFunction|ImportedBinaryJitFunction|"
        r"ImportedFiniteJitFunction|UniformJitFunction|StatementJitFunction|"
        r"SpecializedStatementJitFunction|ClosedUnaryJitFunction|ProcedureJitFunction)\s*:=",
        header)
    if helper_type is None:
        return header
    match = re.search(
        r"\bwith_triton_params\s*(\([^)]*\))\s+constexpr_args\s*(\[[^]]*\])"
        r"(?:\s+(signature|rebinding_signature)\s+([A-Za-z_]\w*))?\s+in\s+triton\s*$",
        header)
    if match is None:
        raise ValueError("typed helper requires its complete parameter declaration")
    if match[3]:
        keyword = "triton_rebinding_procedure" if match[3] == "rebinding_signature" else "triton_procedure"
        replacement = f"{keyword} {match[4]} {match[1]} constexpr_args {match[2]} "
    else:
        empty_only = {"PureJitFunction", "BinaryPureJitFunction", "UniformJitFunction",
                      "ImportedBinaryJitFunction", "ImportedFiniteJitFunction"}
        if helper_type[1] in empty_only:
            if match[2].strip() != "[]":
                raise ValueError("pure helper requires an empty constexpr list")
            replacement = f"triton_function {match[1]} "
        else:
            replacement = f"triton_function {match[1]} constexpr_args {match[2]} "
    return header[:match.start()] + replacement


def quotation(text: str, definition: str, *, raw_python: bool = False) -> tuple[str, str]:
    text, definition = source_scope(text, definition)
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
    begin = re.search(r"\b(?:triton|triton_function|triton_procedure|triton_rebinding_procedure|triton_source_procedure)\b", code)
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
                        return helper_header(selected[:opening]), selected[opening + 1:stop - 1]
        except (tokenize.TokenError, IndentationError) as error:
            raise ValueError("invalid Python quotation delimiters") from error
        raise ValueError("unterminated quotation")
    depth = 1
    for i in range(opening + 1, len(code)):
        depth += (code[i] == "{") - (code[i] == "}")
        if depth == 0:
            if code[i + 1:].strip():
                raise ValueError("unexpected code after quotation")
            return helper_header(selected[:opening]), strip_lean_comments(selected[opening + 1:i])
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
        source = re.sub(r"(?<![\w.])\.(real|fp16|bf16|fp32|fp64|f8e4|f8e5)\b",
                        r"FloatDType.\1", floating[1])
        tree = ast.parse(source, mode="eval").body
        if not isinstance(tree, ast.List):
            raise ValueError("unsupported polymorphic pointer signature")
        pairs = []
        for pair in tree.elts:
            if not (isinstance(pair, ast.Tuple) and len(pair.elts) == 2
                    and isinstance(pair.elts[0], ast.Constant) and type(pair.elts[0].value) is str):
                raise ValueError("unsupported floating pointer signature")
            dtype = pair.elts[1]
            if isinstance(dtype, ast.Name) and dtype.id in floating_parameters:
                selected = dtype.id
            elif (isinstance(dtype, ast.Attribute) and isinstance(dtype.value, ast.Name)
                  and dtype.value.id == "FloatDType"
                  and dtype.attr in {"real", "fp16", "bf16", "fp32", "fp64", "f8e4", "f8e5"}):
                selected = dtype.attr
            else:
                raise ValueError("floating pointers require declared dtype parameters or explicit FloatDType constructors")
            pairs.append([pair.elts[0].value, selected])
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
          expected_arange_args: dict[str, str] | None = None,
          expected_binary_functions: dict[str, str] | None = None,
          expected_ordered_sum_trees: dict[str, str] | None = None,
          expected_namespace_imports: dict[str, str] | None = None,
          expected_constexpr_bindings: dict[str, str] | None = None,
          expected_philox_call: dict | None = None,
          expected_scalar_infinity_where: bool = False,
          expected_welford_tree: dict[str, str] | None = None,
          expected_jit_functions: dict[str, str] | None = None,
          expected_natural_reads: list[str] | None = None,
          expected_jit_procedures: dict[str, str] | None = None,
          expected_parameter_rebinding: bool = False,
          expected_fp32_constants: list[str] | None = None,
          expected_source_procedures: dict[str, str] | None = None) -> dict:
    if expected_parameter_rebinding and expected_jit_procedures is None:
        raise ValueError("parameter rebinding requires the reviewed procedure invocation clause")
    helper_clauses = [expected_pure_functions, expected_binary_functions, expected_jit_functions,
                      expected_jit_procedures, expected_source_procedures]
    if sum(clause is not None for clause in helper_clauses) > 1:
        raise ValueError("unary, binary and statement helper clauses cannot be mixed")
    helper_functions = next((clause for clause in helper_clauses if clause is not None), None)
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
    projection = re.findall(r"\bwith_triton_natural_reads\s*\[([^\]]*)\]\s+in\b", signature)
    if expected_natural_reads is None:
        if projection:
            raise ValueError("natural read projections require reviewed caller metadata")
    else:
        if len(projection) != 1 or not expected_natural_reads:
            raise ValueError("expected one nonempty natural read projection")
        selected = [name.strip() for name in projection[0].split(',')]
        if selected != expected_natural_reads or len(set(selected)) != len(selected):
            raise ValueError("natural read projections differ from the reviewed caller")
        for name in selected:
            if not re.fullmatch(r"[A-Za-z_]\w*", name) or not re.search(
                    r"\([^()\n]*\b" + re.escape(name) + r"\b[^()\n]*:\s*Region\s+\.nat\s*\)", signature):
                raise ValueError("natural read projections require typed natural caller regions")
    profile = None
    if signature_definition is not None:
        if expected_signature is None:
            raise ValueError("pasted body requires an independently reviewed call signature")
        if re.search(r"\btl\.(region|param|call_signature)\b|\$", pasted_code_mask(body)):
            raise ValueError("pasted body must not contain metadata directives or Lean antiquotes")
        profile = call_signature(source_scope(lean_text, lean_definition)[0], signature_definition)
        if expected_natural_reads is not None:
            kinds = dict(profile['pointers'])
            if any(kinds.get(name) != 'i32' for name in expected_natural_reads):
                raise ValueError("natural read projections retain the signed32 storage ABI")
            if set(expected_natural_reads) & set(profile.get('writePorts', [])):
                raise ValueError("natural read projections must be read-only")
        dtype_arguments = profile.get("dtype_parameters", [])
        selected_signature = (r"\(\s*" + re.escape(signature_definition) + r"\s+" +
                              r"\s+".join(map(re.escape, dtype_arguments)) + r"\s*\)"
                              if dtype_arguments else re.escape(signature_definition))
        offset_clause = (r"\s+pointer_offsets\s*\[([^\]]*)\]"
                         if expected_pointer_offsets is not None else "")
        if expected_fp32_constants is not None:
            if (not isinstance(expected_fp32_constants, list)
                    or (not expected_fp32_constants and expected_source_procedures is None)
                    or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name)
                           for name in expected_fp32_constants)
                    or len(set(expected_fp32_constants)) != len(expected_fp32_constants)
                    or not set(expected_fp32_constants) <= set(profile["constexpr"])
                    or expected_call_bindings is None or offset_clause):
                raise ValueError("symbolic FP32 constants require distinct reviewed constexpr names and a scalar call")
            if any(isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                   and node.id in expected_fp32_constants for node in ast.walk(function)):
                raise ValueError("symbolic constexpr parameters must remain immutable")
            offset_clause = (r"\s+fp32_constants\s*\[\s*"
                             + r"\s*,\s*".join(map(re.escape, expected_fp32_constants))
                             + r"\s*\]")
        if expected_pointer_cells is not None:
            if expected_pointer_offsets is not None or expected_call_bindings is not None:
                raise ValueError("pointer cell views require a signature without scalar bindings or pointer offsets")
            offset_clause = r"\s+pointer_cells\s*\[([^\]]*)\]"
        if expected_optional_pointers is not None:
            if expected_pointer_offsets is not None or expected_pointer_cells is not None:
                raise ValueError("optional pointer metadata cannot be mixed with coordinate bindings")
            offset_clause = r"\s+(?:guarded_)?optional_pointers\s*\[([^\]]*)\]"
        if expected_static_strides is not None:
            if (expected_call_bindings is None or expected_pointer_offsets is not None
                    or expected_pointer_cells is not None or expected_optional_pointers is not None):
                raise ValueError("static strides require the reviewed scalar-only invocation")
            offset_clause = r"\s+static_strides"
        if helper_functions is not None:
            if (expected_call_bindings is None
                    or expected_pointer_cells is not None or expected_optional_pointers is not None
                    or expected_static_strides is not None
                    or (not helper_functions and not expected_parameter_rebinding)):
                raise ValueError("pure JIT helpers require a reviewed scalar-only call")
            if expected_pointer_offsets is not None:
                offset_clause = offset_clause.replace("pointer_offsets", "signed_pointer_offsets")
            label = ("source_procedures" if expected_source_procedures is not None else
                     ("rebinding_jit_procedures" if expected_parameter_rebinding else "jit_procedures")
                     if expected_jit_procedures is not None else
                     "jit_functions" if expected_jit_functions is not None else
                     "binary_functions" if expected_binary_functions is not None else "pure_functions")
            offset_clause += r"\s+" + label + r"\s*\[([^\]]*)\]"
        if expected_integer_args is not None or expected_arange_args is not None:
            if (expected_integer_args is None or expected_arange_args is None
                    or expected_call_bindings is None or expected_pointer_offsets is not None
                    or expected_pointer_cells is not None or expected_optional_pointers is not None
                    or expected_static_strides is not None or helper_functions is not None):
                raise ValueError("integer/arange arguments require a reviewed scalar-only call")
            offset_clause = (r"\s+integer_args\s*\[([^\]]*)\]"
                             r"\s+arange_args\s*\[([^\]]*)\]")
        if expected_ordered_sum_trees is not None:
            if (not expected_ordered_sum_trees or expected_call_bindings is None or offset_clause):
                raise ValueError("ordered sum trees require a reviewed scalar-only call")
            offset_clause = r"\s+ordered_sum_trees\s*\[([^\]]*)\]"
        if expected_philox_call is not None:
            if (set(expected_philox_call) != {"fp32_bits", "i64_args", "arange_args", "uniform_helpers"}
                    or expected_call_bindings is None or offset_clause):
                raise ValueError("Philox arguments require a reviewed scalar-only call")
            offset_clause = (r"\s+fp32_bits\s*\[([^\]]*)\]"
                             r"\s+i64_args\s*\[([^\]]*)\]"
                             r"\s+arange_args\s*\[([^\]]*)\]"
                             r"\s+uniform_helpers\s*\[([^\]]*)\]")
        if expected_scalar_infinity_where:
            if expected_call_bindings is None or offset_clause:
                raise ValueError("scalar infinity where requires a reviewed scalar-only call")
            offset_clause = r"\s+scalar_infinity_where"
        if expected_welford_tree is not None:
            if (set(expected_welford_tree) != {"argument", "width"}
                    or expected_static_strides is None or expected_call_bindings is None
                    or any(x is not None for x in [expected_pointer_offsets, expected_pointer_cells,
                                                   expected_optional_pointers, helper_functions,
                                                   expected_integer_args, expected_ordered_sum_trees,
                                                   expected_philox_call]) or expected_scalar_infinity_where):
                raise ValueError("Welford trees require a reviewed scalar call and immutable signed strides")
            tree_arg, tree_width = expected_welford_tree["argument"], expected_welford_tree["width"]
            if (not re.fullmatch(r"[A-Za-z_]\w*", tree_arg)
                    or tree_width not in constants
                    or not re.search(r"\(\s*" + re.escape(tree_arg) +
                                     r"\s*:\s*DSL\.WelfordSource\.Tree\s+" +
                                     re.escape(tree_width) + r"\s*\)", signature)):
                raise ValueError("Welford tree must be a typed caller argument for the source block width")
            helper_calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                            and ast.unparse(n.func) == "triton_helpers.welford"]
            if (len(helper_calls) != 1 or len(helper_calls[0].args) != 4
                    or helper_calls[0].keywords or not isinstance(helper_calls[0].args[3], ast.Constant)
                    or type(helper_calls[0].args[3].value) is not int
                    or helper_calls[0].args[3].value != 1):
                raise ValueError("Welford metadata requires the original single-row axis 1 helper")
            offset_clause = r"\s+welford_tree\s+" + re.escape(tree_arg)
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
            binding_text = calls[0][1].strip()
            for entry in binding_text.split(",") if binding_text else []:
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|[0-9]+)\s*", entry)
                if not pair or pair[1] in bindings:
                    raise ValueError("runtime call bindings require distinct named caller arguments")
                bindings[pair[1]] = pair[2]
            if bindings != expected_call_bindings:
                raise ValueError("runtime call bindings differ from the reviewed invocation")
            integer_args = {n for n, t in profile["scalars"] if t in {"integer", "i32", "i64"}}
            fp32_args = {n for n, t in profile["scalars"] if t == "fp32"}
            handled = set(expected_integer_args or {}) | set((expected_philox_call or {}).get("i64_args", {}))
            if (not handled <= integer_args or not (integer_args-handled) <= set(bindings)
                    or set(bindings) & handled or not set(bindings) <= integer_args | fp32_args):
                raise ValueError("runtime call bindings must include all integer parameters and only supported scalar parameters")
        if expected_philox_call is not None:
            for group, label in enumerate(["fp32_bits", "i64_args", "arange_args", "uniform_helpers"], start=2):
                values = {}
                for entry in calls[0][group].replace("«to»", "to").split(","):
                    pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*)\s*", entry)
                    if not pair or pair[1] in values:
                        raise ValueError("Philox arguments require distinct named bindings")
                    values[pair[1]] = pair[2]
                if values != expected_philox_call[label]:
                    raise ValueError("Philox arguments differ from the reviewed call")
            for label, dtype, caller_type in [("fp32_bits", "fp32", "Float32Bits"), ("i64_args", "i64", "Int")]:
                for name, caller in expected_philox_call[label].items():
                    if (dict(profile["scalars"]).get(name) != dtype or not any(
                            caller in names.split() for names in re.findall(
                                r"\(([A-Za-z_][\w\s]*)\s*:\s*"+caller_type+r"\s*\)", signature))):
                        raise ValueError("Philox caller type differs from the reviewed ABI")
            for size, proof in expected_philox_call['arange_args'].items():
                if size not in constants or not re.search(r"\([^()\n]*\b"+re.escape(proof)+
                        r"\s*:\s*Random\.ArangeSizeDomain\s+"+re.escape(size)+r"\s*\)", signature):
                    raise ValueError("Philox arange requires its checked source size")
            module = ast.parse(python_text)
            for helper in expected_philox_call['uniform_helpers']:
                declarations = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == helper]
                if len(declarations) != 1:
                    raise ValueError("uniform helper requires one retained fallback body")
                selected = [n for n in module.body if isinstance(n, ast.Try) and len(n.body) == 1
                            and len(n.handlers) == 1 and not n.orelse and not n.finalbody
                            and isinstance(n.body[0], ast.Assign) and len(n.body[0].targets) == 1
                            and ast.unparse(n.body[0].targets[0]) == helper
                            and ast.unparse(n.body[0].value) == 'tl.uint_to_uniform_float'
                            and isinstance(n.handlers[0].type, ast.Name) and n.handlers[0].type.id == 'AttributeError'
                            and n.handlers[0].name is None and n.handlers[0].body == declarations]
                writes = [n for n in ast.walk(module) if isinstance(n, ast.Name)
                          and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == helper]
                if len(selected) != 1 or writes != selected[0].body[0].targets:
                    raise ValueError("uniform helper must retain the exact builtin/AttributeError fallback selection")
                if any(isinstance(n, ast.arg) and n.arg == helper for n in ast.walk(function)):
                    raise ValueError("uniform helper cannot be shadowed")
        if expected_ordered_sum_trees is not None:
            trees = {}
            for entry in calls[0][2].split(','):
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*)\s*", entry)
                if not pair or pair[1] in trees:
                    raise ValueError("ordered sum trees require distinct named destinations")
                trees[pair[1]] = pair[2]
            if trees != expected_ordered_sum_trees:
                raise ValueError("ordered sum trees differ from the reviewed call")
            sums = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == 'tl.sum']
            destinations = []
            for statement in ast.walk(function):
                if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and isinstance(statement.value, ast.Subscript)):
                    continue
                view = statement.value
                if not (isinstance(view.value, ast.Call) and view.value in sums):
                    continue
                call = view.value
                if (ast.unparse(view.slice) != '(:, None)' or len(call.args) != 2
                        or call.keywords or not isinstance(call.args[1], ast.Constant)
                        or type(call.args[1].value) is not int or call.args[1].value != 1):
                    raise ValueError("ordered tree requires the original single-row axis-one sum and view")
                destinations.append(statement.targets[0].id)
            if (len(destinations) != len(sums) or len(destinations) != len(set(destinations))
                    or set(destinations) != set(trees)):
                raise ValueError("every source sum must have exactly one reviewed ordered tree")
        if expected_constexpr_bindings is not None:
            for name, value in expected_constexpr_bindings.items():
                binder = r"(?:" + re.escape(name) + r"\b|«" + re.escape(name) + r"»)"
                prefix = r"\blet\s+" + binder
                if re.fullmatch(r"[0-9]+", value):
                    declaration = prefix + r"\s*:\s*Nat\s*:=\s*" + re.escape(value) + r"\b"
                elif value in {"FloatDType.real", "FloatDType.fp32", "FloatDType.fp16",
                               "FloatDType.bf16", "FloatDType.f8e4", "FloatDType.f8e5"}:
                    declaration = (prefix + r"\s*:\s*FloatDType\s*:=\s*(?:\.|FloatDType\.)" +
                                   re.escape(value.split(".")[1]) + r"\b")
                elif value in {"SourceDType.i8", "SourceDType.i16", "SourceDType.i32", "SourceDType.i64",
                               "SourceDType.u8", "SourceDType.u16", "SourceDType.u32", "SourceDType.u64"}:
                    declaration = (prefix + r"\s*:\s*SourceDType\s*:=\s*(?:\.|SourceDType\.)" +
                                   re.escape(value.split(".")[1]) + r"\b")
                else:
                    raise ValueError("unsupported closed constexpr specialization")
                if (name not in constants
                        or len(re.findall(r"\b(?:let|have)\s+" + binder, signature)) != 1
                        or len(re.findall(declaration, signature)) != 1):
                    raise ValueError("closed constexpr binding differs from the reviewed specialization")
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
        if helper_functions is not None:
            bindings = {}
            helper_group = 3 if expected_pointer_offsets is not None else 2
            clause = calls[0][helper_group]
            for item in clause.split(",") if clause.strip() else []:
                pair = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*:=\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*", item)
                if not pair or pair[1] in bindings:
                    raise ValueError("pure JIT functions require distinct named source definitions")
                bindings[pair[1]] = pair[2]
            if bindings != helper_functions:
                raise ValueError("pure JIT function bindings differ from the reviewed call")
            builtin_names = {"range", "float", "int"}
            body_nodes = [node for statement in function.body for node in ast.walk(statement)]
            min_calls = [n for n in body_nodes if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "min"]
            if min_calls and "min" not in bindings:
                if any(len(call.args) != 2 or call.keywords for call in min_calls):
                    raise ValueError("builtin min requires exactly two positional arguments")
                module = ast.parse(python_text)
                shadowed = any(
                    (isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                     and node.id == "min")
                    or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.name == "min")
                    or (isinstance(node, ast.Import) and any(
                        (alias.asname or alias.name.split('.')[0]) == "min" for alias in node.names))
                    or (isinstance(node, ast.ImportFrom) and any(
                        alias.name == "*" or (alias.asname or alias.name) == "min" for alias in node.names))
                    for node in ast.walk(module))
                if shadowed or any(arg.arg == "min" for arg in function.args.args):
                    raise ValueError("builtin min must not be shadowed or rebound")
                builtin_names.add("min")
            actual = {n.func.id for n in body_nodes if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id not in builtin_names}
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
        for name, target in (expected_namespace_imports or {}).items():
            if (name, target) not in {
                    ('libdevice', 'torch._inductor.runtime.triton_helpers.libdevice'),
                    ('libdevice', 'triton.language.extra.libdevice'),
                    ('libdevice', 'triton.language.extra.cuda.libdevice'),
                    ('triton_helpers', 'torch._inductor.runtime.triton_helpers')}:
                raise ValueError("unsupported source intrinsic namespace")
            imports = [node.module + '.' + alias.name for node in module.body
                       if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                       and node.lineno < function.lineno for alias in node.names
                       if (alias.asname or alias.name) == name]
            def namespace_root(node):
                while isinstance(node, ast.Attribute):
                    node = node.value
                return node.id if isinstance(node, ast.Name) else None
            mutated = any(isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                          and namespace_root(node) == name for node in ast.walk(module))
            local_writes = any(isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                               and node.id == name for node in ast.walk(function))
            local_imports = any((alias.asname or alias.name.split('.')[0]) == name
                                for node in ast.walk(function) if isinstance(node, (ast.Import, ast.ImportFrom))
                                for alias in node.names)
            if (imports != [target] or writes.wildcard_import or writes.names.count(name) != 1
                    or mutated or local_writes or local_imports or name in {a.arg for a in function.args.args}):
                raise ValueError("source intrinsic namespace differs, is shadowed or is rebound")
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
            if literal.startswith("math."):
                # Preserve the closed host expression as a source tag. Real and
                # native interpretations bind it separately; do not assert that
                # a host floating computation equals its Real interpretation.
                declared = ast.parse(literal, mode="eval").body
                def closed_math(node):
                    if isinstance(node, ast.Constant):
                        return type(node.value) in (int, float) and (type(node.value) is int or math.isfinite(node.value))
                    if isinstance(node, ast.Attribute):
                        return ast.unparse(node) == "math.pi"
                    if isinstance(node, ast.Call):
                        return (ast.unparse(node.func) == "math.sqrt" and len(node.args) == 1
                                and not node.keywords and closed_math(node.args[0]))
                    if isinstance(node, ast.UnaryOp):
                        return isinstance(node.op, ast.USub) and closed_math(node.operand)
                    if isinstance(node, ast.BinOp):
                        return (isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
                                and closed_math(node.left) and closed_math(node.right))
                    return False
                if not closed_math(declared) or ast.dump(declared) != ast.dump(value.args[0]):
                    raise ValueError("source module math expression differs or is not closed")
                def host_math(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    if isinstance(node, ast.Attribute):
                        return math.pi
                    if isinstance(node, ast.Call):
                        return math.sqrt(host_math(node.args[0]))
                    if isinstance(node, ast.UnaryOp):
                        return -host_math(node.operand)
                    a, b = host_math(node.left), host_math(node.right)
                    if isinstance(node.op, ast.Add):
                        return a + b
                    if isinstance(node.op, ast.Sub):
                        return a - b
                    if isinstance(node.op, ast.Mult):
                        return a * b
                    return a / b
                try:
                    host_value = host_math(declared)
                    if not math.isfinite(host_value):
                        raise ValueError("nonfinite host constant")
                except (ValueError, ZeroDivisionError, OverflowError) as error:
                    raise ValueError("module math expression needs a defined finite host value") from error
                imports = [node for node in module.body if isinstance(node, ast.Import)
                           for alias in node.names if alias.name == "math" and alias.asname is None
                           and node.lineno < assignments[0].lineno]
                def math_root(node):
                    while isinstance(node, ast.Attribute):
                        node = node.value
                    return isinstance(node, ast.Name) and node.id == "math"
                mutated = any(isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del))
                              and math_root(node) for node in ast.walk(module))
                if len(imports) != 1 or writes.names.count("math") != 1 or mutated:
                    raise ValueError("module math expression needs one immutable preceding math import")
                continue
            try:
                original_value, declared_value = ast.literal_eval(value.args[0]), ast.literal_eval(literal)
            except (ValueError, SyntaxError) as error:
                raise ValueError("source constant must contain a literal integer or finite floating literal") from error
            literal_type = type(original_value)
            if (literal_type not in (int, float) or type(declared_value) is not literal_type
                    or original_value != declared_value
                    or (literal_type is float and (not math.isfinite(original_value)
                        or original_value.hex() != declared_value.hex()))):
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
        scalar_types = {"integer": "int", "fp32": "tl.float32", "fp64": "tl.float64",
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
            if t == "int" and (writes.wildcard_import or "int" in writes.names
                               or any(a.arg == "int" for a in function.args.args)):
                raise ValueError("builtin int annotation is shadowed or rebound")
    if profile is not None:
        # Parse the pasted body as Python directly: braces, Lean assignment
        # syntax and other legacy spelling changes cannot become a copy pass.
        statements, parameters = ast.parse("def __veritile_pasted__():\n" + body).body[0].body, {}
    else:
        statements, parameters = source_statements(body, constants)
    if profile is None and parameters != runtime:
        raise ValueError(f"runtime signature mismatch: Python {runtime}; Lean {parameters}")
    for name in constants:
        if not re.search(r"\b" + re.escape(name) + r"\b", signature):
            raise ValueError(f"missing constexpr binder: {name}")
    left = ast.Module(copy.deepcopy(function.body), [])
    right = ast.Module(statements, [])
    # A function docstring is compiler metadata. Reindent its descriptive text
    # with Python's docstring rule, without touching executable string literals.
    # Wrapping the pasted body above preserves every other multiline string.
    for program in (left, right):
        if (program.body and isinstance(program.body[0], ast.Expr)
                and isinstance(program.body[0].value, ast.Constant)
                and isinstance(program.body[0].value.value, str)):
            program.body[0].value.value = inspect.cleandoc(program.body[0].value.value)
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
                        lean_definition: str, arity: int = 1,
                        expected_imports: dict[str, list[str]] | None = None) -> dict:
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d) == "triton.jit" for d in functions[0].decorator_list):
        raise ValueError("expected one explicitly selected @triton.jit helper")
    fn = functions[0]
    statements = fn.body
    if (statements and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        statements = statements[1:]
    if (arity not in (1, 2) or len(fn.args.args) != arity or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg
            or fn.args.kwarg or fn.args.defaults or fn.args.kw_defaults
            or any(arg.annotation for arg in fn.args.args) or len(statements) != 1 or not isinstance(statements[0], ast.Return)):
        raise ValueError("pure JIT helper requires one unannotated parameter and one return")
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    arguments = [a.arg for a in fn.args.args]
    parameters = r"\s*,\s*".join(map(re.escape, arguments))
    if not re.search(r"\btriton_function\s*\(\s*" + parameters + r"\s*\)\s*$", header):
        raise ValueError("pure JIT quotation parameter differs from Python")
    if expected_imports is not None:
        if not expected_imports:
            raise ValueError("imported helper requires nonempty extern bindings")
        clause = re.search(r'with_triton_imports\s*\[([^\]]*)\]\s+in\s+triton_function', header)
        if not clause:
            raise ValueError("imported helper must declare its source aliases outside the body")
        declared = {}
        for item in clause[1].split(','):
            match = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*:=\s*("[^"\n]*")\s*', item)
            if not match or match[1] in declared:
                raise ValueError("imported helper aliases must be distinct static strings")
            declared[match[1]] = json.loads(match[2])
        if declared != {name: targets[0] for name, targets in expected_imports.items() if targets}:
            raise ValueError("imported helper aliases differ from reviewed primary imports")
        module = ast.parse(python_text)
        def import_chain(node, name):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if len(node.names) != 1 or (node.names[0].asname or node.names[0].name) != name:
                    raise ValueError("extern fallback requires one exact import alias")
                return [node.module + '.' + node.names[0].name]
            if (isinstance(node, ast.Try) and len(node.body) == len(node.handlers) == 1
                    and not node.orelse and not node.finalbody
                    and isinstance(node.handlers[0].type, ast.Name)
                    and node.handlers[0].type.id == 'ImportError'
                    and node.handlers[0].name is None and len(node.handlers[0].body) == 1):
                return import_chain(node.body[0], name) + import_chain(node.handlers[0].body[0], name)
            raise ValueError("extern alias requires a direct ImportError fallback chain")
        for name, targets in expected_imports.items():
            statements = [node for node in module.body if any(
                isinstance(child, ast.ImportFrom) and any((a.asname or a.name) == name for a in child.names)
                for child in ast.walk(node))]
            writes = [node for node in ast.walk(module) if
                      (isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name)
                      or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name)]
            other_imports = [node for node in ast.walk(module) if
                            (isinstance(node, ast.Import) and any((a.asname or a.name.split('.')[0]) == name for a in node.names))
                            or (isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names))]
            if len(statements) != 1 or writes or other_imports or import_chain(statements[0], name) != targets:
                raise ValueError("imported helper extern chain differs from Python")
            intrinsic = targets[0].rsplit('.', 1)[-1]
            if intrinsic not in ({'finitef', 'isfinited'} if arity == 1 else {'pow'}) or targets != [
                    'triton.language.' + location + '.' + intrinsic
                    for location in ('extra.cuda.libdevice', 'math', 'libdevice')]:
                raise ValueError("unsupported helper extern import chain")
    elif re.search(r'\bwith_triton_imports\b', header):
        raise ValueError("imported helper requires independently reviewed extern imports")
    code = pasted_code_mask(body)
    if "$" in code or re.search(r"\btl\.(region|param|source_helper|call_signature)\b", code):
        raise ValueError("pure JIT body must contain only pasted Python")
    if ast.dump(ast.Module(fn.body, [])) != ast.dump(ast.Module(ast.parse("def __veritile_pasted__():\n" + body).body[0].body, [])):
        raise ValueError("pure JIT return body differs from Python")
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body", "kind": "binary_pure_jit_function" if arity == 2 else "pure_jit_function",
            "statements": len(fn.body), "runtime_parameters": {a: "inferred" for a in arguments}, "constexpr_parameters": []}


def check_statement_function(python_text: str, lean_text: str, python_function: str,
                             lean_definition: str) -> dict:
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d) == "triton.jit" for d in functions[0].decorator_list):
        raise ValueError("expected one explicitly selected @triton.jit statement helper")
    fn = functions[0]
    if (fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg
            or fn.args.defaults or fn.args.kw_defaults or not fn.args.args):
        raise ValueError("unsupported statement JIT helper signature")
    parameters = [a.arg for a in fn.args.args]
    constants = []
    for arg in fn.args.args:
        if arg.annotation is None:
            continue
        annotation = arg.annotation.value if isinstance(arg.annotation, ast.Constant) else ast.unparse(arg.annotation)
        if annotation != "tl.constexpr":
            raise ValueError("statement helper runtime arguments must retain inferred call types")
        constants.append(arg.arg)
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    declaration = re.search(r"\btriton_function\s*\(([^)]*)\)\s+constexpr_args\s*\[([^\]]*)\]\s*$", header)
    if not declaration:
        raise ValueError("statement helper requires its explicit parameter and constexpr lists")
    declared = [[name.strip() for name in group.split(",") if name.strip()] for group in declaration.groups()]
    if declared != [parameters, constants]:
        raise ValueError("statement helper parameters or constexpr annotations differ from Python")
    if re.search(r"\btl\.(region|param|source_helper|call_signature)\b|\$", pasted_code_mask(body)):
        raise ValueError("statement helper must contain only pasted Python")
    pasted = ast.parse("def __veritile_helper__():\n" + body).body[0].body
    left, right = ast.Module(copy.deepcopy(fn.body), []), ast.Module(pasted, [])
    for program in (left, right):
        if (program.body and isinstance(program.body[0], ast.Expr)
                and isinstance(program.body[0].value, ast.Constant)
                and isinstance(program.body[0].value.value, str)):
            program.body[0].value.value = inspect.cleandoc(program.body[0].value.value)
    if ast.dump(left) != ast.dump(right):
        raise ValueError("statement helper body differs from Python")
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body", "kind": "statement_jit_function",
            "statements": len(fn.body), "runtime_parameters": {a: "inferred" for a in parameters if a not in constants},
            "constexpr_parameters": constants}


def check_source_procedure(python_text: str, lean_text: str, python_function: str,
                           lean_definition: str, expected_fp32_constants: list[str]) -> dict:
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    parameters = r"(\([^)]*\)\s+constexpr_args\s*\[[^\]]*\])"
    fp32_constants = r"\s+fp32_constants\s*\[([^\]]*)\]"
    declaration = re.search(r"\bwith_triton_params\s*" + parameters
                            + r"(?:" + fp32_constants + r")?\s+in\s+triton\s*$", header)
    if declaration is None:
        declaration = re.search(r"\btriton_source_procedure\s*" + parameters
                                + fp32_constants + r"\s*$", header)
    if not declaration:
        raise ValueError("source procedure requires exact parameters, constexprs and materialized constants")
    selected = f"def {lean_definition} := triton_function {declaration[1]} {{" + body + "}"
    result = check_statement_function(python_text, selected, python_function, lean_definition)
    materialized = [name.strip() for name in (declaration[2] or '').split(',') if name.strip()]
    if (materialized != expected_fp32_constants or len(set(materialized)) != len(materialized)
            or not set(materialized) <= set(result['constexpr_parameters'])
            or any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in materialized)):
        raise ValueError("source procedure materialized constants differ from reviewed constexpr subset")
    fn = next(n for n in ast.parse(python_text).body if isinstance(n, ast.FunctionDef) and n.name == python_function)
    if any(isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
           and node.id in result['constexpr_parameters'] for node in ast.walk(fn)):
        raise ValueError("source procedure constexpr parameters must remain immutable")
    result['kind'] = 'source_procedure'
    result['fp32_constants'] = materialized
    return result


def check_procedure_function(python_text: str, lean_text: str, python_function: str,
                             lean_definition: str, signature_definition: str,
                             expected_signature: dict, expected_parameter_rebinding: bool = False) -> dict:
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    keyword = "triton_rebinding_procedure" if expected_parameter_rebinding else "triton_procedure"
    declaration = re.search(r"\b" + keyword + r"\s+" + re.escape(signature_definition)
                            + r"\s*(\([^)]*\)\s+constexpr_args\s*\[[^\]]*\])\s*$", header)
    if not declaration:
        raise ValueError("procedure must declare its reviewed ABI, parameter order and constexprs")
    profile = call_signature(source_scope(lean_text, lean_definition)[0], signature_definition)
    if profile != expected_signature:
        raise ValueError("procedure call signature differs from the reviewed ABI")
    # Reuse the exact body/parameter comparison, including terminal empty return.
    # Only the declaration header is adapted; no source expression is changed.
    parameter_header = declaration[1].replace("«do»", "do")
    selected = f"def {lean_definition} := triton_function {parameter_header} {{" + body + "}"
    result = check_statement_function(python_text, selected, python_function, lean_definition)
    parameters = list(result["runtime_parameters"]) + result["constexpr_parameters"]
    declared = [name for name, _ in profile["pointers"] + profile["scalars"]] + profile["constexpr"]
    if (len(declared) != len(parameters) or set(declared) != set(parameters)
            or profile["constexpr"] != result["constexpr_parameters"]):
        raise ValueError("procedure ABI must cover all and only original Python parameters")
    result["kind"] = "procedure_jit_function"
    result["runtime_parameters"] = dict(profile["pointers"] + profile["scalars"])
    return result


def check_uniform_function(python_text: str, lean_text: str, python_function: str, lean_definition: str) -> dict:
    module = ast.parse(python_text)
    functions = [n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d) == 'triton.jit' for d in functions[0].decorator_list):
        raise ValueError("uniform decoder requires one copied JIT helper")
    fn = functions[0]
    if (len(fn.args.args) != 1 or fn.args.args[0].arg != 'x' or fn.args.args[0].annotation
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg
            or fn.args.defaults or fn.args.kw_defaults):
        raise ValueError("uniform decoder requires its original scalar argument")
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    if not re.search(r"\btriton_function\s*\(\s*x\s*\)\s*$", header):
        raise ValueError("uniform decoder quotation parameter differs")
    pasted = ast.parse('def helper():\n' + body).body[0].body
    if '$' in body or ast.dump(ast.Module(fn.body, [])) != ast.dump(ast.Module(pasted, [])):
        raise ValueError("uniform decoder complete body differs from Python")
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body", "kind": "uniform_jit_function",
            "statements": len(fn.body), "runtime_parameters": {"x": "uint32 selected by Philox4x32"},
            "constexpr_parameters": []}


def check_control_source(python_text: str, lean_text: str, python_function: str,
                         lean_definition: str, signature_definition: str,
                         expected_signature: dict, expected_constexpr_bools: dict,
                         expected_constexpr_dtypes: dict | None = None,
                         expected_jit_calls: dict | None = None) -> dict:
    """Check the full pasted control tree and its separately reviewed caller ABI.

    This source gate never specializes a branch. The typed control backend must
    check and implement the signature and constexpr bindings independently.
    """
    functions = [n for n in ast.parse(python_text).body
                 if isinstance(n, ast.FunctionDef) and n.name == python_function]
    if len(functions) != 1 or not any(ast.unparse(d.func if isinstance(d, ast.Call) else d) == "triton.jit" for d in functions[0].decorator_list):
        raise ValueError("expected one explicitly selected control-source JIT")
    fn = functions[0]
    if fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        raise ValueError("unsupported control-source Python signature")
    profile = call_signature(source_scope(lean_text, lean_definition)[0], signature_definition)
    if profile != expected_signature:
        raise ValueError("control-source caller signature differs from reviewed ABI")
    parameters = [a.arg for a in fn.args.args]
    constants = [a.arg for a in fn.args.args if a.annotation is not None
                 and ast.unparse(a.annotation) == "tl.constexpr"]
    if any(a.annotation is not None and a.arg not in constants for a in fn.args.args):
        raise ValueError("unreviewed control-source runtime annotation")
    declared = [name for name, _ in profile.get("pointers", []) + profile.get("scalars", [])]
    runtime = [p for p in parameters if p not in constants]
    pointer_names = [n for n, _ in profile.get("pointers", [])]
    scalar_names = [n for n, _ in profile.get("scalars", [])]
    if (profile.get("constexpr", []) != constants or len(declared) != len(set(declared))
            or set(declared) != set(runtime)
            or pointer_names != [p for p in runtime if p in pointer_names]
            or scalar_names != [p for p in runtime if p in scalar_names]):
        raise ValueError("control-source parameter order or constexpr annotation differs")
    if not isinstance(expected_constexpr_bools, dict) or any(
            name not in constants or type(value) is not bool
            for name, value in expected_constexpr_bools.items()):
        raise ValueError("control-source boolean bindings must name reviewed constexpr parameters")
    defaults = dict(zip(parameters[len(parameters)-len(fn.args.defaults):], fn.args.defaults)) if fn.args.defaults else {}
    if set(defaults) != set(expected_constexpr_bools) or any(
            not isinstance(defaults[name], ast.Constant) or type(defaults[name].value) is not bool
            or defaults[name].value != value for name, value in expected_constexpr_bools.items()):
        raise ValueError("control-source default flags differ from the selected caller")
    header, body = quotation(lean_text, lean_definition, raw_python=True)
    linked = expected_constexpr_dtypes is not None or expected_jit_calls is not None
    if linked:
        if not isinstance(expected_constexpr_dtypes, dict) or not isinstance(expected_jit_calls, dict):
            raise ValueError("linked control functions require reviewed dtype and helper bindings")
        prefix = r"\bwith_control_function\s+" + re.escape(json.dumps(python_function)) + r"\s+"
        suffix = r"\s+dtype_constants\s*\[([^\]]*)\]\s+jit_calls\s*\[([^\]]*)\]"
    else:
        prefix, suffix = r"\bwith_control_call\s+", ""
    match = re.search(prefix + re.escape(signature_definition)
                      + r"\s+constexpr_bools\s*\[([^\]]*)\]" + suffix + r"\s+in\s+triton\s*$", header)
    if not match:
        raise ValueError("control-source body must use its reviewed caller signature")
    bindings = []
    for entry in (match[1].split(",") if match[1].strip() else []):
        binding = re.fullmatch(r"\s*(\w+)\s*:=\s*(true|false)\s*", entry)
        if not binding:
            raise ValueError("control-source boolean bindings must be explicit literals")
        bindings.append((binding[1], binding[2] == "true"))
    if len(dict(bindings)) != len(bindings) or dict(bindings) != expected_constexpr_bools:
        raise ValueError("control-source constexpr bindings differ from the reviewed caller")
    if linked:
        def entries(text, value_pattern):
            result = []
            for raw in (text.split(",") if text.strip() else []):
                item = re.fullmatch(r"\s*(\w+)\s*:=\s*" + value_pattern + r"\s*", raw)
                if not item:
                    raise ValueError("invalid linked control-function binding")
                result.append((item[1], item[2]))
            if len(dict(result)) != len(result):
                raise ValueError("duplicate linked control-function binding")
            return dict(result)
        if entries(match[2], r"\.(\w+)") != expected_constexpr_dtypes:
            raise ValueError("control-function dtype bindings differ from the reviewed caller")
        if entries(match[3], r'"([A-Za-z_]\w*)"') != expected_jit_calls:
            raise ValueError("control-function helper links differ from the reviewed caller")
        if any(name not in constants for name in expected_constexpr_dtypes):
            raise ValueError("control dtype binding must name a source constexpr")
        calls = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id not in {"range", "float", "int"}}
        if calls != set(expected_jit_calls):
            raise ValueError("control-function helper calls are not completely linked")
    if re.search(r"\btl\.(region|param|call_signature|source_helper)\b|\$", pasted_code_mask(body)):
        raise ValueError("control-source body must contain only pasted Python")
    copied = ast.parse("def __veritile_control__():\n" + body).body[0].body
    if ast.dump(ast.Module(fn.body, [])) != ast.dump(ast.Module(copied, [])):
        raise ValueError("control-source body differs from original Python")
    return {"python_function": python_function, "lean_definition": lean_definition,
            "status": "PASS", "mode": "pasted_body", "kind": "control_source",
            "statements": len(fn.body), "runtime_parameters": dict(profile.get("pointers", []) + profile.get("scalars", [])),
            "constexpr_parameters": constants, "constexpr_bools": dict(bindings)}


def check_manifest(root: Path, manifest: Path) -> list[dict]:
    return check_records(root, json.loads(manifest.read_text()))


def check_records(root: Path, records: list[dict]) -> list[dict]:
    """Check records supplied by a snapshot or the source-discovery tool."""
    results = []
    for r in records:
        py, lean = (root / r["python"]).read_text(), (root / r["lean"]).read_text()
        if r.get("kind") == "control_source":
            if (set(r) - {"expected_constexpr_dtypes", "expected_jit_calls"}) != {"kind", "python", "lean", "python_function", "lean_definition",
                          "signature_definition", "expected_signature", "expected_constexpr_bools"}:
                raise ValueError("control-source record requires its exact reviewed ABI and boolean bindings")
            for callee in r.get("expected_jit_calls", {}):
                linked = [c for c in records if c.get("python") == r["python"]
                          and c.get("python_function") == callee and c.get("kind") == "control_source"]
                if len(linked) != 1:
                    raise ValueError("every linked control helper requires exactly one complete source pair")
            results.append(check_control_source(py, lean, r["python_function"], r["lean_definition"],
                                               r["signature_definition"], r["expected_signature"],
                                               r["expected_constexpr_bools"],
                                               r.get("expected_constexpr_dtypes"), r.get("expected_jit_calls")))
            continue
        if r.get("kind") == "source_procedure":
            if set(r) != {"kind", "python", "lean", "python_function", "lean_definition", "expected_fp32_constants"}:
                raise ValueError("source procedure requires exact original parameters and reviewed constants")
            results.append(check_source_procedure(py, lean, r["python_function"], r["lean_definition"],
                                                  r["expected_fp32_constants"]))
            continue
        if r.get("kind") == "procedure_jit_function":
            if set(r) - {"expected_parameter_rebinding"} != {"kind", "python", "lean", "python_function", "lean_definition",
                          "signature_definition", "expected_signature"}:
                raise ValueError("procedure helper requires its exact independently reviewed ABI")
            results.append(check_procedure_function(py, lean, r["python_function"], r["lean_definition"],
                                                     r["signature_definition"], r["expected_signature"],
                                                     r.get("expected_parameter_rebinding", False)))
            continue
        if r.get("kind") == "statement_jit_function":
            if set(r) != {"kind", "python", "lean", "python_function", "lean_definition"}:
                raise ValueError("statement helper cannot override invocation metadata")
            results.append(check_statement_function(py, lean, r["python_function"], r["lean_definition"]))
            continue
        if r.get("kind") == "uniform_jit_function":
            if set(r) != {"kind", "python", "lean", "python_function", "lean_definition"}:
                raise ValueError("uniform helper cannot override invocation metadata")
            results.append(check_uniform_function(py, lean, r['python_function'], r['lean_definition']))
            continue
        if r.get("kind") in {"pure_jit_function", "binary_pure_jit_function"}:
            if (set(r) - {"expected_imports"}) != {"kind", "python", "lean", "python_function", "lean_definition"}:
                raise ValueError("pure JIT helper record cannot override kernel call metadata")
            results.append(check_pure_function(py, lean, r["python_function"], r["lean_definition"],
                                               2 if r["kind"] == "binary_pure_jit_function" else 1, r.get("expected_imports")))
            continue
        results.append(check(py, lean, r["python_function"], r["lean_definition"],
                             r.get("signature_definition"), r.get("expected_signature"),
                             r.get("expected_call_bindings"), r.get("expected_pointer_offsets"),
                             r.get("expected_pointer_cells"), r.get("expected_optional_pointers"),
                             r.get("expected_static_strides"), r.get("expected_pure_functions"),
                             r.get("expected_integer_args"), r.get("expected_arange_args"), r.get("expected_binary_functions"),
                             r.get("expected_ordered_sum_trees"), r.get("expected_namespace_imports"),
                             r.get("expected_constexpr_bindings"), r.get("expected_philox_call"),
                             r.get("expected_scalar_infinity_where", False),
                             r.get("expected_welford_tree"), r.get("expected_jit_functions"),
                             r.get("expected_natural_reads"), r.get("expected_jit_procedures"),
                             r.get("expected_parameter_rebinding", False),
                             r.get("expected_fp32_constants"), r.get("expected_source_procedures")))
        for name, definition in r.get('expected_philox_call', {}).get('uniform_helpers', {}).items():
            helpers = [h for h in records if h.get('kind') == 'uniform_jit_function'
                       and h['python'] == r['python'] and h['lean'] == r['lean']
                       and h['python_function'] == name and h['lean_definition'] == definition]
            if len(helpers) != 1:
                raise ValueError("uniform call requires its separately checked complete helper")
        helper_kind = ("source_procedure" if "expected_source_procedures" in r else
                       "procedure_jit_function" if "expected_jit_procedures" in r else
                       "statement_jit_function" if "expected_jit_functions" in r else
                       "binary_pure_jit_function" if "expected_binary_functions" in r else "pure_jit_function")
        for name, definition in r.get("expected_source_procedures", r.get("expected_jit_procedures", r.get("expected_jit_functions", r.get("expected_binary_functions", r.get("expected_pure_functions", {}))))).items():
            helpers = [h for h in records if h.get("kind") == helper_kind
                       and h["lean"] == r["lean"] and h["lean_definition"] == definition]
            # Independently retained Python files may copy the same unchanged
            # helper into one shared Lean declaration. Bind each local call to
            # its own checked Python body; an imported helper still needs the
            # unique separately checked import below.
            local_helpers = [h for h in helpers if h["python"] == r["python"]]
            if local_helpers:
                helpers = local_helpers
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
