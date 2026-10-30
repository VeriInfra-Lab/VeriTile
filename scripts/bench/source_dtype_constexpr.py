"""Recorded scalar dtype specializations for source comparison only.

The original Python file and a successful native compiler profile are pinned.
Only direct dtype operands are substituted; no branch, arithmetic expression,
signature, wrapper, or failed native profile is rewritten or certified.
"""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import ast
import hashlib
import json
from pathlib import Path
import re

FIELD = 'dtype-constexpr'
ROOT = Path(__file__).resolve().parent.parent.parent


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _registry():
    from scripts.bench import audit_source as source
    try:
        registry = source._source_profile_json(
            (ROOT / 'bench/tritonbench_g/source_dtype_constexpr_profiles.json').read_text(), FIELD)
    except OSError as error:
        raise ValueError('Missing reviewed dtype constexpr evidence registry') from error
    if (not isinstance(registry, dict) or set(registry) != {'schema', 'profiles'}
            or type(registry['schema']) is not int or registry['schema'] != 1
            or not isinstance(registry['profiles'], dict)):
        raise ValueError('Unsupported dtype constexpr evidence registry')
    return registry['profiles']


def _evidence(record, python_text):
    """Read only the reviewed repository registry's evidence directory."""
    relative = Path(record['evidence'])
    if (relative.is_absolute() or '..' in relative.parts
            or relative.parts[:2] != ('bench', 'review-checkpoints')):
        raise ValueError('Dtype constexpr evidence must be a reviewed repository checkpoint')
    base = ROOT / relative
    receipt_bytes = (base / 'receipt.json').read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != record['receipt_sha256']:
        raise ValueError('Dtype constexpr native receipt digest mismatch')
    receipt = json.loads(receipt_bytes)
    if (receipt['source_sha256'] != hashlib.sha256(python_text.encode()).hexdigest()
            or receipt['triton_version'] != '3.8.0' or receipt['target'] != 'cuda sm90'
            or receipt['original_kernel_ast_unchanged'] is not True
            or receipt['kernel_executed'] is not False):
        raise ValueError('Dtype constexpr original source/compiler scope mismatch')
    for name, digest in receipt['artifact_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid dtype constexpr artifact path')
        if hashlib.sha256((base / path).read_bytes()).hexdigest() != digest:
            raise ValueError('Dtype constexpr native artifact digest mismatch: ' + name)
    results = receipt['results']
    if [row['dtype'] for row in results] != [record['selected'], *record['unsupported']]:
        raise ValueError('Dtype constexpr native profile inventory mismatch')
    selected = results[0]
    if (selected['compiled'] is not True
            or any(selected['constexprs'].get(name) != {'tl.float16': 'fp16'}[dtype]
                   for name, dtype in record['python'].items())):
        raise ValueError('Dtype constexpr requires the recorded successful fp16 configuration')
    if any(row['compiled'] is not False or 'out_dtype=bfloat16 is unsupported' not in row['error']
           for row in results[1:]):
        raise ValueError('Dtype constexpr failed configurations must remain recorded failures')
    return selected


def source_dtype_constexpr_profile(python_text, lean_text):
    from scripts.bench import audit_source as source
    declared = source.source_audit_metadata(lean_text, FIELD)
    if declared is None:
        return ()
    # Parameter renaming composes because it never touches this exact constexpr.
    # Other transformations are rejected until their combined source domain is reviewed.
    if any(source.source_audit_metadata(lean_text, key) is not None for key in
           ('constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes', 'materializations')):
        raise ValueError('Dtype constexpr requires the untransformed original Python profile')
    profile = source._source_profile_json(declared, FIELD)
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean', 'native'}
            or not isinstance(profile['python'], dict) or not profile['python']
            or not isinstance(profile['native'], dict)
            or set(profile['native']) != {'profile', 'sha256'}
            or not isinstance(profile['native']['profile'], str)):
        raise ValueError('Unsupported dtype constexpr profile schema')
    source._validate_source_profile_binding(lean_text, profile['lean'], FIELD)
    record = _registry().get(profile['native']['profile'])
    if (not isinstance(record, dict) or set(record) != {
            'jit', 'python', 'lean', 'evidence', 'receipt_sha256', 'selected', 'unsupported'}
            or _digest(record) != profile['native']['sha256']
            or record['python'] != profile['python'] or record['lean'] != profile['lean']
            or record['selected'] != 'fp16' or record['unsupported'] != ['bf16']
            or any(value != 'tl.float16' for value in profile['python'].values())):
        raise ValueError('Dtype constexpr profile does not match reviewed fp16 evidence')
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    parameters = {a.arg: a for a in kernel.args.posonlyargs + kernel.args.args + kernel.args.kwonlyargs}
    source._python_parameter_names(kernel)
    if kernel.name != record['jit'] or 'tl' in parameters:
        raise ValueError('Dtype constexpr original JIT or namespace mismatch')
    for name in profile['python']:
        argument = parameters.get(name)
        if argument is None or argument.annotation is None or ast.unparse(argument.annotation) != 'tl.constexpr':
            raise ValueError('Dtype constexpr requires an exact tl.constexpr parameter')
    nodes = [n for statement in kernel.body for n in ast.walk(statement)]
    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
                   ast.ExceptHandler, ast.Match)
    if any(isinstance(n, unsupported) for n in nodes):
        raise ValueError('Dtype constexpr rejects nested scopes or indirect binding')
    parents = {child: node for node in ast.walk(kernel) for child in ast.iter_child_nodes(node)}
    used = set()
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in profile['python']:
            continue
        parent = parents.get(node)
        call = parents.get(parent) if isinstance(parent, ast.keyword) else parent
        allowed = False
        if isinstance(node.ctx, ast.Load) and isinstance(call, ast.Call):
            if isinstance(parent, ast.keyword):
                allowed = ((ast.unparse(call.func), parent.arg) in
                           {('tl.zeros', 'dtype'), ('tl.full', 'dtype'), ('tl.dot', 'out_dtype')})
            else:
                allowed = (isinstance(call.func, ast.Attribute) and call.func.attr == 'to'
                           and call.args and call.args[0] is node and len(call.args) == 1
                           and not call.keywords)
        if not allowed:
            raise ValueError('Dtype constexpr parameter has a non-dtype use or rebinding: ' + node.id)
        used.add(node.id)
    if used != set(profile['python']):
        raise ValueError('Dtype constexpr parameter has no supported dtype operand')
    try:
        native = _evidence(record, python_text)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Missing or malformed dtype constexpr native evidence') from error
    constexprs = {name for name, arg in parameters.items()
                 if arg.annotation is not None and ast.unparse(arg.annotation) == 'tl.constexpr'}
    if (set(native['constexprs']) != constexprs
            or set(native['signature']) != set(parameters) - constexprs):
        raise ValueError('Dtype constexpr evidence omits original signature parameters')
    return tuple(profile['python'].items())


def python_source_with_dtype_constexpr_profile(python_text, lean_text):
    """Replace only checked Name locations; signatures and wrappers stay original."""
    from scripts.bench import audit_source as source
    constants = dict(source_dtype_constexpr_profile(python_text, lean_text))
    if not constants:
        return python_text
    replacements = {**dict(source.source_parameter_map(python_text, lean_text)), **constants}
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    offsets, total = [], 0
    for line in python_text.encode().splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    edits = []
    for statement in kernel.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id in replacements:
                edits.append((offsets[node.lineno - 1] + node.col_offset,
                              offsets[node.end_lineno - 1] + node.end_col_offset,
                              replacements[node.id].encode()))
    result = python_text.encode()
    for start, end, value in sorted(edits, reverse=True):
        result = result[:start] + value + result[end:]
    result = result.decode()
    _match_selected_source(result, lean_text)
    return result


def _match_selected_source(python_text, lean_text):
    """Compare complete expressions after only the recorded DSL syntax mapping.

    Existing general lexical gates intentionally do not compare every dot
    option or arithmetic RHS. A dtype profile must additionally retain both.
    This parser admits only this profile's direct parameter splices, the one
    range loop, and tuple/list notation for tl.advance's offset vector.
    """
    from scripts.bench import audit_source as source
    binders = set(source.lean_source_parameter_names(lean_text))
    body = source.lean_first_triton_body(lean_text)

    def typed(match):
        expression, carrier = match.groups()
        names = expression.split(' + ')
        if (any(name not in binders for name in names)
                or (carrier == 'ℝ') != (names == ['sm_scale'])
                or (len(names) > 1 and expression != 'N_CTX + P_SEQ')):
            raise ValueError('Dtype constexpr unsupported typed parameter splice')
        return '(' + expression + ')'

    body = re.sub(r'\$\(\(([A-Za-z_][A-Za-z0-9_]*(?: \+ [A-Za-z_][A-Za-z0-9_]*)?) : (Nat|ℝ)\)\)', typed, body)

    def plain(match):
        name = match[1]
        if name != '0' and name not in binders:
            raise ValueError('Dtype constexpr splice is not a source parameter')
        return name

    body = re.sub(r'\$\(([A-Za-z_][A-Za-z0-9_]*|0)\)', plain, body)
    if '$' in body:
        raise ValueError('Dtype constexpr unsupported Lean antiquotation')
    lines, pending, depth, indent, loops = [], '', 0, 0, 0
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == '}' and not pending:
            if indent != 1:
                raise ValueError('Dtype constexpr unsupported Lean block structure')
            indent = 0
            continue
        pending += (' ' if pending else '') + stripped
        depth += stripped.count('(') + stripped.count('[') - stripped.count(')') - stripped.count(']')
        if depth < 0:
            raise ValueError('Dtype constexpr unbalanced Lean expression')
        if depth or pending.endswith('+'):
            continue
        if pending.endswith('{'):
            if indent or pending != 'for start_n in range(lo, hi, BLOCK_N) {':
                raise ValueError('Dtype constexpr unsupported Lean loop header')
            lines.append('for start_n in range(lo, hi, BLOCK_N):')
            indent = 1
            loops += 1
        else:
            if '{' in pending or '}' in pending:
                raise ValueError('Dtype constexpr unsupported Lean statement')
            lines.append('    ' * indent + pending)
        pending = ''
    if pending or depth or indent or loops != 1:
        raise ValueError('Dtype constexpr incomplete Lean source body')
    try:
        actual = ast.parse('\n'.join(lines))
    except SyntaxError as error:
        raise ValueError('Dtype constexpr unsupported Lean expression syntax') from error

    class AdvanceVector(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (ast.unparse(node.func) == 'tl.advance' and len(node.args) == 2
                    and isinstance(node.args[1], ast.List)):
                node.args[1] = ast.Tuple(elts=node.args[1].elts, ctx=ast.Load())
            return node

    actual = AdvanceVector().visit(actual)
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    expected = ast.Module(body=kernel.body, type_ignores=[])
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        raise ValueError('Dtype constexpr complete selected source/RHS AST mismatch')
