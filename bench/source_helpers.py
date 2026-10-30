"""Explicit, checked expansion of local pure-return JIT arithmetic helpers.

This constructs a selected source comparison view, not an equivalence proof.
Every expanded RHS must match its Lean assignment as an expression tree.
Only parameter leaves, integer literals, arithmetic and tl.cdiv are supported;
unknown calls and binding constructs are errors rather than ignored effects.
"""
import ast
import copy
import json
import keyword
import re
from functools import lru_cache


def _attribute(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute(node.value)
        return base + '.' + node.attr if base else None
    return None


def _expression(node, parameters):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in parameters:
        return
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        _expression(node.left, parameters)
        _expression(node.right, parameters)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _expression(node.operand, parameters)
        return
    if (isinstance(node, ast.Call) and _attribute(node.func) == 'tl.cdiv'
            and len(node.args) == 2 and not node.keywords):
        for argument in node.args:
            _expression(argument, parameters)
        return
    raise ValueError('Source-audit helpers requires a pure arithmetic return expression')


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate Source-audit helpers JSON key: ' + name)
        result[name] = value
    return result


def _bindings(module, helpers):
    """Reject explicit shadowing of helper names and the canonical tl namespace."""
    protected = set(helpers) | {'triton', 'tl'}
    found = {name: [] for name in protected}
    dynamic = {'exec', 'eval', 'globals', 'locals', 'vars', 'setattr', 'delattr',
               '__dict__', '__getattribute__', '__setattr__', '__delattr__',
               '__import__', '__builtins__', '__globals__', '__closure__',
               'modules', 'import_module'}
    parents = {child: node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(module):
        if (isinstance(node, ast.Name) and node.id in dynamic
                or isinstance(node, ast.Attribute) and node.attr in dynamic):
            raise ValueError('Source-audit helpers does not support dynamic namespace access')
        if isinstance(node, (ast.Global, ast.Nonlocal)) and protected.intersection(node.names):
            raise ValueError('Source-audit helpers does not support indirect global bindings')
        if isinstance(node, ast.Name) and node.id in protected and not isinstance(node.ctx, ast.Load):
            raise ValueError('Source-audit helpers rejects rebinding: ' + node.id)
        if isinstance(node, ast.Name) and node.id in protected and isinstance(node.ctx, ast.Load):
            parent = parents.get(node)
            if node.id in helpers:
                allowed = isinstance(parent, ast.Call) and parent.func is node
            else:
                allowed = isinstance(parent, ast.Attribute) and parent.value is node
            if not allowed:
                raise ValueError('Source-audit helpers rejects aliases of helper/Triton bindings')
        if isinstance(node, ast.Attribute) and _attribute(node) in {'tl.cdiv', 'triton.language', 'triton.jit'}:
            parent = parents.get(node)
            allowed = isinstance(parent, ast.Call) and parent.func is node
            if _attribute(node) == 'triton.jit':
                allowed |= isinstance(parent, ast.FunctionDef) and node in parent.decorator_list
            if _attribute(node) == 'triton.language':
                # The canonical language namespace is already imported as tl.
                # Other attribute paths can leak its mutable function objects.
                allowed = False
            if not allowed:
                raise ValueError('Source-audit helpers rejects aliases of arithmetic/namespace attributes')
        if isinstance(node, ast.arg) and node.arg in protected:
            raise ValueError('Source-audit helpers rejects shadowed parameters: ' + node.arg)
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            base = _attribute(node.value)
            if node.attr in protected or base and any(
                    base == name or base.startswith(name + '.') for name in protected):
                raise ValueError('Source-audit helpers rejects mutated helper/Triton attributes')
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in protected:
            found[node.name].append(node)
        if isinstance(node, ast.ExceptHandler) and node.name in protected:
            raise ValueError('Source-audit helpers rejects exception binding')
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in protected:
            raise ValueError('Source-audit helpers rejects pattern binding')
        if isinstance(node, ast.MatchMapping) and node.rest in protected:
            raise ValueError('Source-audit helpers rejects pattern rest binding')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == '*':
                    raise ValueError('Source-audit helpers rejects wildcard imports')
                name = alias.asname or alias.name.split('.')[0]
                origin = alias.name if isinstance(node, ast.Import) else node.module or ''
                if origin.split('.')[0] in {'sys', 'builtins', 'importlib'}:
                    raise ValueError('Source-audit helpers rejects dynamic namespace imports')
                if isinstance(node, ast.Import):
                    triton_import = alias.name == 'triton' or alias.name.startswith('triton.')
                    canonical = (alias.name == 'triton' and alias.asname in (None, 'triton')
                                 or alias.name == 'triton.language' and alias.asname == 'tl')
                else:
                    triton_import = node.module == 'triton' or (node.module or '').startswith('triton.')
                    canonical = (node.level == 0 and node.module == 'triton'
                                 and alias.name == 'language' and alias.asname == 'tl')
                if triton_import and not canonical:
                    raise ValueError('Source-audit helpers rejects noncanonical Triton imports')
                if name in protected:
                    found[name].append(node)

    for name in ('triton', 'tl'):
        if len(found[name]) != 1 or found[name][0] not in module.body:
            raise ValueError('Source-audit helpers requires a unique top-level ' + name + ' import')
        node = found[name][0]
        if name == 'triton':
            valid = isinstance(node, ast.Import) and any(
                a.name == 'triton' and a.asname in (None, 'triton') for a in node.names)
        else:
            valid = (isinstance(node, ast.Import) and any(
                a.name == 'triton.language' and a.asname == 'tl' for a in node.names)
                or isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'triton'
                and any(a.name == 'language' and a.asname == 'tl' for a in node.names))
        if not valid:
            raise ValueError('Source-audit helpers requires the canonical ' + name + ' import')
    for name in helpers:
        if (len(found[name]) != 1 or not isinstance(found[name][0], ast.FunctionDef)
                or found[name][0] not in module.body):
            raise ValueError('Source-audit helpers requires one top-level JIT definition: ' + name)
    return {name: found[name][0] for name in helpers}


@lru_cache(maxsize=32)
def python_source_with_helpers(python_text, lean_text):
    # Imported lazily to keep the shared extractor's dependency acyclic.
    from bench import audit_source as source

    declared = source.source_audit_metadata(lean_text, 'helpers')
    if declared is None:
        return python_text
    try:
        profile = json.loads(declared, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError('Invalid Source-audit helpers JSON') from error
    if isinstance(profile, dict) and 'kind' in profile:
        from bench.source_finite_helpers import python_source_with_finite_helper
        return python_source_with_finite_helper(python_text, lean_text, profile)
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean'}
            or not isinstance(profile['python'], list) or not profile['python']
            or not isinstance(profile['lean'], dict)
            or set(profile['lean']) not in ({'source', 'args'}, {'source', 'args', 'symbolic'})):
        raise ValueError('Unsupported Source-audit helpers schema')
    names, binding = profile['python'], profile['lean']
    if any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
           or name in {'_', 'tl', 'triton'} for name in names) or len(set(names)) != len(names):
        raise ValueError('Invalid or duplicate Source-audit helper name')
    selected = source.select_lean_audit_source(lean_text)
    arguments = source._macro_arguments(selected)
    formals = source._macro_parameters(source.lean_first_preamble(selected))
    symbolic = binding.get('symbolic', [])
    if (not isinstance(symbolic, list) or any(not isinstance(name, str) for name in symbolic)
            or len(set(symbolic)) != len(symbolic)
            or set(arguments) & set(symbolic)
            or set(arguments) | set(symbolic) != set(formals)
            or any(formals.get(name) != 'tritonDType' for name in symbolic)):
        raise ValueError('Source-audit helpers requires every macro argument fixed or an explicit symbolic dtype')
    expected = {'source': source.source_audit_metadata(lean_text, 'Lean'), 'args': arguments}
    # JSON booleans and integers must not compare equal as Python values.
    fixed = {key: binding[key] for key in ('source', 'args')}
    if json.dumps(fixed, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('Source-audit helpers does not match the selected Lean profile')
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    parameters = source._python_parameter_names(kernel)
    if kernel.name in names:
        raise ValueError('Source-audit helpers cannot expand the selected kernel recursively')
    # Validate mappings and all constexpr arms against the original source.
    source.source_parameter_map(python_text, lean_text)
    module = ast.parse(python_text)
    helpers = _bindings(module, names)
    returns = {}
    for name, function in helpers.items():
        args = function.args
        if (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                or args.defaults or args.kw_defaults or function.returns is not None
                or len(function.decorator_list) != 1
                or _attribute(function.decorator_list[0]) != 'triton.jit'
                or any(a.annotation is not None for a in args.args)):
            raise ValueError('Source-audit helpers requires an undecorated-signature @triton.jit helper')
        formals = [a.arg for a in args.args]
        if not formals or len(formals) != len(set(formals)):
            raise ValueError('Source-audit helpers requires distinct formal parameters')
        body = function.body
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            raise ValueError('Source-audit helpers requires one return statement')
        _expression(body[0].value, set(formals))
        returns[name] = (formals, body[0].value)

    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.ExceptHandler,
                   ast.Match, ast.With, ast.AsyncWith)
    nodes = [node for statement in kernel.body for node in ast.walk(statement)]
    if any(isinstance(node, unsupported) for node in nodes):
        raise ValueError('Source-audit helpers cannot expand nested scopes or indirect bindings')
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    sites = []
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in helpers:
            continue
        call = parents.get(node)
        statement = parents.get(call)
        if (not isinstance(call, ast.Call) or call.func is not node or call.keywords
                or not isinstance(statement, ast.Assign) or statement.value is not call
                or len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name)):
            raise ValueError('Source-audit helpers requires a direct assignment RHS call')
        lhs = statement.targets[0].id
        if lhs in parameters or sum(isinstance(n, ast.Name) and n.id == lhs
                                   and not isinstance(n.ctx, ast.Load) for n in nodes) != 1:
            raise ValueError('Source-audit helpers requires an unambiguous fresh assignment target')
        formals, value = returns[node.id]
        if len(call.args) != len(formals) or any(
                not isinstance(arg, ast.Name) or not isinstance(arg.ctx, ast.Load) for arg in call.args):
            raise ValueError('Source-audit helpers only substitutes bare positional names')
        substitution = dict(zip(formals, call.args))

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, leaf):
                return copy.deepcopy(substitution.get(leaf.id, leaf))

        sites.append((node.id, lhs, call, Substitute().visit(copy.deepcopy(value))))
    if {name for name, _, _, _ in sites} != set(names):
        raise ValueError('Source-audit helpers contains an unused helper')

    lines = python_text.encode().splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    replacements = [(offsets[call.lineno - 1] + call.col_offset,
                     offsets[call.end_lineno - 1] + call.end_col_offset,
                     ast.unparse(value).encode()) for _, _, call, value in sites]
    expanded = python_text.encode()
    for start, end, value in sorted(replacements, reverse=True):
        expanded = expanded[:start] + value + expanded[end:]
    expanded = expanded.decode()
    selected = source.python_source_with_parameter_map(expanded, lean_text)
    selected = source.python_source_with_constexpr_profile(selected, lean_text)
    selected_kernel = source._explicit_python_audit_kernel(selected, lean_text)
    assignments = {n.targets[0].id: n.value for n in ast.walk(selected_kernel)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name)}
    lean_body = source.lean_first_triton_body(lean_text)
    for _, lhs, _, _ in sites:
        if lhs not in assignments:
            raise ValueError('Source-audit helpers cannot discard an expanded helper site')
        matches = re.findall(r'^\s*' + re.escape(lhs) + r'\s*=\s*([^\n]+)', lean_body, re.M)
        if len(matches) != 1:
            raise ValueError('Source-audit helpers needs one matching Lean assignment: ' + lhs)
        rhs = re.sub(r'\$\(([A-Za-z_][A-Za-z0-9_]*)\)', r'\1', matches[0]).strip()
        try:
            lean_value = ast.parse(rhs, mode='eval').body
        except SyntaxError as error:
            raise ValueError('Unsupported Lean helper RHS: ' + lhs) from error
        if ast.dump(lean_value) != ast.dump(assignments[lhs]):
            raise ValueError('Source-audit helper expression mismatch: ' + lhs)
    return expanded
