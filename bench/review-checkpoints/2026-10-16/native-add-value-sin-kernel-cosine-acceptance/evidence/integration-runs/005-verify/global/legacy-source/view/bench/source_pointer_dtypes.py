"""Recorded pointer-storage comparisons for a selected source-audit profile.

This does not infer a boolean constexpr from a Lean dtype. It checks the whole
original Python source, native signature and selected Lean region declarations
before selecting a direct pointer-dtype equality. Every original arm is checked
for rebinding and unsupported dtype use before any arm can be removed. Other
runtime guards remain present. The result is comparison text, never execution.
"""
import ast
import hashlib
import json
import re
from pathlib import Path


DTYPES = {
    'tl.float16': ('*fp16', 'f16'),
    'tl.float8e4nv': ('*fp8e4nv', 'f8E4M3FN'),
}
SCALARS = {'i32': 'i32'}
FIELD = 'pointer-dtypes'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _registry():
    """Only this reviewed registry is readable; source metadata supplies no path."""
    from bench import audit_source as source
    path = Path(__file__).resolve().parent / 'tritonbench_g/source_pointer_dtype_profiles.json'
    try:
        data = source._source_profile_json(path.read_text(), FIELD + ' evidence')
    except OSError as error:
        raise ValueError('Missing reviewed pointer dtype evidence registry') from error
    if (not isinstance(data, dict) or set(data) != {'schema', 'profiles'}
            or type(data['schema']) is not int or data['schema'] != 1
            or not isinstance(data['profiles'], dict)):
        raise ValueError('Unsupported pointer dtype evidence registry')
    return data['profiles']


def _pointer_attribute(node):
    if (isinstance(node, ast.Attribute) and node.attr == 'element_ty'
            and isinstance(node.value, ast.Attribute) and node.value.attr == 'dtype'
            and isinstance(node.value.value, ast.Name)):
        return node.value.value.id
    return None


def _comparison(node, pointers):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1):
        return None
    pointer = _pointer_attribute(node.left)
    right = node.comparators[0]
    if (pointer not in pointers or not isinstance(right, ast.Attribute)
            or not isinstance(right.value, ast.Name) or right.value.id != 'tl'):
        return None
    dtype = 'tl.' + right.attr
    return (pointer, dtype) if dtype in DTYPES else None


def source_pointer_dtype_profile(python_text, lean_text):
    from bench import audit_source as source
    declared = source.source_audit_metadata(lean_text, FIELD)
    if declared is None:
        return ()
    # None of the currently reviewed pointer records also rewrites parameters,
    # helpers or scalars. Reject composition rather than validate a changed
    # source against the native receipt or accidentally omit an original arm.
    if any(source.source_audit_metadata(lean_text, key) is not None
           for key in ('params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern')):
        raise ValueError('Pointer dtype profiles require an untransformed original Python source')
    profile = source._source_profile_json(declared, FIELD)
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean', 'native'}
            or not isinstance(profile['python'], dict) or not profile['python']
            or not isinstance(profile['native'], dict)
            or set(profile['native']) != {'profile', 'sha256'}
            or not isinstance(profile['native']['profile'], str)
            or not isinstance(profile['native']['sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', profile['native']['sha256'])):
        raise ValueError('Unsupported Source-audit pointer-dtypes schema')
    pointers = profile['python']
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)
           or not isinstance(dtype, str) or dtype not in DTYPES
           for name, dtype in pointers.items()):
        raise ValueError('Pointer dtype profiles require named fp16 or float8e4nv pointers')
    selected = source._validate_source_profile_binding(lean_text, profile['lean'], FIELD)
    try:
        kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    except SyntaxError as error:
        raise ValueError('Invalid original Python source for pointer dtype profile') from error
    names = source._python_parameter_names(kernel)
    parameters = {argument.arg: argument for argument in
                  kernel.args.posonlyargs + kernel.args.args + kernel.args.kwonlyargs}
    constants = {name for name, arg in parameters.items()
                 if isinstance(arg.annotation, ast.Attribute) and arg.annotation.attr == 'constexpr'
                 and isinstance(arg.annotation.value, ast.Name) and arg.annotation.value.id == 'tl'}
    if any(name not in parameters or parameters[name].annotation is not None for name in pointers):
        raise ValueError('Pointer dtype selection requires an original unannotated runtime pointer parameter')
    if 'tl' in names:
        raise ValueError('Pointer dtype source shadows the Triton language namespace')
    record = _registry().get(profile['native']['profile'])
    if (not isinstance(record, dict) or set(record) != {
            'jit', 'python', 'lean', 'receipt', 'receipt_sha256', 'native_dtype', 'ttir'}):
        raise ValueError('Unknown or malformed reviewed pointer dtype evidence')
    if _digest(record) != profile['native']['sha256']:
        raise ValueError('Pointer dtype native evidence digest mismatch')
    if (record['jit'] != kernel.name or record['python'] != pointers
            or _digest(record['lean']) != _digest(profile['lean'])):
        raise ValueError('Pointer dtype native evidence does not bind this selected source/profile')
    receipt = record['receipt']
    if (not isinstance(receipt, dict) or set(receipt) != {
            'source_sha256', 'triton_version', 'target', 'scope', 'results'}
            or _digest(receipt) != record['receipt_sha256']
            or receipt['source_sha256'] != hashlib.sha256(python_text.encode()).hexdigest()
            or receipt['triton_version'] != '3.7.1'
            or _digest(receipt['target']) != _digest({'backend': 'cuda', 'arch': 90, 'warp_size': 32})
            or not isinstance(receipt['results'], list)
            or not isinstance(record['ttir'], str)):
        raise ValueError('Pointer dtype original source/compiler/receipt evidence mismatch')
    rows = receipt['results']
    if (not rows or any(not isinstance(row, dict) for row in rows)
            or any(not isinstance(row.get('dtype'), str) for row in rows)
            or len({row['dtype'] for row in rows}) != len(rows)):
        raise ValueError('Ambiguous pointer dtype native result inventory')
    matches = [row for row in rows if row['dtype'] == record['native_dtype']]
    if len(matches) != 1:
        raise ValueError('Pointer dtype evidence has no unique selected native result')
    native = matches[0]
    if (set(native) != {'dtype', 'constexprs', 'signature', 'num_warps', 'num_stages',
                       'status', 'ttir_sha256', 'ptx_bytes', 'dot_lines', 'signed_minus_one', 'loop_present'}
            or native['status'] != 'compiled'
            or not isinstance(native['constexprs'], dict) or set(native['constexprs']) != constants
            or any(type(value) is not int or value <= 0 for value in native['constexprs'].values())
            or not isinstance(native['signature'], dict) or set(native['signature']) != set(names) - constants
            or any(type(native[name]) is not int or native[name] <= 0
                   for name in ('num_warps', 'num_stages', 'ptx_bytes'))
            or native['ttir_sha256'] != hashlib.sha256(record['ttir'].encode()).hexdigest()):
        raise ValueError('Pointer dtype native signature/configuration/TTIR evidence mismatch')
    if (any(DTYPES[dtype][0] != '*' + native['dtype'] for dtype in pointers.values())
            or native['dot_lines'] != [line.strip() for line in record['ttir'].splitlines() if 'tt.dot' in line]
            or type(native['signed_minus_one']) is not bool
            or native['signed_minus_one'] != ('-1 : i32' in record['ttir'])
            or type(native['loop_present']) is not bool
            or native['loop_present'] != ('scf.for' in record['ttir'])):
        raise ValueError('Pointer dtype native result observations disagree with actual TTIR')
    headers = re.findall(r'^\s*tt\.func public @' + re.escape(kernel.name) +
                         r'\(([^\n]*)\) attributes \{[^\n]*\} \{\s*$', record['ttir'], re.M)
    if len(headers) != 1:
        raise ValueError('Pointer dtype evidence requires one actual named native TTIR signature')
    # Every runtime parameter must occur once, in original order, at its exact
    # native type. Native pointer storage also binds every corresponding Lean
    # region, not only the pointer which happens to guard this branch.
    observed = re.findall(r'%([A-Za-z_][A-Za-z0-9_]*):\s*(!tt\.ptr<[^>]+>|[A-Za-z0-9]+)', headers[0])
    if [name for name, _ in observed] != [name for name in names if name not in constants]:
        raise ValueError('Pointer dtype TTIR parameter inventory mismatch')
    observed = dict(observed)
    regions = source.declared_region_dtypes(source.lean_first_triton_body(selected))
    binders = set(source.lean_source_parameter_names(selected))
    native_pointer_types = {}
    inverse = {abi: (dtype, ir) for dtype, (abi, ir) in DTYPES.items()}
    for name, abi in native['signature'].items():
        if not isinstance(abi, str):
            raise ValueError('Unsupported pointer dtype native ABI')
        if abi in inverse:
            dtype, ir = inverse[abi]
            native_pointer_types[name] = dtype
            if name not in binders or regions.get(name) != {dtype}:
                raise ValueError('Pointer dtype native storage disagrees with selected Lean region dtype')
            expected = '!tt.ptr<' + ir + '>'
        elif abi in SCALARS:
            expected = SCALARS[abi]
        else:
            raise ValueError('Unsupported pointer dtype native ABI')
        if observed.get(name) != expected:
            raise ValueError('Pointer dtype native ABI disagrees with actual TTIR signature')
    if any(native_pointer_types.get(name) != dtype for name, dtype in pointers.items()):
        raise ValueError('Pointer dtype selected profile disagrees with native pointer storage')
    nodes = [node for statement in kernel.body for node in ast.walk(statement)]
    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
                   ast.ExceptHandler, ast.Match)
    if any(isinstance(node, unsupported) for node in nodes):
        raise ValueError('Pointer dtype profile rejects nested scopes or indirect bindings')
    if any(isinstance(node, ast.Name) and node.id in {*native_pointer_types, 'tl'}
           and not isinstance(node.ctx, ast.Load) for node in nodes):
        raise ValueError('Pointer dtype parameter or language namespace is rebound in an original arm')
    guards, used = set(), set()
    for node in nodes:
        if isinstance(node, ast.If) and (pair := _comparison(node.test, pointers)) is not None:
            guards.add(id(node.test.left))
            guards.add(id(node.test.left.value))
            used.add(pair[0])
    for node in nodes:
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in pointers and id(node) not in guards:
                raise ValueError('Pointer dtype parameter has an unsupported attribute use in an original arm')
            if isinstance(root, ast.Name) and root.id in {*native_pointer_types, 'tl'} and not isinstance(node.ctx, ast.Load):
                raise ValueError('Pointer dtype source mutates a pointer/language attribute')
    parents = {child: node for node in ast.walk(kernel) for child in ast.iter_child_nodes(node)}
    for node in nodes:
        if not isinstance(node, ast.Name):
            continue
        parent = parents.get(node)
        if node.id == 'tl' and not isinstance(parent, ast.Attribute):
            raise ValueError('Pointer dtype source has an indirect language namespace use')
        if node.id not in pointers:
            continue
        if isinstance(parent, ast.Attribute) and id(parent) in guards:
            continue
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Sub)) and parent.left is node:
            continue
        if (isinstance(parent, ast.Call) and ast.unparse(parent.func) in {'tl.load', 'tl.store'}
                and parent.args and parent.args[0] is node):
            continue
        raise ValueError('Pointer dtype parameter has an indirect or unsupported use in an original arm')
    if used != set(pointers):
        raise ValueError('Pointer dtype profile has no supported direct equality if guard')
    return tuple(pointers.items())


def python_source_with_pointer_dtype_profile(python_text, lean_text):
    from bench import audit_source as source
    pointers = dict(source_pointer_dtype_profile(python_text, lean_text))
    if not pointers:
        return python_text
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    first_line, last_line = kernel.body[0].lineno, kernel.end_lineno
    if (isinstance(kernel.body[0], ast.Expr) and isinstance(kernel.body[0].value, ast.Constant)
            and isinstance(kernel.body[0].value.value, str)):
        kernel.body = kernel.body[1:]

    class SelectPointerGuards(ast.NodeTransformer):
        def visit_If(self, node):
            pair = _comparison(node.test, pointers)
            if pair is None:
                return self.generic_visit(node)
            pointer, dtype = pair
            selected = node.body if pointers[pointer] == dtype else node.orelse
            result = []
            for statement in selected:
                value = self.visit(statement)
                result.extend(value if isinstance(value, list) else [value])
            return result

    kernel = SelectPointerGuards().visit(kernel)
    if not kernel.body:
        raise ValueError('Pointer dtype selection produced an empty kernel')
    body = ast.unparse(ast.Module(body=kernel.body, type_ignores=[]))
    lines = python_text.splitlines(keepends=True)
    result = (''.join(lines[:first_line - 1]) +
              '\n'.join('    ' + line for line in body.splitlines()) + '\n' +
              ''.join(lines[last_line:]))
    try:
        ast.parse(result)
    except SyntaxError as error:
        raise ValueError('Empty or malformed pointer dtype branch result') from error
    return result
