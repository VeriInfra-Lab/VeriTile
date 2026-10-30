"""Explicit identity materializations with complete straight-line RHS comparison.

This is a source comparison, not native ABI admission or proof replay. No
statement is erased to fix an inventory: the Python comparison body gains the
same checked identity assignments as Lean. All original statements, runtime
parameters, arithmetic trees and eager call arguments are compared afterwards.
Unsupported syntax and profile composition fail before any transformation.
"""
import ast
import copy
import hashlib
import keyword
from pathlib import Path
import re

FIELD = 'materializations'
INTEGER_CARRIERS = {'tl.int32': 'Int', 'tl.int64': 'Int',
                    'tl.uint32': 'Nat', 'tl.uint64': 'Nat'}
STORAGE_DTYPES = {'tl.float32', 'tl.float16', 'tl.bfloat16', 'tl.float64',
                  'tl.float8e4nv', 'tl.float8e5'}
ADAPTER = '_root_.VeriTile.Triton.Random.i32ArangeSize'
ADAPTER_PATH = 'VeriTile/Triton/DSL/RandomProfile.lean'
# Reviewed random-frontend source, not an arbitrary path supplied by metadata.
# A library change requires renewed review; a self-updated metadata hash is
# insufficient to bless a different adapter implementation.
# 2026-10-14: i64-to-i32 provenance now requires the actual low-word signed
# cast in the DSL. The complete multinomial source and integer regressions
# passed in the isolated stage. The identity adapter and recognizer did not change.
# 2026-10-13: the identity adapter and its recognizer are unchanged. The
# reviewed frontend adds direct Philox, explicit helper bindings and modular
# unsigned arithmetic. Versioned fixtures preserve each earlier reviewed adapter.
ADAPTER_SHA256 = '61f31027a53ea7305b18341f0c7dced51c9407589bf4fe16b1c7bfded7859d04'
_NAME = r'[A-Za-z_][A-Za-z0-9_]*'
_QUALIFIED = _NAME + r'(?:\.' + _NAME + r')*'


def _fail(message):
    raise ValueError('Source-audit materializations: ' + message)


def _name(value):
    return (isinstance(value, str) and not keyword.iskeyword(value)
            and re.fullmatch(_NAME, value) is not None)


def _adapter_path():
    return Path(__file__).resolve().parents[1] / ADAPTER_PATH


def _check_adapter(digest):
    if digest != ADAPTER_SHA256:
        _fail('unreviewed adapter fingerprint')
    try:
        actual = hashlib.sha256(_adapter_path().read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError('Source-audit materializations: missing reviewed adapter source') from error
    if actual != ADAPTER_SHA256:
        _fail('adapter source differs from reviewed identity definition')


def _binders(selected):
    from bench import audit_source as source
    signature = source.quoted_kernel_signature(selected)
    if not signature:
        _fail('requires an explicit quoted kernel lambda')
    signature = source.original_statement_splices(signature, source.lean_first_preamble(selected))
    result, position = {}, 0
    while position < len(signature):
        if signature[position].isspace():
            position += 1
            continue
        if signature[position] != '(':
            _fail('unsupported lambda binder syntax')
        end, depth = position + 1, 1
        while end < len(signature) and depth:
            depth += (signature[end] == '(') - (signature[end] == ')')
            end += 1
        if depth:
            _fail('unterminated lambda binder')
        parts = source._macro_parts(signature[position + 1:end - 1], ':')
        if len(parts) != 2 or not parts[1]:
            _fail('lambda binders require exact types')
        names = parts[0].split()
        if not names or any(not _name(name) or name in result for name in names):
            _fail('invalid or repeated lambda binder')
        for name in names:
            result[name] = ' '.join(parts[1].split())
        position = end
    return result


def _check_expression(node, defined):
    """Small explicit grammar, preserving every operand and argument position."""
    if isinstance(node, ast.Name):
        if node.id not in defined:
            _fail('unknown or forward expression name: ' + node.id)
    elif isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            _fail('unsupported expression literal')
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        _check_expression(node.left, defined)
        _check_expression(node.right, defined)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _check_expression(node.operand, defined)
    elif isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1 and isinstance(
            node.ops[0], (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq)):
        _check_expression(node.left, defined)
        _check_expression(node.comparators[0], defined)
    elif isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'tl' and node.func.attr in
                {'program_id', 'arange', 'load', 'rand', 'where', 'store'}):
            if any(keyword.arg is None for keyword in node.keywords):
                _fail('expanded keyword arguments are not admitted')
            if len({keyword.arg for keyword in node.keywords}) != len(node.keywords):
                _fail('duplicate call keyword')
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                _check_expression(argument, defined)
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
              and len(node.args) == 1 and not node.keywords
              and isinstance(node.args[0], ast.Attribute)
              and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == 'tl'
              and 'tl.' + node.args[0].attr in STORAGE_DTYPES):
            _check_expression(node.func.value, defined)
        else:
            _fail('unsupported call or dynamic language namespace')
    else:
        _fail('unsupported computation expression: ' + type(node).__name__)


def _check_body(body, parameters):
    defined = set(parameters)
    for statement in body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            name = statement.targets[0].id
            if name in defined or name == 'tl':
                _fail('shadowed or rebound source name: ' + name)
            _check_expression(statement.value, defined)
            defined.add(name)
        elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
              and isinstance(statement.value.func, ast.Attribute)
              and isinstance(statement.value.func.value, ast.Name)
              and statement.value.func.value.id == 'tl' and statement.value.func.attr == 'store'):
            _check_expression(statement.value, defined)
        else:
            _fail('only complete straight-line assignments and stores are admitted')


def _lean_comparison_body(body, profile, binders):
    from bench import audit_source as source
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    regions, position = {}, 0
    while position < len(lines) and lines[position].startswith('tl.region '):
        items = lines[position][len('tl.region '):].split(',')
        for item in items:
            match = re.fullmatch(r'\s*(' + _NAME + r')\s*=\s*(tl\.\w+)\s*', item)
            if match is None or match[1] in regions:
                _fail('ambiguous or unsupported region declaration')
            regions[match[1]] = match[2]
        position += 1
    if regions != profile['regions']:
        _fail('region storage differs from explicitly selected comparison profile')
    runtime, constexpr = profile['bindings']
    seed, seed_local = runtime['python'], runtime['local']
    size, size_local = constexpr['python'], constexpr['local']
    carrier = INTEGER_CARRIERS[runtime['dtype']]
    prefix = [
        re.escape(seed_local) + r'\s*=\s*\$\(\(\s*' + re.escape(seed) +
        r'\s*:\s*' + carrier + r'\s*\)\)',
        r'tl\.source_integer\s+' + re.escape(seed_local) + r'\s*=\s*' + re.escape(runtime['dtype']),
        re.escape(size_local) + r'\s*=\s*\$\(\s*' + re.escape(ADAPTER) +
        r'\s+' + re.escape(size) + r'\s+\(\s*' + re.escape(constexpr['proof']['function']) +
        r'\s+' + re.escape(constexpr['proof']['binder']) + r'\s*\)\s*\)',
    ]
    if len(lines) < position + len(prefix):
        _fail('missing complete materialization prefix')
    for expected, line in zip(prefix, lines[position:position + len(prefix)]):
        if re.fullmatch(expected, line) is None:
            _fail('materialization is not the recorded direct parameter/metadata/identity adapter')
    rest = '\n'.join(lines[position + len(prefix):])

    def external(match):
        if match[1] not in binders:
            _fail('antiquotation is not a lambda parameter')
        return match[1]

    rest = re.sub(r'\$\(\s*(' + _NAME + r')\s*\)', external, rest)
    if '$' in rest or 'tl.source_integer' in rest or 'tl.region' in rest:
        _fail('unrecorded antiquotation or metadata in computation body')
    normalized = f'{seed_local} = {seed}\n{size_local} = {size}\n' + rest
    try:
        return ast.parse(normalized)
    except SyntaxError as error:
        raise ValueError('Source-audit materializations: unsupported Lean computation syntax') from error


def python_source_with_materializations(python_text, lean_text):
    from bench import audit_source as source
    declared = source.source_audit_metadata(lean_text, FIELD)
    if declared is None:
        return python_text
    profile = source._source_profile_json(declared, FIELD)
    if isinstance(profile, dict) and profile.get('kind') == 'direct_uniform_identity_v1':
        from bench.source_uniform_materializations import python_source_with_uniform_materialization
        return python_source_with_uniform_materialization(python_text, lean_text, profile)
    if any(source.source_audit_metadata(lean_text, key) is not None for key in
           ('params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes')):
        _fail('profile composition requires a separate review of the unchanged original')
    if isinstance(profile, dict) and profile.get('kind') == 'guarded_constexpr_identity_v1':
        from bench.source_control_materializations import python_source_with_guarded_materialization
        return python_source_with_guarded_materialization(python_text, lean_text, profile)
    if (not isinstance(profile, dict) or set(profile) != {'source_sha256', 'lean', 'regions', 'bindings'}
            or profile['source_sha256'] != hashlib.sha256(python_text.encode()).hexdigest()
            or not isinstance(profile['regions'], dict) or not profile['regions']
            or any(not _name(name) or not isinstance(dtype, str) or dtype not in STORAGE_DTYPES
                   for name, dtype in profile['regions'].items())
            or not isinstance(profile['bindings'], list) or len(profile['bindings']) != 2):
        _fail('invalid profile schema or full original source fingerprint')
    runtime, constexpr = profile['bindings']
    if (not isinstance(runtime, dict) or set(runtime) != {'python', 'local', 'kind', 'dtype'}
            or runtime['kind'] != 'runtime_integer' or not isinstance(runtime['dtype'], str)
            or runtime['dtype'] not in INTEGER_CARRIERS
            or not isinstance(constexpr, dict)
            or set(constexpr) != {'python', 'local', 'kind', 'adapter', 'adapter_sha256', 'proof'}
            or constexpr['kind'] != 'constexpr_identity' or constexpr['adapter'] != ADAPTER
            or not isinstance(constexpr['proof'], dict)
            or set(constexpr['proof']) != {'function', 'binder'}
            or not isinstance(constexpr['proof']['function'], str)
            or re.fullmatch(_QUALIFIED, constexpr['proof']['function']) is None
            or not _name(constexpr['proof']['binder'])):
        _fail('unsupported binding or adapter schema')
    names = [binding[key] for binding in profile['bindings'] for key in ('python', 'local')]
    if any(not _name(name) or name == 'tl' for name in names) or len(set(names)) != len(names):
        _fail('materializations require distinct original parameter and local names')
    _check_adapter(constexpr['adapter_sha256'])
    selected = source._validate_source_profile_binding(lean_text, profile['lean'], FIELD)
    # This validates every macro arm and admission guard before source comparison.
    lean_body = source.lean_first_triton_body(selected)
    binders = _binders(selected)
    try:
        kernel = source._explicit_python_audit_kernel(python_text, lean_text)
        tree = ast.parse(python_text)
    except SyntaxError as error:
        raise ValueError('Source-audit materializations: invalid original Python') from error
    parameters = source._python_parameter_names(kernel)
    if (kernel.args.defaults or kernel.args.kw_defaults or kernel.args.posonlyargs
            or kernel.args.kwonlyargs or len(kernel.decorator_list) != 1
            or not source.is_triton_jit(kernel.decorator_list[0])
            or 'tl' in parameters):
        _fail('unsupported original kernel signature')
    parameters_by_name = {argument.arg: argument for argument in kernel.args.args}
    seed, size = runtime['python'], constexpr['python']
    if (seed not in parameters_by_name or size not in parameters_by_name
            or parameters_by_name[seed].annotation is not None
            or parameters_by_name[size].annotation is None
            or ast.dump(parameters_by_name[size].annotation) != ast.dump(ast.parse('tl.constexpr', mode='eval').body)
            or any(argument.annotation is not None for argument in kernel.args.args if argument.arg != size)):
        _fail('bindings do not preserve runtime seed and original constexpr signature')
    proof = constexpr['proof']['binder']
    if (list(binders) != parameters + [proof] or proof in parameters
            or binders[seed] != INTEGER_CARRIERS[runtime['dtype']]
            or binders[size] != 'Nat' or binders[proof] != 'SourceDomain.LegalBlock ' + size
            or any(name not in parameters or binders[name] != 'RegionName' for name in profile['regions'])):
        _fail('selected lambda carrier, parameters, regions or legal-block proof differ')
    # Exactly one unmodified language import; no dynamic/rebound namespace in
    # the original target is admitted by the complete body grammar below.
    imports = [node for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.name == 'triton.language' and alias.asname == 'tl']
    if len(imports) != 1:
        _fail('requires the original direct Triton language import')
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in {'tl', 'triton'}:
            _fail('original source rebinds the language namespace')
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {'tl', 'triton'}:
                _fail('original source mutates the language namespace')
    locals_ = {binding['local'] for binding in profile['bindings']}
    if locals_ & (set(binders) | {node.id for node in ast.walk(kernel) if isinstance(node, ast.Name)}):
        _fail('materialization locals collide with an original source name')
    _check_body(kernel.body, parameters)
    mapping = {binding['python']: binding['local'] for binding in profile['bindings']}

    class RenameReads(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in mapping:
                node.id = mapping[node.id]
            return node

    changed = copy.deepcopy(kernel)
    changed.body = [ast.Assign(targets=[ast.Name(id=binding['local'], ctx=ast.Store())],
                               value=ast.Name(id=binding['python'], ctx=ast.Load()))
                    for binding in profile['bindings']] + [RenameReads().visit(copy.deepcopy(statement))
                                                          for statement in kernel.body]
    expected_text = ast.unparse(ast.fix_missing_locations(ast.Module(body=changed.body, type_ignores=[])))
    # Only the established implicit destination store conversion is normalized.
    # Every other cast and every computation expression must match exactly.
    for before, after in zip(source.tl_calls(expected_text, 'store'),
                             source.store_calls_with_explicit_destination_casts(expected_text, lean_body)):
        expected_text = expected_text.replace(before, after, 1)
    expected = ast.parse(expected_text)
    actual = _lean_comparison_body(lean_body, profile, binders)
    _check_body(expected.body, parameters)
    _check_body(actual.body, parameters)
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        _fail('complete source statement/expression AST differs after checked identity materialization')
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == kernel.name:
            tree.body[index] = changed
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
