"""Complete guarded-source comparison with one checked constexpr identity.

Version one admits the complete layer-norm forward quotation, including all
ten symbolic flags. It does not specialize or erase either side's branches.
The Python comparison gains the identity statement at the recorded position;
the original signature, decorators, host wrapper and every source statement
remain present. This is a source gate, not native admission or proof replay.
"""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import ast
import copy
import hashlib
import itertools
import re

from scripts.bench import audit_source as source
from scripts.bench import source_materializations as material

KIND = 'guarded_constexpr_identity_v1'
TARGET = '_layer_norm_fwd_1pass_kernel'
SELECTOR = 'layerNormFwdSource%'
# Reviewed complete upstream file, including autotune/heuristics and host code.
# A metadata-only rehash cannot authorize a different original program.
SOURCE_SHA256 = '27dfccf23671d5b9c6f4b749cef39ddf69079793c486a833cbae9c57fe3a4de4'
DTYPE_ARGUMENTS = ('input', 'output', 'weight', 'bias', 'residual', 'input1',
                   'weight1', 'bias1', 'residualOut', 'rowscale')
FLOAT_REGIONS = {'X': 'input', 'Y': 'output', 'W': 'weight', 'B': 'bias',
                 'RESIDUAL': 'residual', 'X1': 'input1', 'W1': 'weight1',
                 'B1': 'bias1', 'Y1': 'output', 'RESIDUAL_OUT': 'residualOut',
                 'ROWSCALE': 'rowscale'}
FIXED_REGIONS = {'SEEDS': 'tl.int64', 'DROPOUT_MASK': 'tl.int1',
                 'Mean': 'tl.float32', 'Rstd': 'tl.float32'}
POINTERS = list(FLOAT_REGIONS)[:8] + ['Y1', 'RESIDUAL_OUT', 'ROWSCALE',
                                    'SEEDS', 'DROPOUT_MASK', 'Mean', 'Rstd']
STRIDES = ['stride_x_row', 'stride_y_row', 'stride_res_row', 'stride_res_out_row',
           'stride_x1_row', 'stride_y1_row']
FLAGS = ('IS_RMS_NORM', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'HAS_BIAS',
         'HAS_DROPOUT', 'STORE_DROPOUT_MASK', 'HAS_ROWSCALE', 'HAS_X1', 'HAS_W1', 'HAS_B1')
PARAMETERS = POINTERS + STRIDES + ['M', 'N', 'eps', 'dropout_p', FLAGS[0], 'BLOCK_N', *FLAGS[1:]]
CALLS = {'program_id', 'arange', 'load', 'rand', 'where', 'store', 'sum', 'sqrt'}
CASTS = material.STORAGE_DTYPES | {'tl.uint32'}
DECORATORS = '''
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4), triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"])
@triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
@triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
@triton.jit
def target(): pass
'''


def _fail(message):
    raise ValueError('Source-audit guarded materialization: ' + message)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _guard(node, values):
    if isinstance(node, ast.Name) and node.id in FLAGS:
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        if isinstance(node.operand, ast.Name) and node.operand.id in FLAGS:
            return not values[node.operand.id]
    _fail('control must retain an original boolean flag or its negation')


def _expression(node, defined, values):
    if isinstance(node, ast.Name):
        if node.id not in defined:
            _fail('unknown or forward expression name: ' + node.id)
    elif isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            _fail('unsupported computation literal')
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        _expression(node.left, defined, values)
        _expression(node.right, defined, values)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _expression(node.operand, defined, values)
    elif isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1 and isinstance(
            node.ops[0], (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq)):
        _expression(node.left, defined, values)
        _expression(node.comparators[0], defined, values)
    elif isinstance(node, ast.IfExp):
        if values is None:
            _guard(node.test, dict.fromkeys(FLAGS, False))
            _expression(node.body, defined, None)
            _expression(node.orelse, defined, None)
        else:
            _expression(node.body if _guard(node.test, values) else node.orelse, defined, values)
    elif isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'tl' and node.func.attr in CALLS):
            if (any(k.arg is None for k in node.keywords)
                    or len({k.arg for k in node.keywords}) != len(node.keywords)):
                _fail('expanded or duplicate call keyword')
            # tl.where is eager: validate every argument, including both values.
            for argument in [*node.args, *(k.value for k in node.keywords)]:
                _expression(argument, defined, values)
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
              and len(node.args) == 1 and not node.keywords
              and isinstance(node.args[0], ast.Attribute)
              and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == 'tl'
              and 'tl.' + node.args[0].attr in CASTS):
            _expression(node.func.value, defined, values)
        else:
            _fail('unsupported call or dynamic language namespace')
    else:
        _fail('unsupported computation expression: ' + type(node).__name__)


def _statements(body, defined, values):
    for statement in body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            name = statement.targets[0].id
            if name in {*FLAGS, 'BLOCK_N', 'tl', 'triton'}:
                _fail('rebound control parameter or namespace: ' + name)
            _expression(statement.value, defined, values)
            defined.add(name)
        elif (isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name)
              and isinstance(statement.op, (ast.Add, ast.Mult))):
            if statement.target.id not in defined or statement.target.id in {*FLAGS, 'BLOCK_N', 'tl', 'triton'}:
                _fail('invalid augmented-assignment target')
            _expression(statement.value, defined, values)
        elif isinstance(statement, ast.If):
            _statements(statement.body if _guard(statement.test, values) else statement.orelse,
                        defined, values)
        elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
              and isinstance(statement.value.func, ast.Attribute)
              and isinstance(statement.value.func.value, ast.Name)
              and statement.value.func.value.id == 'tl' and statement.value.func.attr == 'store'):
            _expression(statement.value, defined, values)
        else:
            _fail('unsupported source statement: ' + type(statement).__name__)


def _validate_tree(body, parameters):
    """Check every arm, including structurally present contradictory guards."""
    all_names = set(parameters) | {n.id for statement in body for n in ast.walk(statement)
                                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

    def grammar(statements):
        for statement in statements:
            if isinstance(statement, ast.If):
                _guard(statement.test, dict.fromkeys(FLAGS, False))
                grammar(statement.body)
                grammar(statement.orelse)
            else:
                # All IfExp arms get a grammar check even in dead branches.
                _statements([statement], set(all_names), None)
    grammar(body)
    # Only name availability uses valuations. The returned/comparison AST keeps
    # all ten flags and every branch, without constant folding or dead-code removal.
    for bits in itertools.product((False, True), repeat=len(FLAGS)):
        _statements(body, set(parameters), dict(zip(FLAGS, bits)))


def _namespace(tree):
    imports = [alias for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.name == 'triton.language' and alias.asname == 'tl']
    if len(imports) != 1:
        _fail('requires one original direct Triton language import')
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in {'tl', 'triton'}:
            _fail('original source rebinds the language namespace')
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {'tl', 'triton'}:
                _fail('original source mutates the language namespace')


def _signature(kernel, binders):
    if (kernel.name != TARGET or source._python_parameter_names(kernel) != PARAMETERS
            or kernel.args.defaults or kernel.args.kw_defaults or kernel.args.posonlyargs
            or kernel.args.kwonlyargs or kernel.returns is not None or getattr(kernel, 'type_params', [])):
        _fail('requires the complete original kernel signature')
    for argument in kernel.args.args:
        expected = 'tl.constexpr' if argument.arg in {*FLAGS, 'BLOCK_N'} else None
        actual = ast.unparse(argument.annotation) if argument.annotation else None
        if actual != expected:
            _fail('original parameter annotation differs: ' + argument.arg)
    expected_decorators = ast.parse(DECORATORS).body[0].decorator_list
    if [_dump(d) for d in kernel.decorator_list] != [_dump(d) for d in expected_decorators]:
        _fail('original autotune, heuristics or JIT decorators differ')
    expected = {name: 'RegionName' for name in POINTERS}
    expected.update(SEEDS='Region .int', DROPOUT_MASK='Region .bool')
    expected.update({name: 'Nat' for name in [*STRIDES, 'M', 'N', 'BLOCK_N']})
    expected.update(eps='ℝ', dropout_p='ℝ')
    expected.update({name: 'Bool' for name in FLAGS})
    expected['hBlock'] = 'Random.ArangeSizeDomain BLOCK_N'
    order = PARAMETERS.copy()
    order.insert(order.index('BLOCK_N') + 1, 'hBlock')
    if list(binders) != order or binders != expected:
        _fail('selected lambda carriers, parameter order or direct arange-domain proof differ')


def _antiquotations(text, binders):
    """Only direct parameters with an optional exact binder type are syntax."""
    out, position = [], 0
    while position < len(text):
        start = text.find('$', position)
        if start < 0:
            return ''.join(out) + text[position:]
        out.append(text[position:start])
        if not text.startswith('$(', start):
            _fail('unresolved computation splice')
        end, depth = start + 2, 1
        while end < len(text) and depth:
            depth += (text[end] == '(') - (text[end] == ')')
            end += 1
        if depth:
            _fail('unterminated computation antiquotation')
        value = text[start + 2:end - 1].strip()
        plain = re.fullmatch(material._NAME, value)
        typed = re.fullmatch(r'\(\s*(' + material._NAME + r')\s*:\s*([^()]+)\s*\)', value)
        name = value if plain else typed[1] if typed else None
        if name not in binders or (typed and ' '.join(typed[2].split()) != binders[name]):
            _fail('antiquotation is not a direct parameter with its exact carrier')
        out.append(name)
        position = end
    return ''.join(out)


def _brace_body(text):
    """Translate only block delimiters/keywords, retaining all expressions."""
    output, current, indent, parens, quote, escaped = [], [], 0, [], None, False

    def flush(block=False):
        statement = ''.join(current).strip()
        current.clear()
        if not statement:
            if block:
                _fail('empty block header')
            return
        if block:
            match = re.fullmatch(r'(?:tl\.)?if\s+(.+)', statement, re.S)
            if match:
                statement = 'if ' + match[1] + ':'
            elif statement == 'else':
                statement = 'else:'
            else:
                _fail('unsupported block header')
        output.append('    ' * indent + statement)

    for char in text:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == chr(92):
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
            current.append(char)
        elif char in '([':
            parens.append(char)
            current.append(char)
        elif char in ')]':
            if not parens or parens.pop() != (')' == char and '(' or '['):
                _fail('unbalanced computation delimiter')
            current.append(char)
        elif char == '{':
            if parens:
                _fail('expression braces are unsupported')
            flush(block=True)
            indent += 1
        elif char == '}':
            if parens or indent == 0:
                _fail('unbalanced source block')
            flush()
            indent -= 1
        elif char == '\n' and not parens:
            flush()
        else:
            current.append(char)
    if quote or parens or indent:
        _fail('unterminated source expression or block')
    flush()
    try:
        return ast.parse('\n'.join(output))
    except SyntaxError as error:
        raise ValueError('Source-audit guarded materialization: unsupported Lean computation syntax') from error


def _lean_body(lean_body, profile, binders):
    lines = lean_body.splitlines()
    regions, computation, started = {}, [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('tl.region '):
            if started:
                _fail('region metadata after computation')
            for item in stripped[len('tl.region '):].split(','):
                match = re.fullmatch(r'\s*(' + material._NAME + r')\s*=\s*(tl\.\w+)\s*', item)
                if match is None or match[1] in regions:
                    _fail('ambiguous or unsupported region declaration')
                regions[match[1]] = match[2]
        else:
            started = started or bool(stripped)
            computation.append(line)
    if regions != profile['regions']:
        _fail('region storage differs from the selected comparison profile')
    exact = ('BLOCK_N_arg = $(' + material.ADAPTER + ' BLOCK_N hBlock)')
    matches = [i for i, line in enumerate(computation) if re.match(r'\s*BLOCK_N_arg\s*=', line)]
    if len(matches) != 1 or ' '.join(computation[matches[0]].strip().split()) != exact:
        _fail('requires the recorded unique direct identity materialization')
    computation[matches[0]] = 'BLOCK_N_arg = BLOCK_N'
    normalized = _antiquotations('\n'.join(computation), binders)
    if any(token in normalized for token in ('tl.region', 'tl.source_integer', '$')):
        _fail('unrecorded computation metadata or splice')
    return _brace_body(normalized)


def python_source_with_guarded_materialization(python_text, lean_text, profile):
    if any(source.source_audit_metadata(lean_text, key) is not None for key in
           ('params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes')):
        _fail('profile composition requires separate review')
    if (not isinstance(profile, dict)
            or set(profile) != {'kind', 'source_sha256', 'lean', 'regions', 'binding'}
            or profile['kind'] != KIND
            or profile['source_sha256'] != SOURCE_SHA256
            or profile['source_sha256'] != hashlib.sha256(python_text.encode()).hexdigest()):
        _fail('invalid schema or full original source fingerprint')
    selected = source._validate_source_profile_binding(lean_text, profile['lean'], 'materializations')
    arguments = profile['lean']['args']
    if (profile['lean']['source'] != SELECTOR or source.source_audit_metadata(lean_text, 'target') != TARGET
            or set(arguments) != {*DTYPE_ARGUMENTS, 'version'} or arguments['version'] != 'original'
            or any(not isinstance(arguments[name], str) or arguments[name] not in material.STORAGE_DTYPES
                   for name in DTYPE_ARGUMENTS)):
        _fail('unsupported selected source or storage configuration')
    regions = {name: arguments[arg] for name, arg in FLOAT_REGIONS.items()} | FIXED_REGIONS
    if profile['regions'] != regions:
        _fail('requires complete floating regions, signed int64 seeds and boolean masks')
    binding = {'python': 'BLOCK_N', 'local': 'BLOCK_N_arg', 'kind': 'constexpr_identity',
               'adapter': material.ADAPTER, 'adapter_sha256': material.ADAPTER_SHA256,
               'proof': 'hBlock', 'before': 'cols'}
    if profile['binding'] != binding:
        _fail('unsupported identity binding, proof or insertion point')
    material._check_adapter(profile['binding']['adapter_sha256'])
    lean_body = source.lean_first_triton_body(selected)
    binders = material._binders(selected)
    try:
        kernel = source._explicit_python_audit_kernel(python_text, lean_text)
        tree = ast.parse(python_text)
    except SyntaxError as error:
        raise ValueError('Source-audit guarded materialization: invalid original Python') from error
    _signature(kernel, binders)
    _namespace(tree)
    if any(isinstance(node, ast.Name) and node.id == 'BLOCK_N_arg' for node in ast.walk(kernel)):
        _fail('identity local collides with an original source name')
    _validate_tree(kernel.body, PARAMETERS)
    anchors = [i for i, statement in enumerate(kernel.body)
               if isinstance(statement, ast.Assign) and len(statement.targets) == 1
               and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == 'cols']
    anchor = ast.parse('cols = tl.arange(0, BLOCK_N)').body[0]
    if len(anchors) != 1 or _dump(kernel.body[anchors[0]]) != _dump(anchor):
        _fail('requires the unique original top-level arange assignment')

    class RenameReads(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id == 'BLOCK_N':
                node.id = 'BLOCK_N_arg'
            return node

    changed = copy.deepcopy(kernel)
    changed.body = [RenameReads().visit(copy.deepcopy(statement)) for statement in kernel.body]
    changed.body.insert(anchors[0], ast.parse('BLOCK_N_arg = BLOCK_N').body[0])
    actual = _lean_body(lean_body, profile, binders)
    expected_text = ast.unparse(ast.fix_missing_locations(ast.Module(body=changed.body, type_ignores=[])))
    typed_actual = '\n'.join('tl.region ' + name + ' = ' + dtype for name, dtype in regions.items())
    typed_actual += '\n' + ast.unparse(actual)
    for before, after in zip(source.tl_calls(expected_text, 'store'),
                             source.store_calls_with_explicit_destination_casts(expected_text, typed_actual)):
        expected_text = expected_text.replace(before, after, 1)
    expected = ast.parse(expected_text)
    _validate_tree(expected.body, PARAMETERS)
    _validate_tree(actual.body, PARAMETERS)
    if _dump(expected) != _dump(actual):
        _fail('complete source statement/expression AST differs after checked identity materialization')
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == kernel.name:
            tree.body[index] = changed
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
