"""Strict complete-source comparison for the direct uniform Philox kernel.

The three inserted assignments are checked identities. Every original kernel
statement remains, including the signed parameter rebindings, tuple, unsigned
update, helper calls and cache hints. The original module's complete try/except
helper and host code are pinned and retained. This gate compares selected source
ASTs; it does not certify native admission, helper equivalence or proof replay.
"""
import ast
import copy
import hashlib
import re

from bench import audit_source as source
from bench import source_materializations as material

KIND = 'direct_uniform_identity_v1'
TARGET = 'uniform_kernel'
SELECTOR = 'uniformSource%'
SOURCE_SHA256 = 'd2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1'
PARAMETERS = ['out_ptr', 'N', 'philox_seed', 'philox_offset', 'from_', 'to', 'BLOCK']
PARAMETER_MAP = {'to': 'upper'}
HELPER = {'python': 'uint_to_uniform_float', 'builtin': 'tl.uint_to_uniform_float',
          'binding': 'original_try_attributeerror'}
LOCALS = ['philox_seed_arg', 'philox_offset_arg', 'BLOCK_arg']
DECORATORS = '''
@triton.heuristics({"BLOCK": heur_block, "num_warps": heur_num_warps})
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def target(): pass
'''


def _fail(message):
    raise ValueError('Source-audit direct uniform materialization: ' + message)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def bindings():
    return [
        {'python': name, 'local': name + '_arg', 'kind': 'runtime_integer',
         'dtype': 'tl.int64', 'initial_cast': name}
        for name in ('philox_seed', 'philox_offset')
    ] + [{'python': 'BLOCK', 'local': 'BLOCK_arg', 'kind': 'constexpr_identity',
          'adapter': material.ADAPTER, 'adapter_sha256': material.ADAPTER_SHA256,
          'proof': {'function': 'arange_domain', 'binder': 'hBlock'}}]


def _signature(kernel, binders):
    if (source._python_parameter_names(kernel) != PARAMETERS or kernel.args.defaults
            or kernel.args.kw_defaults or kernel.args.posonlyargs or kernel.args.kwonlyargs
            or kernel.returns is not None or getattr(kernel, 'type_params', [])):
        _fail('requires the complete original kernel signature')
    for argument in kernel.args.args:
        expected = 'tl.constexpr' if argument.arg == 'BLOCK' else None
        if (ast.unparse(argument.annotation) if argument.annotation else None) != expected:
            _fail('original parameter annotation differs: ' + argument.arg)
    expected = ast.parse(DECORATORS).body[0].decorator_list
    if [_dump(x) for x in kernel.decorator_list] != [_dump(x) for x in expected]:
        _fail('original heuristics or do_not_specialize decorators differ')
    expected = {'out_ptr': 'RegionName', 'N': 'Nat', 'philox_seed': 'Int',
                'philox_offset': 'Int', 'from_': 'ℝ', 'upper': 'ℝ', 'BLOCK': 'Nat',
                'hBlock': 'SourceDomain.LegalBlock BLOCK'}
    if list(binders) != list(expected) or binders != expected:
        _fail('selected lambda parameter order, carriers or domain proof differ')


def _module_binding(tree):
    imports = [alias for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names if alias.name == 'triton.language' and alias.asname == 'tl']
    if len(imports) != 1:
        _fail('requires one original direct Triton language import')
    tries = [node for node in tree.body if isinstance(node, ast.Try)]
    expected = ast.parse('uint_to_uniform_float = tl.uint_to_uniform_float').body[0]
    if (len(tries) != 1 or len(tries[0].body) != 1
            or _dump(tries[0].body[0]) != _dump(expected)
            or tries[0].orelse or tries[0].finalbody or len(tries[0].handlers) != 1):
        _fail('requires the original explicit builtin/helper fallback binding')
    handler = tries[0].handlers[0]
    if (not isinstance(handler.type, ast.Name) or handler.type.id != 'AttributeError'
            or handler.name is not None or len(handler.body) != 1
            or not isinstance(handler.body[0], ast.FunctionDef)
            or handler.body[0].name != HELPER['python']):
        _fail('requires the complete original AttributeError fallback helper')
    # The full-file hard pin above also covers both fallback dtype branches,
    # constants, bitcasts and final fold. No helper branch is selected or erased.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in {'tl', 'triton'}:
                _fail('original source rebinds the language namespace')
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {'tl', 'triton'}:
                _fail('original source mutates the language namespace')


def _expression(node, defined):
    if isinstance(node, ast.Name):
        if node.id not in defined:
            _fail('unknown or forward expression name: ' + node.id)
    elif isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            _fail('unsupported computation literal')
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                                           ast.BitAnd, ast.RShift)):
        _expression(node.left, defined)
        _expression(node.right, defined)
    elif (isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1
          and isinstance(node.ops[0], ast.Lt)):
        _expression(node.left, defined)
        _expression(node.comparators[0], defined)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == HELPER['python']:
            if len(node.args) != 1 or node.keywords:
                _fail('helper requires exactly one original operand')
            _expression(node.args[0], defined)
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == 'to'
              and len(node.args) == 1 and not node.keywords
              and isinstance(node.args[0], ast.Attribute)
              and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == 'tl'
              and 'tl.' + node.args[0].attr in material.STORAGE_DTYPES | {'tl.int64', 'tl.uint32'}):
            _expression(node.func.value, defined)
        elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.value.id == 'tl' and node.func.attr in {'program_id', 'arange', 'philox', 'store'}):
            if (any(k.arg is None for k in node.keywords)
                    or len({k.arg for k in node.keywords}) != len(node.keywords)):
                _fail('expanded or duplicate call keyword')
            for argument in node.args:
                _expression(argument, defined)
            for keyword in node.keywords:
                if keyword.arg == 'eviction_policy' and node.func.attr == 'store':
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value != 'evict_first':
                        _fail('store cache hint differs from the original literal')
                elif keyword.arg == 'mask' and node.func.attr == 'store':
                    _expression(keyword.value, defined)
                else:
                    _fail('unexpected source call keyword')
        else:
            _fail('unsupported call or unbound helper')
    else:
        _fail('unsupported computation expression: ' + type(node).__name__)


def _statements(body, parameters):
    defined = set(parameters)
    protected = {'out_ptr', 'N', 'from_', 'to', 'upper', 'BLOCK', 'tl', 'triton', HELPER['python']}
    for statement in body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                names = [target.id]
            elif (isinstance(target, ast.Tuple) and len(target.elts) == 4
                  and all(isinstance(x, ast.Name) for x in target.elts)
                  and isinstance(statement.value, ast.Call)
                  and isinstance(statement.value.func, ast.Attribute)
                  and isinstance(statement.value.func.value, ast.Name)
                  and statement.value.func.value.id == 'tl' and statement.value.func.attr == 'philox'
                  and len(statement.value.args) == 5 and not statement.value.keywords):
                names = [x.id for x in target.elts]
            else:
                _fail('only the original four-result Philox tuple is admitted')
            if set(names) & protected:
                _fail('rebound protected source name')
            _expression(statement.value, defined)
            defined.update(names)
        elif (isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name)
              and isinstance(statement.op, ast.Add)):
            if statement.target.id not in defined or statement.target.id in protected:
                _fail('invalid unsigned augmented-assignment target')
            _expression(statement.value, defined)
        elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
              and isinstance(statement.value.func, ast.Attribute)
              and isinstance(statement.value.func.value, ast.Name)
              and statement.value.func.value.id == 'tl' and statement.value.func.attr == 'store'):
            _expression(statement.value, defined)
        else:
            _fail('unsupported source statement: ' + type(statement).__name__)


def _lean_body(body, profile, binders):
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    prefix = [
        r'tl\.region\s+out_ptr\s*=\s*' + re.escape(profile['regions']['out_ptr']),
        r'tl\.source_helper\s+uint_to_uniform_float\s*=\s*tl\.uint_to_uniform_float',
    ]
    for name in ('philox_seed', 'philox_offset'):
        prefix += [name + r'_arg\s*=\s*\$\(\(\s*' + name + r'\s*:\s*Int\s*\)\)',
                   r'tl\.source_integer\s+' + name + r'_arg\s*=\s*tl\.int64']
    prefix += [r'BLOCK_arg\s*=\s*\$\(\s*' + re.escape(material.ADAPTER) +
               r'\s+BLOCK\s+\(\s*arange_domain\s+hBlock\s*\)\s*\)']
    if len(lines) <= len(prefix):
        _fail('missing complete metadata/materialization prefix')
    for expected, actual in zip(prefix, lines):
        if re.fullmatch(expected, actual) is None:
            _fail('source metadata or identity prefix differs from the exact recorded binding')

    def external(match):
        if match[1] not in binders:
            _fail('antiquotation is not a direct lambda parameter')
        return match[1]

    def typed_external(match):
        if match[1] not in binders or ' '.join(match[2].split()) != binders[match[1]]:
            _fail('typed antiquotation differs from the direct parameter carrier')
        return match[1]

    rest = re.sub(r'\$\(\(\s*(' + material._NAME + r')\s*:\s*([^()]+)\s*\)\)',
                  typed_external, '\n'.join(lines[len(prefix):]))
    rest = re.sub(r'\$\(\s*(' + material._NAME + r')\s*\)', external, rest)
    if '$' in rest or any(token in rest for token in ('tl.region', 'tl.source_integer', 'tl.source_helper')):
        _fail('unrecorded computation metadata or splice')
    text = 'philox_seed_arg = philox_seed\nphilox_offset_arg = philox_offset\nBLOCK_arg = BLOCK\n' + rest
    try:
        return ast.parse(text)
    except SyntaxError as error:
        raise ValueError('Source-audit direct uniform materialization: unsupported Lean computation syntax') from error


def python_source_with_uniform_materialization(python_text, lean_text, profile):
    if any(source.source_audit_metadata(lean_text, key) is not None for key in
           ('constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes')):
        _fail('unreviewed profile composition')
    if (not isinstance(profile, dict)
            or set(profile) != {'kind', 'source_sha256', 'lean', 'regions', 'bindings', 'helper', 'params'}
            or profile['kind'] != KIND or profile['source_sha256'] != SOURCE_SHA256
            or hashlib.sha256(python_text.encode()).hexdigest() != SOURCE_SHA256):
        _fail('invalid schema or full original source fingerprint')
    if (profile['bindings'] != bindings() or profile['helper'] != HELPER
            or profile['params'] != PARAMETER_MAP
            or dict(source.source_parameter_map(python_text, lean_text)) != PARAMETER_MAP):
        _fail('requires exact integer/identity/helper bindings and to=upper alpha correspondence')
    material._check_adapter(profile['bindings'][2]['adapter_sha256'])
    selected = source._validate_source_profile_binding(lean_text, profile['lean'], 'materializations')
    args = profile['lean']['args']
    if (source.source_audit_metadata(lean_text, 'target') != TARGET
            or profile['lean']['source'] != SELECTOR or set(args) != {'storage', 'version'}
            or args['version'] != 'original' or not isinstance(args['storage'], str)
            or args['storage'] not in material.STORAGE_DTYPES
            or profile['regions'] != {'out_ptr': args['storage']}):
        _fail('unsupported selected source or output storage')
    tree = ast.parse(python_text)
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    binders = material._binders(selected)
    _signature(kernel, binders)
    _module_binding(tree)
    if any(isinstance(node, ast.Name) and node.id in LOCALS for node in ast.walk(kernel)):
        _fail('materialized local collides with an original name')
    _statements(kernel.body, PARAMETERS)
    for index, name in enumerate(('philox_seed', 'philox_offset')):
        anchor = ast.parse(name + ' = ' + name + '.to(tl.int64)').body[0]
        if _dump(kernel.body[index]) != _dump(anchor):
            _fail('original signed parameter rebinding differs')

    class RenameBlockReads(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id == 'BLOCK':
                node.id = 'BLOCK_arg'
            return node

    changed = copy.deepcopy(kernel)
    changed.body = [RenameBlockReads().visit(copy.deepcopy(s)) for s in kernel.body]
    for index, name in enumerate(('philox_seed', 'philox_offset')):
        changed.body[index].value.func.value.id = name + '_arg'
    changed.body = ast.parse('philox_seed_arg = philox_seed\nphilox_offset_arg = philox_offset\nBLOCK_arg = BLOCK').body + changed.body
    actual = _lean_body(source.lean_first_triton_body(selected), profile, binders)

    class AlphaMap(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = PARAMETER_MAP.get(node.id, node.id)
            return node

    expected = AlphaMap().visit(ast.Module(body=copy.deepcopy(changed.body), type_ignores=[]))
    expected_text = ast.unparse(ast.fix_missing_locations(expected))
    typed_actual = 'tl.region out_ptr = ' + args['storage'] + '\n' + ast.unparse(actual)
    for before, after in zip(source.tl_calls(expected_text, 'store'),
                             source.store_calls_with_explicit_destination_casts(expected_text, typed_actual)):
        expected_text = expected_text.replace(before, after, 1)
    expected = ast.parse(expected_text)
    mapped_parameters = [PARAMETER_MAP.get(name, name) for name in PARAMETERS]
    _statements(expected.body, mapped_parameters)
    _statements(actual.body, mapped_parameters)
    if _dump(expected) != _dump(actual):
        _fail('complete source statement/expression AST differs after checked identity materialization')
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == TARGET:
            tree.body[index] = changed
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
