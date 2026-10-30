"""Recorded Pow extern view, deliberately separate from arithmetic helpers.

Only the original fp32 grid-stride profile and its exact single-return helper
are admitted. Validation reads original source and reviewed native evidence;
all rewrites are comparison representations, never executable Python changes.
"""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import ast
import copy
import hashlib
import json
from pathlib import Path
import re

from scripts.bench.source_orders import KERNEL, LEAN, PARAMETERS, UNSUPPORTED, WRAPPER, order_edits

HELPER = 'pow_func_scalar_tensor'
IMPORT_CHAIN = '''try:
    from triton.language.extra.cuda.libdevice import pow as _pow
except ImportError:
    try:
        from triton.language.math import pow as _pow
    except ImportError:
        from triton.language.libdevice import pow as _pow
'''


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _json_equal(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _excerpt(text, node):
    first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return ''.join(text.splitlines(keepends=True)[first - 1:node.end_lineno])


def _registry():
    from scripts.bench import audit_source as source
    path = Path(__file__).resolve().parent / 'tritonbench_g/source_extern_profiles.json'
    try:
        value = source._source_profile_json(path.read_text(), 'extern evidence')
    except OSError as error:
        raise ValueError('Missing reviewed extern evidence registry') from error
    if (not isinstance(value, dict) or set(value) != {'schema', 'profiles'}
            or type(value['schema']) is not int or value['schema'] != 1
            or not isinstance(value['profiles'], dict)):
        raise ValueError('Unsupported extern evidence registry schema')
    return value['profiles']


def _bindings(module, kernel, helper, chain):
    """No alternate binding or namespace path may change the reviewed helper."""
    protected = {'triton', 'tl', '_pow', HELPER, KERNEL, WRAPPER, 'ImportError'}
    wrappers = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == WRAPPER]
    if len(wrappers) != 1:
        raise ValueError('Extern requires one unambiguous original wrapper')
    definitions = {HELPER: helper, KERNEL: kernel, WRAPPER: wrappers[0]}
    found = {name: [] for name in protected}
    imports = {id(n) for n in ast.walk(chain) if isinstance(n, ast.ImportFrom)}
    helper_nodes = {id(n) for n in ast.walk(helper)}
    parents = {child: node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
    dynamic = {'exec', 'eval', 'globals', 'locals', 'vars', 'setattr', 'delattr',
               '__dict__', '__getattribute__', '__setattr__', '__delattr__',
               '__import__', '__builtins__', '__globals__', '__closure__', 'modules', 'import_module'}
    for node in ast.walk(module):
        if (isinstance(node, ast.Name) and node.id in dynamic
                or isinstance(node, ast.Attribute) and node.attr in dynamic):
            raise ValueError('Extern profile rejects dynamic namespace access')
        if isinstance(node, (ast.Global, ast.Nonlocal)) and protected.intersection(node.names):
            raise ValueError('Extern profile rejects indirect bindings')
        if isinstance(node, ast.Name) and node.id in protected:
            parent = parents.get(node)
            if not isinstance(node.ctx, ast.Load):
                raise ValueError('Extern profile rejects rebinding/deletion of protected names')
            if node.id in {'triton', 'tl'}:
                valid = isinstance(parent, ast.Attribute) and parent.value is node
            elif node.id == KERNEL:
                valid = isinstance(parent, ast.Subscript) and parent.value is node
            elif node.id == 'ImportError':
                valid = isinstance(parent, ast.ExceptHandler) and parent.type is node and parent in list(ast.walk(chain))
            else:
                valid = isinstance(parent, ast.Call) and parent.func is node
                if node.id == '_pow': valid &= id(node) in helper_nodes
            if not valid:
                raise ValueError('Extern profile rejects protected-name aliases')
        if isinstance(node, ast.Attribute):
            path = ast.unparse(node)
            base = ast.unparse(node.value)
            if not isinstance(node.ctx, ast.Load) and any(base == name or base.startswith(name + '.') for name in protected):
                raise ValueError('Extern profile rejects protected namespace mutation')
            if path.startswith(('tl.', 'triton.')):
                parent = parents.get(node)
                valid = isinstance(parent, ast.Call) and parent.func is node
                valid |= path in {'tl.tensor', 'tl.constexpr', 'tl.float32'}
                valid |= path == 'triton.jit' and isinstance(parent, ast.FunctionDef) and node in parent.decorator_list
                if not valid:
                    raise ValueError('Extern profile rejects protected attribute aliases')
        if isinstance(node, ast.arg) and node.arg in protected:
            raise ValueError('Extern profile rejects shadowed parameters')
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in protected:
            if definitions.get(node.name) is not node:
                raise ValueError('Extern profile rejects shadowed definitions')
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name in protected:
            raise ValueError('Extern profile rejects indirect name binding')
        if isinstance(node, ast.MatchMapping) and node.rest in protected:
            raise ValueError('Extern profile rejects pattern binding')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                origin = alias.name if isinstance(node, ast.Import) else node.module or ''
                name = alias.asname or alias.name.split('.')[0]
                if alias.name == '*' or origin.split('.')[0] in {'sys', 'builtins', 'importlib'}:
                    raise ValueError('Extern profile rejects dynamic namespace imports')
                if id(node) in imports:
                    found['_pow'].append(node)
                    continue
                canonical = (isinstance(node, ast.Import) and alias.name == 'triton' and alias.asname in (None, 'triton')
                             or isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == 'triton'
                             and alias.name == 'language' and alias.asname == 'tl')
                if (origin == 'triton' or origin.startswith('triton.')) and not canonical:
                    raise ValueError('Extern profile rejects noncanonical Triton imports')
                if name in protected:
                    if not canonical or node not in module.body:
                        raise ValueError('Extern profile rejects additional protected imports')
                    found[name].append(node)
    if len(found['triton']) != 1 or len(found['tl']) != 1 or len(found['_pow']) != 3:
        raise ValueError('Extern profile canonical import bindings are incomplete')


def _native_record(profile, python_text, kernel, helper, chain):
    record = _registry().get(profile['native']['profile'])
    fields = {'jit', 'helper', 'source_sha256', 'kernel_source_sha256', 'helper_source_sha256',
              'import_chain_sha256', 'lean', 'compiler', 'selected_import', 'runtime_base_profile',
              'core_sha256', 'cuda_libdevice_sha256', 'native', 'ttir', 'runtime_base_specialization',
              'import_evidence', 'generated_module_sha256'}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError('Unknown or malformed reviewed extern evidence')
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != profile['native']['sha256']:
        raise ValueError('Extern native evidence digest mismatch')
    if (record['jit'] != kernel.name or record['helper'] != helper.name
            or not _json_equal(record['lean'], profile['lean'])
            or record['source_sha256'] != _sha(python_text)
            or record['kernel_source_sha256'] != _sha(_excerpt(python_text, kernel))
            or record['helper_source_sha256'] != _sha(_excerpt(python_text, helper))
            or record['import_chain_sha256'] != _sha(_excerpt(python_text, chain))):
        raise ValueError('Extern original source/helper/import fingerprint mismatch')
    generated = ('import triton\nfrom triton import language as tl\n' + _excerpt(python_text, chain) + '\n'
                 + _excerpt(python_text, helper) + '\n' + _excerpt(python_text, kernel) + '\n')
    if record['generated_module_sha256'] != _sha(generated):
        raise ValueError('Extern native module extraction fingerprint mismatch')
    expected_compiler = {'triton_version': '3.7.1', 'target': 'cuda sm90 warp32',
                         'options': {'num_warps': 4, 'num_stages': 3}, 'kernel_launched': False}
    if not _json_equal(record['compiler'], expected_compiler):
        raise ValueError('Unsupported extern compiler configuration')
    if (record['core_sha256'] != '373eebe68a4c1f6959c81ae6c382b65c5c2893d53274a205f5ff9bbb100fb31b'
            or record['cuda_libdevice_sha256'] != '27b2a5d1e8db008bacefe6019f63922bbd65926de90bb1b527ee597477d2f365'):
        raise ValueError('Unreviewed extern compiler/library snapshots')
    if not _json_equal(record['selected_import'], {'branch': 'primary', 'module': 'triton.language.extra.cuda.libdevice', 'symbol': 'pow', 'alias': '_pow'}):
        raise ValueError('Extern fallback imports need separate reviewed evidence')
    if record['runtime_base_profile'] != {'object': 'float', 'abi': 'fp32', 'cast_origin': 'existing helper source; no ABI materialization'}:
        raise ValueError('Extern profile does not admit this runtime base type')
    if record['runtime_base_specialization'] != {'2.0': ['fp32', None], '0.5': ['fp32', None], '7': ['i32', None]}:
        raise ValueError('Extern scalar specialization evidence mismatch')
    evidence = record['import_evidence']
    if (not isinstance(evidence, list) or len(evidence) != 3 or any(not isinstance(row, dict) for row in evidence)
            or not _json_equal(evidence[0], {'module': 'triton.language.extra.cuda.libdevice', 'status': 'accepted',
                               'resolved_module': 'triton.language.extra.cuda.libdevice', 'same_as_selected': True})
            or [row.get('module') for row in evidence[1:]] != ['triton.language.math', 'triton.language.libdevice']
            or any(set(row) != {'module', 'status', 'message'}
                   or row.get('status') != 'ImportError'
                   or not isinstance(row.get('message'), str) or not row['message']
                   for row in evidence[1:])):
        raise ValueError('Extern conditional import evidence mismatch')
    native = record['native']
    signature = {'val0': 'fp32', 'in0_ptr': '*fp32', 'out0_ptr': '*fp32', 'in0_stride0': 'i32',
                 'out0_stride0': 'i32', 's0': 'i32', 'num_tasks': 'i32', 'tiles_per_cta': 'i32'}
    constants = {'in0_stride_order0': 0, 'out0_stride_order0': 0, 'tile_size0': 32, 'one_tile_per_cta': False}
    native_fields = {'profile', 'signature', 'constexprs', 'kernel_launched', 'status',
                     'ttir_sha256', 'llir_sha256', 'ptx_sha256', 'ttir_signature',
                     'extern_lines', 'base_conversion_lines'}
    ir_types = {'fp32': 'f32', '*fp32': '!tt.ptr<f32>', 'i32': 'i32'}
    expected_signature = ('tt.func public @' + KERNEL + '(' + ', '.join(
        '%' + name + ': ' + ir_types[typ] + ' loc("' + name + '"(#loc))'
        for name, typ in signature.items()) + ') attributes {noinline = false} {')
    if (not isinstance(native, dict) or set(native) != native_fields
            or native.get('profile') != 'base_fp32_exp_fp32_single_0'
            or native.get('status') != 'compiled' or native.get('base_conversion_lines') != []
            or any(not isinstance(native.get(key), str) or not re.fullmatch('[0-9a-f]{64}', native[key])
                   for key in ['ttir_sha256', 'llir_sha256', 'ptx_sha256'])
            or native.get('kernel_launched') is not False
            or not _json_equal(native.get('signature'), signature)
            or not _json_equal(native.get('constexprs'), constants)
            or not isinstance(record['ttir'], str) or native.get('ttir_sha256') != _sha(record['ttir'])
            or native.get('ttir_signature') != expected_signature
            or native['ttir_signature'] not in record['ttir']
            or not native['ttir_signature'].startswith('tt.func public @' + KERNEL + '(')
            or '%val0: f32' not in native['ttir_signature']
            or '%in0_ptr: !tt.ptr<f32>' not in native['ttir_signature']
            or '%out0_ptr: !tt.ptr<f32>' not in native['ttir_signature']):
        raise ValueError('Extern native signature/configuration/TTIR mismatch')
    externs = [line.strip() for line in record['ttir'].splitlines() if 'tt.extern_elementwise' in line]
    if (native.get('extern_lines') != externs or len(externs) != 1
            or 'symbol = "__nv_powf"' not in externs[0]
            or '(tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>' not in externs[0]):
        raise ValueError('Extern native powf dispatch mismatch')
    return record


def extern_edits(python_text, lean_text):
    from scripts.bench import audit_source as source
    declared = source.source_audit_metadata(lean_text, 'extern')
    if declared is None:
        return []
    profile = source._source_profile_json(declared, 'extern')
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean', 'native'}
            or profile['python'] != {'helper': HELPER, 'alias': '_pow'}
            or not isinstance(profile['native'], dict) or set(profile['native']) != {'profile', 'sha256'}
            or not isinstance(profile['native']['profile'], str)
            or not isinstance(profile['native']['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', profile['native']['sha256'])):
        raise ValueError('Unsupported Source-audit extern schema')
    source._validate_source_profile_binding(lean_text, profile['lean'], 'extern')
    if not _json_equal(profile['lean'], LEAN):
        raise ValueError('Extern requires the exact recorded fp32 grid-stride profile')
    if any(source.source_audit_metadata(lean_text, name) is not None for name in ['helpers', 'runtime']):
        raise ValueError('Extern helper cannot overlap arithmetic helpers or runtime ABI materialization')
    if source.source_audit_metadata(lean_text, 'orders') is None:
        raise ValueError('Extern requires explicit rank-1 order specialization')
    try:
        module = ast.parse(python_text)
        selected = source._explicit_python_audit_kernel(python_text, lean_text)
        kernels = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == selected.name]
        if len(kernels) != 1:
            raise ValueError('Extern requires one unambiguous original kernel')
        kernel = kernels[0]
    except SyntaxError as error:
        raise ValueError('Invalid original Python source for extern profile') from error
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == HELPER]
    if len(functions) != 1 or kernel.name != KERNEL or source._python_parameter_names(kernel) != PARAMETERS:
        raise ValueError('Extern requires unique original helper and full kernel signature')
    helper = functions[0]
    expected_decorator = ast.parse("triton.jit(do_not_specialize=['val0'])", mode='eval').body
    if len(kernel.decorator_list) != 1 or ast.dump(kernel.decorator_list[0]) != ast.dump(expected_decorator):
        raise ValueError('Extern kernel must retain do_not_specialize=[val0]')
    args = helper.args
    if (len(args.args) != 2 or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or args.defaults or args.kw_defaults or any(a.annotation is not None for a in args.args)
            or helper.returns is not None or getattr(helper, 'type_params', []) or len(helper.decorator_list) != 1
            or ast.unparse(helper.decorator_list[0]) != 'triton.jit'):
        raise ValueError('Extern requires the original two-argument single-return JIT helper')
    formals = [a.arg for a in args.args]
    if len(set(formals)) != 2 or set(formals) & {'tl', 'triton', '_pow', HELPER, KERNEL}:
        raise ValueError('Extern helper formals capture protected names')
    body = helper.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    expected = ast.parse('_pow(' + formals[0] + '.to(tl.float32), ' + formals[1] + ')', mode='eval').body
    if len(body) != 1 or not isinstance(body[0], ast.Return) or ast.dump(body[0].value) != ast.dump(expected):
        raise ValueError('Extern helper must retain exactly its original fp32 source cast and pow argument order')
    expected_chain = ast.parse(IMPORT_CHAIN).body[0]
    chains = [n for n in module.body if isinstance(n, ast.Try) and ast.dump(n) == ast.dump(expected_chain)]
    if len(chains) != 1:
        raise ValueError('Extern requires the complete original conditional import chain')
    chain = chains[0]
    _bindings(module, kernel, helper, chain)
    _native_record(profile, python_text, kernel, helper, chain)
    nodes = [n for statement in kernel.body for n in ast.walk(statement)]
    if any(isinstance(n, UNSUPPORTED) for n in nodes):
        raise ValueError('Extern rejects nested scopes and indirect kernel bindings')
    if any(isinstance(n, ast.Name) and n.id == 'val0' and not isinstance(n.ctx, ast.Load) for n in nodes):
        raise ValueError('Extern base is rebound or deleted')
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    sites = []
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id != HELPER:
            continue
        call = parents.get(node); assignment = parents.get(call)
        if (not isinstance(call, ast.Call) or call.func is not node or call.keywords
                or [ast.unparse(a) for a in call.args] != ['val0', 'in0']
                or not all(isinstance(a, ast.Name) for a in call.args)
                or not isinstance(assignment, ast.Assign) or assignment.value is not call
                or len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name)
                or assignment.targets[0].id != 'out0'):
            raise ValueError('Extern requires the exact original simple RHS helper call')
        sites.append(call)
    branches = [n for n in kernel.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'one_tile_per_cta']
    if len(branches) != 1 or len(sites) != 2:
        raise ValueError('Extern requires both original helper call sites')
    for arm in [branches[0].body, branches[0].orelse]:
        members = {id(n) for stmt in arm for n in ast.walk(stmt)}
        if sum(id(n) in members for n in sites) != 1:
            raise ValueError('Extern requires one helper call in each original arm')
    return [(node, 'tl.extra.cuda.libdevice.pow(val0.to(tl.float32), in0)') for node in sites]


def _apply_edits(text, edits):
    offsets, total = [], 0
    for line in text.encode().splitlines(keepends=True):
        offsets.append(total); total += len(line)
    spans = sorted((offsets[n.lineno - 1] + n.col_offset, offsets[n.end_lineno - 1] + n.end_col_offset,
                    value.encode()) for n, value in edits)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError('Overlapping extern/order comparison edits')
    data = text.encode()
    for start, end, value in reversed(spans):
        data = data[:start] + value + data[end:]
    return data.decode()


def _lean_as_python(body):
    """The narrow selected Pow DSL grammar: typed splices, calls and one loop."""
    from scripts.bench import audit_source as source
    if source.declared_region_dtypes(body) != {'X': {'tl.float32'}, 'Y': {'tl.float32'}}:
        raise ValueError('Extern selected Lean regions differ from the native profile')
    text = source.strip_region_declarations(source.resolve_source_dtype_attributes(body, body))
    text = re.sub(r'\$\(\((base)\s*:\s*ℝ\)\)', r'\1', text)
    text = re.sub(r'\$\((N|B|sx|sy|tiles|tasks|0)\)', r'\1', text)
    text = re.sub(r'\(\[0\]\s*:\s*List Nat\)', '(0,)', text)
    if '$' in text:
        raise ValueError('Unsupported Lean splice in extern comparison')
    lines, pending, depth, indent, loops = [], '', 0, 0, 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == '}' and not pending:
            if indent != 1:
                raise ValueError('Unsupported Lean extern block structure')
            indent = 0
            continue
        pending += (' ' if pending else '') + stripped
        # No string expression is admitted by this selected Pow grammar.
        if any(char in stripped for char in ['"', "'"]):
            raise ValueError('Unsupported quoted expression in Lean extern body')
        depth += stripped.count('(') + stripped.count('[') - stripped.count(')') - stripped.count(']')
        if depth < 0:
            raise ValueError('Unbalanced Lean extern expression')
        if depth:
            continue
        if pending.endswith('{'):
            if indent or pending != 'for j in range(0, tiles) {':
                raise ValueError('Unsupported Lean extern loop header')
            lines.append('for j in range(0, tiles):'); indent = 1; loops += 1
        else:
            if '{' in pending or '}' in pending:
                raise ValueError('Unsupported Lean extern statement')
            lines.append('    ' * indent + pending)
        pending = ''
    if pending or depth or indent or loops != 1:
        raise ValueError('Incomplete Lean extern body')
    try:
        return ast.parse('\n'.join(lines))
    except SyntaxError as error:
        raise ValueError('Unsupported Lean extern expression syntax') from error


def _canonical(module, *, lean):
    """Normalize only declared block-pointer positional/keyword surface syntax."""
    class Normalize(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if ast.unparse(node.func) != 'tl.make_block_ptr':
                return node
            names = ['base', 'shape', 'strides', 'offsets', 'block_shape', 'order']
            if lean:
                if node.args or [k.arg for k in node.keywords] != names:
                    raise ValueError('Unsupported Lean block-pointer signature')
                values = [k.value for k in node.keywords]
                # Lean singleton grouping is its rank-1 vector DSL notation.
                values = [v if i == 0 or isinstance(v, ast.Tuple) else ast.Tuple(elts=[v], ctx=ast.Load()) for i, v in enumerate(values)]
            else:
                if len(node.args) != 5 or [k.arg for k in node.keywords] != ['order']:
                    raise ValueError('Unsupported Python block-pointer signature')
                values = node.args + [node.keywords[0].value]
            if any(not isinstance(v, ast.Tuple) or len(v.elts) != 1 for v in values[1:]):
                raise ValueError('Extern block pointers must retain all rank-1 vector fields')
            node.args = values; node.keywords = []
            return node
    return Normalize().visit(copy.deepcopy(module))


def _match_selected_body(expanded, lean_text):
    from scripts.bench import audit_source as source
    mapped = source.python_source_with_parameter_map(expanded, lean_text)
    original = source._explicit_python_audit_kernel(mapped, lean_text)
    branches = [n for n in original.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'one_tile_per_cta']
    if (len(branches) != 1 or len(branches[0].orelse) != 2
            or not isinstance(branches[0].orelse[1], ast.For)):
        raise ValueError('Extern requires the reviewed original two-arm loop structure')
    # Both original arms share the exact tail, with their original tile-id
    # assignments. Check the unselected arm too; this does not select or certify
    # its separate native profile, and does not erase casts in either arm.
    grid = branches[0].orelse[1]
    single = ast.Module(body=branches[0].body, type_ignores=[])
    expected_single = ast.Module(body=ast.parse('tile_id = pid').body + grid.body[1:], type_ignores=[])
    if ast.dump(single) != ast.dump(expected_single):
        raise ValueError('Extern original single-tile arm differs from the reviewed common tail')
    mapped = source.python_source_with_constexpr_profile(mapped, lean_text)
    body = source.python_named_jit_body(mapped, KERNEL)
    lean_body = source.lean_first_triton_body(lean_text)
    body = source.resolve_source_dtype_attributes(body, lean_body)
    import textwrap
    try:
        python = ast.parse(textwrap.dedent(body))
    except SyntaxError as error:
        raise ValueError('Unsupported selected Python extern body') from error
    lean = _lean_as_python(lean_body)
    if ast.dump(_canonical(python, lean=False)) != ast.dump(_canonical(lean, lean=True)):
        raise ValueError('Extern complete selected source/RHS AST mismatch')


def python_source_with_extern_orders(python_text, lean_text):
    """Validate original profiles, then apply disjoint helper/order Name spans."""
    from scripts.bench import audit_source as source
    orders = order_edits(python_text, lean_text)
    extern = extern_edits(python_text, lean_text)
    if not orders and not extern:
        return python_text
    result = _apply_edits(python_text, orders + extern)
    if extern:
        _match_selected_body(result, lean_text)
    return result
