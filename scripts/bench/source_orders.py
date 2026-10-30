"""Explicit zero-valued rank-1 order profiles; no general constant propagation."""
# Support both module invocation and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import ast
import hashlib

ALIASES = {'val0': 'base', 'in0_ptr': 'X', 'out0_ptr': 'Y', 'in0_stride0': 'sx',
           'out0_stride0': 'sy', 's0': 'N', 'num_tasks': 'tasks',
           'tiles_per_cta': 'tiles', 'tile_size0': 'B'}
ORDERS = {'in0_stride_order0': 0, 'out0_stride_order0': 0}
KERNEL = 'pow_func_scalar_tensor_kernel_rank_1'
WRAPPER = 'pow_func_scalar_tensor_wrapper_rank_1'
LEAN = {'source': 'powScalarFloating%', 'args': {'output': 'tl.float32', 'single': 0, 'candidate': 0}}
PARAMETERS = ['val0', 'in0_ptr', 'out0_ptr', 'in0_stride0', 'in0_stride_order0',
              'out0_stride0', 'out0_stride_order0', 's0', 'num_tasks',
              'tiles_per_cta', 'tile_size0', 'one_tile_per_cta']
UNSUPPORTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
               ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
               ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
               ast.ExceptHandler, ast.Match, ast.With, ast.AsyncWith)


def source_order_profile(python_text, lean_text):
    """Validate the original wrapper and every original kernel arm.

    The exact source hash pins the reviewed wrapper; the AST checks below also
    reject unsupported constructions in synthetic validation fixtures. Original
    signatures remain unchanged by the eventual comparison representation.
    """
    from scripts.bench import audit_source as source
    declared = source.source_audit_metadata(lean_text, 'orders')
    if declared is None:
        return ()
    profile = source._source_profile_json(declared, 'orders')
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean', 'wrapper'}
            or not isinstance(profile['python'], dict) or set(profile['python']) != set(ORDERS)
            or any(type(v) is not int or v != 0 for v in profile['python'].values())
            or not isinstance(profile['wrapper'], dict)
            or set(profile['wrapper']) != {'name', 'launch', 'source_sha256', 'rank'}):
        raise ValueError('Unsupported Source-audit orders schema; only two explicit integer zeros are admitted')
    source._validate_source_profile_binding(lean_text, profile['lean'], 'orders')
    if profile['lean'] != LEAN:
        raise ValueError('Source-audit orders requires the recorded fp32 grid-stride Lean profile')
    wrapper_profile = profile['wrapper']
    if (wrapper_profile['name'] != WRAPPER or wrapper_profile['launch'] != KERNEL
            or type(wrapper_profile['rank']) is not int or wrapper_profile['rank'] != 1
            or wrapper_profile['source_sha256'] != hashlib.sha256(python_text.encode()).hexdigest()):
        raise ValueError('Source-audit orders wrapper/source fingerprint mismatch')
    try:
        module = ast.parse(python_text)
        kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    except SyntaxError as error:
        raise ValueError('Invalid original Python source for orders profile') from error
    if kernel.name != KERNEL or source._python_parameter_names(kernel) != PARAMETERS:
        raise ValueError('Source-audit orders requires the complete original twelve-parameter signature')
    parameters = {a.arg: a for a in kernel.args.args}
    annotations = {'val0': None, 'in0_ptr': 'tl.tensor', 'out0_ptr': 'tl.tensor',
                   'in0_stride0': 'int', 'out0_stride0': 'int', 's0': 'int', 'num_tasks': 'int',
                   'tiles_per_cta': 'int', 'tile_size0': 'tl.constexpr',
                   'one_tile_per_cta': 'tl.constexpr', **{n: 'tl.constexpr' for n in ORDERS}}
    if (kernel.args.posonlyargs or kernel.args.kwonlyargs or kernel.args.defaults
            or kernel.args.kw_defaults or kernel.returns is not None or getattr(kernel, 'type_params', [])
            or any((ast.unparse(parameters[n].annotation) if parameters[n].annotation else None) != annotations[n]
                   for n in PARAMETERS)):
        raise ValueError('Source-audit orders requires the original parameter kinds, defaults and annotations')
    if any(parameters[n].annotation is None or ast.unparse(parameters[n].annotation) != 'tl.constexpr' for n in ORDERS):
        raise ValueError('Source-audit orders requires exact tl.constexpr parameters')
    if dict(source.source_constexpr_profile(python_text, lean_text)) != {'one_tile_per_cta': False}:
        raise ValueError('Source-audit orders requires an explicit false one_tile_per_cta profile')
    if dict(source.source_parameter_map(python_text, lean_text)) != ALIASES:
        raise ValueError('Source-audit orders requires nine complete, nonoverlapping parameter mappings')
    wrappers = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == WRAPPER]
    if len(wrappers) != 1:
        raise ValueError('Source-audit orders requires one original rank-1 wrapper')
    wrapper = wrappers[0]
    expected_wrapper = ast.parse('def wrapper(val0, in0: Union[torch.Tensor, StridedBuffer], /, *, '
                                 'out0: Union[torch.Tensor, StridedBuffer]): pass').body[0]
    if (ast.dump(wrapper.args) != ast.dump(expected_wrapper.args)
            or wrapper.decorator_list or wrapper.returns is not None or getattr(wrapper, 'type_params', [])):
        raise ValueError('Source-audit orders requires the original wrapper signature')
    wrapper_nodes = [n for statement in wrapper.body for n in ast.walk(statement)]
    forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                 ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar,
                 ast.Match, ast.AsyncWith, ast.Return)
    if any(isinstance(n, forbidden) and not (isinstance(n, ast.Return) and n is wrapper.body[-1])
           for n in wrapper_nodes):
        raise ValueError('Source-audit orders rejects conditional or nested wrapper launch/control')
    launches = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == KERNEL]
    if len(launches) != 1 or len(launches[0].args) != 9:
        raise ValueError('Source-audit orders requires one exact wrapper launch')
    launch = launches[0]
    expected_args = ['val0', 'in0', 'out0', 'in0_strides[0]', 'in0_stride_order[0]',
                     'out0_strides[0]', 'out0_stride_order[0]', 'shape[0]', 'num_tasks']
    if ([k.arg for k in launch.keywords] != ['tiles_per_cta', 'tile_size0', 'one_tile_per_cta', 'num_warps']
            or [ast.unparse(a) for a in launch.args] != expected_args
            or [ast.unparse(k.value) for k in launch.keywords] != ['tiles_per_cta', 'tile_sizes[0]', 'one_tile_per_cta', 'num_warps']
            or ast.unparse(launch.func.slice) != 'grid'):
        raise ValueError('Source-audit orders launch keyword/grid binding mismatch')
    parents = {child: node for node in ast.walk(wrapper) for child in ast.iter_child_nodes(node)}
    expression = parents[launch]
    guard = parents.get(expression)
    if (not isinstance(expression, ast.Expr) or not isinstance(guard, ast.With)
            or guard not in wrapper.body or guard.body != [expression] or len(guard.items) != 1
            or guard.items[0].optional_vars is not None
            or ast.unparse(guard.items[0].context_expr) != 'torch.cuda._DeviceGuard(in0.device.index)'):
        raise ValueError('Source-audit orders requires the original top-level guarded launch')
    for name, position in [('in0_stride_order', 4), ('out0_stride_order', 6)]:
        uses = [n for n in ast.walk(wrapper) if isinstance(n, ast.Name) and n.id == name]
        stores = [n for n in uses if isinstance(n.ctx, ast.Store)]
        loads = [n for n in uses if isinstance(n.ctx, ast.Load)]
        if len(stores) != 1 or len(loads) != 1 or len(uses) != 2:
            raise ValueError('Source-audit orders wrapper order is rebound, deleted or aliased')
        assignment = parents[stores[0]]
        if (not isinstance(assignment, ast.Assign) or assignment not in wrapper.body
                or len(assignment.targets) != 1
                or assignment.targets[0] is not stores[0]
                or assignment.lineno >= launch.lineno
                or ast.dump(assignment.value) != ast.dump(ast.parse('(0,)', mode='eval').body)):
            raise ValueError('Source-audit orders wrapper needs a unique literal (0,) assignment')
        expected = ast.parse(name + '[0]', mode='eval').body
        if (ast.dump(launch.args[position]) != ast.dump(expected)
                or loads[0] not in list(ast.walk(launch.args[position]))):
            raise ValueError('Source-audit orders wrapper order is not the exact launch argument')
        for node in ast.walk(wrapper):
            if isinstance(node, (ast.arg, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(node, 'arg', getattr(node, 'name', None)) == name:
                raise ValueError('Source-audit orders rejects indirect wrapper binding')
            if (isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == name
                    or isinstance(node, ast.MatchMapping) and node.rest == name
                    or isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names
                    or isinstance(node, (ast.Import, ast.ImportFrom)) and
                    any((a.asname or a.name.split('.')[0]) == name or a.name == '*' for a in node.names)):
                raise ValueError('Source-audit orders rejects indirect wrapper binding')
    nodes = [n for stmt in kernel.body for n in ast.walk(stmt)]
    if any(isinstance(n, UNSUPPORTED) for n in nodes):
        raise ValueError('Source-audit orders rejects nested scopes and indirect kernel bindings')
    branches = [n for n in kernel.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'one_tile_per_cta']
    if len(branches) != 1 or not branches[0].body or not branches[0].orelse:
        raise ValueError('Source-audit orders requires both original boolean arms')
    parents = {child: node for node in ast.walk(kernel) for child in ast.iter_child_nodes(node)}
    uses = []
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in ORDERS:
            continue
        if not isinstance(node.ctx, ast.Load):
            raise ValueError('Source-audit orders parameter is rebound or deleted')
        value = parents.get(node); keyword = parents.get(value); call = parents.get(keyword)
        incoming = node.id == 'in0_stride_order0'
        if (not isinstance(value, ast.Tuple) or len(value.elts) != 1
                or not isinstance(keyword, ast.keyword) or not isinstance(call, ast.Call)):
            raise ValueError('Unsupported Source-audit orders AST use')
        fn = ast.unparse(call.func)
        if keyword.arg == 'order':
            valid = fn == 'tl.make_block_ptr' and len(call.args) == 5 and ast.unparse(call.args[0]) == ('in0_ptr' if incoming else 'out0_ptr')
        elif keyword.arg == 'boundary_check':
            valid = fn == ('tl.load' if incoming else 'tl.store') and call.args and ast.unparse(call.args[0]) == ('in0_bptr' if incoming else 'out0_bptr')
        else:
            valid = False
        if not valid:
            raise ValueError('Source-audit orders does not match its pointer/boundary role')
        uses.append(node)
    for body in [branches[0].body, branches[0].orelse]:
        branch_nodes = {id(n) for stmt in body for n in ast.walk(stmt)}
        if any(sum(n.id == name and id(n) in branch_nodes for n in uses) != 2 for name in ORDERS):
            raise ValueError('Source-audit orders must retain both order uses in both original arms')
    if len(uses) != 8:
        raise ValueError('Source-audit orders has unexpected extra parameter uses')
    return tuple(ORDERS.items())


def order_edits(python_text, lean_text):
    """Return original AST spans, after full validation, without changing text."""
    from scripts.bench import audit_source as source
    if not source_order_profile(python_text, lean_text):
        return []
    kernel = source._explicit_python_audit_kernel(python_text, lean_text)
    return [(node, '0') for statement in kernel.body for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id in ORDERS and isinstance(node.ctx, ast.Load)]
