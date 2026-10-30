"""Shared source extraction for the TritonBench-G structural audit.

These helpers preserve the audit's source-text conventions. They do not discover
proof obligations or certify translations; Lean inventories and the official
comparator own proof validation. Rule-specific kernel and call selection is
explicit so changes cannot silently alter the scope of another audit rule.
"""
import ast
import hashlib
import io
import json
import keyword
import re
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def source_audit_metadata(text: str, key: str) -> str | None:
    """Read one explicit source-view field; duplicate declarations are errors.

    Fields belong to the entry's preamble (a block or line comment):
      Source-audit target: python_jit_name
      Source-audit Lean: lean_definition_or_macro%
      Source-audit args: version=original, output=tl.float32, single=1
      Source-audit params: python_parameter=lean_binder, BLOCK_SIZE=B

    Args describe a textual macro specialization, not an exemption or proof of
    source coverage. Unknown expressions remain unresolved and fail the audit.
    Parameter renaming requires an explicit, validated params map. Python
    constexpr specialization is never inferred from these fields. A separate
    constexpr JSON field explicitly binds boolean Python guards to an exact
    Lean selector and complete macro-argument configuration.
    """
    values = re.findall(r'^[ \t]*(?:--[ \t]*)?Source-audit ' + re.escape(key) + r':[ \t]*([^\n]*)',
                        lean_first_preamble(text), re.M)
    if len(values) > 1:
        raise ValueError(f'Duplicate Source-audit {key} declarations')
    if values and not values[0].strip():
        raise ValueError(f'Empty Source-audit {key} declaration')
    return values[0].strip() if values else None


def select_lean_audit_source(text: str, entry: str | None = None) -> str:
    """Select an explicitly named literal source body, never arbitrary proofs.

    The selected definition/macro must itself contain a Triton quotation. A
    wrapper requiring Lean evaluation is deliberately not guessed. The entry
    preamble is retained for Python target and macro argument metadata.
    """
    entry = text if entry is None else entry
    name = source_audit_metadata(entry, 'Lean')
    if name is None:
        return text
    if not re.fullmatch(r'[A-Za-z_][\w.\'%]*', name):
        raise ValueError(f'Invalid Source-audit Lean selector: {name}')
    code = lean_code_mask(text)
    declaration = (r'^(?:(?:private|protected|noncomputable)\s+)*'
                   r'(?:def|abbrev)\s+' + re.escape(name) + r'(?=\s|\()')
    macro = r'^\s*\|\s*`\(' + re.escape(name) + r'(?=\s|\[|\))'
    matches = list(re.finditer(declaration + '|' + macro, code, re.M))
    if len(matches) != 1:
        raise ValueError(f'Source-audit Lean selector {name}: expected one definition/macro, '
                         f'found {len(matches)}')
    start = matches[0].start()
    selected = code[start:]
    # The next top-level declaration ends this definition or macro rule.
    boundary = re.search(r'\n(?:(?:private|protected|noncomputable)\s+)*'
                         r'(?:def|abbrev|inductive|structure|theorem|lemma|specification|example|opaque|'
                         r'axiom|syntax|macro|macro_rules|elab|elab_rules|instance|end|namespace)\b',
                         selected)
    end = start + boundary.start() if boundary else len(code)
    if boundary:
        selected = selected[:boundary.start()]
    if 'triton {' not in selected:
        raise ValueError(f'Source-audit Lean selector {name} has no literal Triton body')
    selected = strip_lean_comments(text[start:end])
    preamble = lean_first_preamble(entry)
    stripped = strip_lean_comments(preamble)
    comments = ''.join(original if original != masked else ('\n' if original == '\n' else ' ')
                       for original, masked in zip(preamble, stripped))
    rule_header = 'macro_rules\n' if selected.lstrip().startswith('|') else ''
    helpers = _selected_macro_helper_context(text, start, selected) if rule_header else ''
    return comments + '\n' + helpers + '\n' + rule_header + selected


_LEAN_DECLARATION = (r'^(?:(?:private|protected|noncomputable)\s+)*'
                     r'(?:def|abbrev|inductive|structure|theorem|lemma|specification|example|'
                     r'opaque|axiom|syntax|macro|macro_rules|elab|elab_rules|instance|end|namespace)\b')


def _lean_namespace_at(code: str, offset: int) -> tuple[str, ...]:
    """Conservatively track explicit namespace/section scopes, never open names."""
    scopes = []
    for command in re.finditer(r'^(namespace\s+([\w.]+)|(?:noncomputable\s+)?section(?:\s+\w+)?|end(?:\s+[\w.]+)?)\s*$',
                               code[:offset], re.M):
        if command[1].startswith('end'):
            if scopes:
                scopes.pop()
        else:
            scopes.append(command[2] or '')
    return tuple(scope for scope in scopes if scope)


def _selected_macro_helper_context(text: str, start: int, selected: str) -> str:
    """Retain referenced, preceding same-namespace helper definitions verbatim.

    Their signatures and bodies are separately validated by the small macro
    evaluator. Ambiguous names and inaccessible helpers stay unresolved; no
    helper is recognized by its spelling alone.
    """
    code = lean_code_mask(text)
    declarations = list(re.finditer(_LEAN_DECLARATION, code, re.M))
    candidates = {}
    for index, declaration in enumerate(declarations):
        header = re.match(r'(?:(?:private|protected|noncomputable)\s+)*(?:def|abbrev)\s+(\w+)\b',
                          code[declaration.start():])
        if header is None:
            continue
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(code)
        candidates.setdefault(header[1], []).append((declaration.start(), end))
    needed, pending = {}, [lean_first_preamble(selected)]
    namespace = _lean_namespace_at(code, start)
    while pending:
        fragment = pending.pop()
        calls = set(re.findall(r'(?<![\w.`$])(\w+)[ \t]+(?=\(|[A-Za-z_]\w*\b)',
                               lean_code_mask(fragment)))
        for name in calls & candidates.keys():
            if name in needed:
                continue
            matches = candidates[name]
            if len(matches) != 1:
                raise UnresolvedSourceMacro(f'Ambiguous source macro helper: {name}')
            begin, end = matches[0]
            if begin >= start or _lean_namespace_at(code, begin) != namespace:
                raise UnresolvedSourceMacro(f'Inaccessible source macro helper: {name}')
            helper = strip_lean_comments(text[begin:end]).strip()
            if 'triton {' in lean_code_mask(helper):
                raise UnresolvedSourceMacro(f'Source macro helper is not a scalar helper: {name}')
            needed[name] = (begin, helper)
            pending.append(helper.split(':=', 1)[-1])
    return '\n'.join(helper for _, helper in sorted(needed.values()))


@lru_cache(maxsize=32)
def lean_code_mask(text: str, *, strings: bool = True) -> str:
    """Blank comments/strings while preserving offsets and line structure.

    Ordinary escaped strings and Lean raw strings (r"..." or r#"..."#) are
    lexical literals, so their ``triton {`` text and braces are not syntax.
    Macro quotations remain code; only their literal strings are masked.
    """
    out, i, n = list(text), 0, len(text)

    def blank(start, end):
        for index in range(start, end):
            if text[index] != '\n':
                out[index] = ' '

    while i < n:
        if text.startswith('--', i):
            end = text.find('\n', i)
            end = n if end < 0 else end
            blank(i, end)
            i = end
        elif text.startswith('/-', i):
            start, depth = i, 1
            i += 2
            while i < n and depth:
                if text.startswith('/-', i):
                    depth += 1
                    i += 2
                elif text.startswith('-/', i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError('Unterminated Lean block comment')
            blank(start, i)
        else:
            character = re.match(r"'(?:\\(?:u\{[0-9A-Fa-f]+\}|.)|[^'\\\n])'", text[i:]) if (
                text[i] == "'" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_'"))
            ) else None
            if character:
                if strings:
                    blank(i, i + character.end())
                i += character.end()
                continue
            raw = re.match(r'r(#+)?"', text[i:]) if (
                text[i] == 'r' and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_'))
            ) else None
            if raw:
                closing = '"' + (raw[1] or '')
                end = text.find(closing, i + raw.end())
                if end < 0:
                    raise ValueError('Unterminated Lean raw string')
                end += len(closing)
                if strings:
                    blank(i, end)
                i = end
            elif text[i] == '"':
                start = i
                i += 1
                while i < n:
                    if text[i] == "\\":
                        i += 2
                    elif text[i] == '"':
                        i += 1
                        break
                    else:
                        i += 1
                else:
                    raise ValueError('Unterminated Lean string')
                if strings:
                    blank(start, min(i, n))
            else:
                i += 1
    return ''.join(out)


def strip_lean_comments(text: str) -> str:
    """Blank nested block/line comments, retaining string contents and offsets."""
    return lean_code_mask(text, strings=False)


def is_triton_jit(dec: ast.expr) -> bool:
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (
        isinstance(dec, ast.Attribute) and dec.attr == "jit"
    ) or (
        isinstance(dec, ast.Name) and dec.id == "jit"
    )


def python_jit_names(text: str) -> set[str]:
    tree = ast.parse(text)
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(is_triton_jit(dec) for dec in node.decorator_list)
    }


def python_block_load_padding(text: str) -> list[dict]:
    """Inventory checked block-pointer loads in JIT bodies without executing
    Python. Omitted/empty padding is unspecified in Triton, not zero. This is
    a source inventory, not a proof that an out-of-bounds lane is reached.
    Dynamic boundary checks remain visible; literal empty checks are harmless.
    """
    tree = ast.parse(text)
    loads = []
    aliases = {"tl"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names
                           if a.name == "triton.language")
    for kernel in tree.body:
        if not isinstance(kernel, ast.FunctionDef) or not any(
            is_triton_jit(dec) for dec in kernel.decorator_list
        ):
            continue
        for node in ast.walk(kernel):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "load" or ast.unparse(node.func.value) not in aliases:
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            boundary = kwargs.get("boundary_check")
            if boundary is None:
                continue
            if isinstance(boundary, (ast.Tuple, ast.List)) and not boundary.elts:
                continue
            padding = kwargs.get("padding_option")
            if padding is None or (isinstance(padding, ast.Constant) and padding.value == ""):
                policy = "unspecified"
            elif isinstance(padding, ast.Constant) and padding.value in ("zero", "nan"):
                policy = padding.value
            else:
                policy = "dynamic_or_invalid"
            loads.append({"kernel": kernel.name, "line": node.lineno,
                          "boundary_check": ast.unparse(boundary), "padding": policy})
    return sorted(loads, key=lambda load: load["line"])


def python_jit_names_text(text: str) -> set[str]:
    """Read JIT names with the control-flow audit's textual body-selection policy."""
    names, pending = set(), False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@triton.jit"):
            pending = True
            continue
        if pending and stripped.startswith("def "):
            names.add(stripped[4:].split("(", 1)[0].strip())
            pending = False
            continue
        if pending and stripped and not stripped.startswith("@"):
            pending = False
    return names


def lean_first_preamble(text: str) -> str:
    idx = lean_code_mask(text).find("triton {")
    return text[:idx] if idx >= 0 else text


def correctness_surface_evidence(source: Path) -> tuple[list[Path], list[str]]:
    """Find named surfaces in code and blocker markers in each file's preamble.

    Split-port declarations may precede an imported source macro in the file
    traversal. A cutoff at the first Triton body in concatenated text would
    discard those declarations. Comments and strings are not surface evidence.
    This is a lexical inventory, not validation of a theorem's statement.
    """
    from bench.port_layout import proof_files

    surface = re.compile(
        r"ComputeCorrect\.Realizes|ComputeRefine\.Realizes|ComputeCorrect\.General"
        r"|correct_target|⊨"
    )
    marker = re.compile(r"^\s*(?:--\s*)?Correctness-surface blocker:(.*)$", re.M)
    files, markers = [], []
    for path in proof_files(source.parent):
        text = path.read_text()
        if surface.search(lean_code_mask(text)):
            files.append(path)
        markers.extend(marker.findall(lean_first_preamble(text)))
    return files, markers


def target_kernel_candidates(preamble: str):
    """Every ``<file>.py`'s `<name>`'' phrase in the preamble, in order."""
    explicit = source_audit_metadata(preamble, 'target')
    if explicit is not None:
        return [explicit]
    legacy = re.findall(r"\.py`'s[^`]*`([^`]+)`", preamble, re.S)
    if legacy:
        return legacy
    named = re.findall(r"Source:\s*\w+\.py,\s*(\w+)", preamble)
    named += re.findall(r"Complete typed source for `([^`]+)` in\s*`[^`]+\.py`", preamble)
    return named


def python_audit_kernel_body(text: str, lean_text: str, *,
                             include_block_pointers: bool = False) -> str:
    """Use the declared Python JIT consistently across all sequence gates.

    The target-name gate separately rejects missing documentation. Its absence
    preserves the historical primary-memory-kernel fallback; a wrong explicit
    target never falls back to another kernel.
    """
    from bench.source_helpers import python_source_with_helpers
    from bench.source_extern import python_source_with_extern_orders
    from bench.source_pointer_dtypes import python_source_with_pointer_dtype_profile
    from bench.source_materializations import python_source_with_materializations
    from bench.source_dtype_constexpr import python_source_with_dtype_constexpr_profile

    text = python_source_with_materializations(text, lean_text)
    text = python_source_with_pointer_dtype_profile(text, lean_text)
    text = python_source_with_extern_orders(text, lean_text)
    text = python_source_with_runtime_scalar_profile(text, lean_text)
    text = python_source_with_helpers(text, lean_text)
    if source_audit_metadata(lean_text, 'dtype-constexpr') is not None:
        text = python_source_with_dtype_constexpr_profile(text, lean_text)
    else:
        text = python_source_with_parameter_map(text, lean_text)
    text = python_source_with_constexpr_profile(text, lean_text)
    preamble = lean_first_preamble(lean_text)
    names = python_jit_names(text)
    target = target_kernel_name(preamble, names)
    if target is not None:
        return python_named_jit_body(text, target)
    if source_audit_metadata(preamble, 'target') is not None:
        raise ValueError('Source-audit target is not a Python JIT')
    return python_first_kernel_body(text, include_block_pointers=include_block_pointers)


def lean_source_parameter_names(text: str) -> list[str]:
    """Read exact binders of the selected definition or quoted kernel lambda.

    This deliberately supports explicit typed binder groups only. It does not
    guess names from the kernel body, theorem premises, implicit elaboration,
    helper functions, or historical sources preceding an explicit selector.
    """
    text = select_lean_audit_source(text)
    signature = quoted_kernel_signature(text)
    if not signature:
        code = lean_code_mask(lean_first_preamble(text))
        declarations = list(re.finditer(r'^(?:(?:private|protected|noncomputable)\s+)*'
                                       r'(?:def|abbrev)\s+[\w.\']+\s*', code, re.M))
        if not declarations:
            raise ValueError('Cannot read selected Lean kernel binders')
        signature = code[declarations[-1].end():]
        pieces = signature.split(':=', 1)
        if len(pieces) != 2:
            raise ValueError('Unsupported selected Lean definition signature')
        signature = pieces[0]
    signature = lean_code_mask(signature)
    names, position = [], 0
    pairs = {'(': ')', '{': '}', '[': ']'}
    while position < len(signature):
        if signature[position].isspace():
            position += 1
            continue
        if signature[position] == ':':
            # A definition's result type is not a list of parameter binders.
            break
        opening = signature[position]
        if opening not in pairs:
            raise ValueError('Unsupported selected Lean kernel binder syntax')
        stack, end = [pairs[opening]], position + 1
        while end < len(signature) and stack:
            char = signature[end]
            if char in pairs:
                stack.append(pairs[char])
            elif char in ')}]':
                if char != stack.pop():
                    raise ValueError('Unbalanced selected Lean binder')
            end += 1
        if stack:
            raise ValueError('Unterminated selected Lean binder')
        group = signature[position + 1:end - 1]
        parts = _macro_parts(group, ':')
        if len(parts) != 2 or not parts[1]:
            raise ValueError('Selected Lean binders must have explicit types')
        group_names = parts[0].split()
        if not group_names or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) or name == '_'
                                  for name in group_names):
            raise ValueError('Unsupported selected Lean binder name')
        names.extend(group_names)
        position = end
    if len(names) != len(set(names)):
        raise ValueError('Duplicate selected Lean kernel binder')
    return names


def _explicit_python_audit_kernel(text: str, lean_text: str) -> ast.FunctionDef:
    target = source_audit_metadata(lean_text, 'target')
    if target is None or source_audit_metadata(lean_text, 'Lean') is None:
        raise ValueError('Source-audit params requires explicit target and Lean selectors')
    functions = [node for node in ast.parse(text).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef) or not any(
            is_triton_jit(decorator) for decorator in functions[0].decorator_list):
        raise ValueError('Source-audit params target must name one unique Python JIT')
    return functions[0]


def _python_parameter_names(kernel: ast.FunctionDef) -> list[str]:
    if kernel.args.vararg or kernel.args.kwarg:
        raise ValueError('Source-audit params does not support variadic Python kernels')
    names = [argument.arg for argument in kernel.args.posonlyargs + kernel.args.args + kernel.args.kwonlyargs]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate Python kernel parameter')
    return names


def _source_profile_json(declared: str, label: str):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate Source-audit ' + label + ' JSON key: ' + key)
            result[key] = value
        return result
    try:
        return json.loads(declared, object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise ValueError('Invalid Source-audit ' + label + ' JSON') from error


def _validate_source_profile_binding(lean_text: str, binding, label: str) -> str:
    if (not isinstance(binding, dict) or set(binding) != {'source', 'args'}
            or not isinstance(binding['args'], dict)):
        raise ValueError('Unsupported Source-audit ' + label + ' Lean binding')
    selected = select_lean_audit_source(lean_text)
    if binding['source'] != source_audit_metadata(lean_text, 'Lean'):
        raise ValueError('Source-audit ' + label + ' Lean selector does not match selected source')
    preamble = lean_first_preamble(selected)
    arguments = _macro_arguments(preamble)
    if set(arguments) != set(_macro_parameters(preamble)):
        raise ValueError('Source-audit ' + label + ' requires every selected Lean macro argument')
    if (set(binding['args']) != set(arguments)
            or any(type(value) is not type(arguments[name]) or value != arguments[name]
                   for name, value in binding['args'].items())):
        raise ValueError('Source-audit ' + label + ' Lean args do not match selected configuration')
    return selected


def _runtime_scalar_registry():
    """Read reviewed native evidence; metadata never supplies an arbitrary path.

    Reading on each validation prevents a cached result from hiding changed
    evidence. The full corpus audit fingerprints this JSON under its ports root.
    """
    path = Path(__file__).resolve().parent / 'tritonbench_g/source_runtime_abi_profiles.json'
    try:
        registry = _source_profile_json(path.read_text(), 'runtime evidence')
    except OSError as error:
        raise ValueError('Missing reviewed runtime scalar evidence registry') from error
    if (not isinstance(registry, dict) or set(registry) != {'schema', 'profiles'}
            or type(registry['schema']) is not int or registry['schema'] != 1
            or not isinstance(registry['profiles'], dict)):
        raise ValueError('Unsupported runtime scalar evidence registry')
    return registry['profiles']


def source_runtime_scalar_profile(python_text: str, lean_text: str) -> tuple[tuple[str, str], ...]:
    """Admit only recorded Python-float/fp32 ABI views of runtime scalars.

    Native source, signature/configuration, TTIR and reviewed-record hashes are
    checked independently of the later comparison rewrite. This is not an
    admission of integer objects, all runtime types, shapes or numerical values.
    """
    declared = source_audit_metadata(lean_text, 'runtime')
    if declared is None:
        return ()
    profile = _source_profile_json(declared, 'runtime')
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean', 'native'}
            or not isinstance(profile['python'], dict) or not profile['python']
            or not isinstance(profile['native'], dict)
            or set(profile['native']) != {'profile', 'sha256'}
            or not isinstance(profile['native']['profile'], str)
            or not isinstance(profile['native']['sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', profile['native']['sha256'])):
        raise ValueError('Unsupported Source-audit runtime schema')
    scalars = profile['python']
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) or keyword.iskeyword(name)
           or declaration != {'object': 'float', 'abi': 'fp32'}
           for name, declaration in scalars.items()):
        raise ValueError('Source-audit runtime supports only explicit Python float / fp32 scalar profiles')
    try:
        kernel = _explicit_python_audit_kernel(python_text, lean_text)
    except SyntaxError as error:
        raise ValueError('Invalid original Python source for runtime profile') from error
    names = _python_parameter_names(kernel)
    parameters = {argument.arg: argument for argument in
                  kernel.args.posonlyargs + kernel.args.args + kernel.args.kwonlyargs}
    for name in scalars:
        argument = parameters.get(name)
        if argument is None or argument.annotation is not None:
            raise ValueError('Source-audit runtime requires an original unannotated non-constexpr parameter: ' + name)
    selected = _validate_source_profile_binding(lean_text, profile['lean'], 'runtime')
    constants = dict(source_constexpr_profile(python_text, lean_text))
    if set(constants) & set(scalars):
        raise ValueError('Runtime scalar and constexpr profiles overlap')
    mapping = dict(source_parameter_map(python_text, lean_text))
    real_binders = set()
    # These admitted templates expose runtime real inputs in the quoted lambda.
    # Other binder forms are not inferred from kernel expressions or names.
    for match in re.finditer(r'\(([A-Za-z_][A-Za-z0-9_\s]*)\s*:\s*ℝ\s*\)',
                             quoted_kernel_signature(selected)):
        real_binders.update(match[1].split())
    if any(mapping.get(name, name) not in real_binders for name in scalars):
        raise ValueError('Runtime scalar does not map to a selected quoted Lean real binder')
    record = _runtime_scalar_registry().get(profile['native']['profile'])
    if not isinstance(record, dict) or set(record) != {
            'jit', 'python', 'lean', 'compiler', 'specialization', 'native', 'ttir'}:
        raise ValueError('Unknown or malformed reviewed runtime scalar evidence')
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != profile['native']['sha256']:
        raise ValueError('Runtime scalar native evidence digest mismatch')
    if record['jit'] != kernel.name or record['python'] != scalars or record['lean'] != profile['lean']:
        raise ValueError('Runtime scalar native evidence does not bind this selected source/profile')
    compiler = {'triton_version': '3.7.1', 'target': 'cuda sm90 warp32',
                'kernel_launched': False, 'options': {'num_warps': 4, 'num_stages': 3}}
    if json.dumps(record['compiler'], sort_keys=True) != json.dumps(compiler, sort_keys=True):
        raise ValueError('Unsupported runtime scalar native compiler profile')
    if record['specialization'] != {'1e-06': ['fp32', None], '1.0': ['fp32', None],
                                    '0.0': ['fp32', None], '7': ['i32', None]}:
        raise ValueError('Runtime scalar specialization evidence is incomplete or altered')
    native = record['native']
    if (not isinstance(native, dict) or native.get('status') != 'compiled'
            or native.get('eps_is_constexpr') is not False
            or native.get('source_sha256') != hashlib.sha256(python_text.encode()).hexdigest()
            or not isinstance(record['ttir'], str)
            or native.get('ttir_sha256') != hashlib.sha256(record['ttir'].encode()).hexdigest()
            or not isinstance(native.get('signature'), dict)
            or not isinstance(native.get('constexprs'), dict)
            or not isinstance(native.get('ttir_signature'), str)
            or native['ttir_signature'] not in record['ttir']):
        raise ValueError('Runtime scalar native source/signature/TTIR evidence mismatch')
    constexpr_names = {name for name, arg in parameters.items()
                       if isinstance(arg.annotation, ast.Attribute) and arg.annotation.attr == 'constexpr'
                       and isinstance(arg.annotation.value, ast.Name) and arg.annotation.value.id == 'tl'}
    if (set(native['constexprs']) != constexpr_names
            or set(native['signature']) != set(names) - constexpr_names
            or not re.search(r'\btt\.func public @' + re.escape(kernel.name) + r'\(', native['ttir_signature'])
            or any(native['signature'].get(name) != 'fp32'
                   or not re.search(r'%' + re.escape(name) + r':\s*f32\b', native['ttir_signature'])
                   for name in scalars)
            or any(type(native['constexprs'].get(name)) is not type(value)
                   or native['constexprs'][name] != value for name, value in constants.items())):
        raise ValueError('Runtime scalar evidence has a different JIT signature or constexpr configuration')
    nodes = [node for statement in kernel.body for node in ast.walk(statement)]
    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
                   ast.ExceptHandler, ast.Match)
    if any(isinstance(node, unsupported) for node in nodes):
        raise ValueError('Runtime scalar view rejects nested scopes or indirect bindings')
    parents = {child: node for node in ast.walk(kernel) for child in ast.iter_child_nodes(node)}
    used = set()
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in scalars:
            continue
        if not isinstance(node.ctx, ast.Load):
            raise ValueError('Runtime scalar parameter is rebound or deleted: ' + node.id)
        used.add(node.id)
        current = node
        while True:
            parent = parents.get(current)
            if (isinstance(parent, ast.Assign) and parent.value is current
                    and len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name)
                    and current is not node):
                break
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                current = parent
                continue
            if (isinstance(parent, ast.Call) and not parent.keywords
                    and ((ast.unparse(parent.func) in {'tl.sqrt', 'tl.log'} and len(parent.args) == 1)
                         or (ast.unparse(parent.func) == 'tl.maximum' and len(parent.args) == 2))
                    and current in parent.args):
                current = parent
                continue
            raise ValueError('Unsupported runtime scalar AST use: ' + node.id)
    if used != set(scalars):
        raise ValueError('Runtime scalar profile has no supported parameter use')
    return tuple((name, 'tl.float32') for name in scalars)


def python_source_with_runtime_scalar_profile(python_text: str, lean_text: str) -> str:
    """Materialize the reviewed scalar ABI conversion at actual Name loads.

    This comparison representation changes only original UTF-8 Name spans.
    Signatures, tensor casts, comments, strings, keyword/attribute labels and
    unselected functions are untouched. Native evidence checks the original
    Python source before parameter mapping and boolean branch selection.
    """
    scalars = dict(source_runtime_scalar_profile(python_text, lean_text))
    if not scalars:
        return python_text
    kernel = _explicit_python_audit_kernel(python_text, lean_text)
    offsets, total = [], 0
    for line in python_text.encode().splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    edits = []
    for statement in kernel.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id in scalars:
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                edits.append((start, end, (node.id + '.to(' + scalars[node.id] + ')').encode()))
    result = python_text.encode()
    for start, end, value in sorted(edits, reverse=True):
        result = result[:start] + value + result[end:]
    return result.decode()


@lru_cache(maxsize=32)
def source_constexpr_profile(python_text: str, lean_text: str) -> tuple[tuple[str, bool], ...]:
    """Validate an explicit boolean profile against both selected source views.

    The JSON schema is {"python": {"FLAG": false}, "lean": {"source":
    "macro%", "args": {"weight": "half", "version": "original"}}}.
    The complete Lean configuration is pinned, not inferred from a Python
    default, dtype or identifier spelling. This is a reviewed correspondence
    used for source comparison, not proof that the remaining profiles match.
    """
    declared = source_audit_metadata(lean_text, 'constexpr')
    if declared is None:
        return ()

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate Source-audit constexpr JSON key: ' + key)
            result[key] = value
        return result

    try:
        profile = json.loads(declared, object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise ValueError('Invalid Source-audit constexpr JSON') from error
    if (not isinstance(profile, dict) or set(profile) != {'python', 'lean'}
            or not isinstance(profile['python'], dict) or not profile['python']
            or not isinstance(profile['lean'], dict)
            or set(profile['lean']) != {'source', 'args'}
            or not isinstance(profile['lean']['args'], dict)):
        raise ValueError('Unsupported Source-audit constexpr schema')
    constants, binding = profile['python'], profile['lean']
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)
           or keyword.iskeyword(name) or type(value) is not bool
           for name, value in constants.items()):
        raise ValueError('Source-audit constexpr only accepts named boolean values')
    kernel = _explicit_python_audit_kernel(python_text, lean_text)
    _python_parameter_names(kernel)
    parameters = {argument.arg: argument for argument in
                  kernel.args.posonlyargs + kernel.args.args + kernel.args.kwonlyargs}
    for name in constants:
        argument = parameters.get(name)
        if argument is None or not (isinstance(argument.annotation, ast.Attribute)
                and argument.annotation.attr == 'constexpr'
                and isinstance(argument.annotation.value, ast.Name)
                and argument.annotation.value.id == 'tl'):
            raise ValueError('Source-audit constexpr key is not an exact tl.constexpr parameter: ' + name)
    selected = select_lean_audit_source(lean_text)
    if binding['source'] != source_audit_metadata(lean_text, 'Lean'):
        raise ValueError('Source-audit constexpr Lean selector does not match selected source')
    preamble = lean_first_preamble(selected)
    arguments = _macro_arguments(preamble)
    if set(arguments) != set(_macro_parameters(preamble)):
        raise ValueError('Source-audit constexpr requires every selected Lean macro argument')
    if (set(binding['args']) != set(arguments)
            or any(type(value) is not type(arguments[name]) or value != arguments[name]
                   for name, value in binding['args'].items())):
        raise ValueError('Source-audit constexpr Lean args do not match selected configuration')
    # Validate all arms before selecting any branch. An assignment or an
    # unsupported use cannot disappear merely because the chosen arm is empty.
    nodes = [node for statement in kernel.body for node in ast.walk(statement)]
    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
                   ast.ExceptHandler, ast.Match)
    if any(isinstance(node, unsupported) for node in nodes):
        raise ValueError('Source-audit constexpr cannot normalize nested scopes or indirect bindings')
    guards = set()
    used = set()
    for node in nodes:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            test = test.operand
        if isinstance(test, ast.Name) and test.id in constants:
            guards.add(id(test))
            used.add(test.id)
    for node in nodes:
        if isinstance(node, ast.Name) and node.id in constants and (
                not isinstance(node.ctx, ast.Load) or id(node) not in guards):
            raise ValueError('Source-audit constexpr parameter has a non-guard use or rebinding: ' + node.id)
    if used != set(constants):
        raise ValueError('Source-audit constexpr parameter has no supported if guard')
    return tuple(constants.items())


def python_source_with_constexpr_profile(python_text: str, lean_text: str) -> str:
    """Select only explicitly pinned boolean if/not-if arms for comparison.

    No arithmetic, dtype, helper call or runtime condition is evaluated. The
    chosen body is printed from its AST; the original JIT signature and all
    other functions stay unchanged. This result is not an executable rewrite.
    """
    constants = dict(source_constexpr_profile(python_text, lean_text))
    if not constants:
        return python_text
    kernel = _explicit_python_audit_kernel(python_text, lean_text)
    first_line, last_line = kernel.body[0].lineno, kernel.end_lineno
    # Record the original function boundary before removing only its actual
    # docstring. A string later selected out of an if arm is not a docstring.
    if (isinstance(kernel.body[0], ast.Expr)
            and isinstance(kernel.body[0].value, ast.Constant)
            and isinstance(kernel.body[0].value.value, str)):
        kernel.body = kernel.body[1:]

    class SelectGuards(ast.NodeTransformer):
        def visit_If(self, node):
            test, negate = node.test, False
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                test, negate = test.operand, True
            if isinstance(test, ast.Name) and test.id in constants:
                selected = node.body if constants[test.id] != negate else node.orelse
                result = []
                for statement in selected:
                    value = self.visit(statement)
                    result.extend(value if isinstance(value, list) else [value])
                return result
            return self.generic_visit(node)

    kernel = SelectGuards().visit(kernel)
    if not kernel.body:
        raise ValueError('Source-audit constexpr produced an empty selected kernel')
    body = ast.unparse(ast.Module(body=kernel.body, type_ignores=[]))
    lines = python_text.splitlines(keepends=True)
    result = (''.join(lines[:first_line - 1]) +
              '\n'.join('    ' + line for line in body.splitlines()) + '\n' +
              ''.join(lines[last_line:]))
    try:
        ast.parse(result)
    except SyntaxError as error:
        raise ValueError('Unsupported empty or malformed constexpr branch result') from error
    return result


@lru_cache(maxsize=32)
def source_parameter_map(python_text: str, lean_text: str) -> tuple[tuple[str, str], ...]:
    """Validate an explicit, injective, capture-free Python-to-Lean name map.

    Missing entries are identities, never abbreviations or inferred aliases.
    This is a parameter correspondence for lexical comparison, not a proof of
    program equivalence or permission to specialize a constexpr parameter.
    """
    from bench.source_dtype_constexpr import source_dtype_constexpr_profile
    constants = dict(source_constexpr_profile(python_text, lean_text))
    constants.update(source_dtype_constexpr_profile(python_text, lean_text))
    declared = source_audit_metadata(lean_text, 'params')
    if declared is None:
        return ()
    mapping = {}
    for item in declared.split(','):
        pair = re.fullmatch(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*', item)
        if pair is None or pair[1] in mapping:
            raise ValueError('Invalid or duplicate Source-audit params entry: ' + item)
        if any(keyword.iskeyword(name) or name == '_' for name in pair.groups()):
            raise ValueError('Unsupported Source-audit params identifier: ' + item)
        mapping[pair[1]] = pair[2]
    kernel = _explicit_python_audit_kernel(python_text, lean_text)
    parameters = _python_parameter_names(kernel)
    if set(constants) & (set(mapping) | set(mapping.values())):
        raise ValueError('Source-audit params must not rename or capture constexpr profile parameters')
    binders = lean_source_parameter_names(lean_text)
    unknown = set(mapping) - set(parameters)
    if unknown:
        raise ValueError('Source-audit params keys are not Python parameters: ' + ', '.join(sorted(unknown)))
    unknown = set(mapping.values()) - set(binders)
    if unknown:
        raise ValueError('Source-audit params targets are not selected Lean binders: ' + ', '.join(sorted(unknown)))
    effective = [mapping.get(parameter, parameter) for parameter in parameters if parameter not in constants]
    if len(effective) != len(set(effective)):
        raise ValueError('Source-audit params mapping is not injective, including unchanged parameters')
    body_nodes = [node for statement in kernel.body for node in ast.walk(statement)]
    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.ExceptHandler,
                   ast.Match, ast.MatchAs, ast.MatchStar)
    if any(isinstance(node, unsupported) for node in body_nodes):
        raise ValueError('Source-audit params cannot normalize nested scopes or indirect Python bindings')
    # A renamed parameter must not capture a local variable, module/helper name,
    # or another free name in the original body. Parameter permutations are
    # safe because replacements below use original AST locations simultaneously.
    occupied = {node.id for node in body_nodes if isinstance(node, ast.Name)} - set(parameters)
    capturing = set(mapping.values()) & occupied
    if capturing:
        raise ValueError('Source-audit params would capture Python names: ' + ', '.join(sorted(capturing)))
    return tuple(mapping.items())


def python_source_parameter_names(python_text: str, lean_text: str) -> list[str]:
    from bench.source_orders import source_order_profile
    from bench.source_dtype_constexpr import source_dtype_constexpr_profile
    mapping = dict(source_parameter_map(python_text, lean_text))
    constants = dict(source_constexpr_profile(python_text, lean_text))
    constants.update(source_order_profile(python_text, lean_text))
    constants.update(source_dtype_constexpr_profile(python_text, lean_text))
    kernel = _explicit_python_audit_kernel(python_text, lean_text)
    return [mapping.get(name, name) for name in _python_parameter_names(kernel) if name not in constants]


def python_source_with_parameter_map(python_text: str, lean_text: str) -> str:
    """Rename only actual parameter Name nodes, retaining source spelling.

    Strings, comments, attribute names and call keyword labels are untouched.
    The result is a comparison view, not executable rewritten Python: its
    signature, annotations, defaults and decorators deliberately stay original.
    AST columns are UTF-8 byte offsets; applying replacements to bytes also
    preserves correct positions when earlier text on a line contains Unicode.
    """
    mapping = dict(source_parameter_map(python_text, lean_text))
    if not mapping:
        return python_text
    kernel = _explicit_python_audit_kernel(python_text, lean_text)
    lines = python_text.encode().splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    replacements = []
    for statement in kernel.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id in mapping:
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                replacements.append((start, end, mapping[node.id].encode()))
    out = python_text.encode()
    for start, end, value in sorted(replacements, reverse=True):
        out = out[:start] + value + out[end:]
    return out.decode()


def target_kernel_name(preamble: str, jit_names):
    """The declared target JIT: the LAST preamble phrase that names a real
    `@triton.jit` kernel. Prose legitimately reuses the same phrasing for
    non-kernel snippets (a transcription note quoting `labels_ptr += row_idx`,
    say); such a phrase must not shadow the declaration, which is what taking
    the last phrase unconditionally used to do."""
    named = [name for name in target_kernel_candidates(preamble) if name in jit_names]
    return named[-1] if named else None


def python_first_kernel_body(text: str, *, include_block_pointers: bool = False) -> str:
    lines = text.splitlines()

    def body_for_name(target: str) -> str:
        i = 0
        while i < len(lines):
            if not lines[i].strip().startswith("@triton.jit"):
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip().startswith("@"):
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("def " + target + "("):
                i += 1
                continue

            start = None
            parens = 0
            for j in range(i, len(lines)):
                parens += lines[j].count("(") - lines[j].count(")")
                if parens <= 0 and lines[j].rstrip().endswith(":"):
                    start = j + 1
                    break
            if start is None:
                return ""

            body = []
            k = start
            while k < len(lines):
                line = lines[k]
                if line and not line.startswith((" ", "\t")):
                    break
                body.append(line)
                k += 1
            body_text = "\n".join(body)
            if any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
                return body_text
            i = k
        return ""

    # Preserve the primary-kernel preference used by the surface checks.
    # Block-pointer-only bodies participate in the fallback scan only.
    fallback_tokens = ("tl.program_id", "tl.load", "tl.store")
    if include_block_pointers:
        fallback_tokens += ("tl.make_block_ptr",)

    preferred = body_for_name("_attn_fwd")
    if preferred:
        return preferred
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue
        start = None
        parens = 0
        for j in range(i, len(lines)):
            parens += lines[j].count("(") - lines[j].count(")")
            if parens <= 0 and lines[j].rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            return ""
        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if any(token in body_text for token in fallback_tokens):
            return body_text
        i = k
    return ""


def python_jit_kernel_bodies(text: str, *, all_jit: bool = False) -> list[str]:
    """Select memory kernels, preferring `_attn_fwd`, or every JIT body.

    Helper-coverage checks need all bodies; sequence checks use the primary
    kernel policy. These choices intentionally remain distinct.
    """
    lines = text.splitlines()
    bodies = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("@triton.jit"):
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith("@"):
            i += 1
        if i >= len(lines) or not lines[i].strip().startswith("def "):
            continue

        def_i = i
        start = None
        parens = 0
        for j in range(def_i, len(lines)):
            line = lines[j]
            parens += line.count("(") - line.count(")")
            if parens <= 0 and line.rstrip().endswith(":"):
                start = j + 1
                break
        if start is None:
            break

        body = []
        k = start
        while k < len(lines):
            line = lines[k]
            if line and not line.startswith((" ", "\t")):
                break
            body.append(line)
            k += 1
        body_text = "\n".join(body)
        if all_jit or any(token in body_text for token in ("tl.program_id", "tl.load", "tl.store")):
            name = lines[def_i].strip().split("def ", 1)[1].split("(", 1)[0]
            if not all_jit and name == "_attn_fwd":
                return [body_text]
            bodies.append(body_text)
        i = k
    return bodies


class UnresolvedSourceMacro(ValueError):
    """A metaprogram needs an explicit source profile or parser support."""


@dataclass(frozen=True)
class _SourceCarrierType:
    """A checked closed type quotation, distinct from scalar/dtype strings."""
    name: str


def _quoted_syntax(text: str, start: int, category: str) -> tuple[str, int]:
    marker = '`(' + category + '|'
    if not text.startswith(marker, start):
        raise UnresolvedSourceMacro(f'Expected a quoted {category}')
    depth, in_string = 1, False
    end = start + len(marker)
    while end < len(text):
        char = text[end]
        if in_string:
            if char == "\\":
                end += 2
                continue
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return text[start + len(marker):end].strip(), end + 1
        end += 1
    raise UnresolvedSourceMacro(f'Unterminated quoted {category}')


def _quoted_statement(text: str, start: int) -> tuple[str, int]:
    return _quoted_syntax(text, start, 'tritonStmt')


def _macro_parameters(preamble: str) -> dict[str, str]:
    rules = list(re.finditer(r'\|\s*`\(([^`]+?)\)\s*=>', strip_lean_comments(preamble)))
    # Local dtype matches also use ``| `(tritonDType| ...) =>``. They are not
    # macro argument declarations and must not shadow the surrounding rule.
    declarations = [dict(re.findall(r'\$(\w+)\s*:\s*(\w+)', rule[1])) for rule in rules]
    return next((parameters for parameters in reversed(declarations) if parameters), {})


_MACRO_DTYPES = {'tl.' + item for item in ('float64', 'float32', 'float16', 'bfloat16',
                'float8e4nv', 'float8e5', 'int1', 'int8', 'int16', 'int32', 'int64',
                'uint8', 'uint16', 'uint32', 'uint64')} | {'OUT_DTYPE', 'DTYPE'}


def _macro_arguments(preamble: str) -> dict[str, object]:
    values = {}
    declared = source_audit_metadata(preamble, 'args')
    if declared is None:
        return values
    for item in declared.split(','):
        match = re.fullmatch(r'\s*(\w+)\s*=\s*([\w.]+)\s*', item)
        if match is None or match[1] in values:
            raise ValueError(f'Invalid or duplicate Source-audit args entry: {item}')
        value = match[2]
        values[match[1]] = int(value) if value.isdigit() else value
    arguments = _macro_parameters(preamble)
    unknown = set(values) - arguments.keys()
    if unknown:
        raise ValueError('Source-audit args must name actual macro parameters: ' +
                         ', '.join(sorted(unknown)))
    for name, value in values.items():
        category = arguments[name]
        if ((category == 'num' and type(value) is not int)
                or (category in ('ident', 'tritonDType') and not isinstance(value, str))
                or category not in ('num', 'ident', 'tritonDType')):
            raise ValueError(f'Unsupported Source-audit argument {name} for {category}')
        # Literal productions in VeriTile/Triton/DSL/Syntax.lean. In particular,
        # a typo must not silently select a dtype match's wildcard branch.
        if category == 'tritonDType' and value not in _MACRO_DTYPES:
            raise ValueError(f'Unknown Source-audit dtype literal: {value}')
    return values


def _macro_parts(expr: str, separator: str) -> list[str]:
    """Split a tiny Lean expression outside parentheses, arrays and literals."""
    masked = lean_code_mask(expr)
    parts, depth, start, index = [], 0, 0, 0
    while index < len(masked):
        char = masked[index]
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
            if depth < 0:
                raise UnresolvedSourceMacro('Unbalanced source macro expression')
        elif depth == 0 and masked.startswith(separator, index):
            if separator == '|' and ((index and masked[index - 1] == '|') or
                                     masked[index + 1:index + 2] == '|'):
                index += 1
                continue
            parts.append(expr[start:index].strip())
            index += len(separator)
            start = index
            continue
        index += 1
    if depth:
        raise UnresolvedSourceMacro('Unbalanced source macro expression')
    return parts + [expr[start:].strip()]


def _macro_if_parts(expr: str) -> tuple[str, str, str]:
    depth, nested, then, otherwise = 0, 0, None, None
    for token in re.finditer(r'\(|\)|\b(?:if|then|else)\b', lean_code_mask(expr)):
        if token[0] == '(':
            depth += 1
        elif token[0] == ')':
            depth -= 1
        elif depth == 0:
            if token[0] == 'if':
                nested += 1
            elif token[0] == 'then' and nested == 1 and then is None:
                then = token
            elif token[0] == 'else':
                nested -= 1
                if nested == 0:
                    otherwise = token
                    break
    if then is None or otherwise is None:
        raise UnresolvedSourceMacro('Source macro if lacks then/else')
    return (expr[3:then.start()].strip(), expr[then.end():otherwise.start()].strip(),
            expr[otherwise.end():].strip())


def _dtype_match(expr: str, values: dict, evaluate):
    """Evaluate explicit dtype quotation matches, including tuple/or patterns."""
    header = re.match(r'match\s+([\w\s,]+)\s+with\s*', expr)
    if header is None:
        raise UnresolvedSourceMacro('Unsupported source macro match: ' + expr)
    names = [name.strip() for name in header[1].split(',')]
    if any(not re.fullmatch(r'\w+', name) or not isinstance(values.get(name), str)
           for name in names):
        raise UnresolvedSourceMacro('Unresolved source macro match ' + header[0].strip())
    pieces = _macro_parts(expr[header.end():], '|')
    if pieces[0]:
        raise UnresolvedSourceMacro('Source macro match lacks pattern arms')
    patterns, result, found = [], None, False
    for piece in pieces[1:]:
        arm = _macro_parts(piece, '=>')
        if len(arm) not in (1, 2):
            raise UnresolvedSourceMacro('Unsupported source macro match arm')
        pattern = _macro_parts(arm[0], ',')
        if len(pattern) != len(names):
            raise UnresolvedSourceMacro('Source macro match pattern arity differs')
        matches = True
        for name, item in zip(names, pattern):
            if item == '_':
                continue
            quoted = re.fullmatch(r'`\(tritonDType\|\s*(tl\.\w+)\s*\)', item)
            if quoted is None or quoted[1] not in _MACRO_DTYPES:
                raise UnresolvedSourceMacro('Unsupported source dtype pattern: ' + item)
            matches = matches and values[name] == quoted[1]
        patterns.append(matches)
        if len(arm) == 2:
            rejected = re.fullmatch(r'Macro\.throwError\s+"(?:[^"\\]|\\.)*"', arm[1])
            # Validate every non-error arm, even those not selected by this view.
            value = None if rejected else evaluate(arm[1])
            if any(patterns) and not found:
                if rejected:
                    raise UnresolvedSourceMacro('Source macro profile rejected by dtype helper')
                result, found = value, True
            patterns = []
    if patterns or not found:
        raise UnresolvedSourceMacro('Incomplete or unmatched source dtype match')
    return result


def _macro_helpers(preamble: str) -> dict:
    helpers = {}
    code = strip_lean_comments(preamble)
    declarations = list(re.finditer(_LEAN_DECLARATION, lean_code_mask(code), re.M))
    for index, declaration in enumerate(declarations):
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(code)
        definition = code[declaration.start():end].strip()
        match = re.fullmatch(r'(?:private\s+)?def\s+(\w+)\s*'
                             r'\(([\w\s]+)\s*:\s*(Nat|TSyntax\s+`tritonDType)\)\s*'
                             r':\s*(MacroM\s+Nat|Nat)\s*:=\s*(.+)', definition, re.S)
        if match is None:
            continue
        name, parameters, category, output, expression = match.groups()
        if name in helpers:
            raise UnresolvedSourceMacro('Ambiguous source macro helper: ' + name)
        names = parameters.split()
        if len(set(names)) != len(names) or (category != 'Nat' and len(names) != 1):
            raise UnresolvedSourceMacro('Unsupported source macro helper signature: ' + name)
        helpers[name] = (names, category, expression.strip())
    return helpers


def _macro_value(expr: str, values: dict, helpers: dict, active=frozenset()):
    """Interpret a closed Bool/Nat expression, with validated local helpers only."""
    expr = expr.strip()
    if expr.startswith('pure '):
        return _macro_value(expr[5:], values, helpers, active)
    if expr.startswith('(') and expr.endswith(')'):
        # Remove a complete enclosing pair only; `(f x) == (g x)` is not one.
        try:
            _macro_parts(expr[1:-1], ',')
        except UnresolvedSourceMacro:
            pass
        else:
            return _macro_value(expr[1:-1], values, helpers, active)
    evaluate = lambda part: _macro_value(part, values, helpers, active)
    if expr.startswith('match '):
        return _dtype_match(expr, values, evaluate)
    if expr.startswith('if '):
        condition, yes, no = _macro_if_parts(expr)
        chosen, first, second = evaluate(condition), evaluate(yes), evaluate(no)
        if type(chosen) is not bool or type(first) is not type(second):
            raise UnresolvedSourceMacro('Ill-typed source macro if')
        return first if chosen else second
    for operator, combine in [('||', any), ('&&', all)]:
        parts = _macro_parts(expr, operator)
        if len(parts) > 1:
            # Evaluate every arm: an unknown arm must not hide behind a known
            # short-circuit result and conceal unsupported metaprogram syntax.
            arguments = [evaluate(part) for part in parts]
            if any(type(value) is not bool for value in arguments):
                raise UnresolvedSourceMacro('Source macro Bool operator has a non-Bool argument')
            return combine(arguments)
    for operator in ('==', '='):
        parts = _macro_parts(expr, operator)
        if len(parts) == 2:
            left, right = map(evaluate, parts)
            if type(left) is not type(right):
                raise UnresolvedSourceMacro('Source macro equality has incompatible arguments')
            return left == right
        if len(parts) > 2:
            raise UnresolvedSourceMacro('Unsupported chained source macro equality')
    if expr.startswith('!'):
        value = evaluate(expr[1:])
        if type(value) is not bool:
            raise UnresolvedSourceMacro('Source macro negation has a non-Bool argument')
        return not value
    if expr in ('Bool.true', 'true'):
        return True
    if expr in ('Bool.false', 'false'):
        return False
    if expr.isdigit():
        return int(expr)
    if re.fullmatch(r'`\w+', expr):
        return expr[1:]
    if expr in values:
        return values[expr]
    match = re.fullmatch(r'(\w+)\.(getId|getNat|raw(?:\.reprint)?)', expr)
    if match:
        name, kind = match.groups()
        # Preserve the established original-variant convention only for an
        # actual getId selector. Other unknown arguments are never guessed.
        actual = values.get(name, 'original' if kind == 'getId' and name == 'version' else None)
        if actual is not None:
            if (kind == 'getNat' and type(actual) is not int) or (kind != 'getNat' and
                                                                              not isinstance(actual, str)):
                raise UnresolvedSourceMacro('Source macro argument accessor has incompatible type')
            return actual
    application = re.match(r'(\w+)\s+(.+)', expr, re.S)
    if application and application[1] in helpers:
        name, arguments = application.groups()
        if name in active:
            raise UnresolvedSourceMacro('Circular source macro helper: ' + name)
        parameters, category, expression = helpers[name]
        # Normalize whitespace outside parentheses so literal application is
        # the only admitted calling convention; no lambdas or partial calls.
        items = [part for part in _macro_parts(re.sub(r'\s+', ' ', arguments), ' ') if part]
        if len(items) != len(parameters):
            raise UnresolvedSourceMacro('Source macro helper arity differs: ' + name)
        args = list(map(evaluate, items))
        expected = int if category == 'Nat' else str
        if any(type(value) is not expected for value in args):
            raise UnresolvedSourceMacro('Source macro helper argument type differs: ' + name)
        value = _macro_value(expression, dict(zip(parameters, args)), helpers, active | {name})
        if type(value) is not int:
            raise UnresolvedSourceMacro('Source macro Nat helper returned another type: ' + name)
        return value
    raise UnresolvedSourceMacro(f'Unresolved source macro condition: {expr}')


def _macro_condition(expr: str, values: dict, helpers: dict | None = None) -> bool:
    value = _macro_value(expr, values, helpers or {})
    if type(value) is not bool:
        raise UnresolvedSourceMacro('Source macro condition is not Bool: ' + expr)
    return value


def original_statement_splices(body: str, preamble: str) -> str:
    """Expand closed quotations and resolvable literal statement/array splices.

    This is intentionally a small lexical evaluator, not a Lean interpreter.
    Unknown dtype helpers, match expressions and constexpr paths remain errors
    instead of disappearing from the call/statement inventory. Metadata can
    bind macro *arguments*; it cannot override local computed bindings. Lean
    elaboration and the trust gate still validate the actual macro expansion.
    """
    values = _macro_arguments(preamble)
    parameters = _macro_parameters(preamble)
    helpers = _macro_helpers(preamble)
    code = strip_lean_comments(preamble)
    # Do not reuse a same-named binding from an earlier, unrelated macro.
    scope = code.rfind('macro_rules')
    if scope >= 0:
        code = code[scope:]
    definitions = {}
    definition_ranges = []
    definition_positions = {}
    pattern = re.compile(r'^(?P<indent>[ \t]*)let\s+(?P<mutable>mut\s+)?(?P<name>\w+)\s*'
                         r'(?::\s*(?P<type>[^\n←]*?))?\s*(?:←|:=)\s*', re.M)
    for match in pattern.finditer(code):
        tail = code[match.end():]
        boundary = re.search(r'\n' + re.escape(match['indent']) +
                             r'(?=let\s|unless\s|return\s|`\()', tail)
        expression = tail[:boundary.start()] if boundary else tail
        definition_ranges.append((match.start(), match.end() + len(expression)))
        if match['name'] in definitions or match['name'] in parameters:
            raise UnresolvedSourceMacro(f'Shadowed source macro binding: {match["name"]}')
        definitions[match['name']] = (match['type'] or '', expression.strip(), bool(match['mutable']))
        definition_positions[match['name']] = match.start()
    categories = {name: re.search(r'`(tritonStmt|tritonExpr|tritonDType|term)\b', typ)
                  for name, (typ, _, _) in definitions.items()}
    active = set()
    visibility = [len(code)]

    def substitute(text):
        def splice(match):
            name, annotation, star = match.groups()
            if name not in definitions and name not in values:
                if parameters.get(name) == 'tritonDType' and not star:
                    return match[0]
                raise UnresolvedSourceMacro(f'Unresolved source macro splice: ${name}')
            value = resolve(name)
            if isinstance(value, _SourceCarrierType):
                if star or annotation not in (None, 'term'):
                    raise UnresolvedSourceMacro('Type quotation used as another syntax category')
                return value.name
            if isinstance(value, list):
                if not star:
                    raise UnresolvedSourceMacro(f'Array splice ${name} lacks *')
                return '\n' + '\n'.join(value) + '\n'
            if isinstance(value, bool) or isinstance(value, int):
                raise UnresolvedSourceMacro(f'Unsupported literal splice: ${name}')
            if categories.get(name) and categories[name][1] == 'tritonStmt':
                return '\n' + str(value) + '\n'
            return str(value)
        parts, end = [], 0
        for match in re.finditer(r'\$(\w+)(?::(tritonStmt|tritonExpr|tritonDType|term))?(\*)?',
                                 lean_code_mask(text)):
            parts.extend((text[end:match.start()], splice(match)))
            end = match.end()
        return ''.join(parts) + text[end:]

    def syntax_expression(expr, category):
        expr = expr.strip()
        if expr.startswith('match '):
            dependencies(expr.split('with', 1)[0])
            return _dtype_match(expr, values, lambda arm: syntax_expression(arm, category))
        if expr.startswith('if '):
            split = re.search(r'\s+then\s+', expr)
            if split is None:
                raise UnresolvedSourceMacro('Source macro if lacks then')
            condition = expr[3:split.start()]
            # The supported arms are closed quotations or literal arrays. Read
            # both syntactically even when only one branch is selected.
            yes, end = syntax_atom(expr, split.end(), category)
            other = re.match(r'\s*else\s*', expr[end:])
            if other is None:
                raise UnresolvedSourceMacro('Source macro if lacks else')
            no, finish = syntax_atom(expr, end + other.end(), category)
            if expr[finish:].strip():
                raise UnresolvedSourceMacro('Unsupported trailing source macro expression')
            dependencies(condition)
            return yes if _macro_condition(condition, values, helpers) else no
        result, end = syntax_atom(expr, 0, category)
        while re.match(r'\s*\+\+', expr[end:]):
            join = re.match(r'\s*\+\+\s*', expr[end:])
            other, end = syntax_atom(expr, end + join.end(), category)
            if not isinstance(result, list) or not isinstance(other, list):
                raise UnresolvedSourceMacro('Source macro ++ requires literal statement arrays')
            result = result + other
        if expr[end:].strip():
            raise UnresolvedSourceMacro('Unsupported source macro expression: ' + expr[end:].strip())
        return result

    def syntax_atom(expr, pos, category):
        while pos < len(expr) and expr[pos].isspace():
            pos += 1
        if expr.startswith('pure ', pos):
            pos += len('pure ')
            while pos < len(expr) and expr[pos].isspace():
                pos += 1
        if expr.startswith('(', pos):
            depth, end = 1, pos + 1
            while end < len(expr) and depth:
                depth += (expr[end] == '(') - (expr[end] == ')')
                end += 1
            if depth:
                raise UnresolvedSourceMacro('Unterminated parenthesized source macro expression')
            return syntax_expression(expr[pos + 1:end - 1], category), end
        if expr.startswith('#[', pos):
            if category == 'term':
                raise UnresolvedSourceMacro('Carrier type quotation cannot be an array')
            depth, end = 1, pos + 2
            while end < len(expr) and depth:
                depth += (expr[end] == '[') - (expr[end] == ']')
                end += 1
            if depth:
                raise UnresolvedSourceMacro('Unterminated source macro array')
            items = split_top_level_args(expr[pos + 2:end - 1])
            result = []
            for item in items:
                item = item.lstrip('←').strip()
                if re.fullmatch(r'\w+', item):
                    value = resolve(item)
                    if not isinstance(value, str):
                        raise UnresolvedSourceMacro('Array member must be a statement quotation')
                    result.append(value)
                else:
                    value, finish = _quoted_syntax(item, 0, category)
                    if item[finish:].strip():
                        raise UnresolvedSourceMacro('Unsupported source macro array member')
                    result.append(substitute(value))
            return result, end
        identifier = re.match(r'\w+', expr[pos:])
        if identifier:
            return resolve(identifier[0]), pos + identifier.end()
        if category == 'term':
            quoted = re.match(r'`\(\s*(?:term\|\s*)?(Int|Nat)\s*\)', expr[pos:])
            if quoted is None:
                raise UnresolvedSourceMacro('Only closed Int/Nat carrier quotations are supported')
            return _SourceCarrierType(quoted[1]), pos + quoted.end()
        value, end = _quoted_syntax(expr, pos, category)
        return substitute(value), end

    def dependencies(expr, exclude=None):
        for dependency in definition_positions:
            if dependency != exclude and re.search(r'(?<![\w.`$])' + re.escape(dependency) + r'\b',
                                                    lean_code_mask(expr)):
                resolve(dependency)

    def mutable_array(name, expr, category):
        lines = [line.strip() for line in expr.splitlines() if line.strip()]
        if category != 'tritonStmt' or not lines or lines[0] != '#[]':
            raise UnresolvedSourceMacro('Unsupported mutable source macro binding: ' + name)
        result, previous, taken = [], False, False
        for line in lines[1:]:
            push = re.fullmatch(r'(else\s+)?if\s+(.+?)\s+then\s+(\w+)\s*:=\s*'
                                r'(\w+)\.push\s*\(\s*←\s*(.+)\)', line)
            if push is None or push[3] != name or push[4] != name:
                raise UnresolvedSourceMacro('Unsupported mutable source macro update: ' + line)
            continuation, condition, _, _, quoted = push.groups()
            if continuation and not previous:
                raise UnresolvedSourceMacro('Source macro else-if has no preceding if')
            if not continuation:
                taken = False
            dependencies(condition)
            enabled = _macro_condition(condition, values, helpers)
            item, end = _quoted_syntax(quoted, 0, category)
            if quoted[end:].strip():
                raise UnresolvedSourceMacro('Unsupported mutable source macro push')
            item = substitute(item)
            if enabled and not taken:
                result.append(item)
                taken = True
            previous = True
        return result

    def resolve(name):
        if name in active:
            raise UnresolvedSourceMacro(f'Circular source macro binding: {name}')
        if name in definition_positions and definition_positions[name] >= visibility[-1]:
            raise UnresolvedSourceMacro(f'Forward reference to source macro binding: {name}')
        if name not in definitions:
            if name in values:
                return values[name]
            raise UnresolvedSourceMacro(f'Unknown source macro binding: {name}')
        active.add(name)
        visibility.append(definition_positions[name])
        typ, expr, mutable = definitions[name]
        category = re.search(r'`(tritonStmt|tritonExpr|tritonDType|term)\b', typ)
        if category:
            if category[1] == 'term' and re.fullmatch(r'TSyntax\s+`term', typ.strip()) is None:
                raise UnresolvedSourceMacro('Carrier quotation requires exact TSyntax term annotation')
            value = mutable_array(name, expr, category[1]) if mutable else syntax_expression(expr, category[1])
            if category[1] == 'term':
                if not isinstance(value, _SourceCarrierType):
                    raise UnresolvedSourceMacro('Carrier quotation has an incompatible syntax category')
            else:
                if isinstance(value, _SourceCarrierType):
                    raise UnresolvedSourceMacro('Carrier quotation used as computation syntax')
                value = [substitute(item) for item in value] if isinstance(value, list) else substitute(value)
        else:
            if mutable:
                raise UnresolvedSourceMacro('Only mutable statement arrays are supported')
            dependencies(expr, name)
            value = _macro_value(expr, values, helpers)
        active.remove(name)
        visibility.pop()
        values[name] = value
        # Remove evaluated definitions so recursive expansion remains bounded.
        del definitions[name]
        return value

    # Check the exact admission guards too: explicit arguments must describe a
    # profile the Lean macro accepts, rather than merely choosing an else arm.
    guards = list(re.finditer(r'^(?P<indent>[ \t]*)unless\s+([\s\S]+?)\s+do\s*\n'
                              r'[ \t]+Macro\.throwError\s+"(?:[^"\\]|\\.)*"[ \t]*$', code, re.M))
    loops = list(re.finditer(r'^(?P<indent>[ \t]*)for\s+(\w+)\s+in\s+\[([\w\s,]+)\]\s+do\s*$',
                             code, re.M))
    for guard in guards:
        condition = guard[2]
        visibility.append(guard.start())
        dependencies(condition)
        local = values
        enclosing = [loop for loop in loops if loop.end() < guard.start() and
                     len(loop['indent']) < len(guard['indent']) and
                     not code[loop.end():guard.start()].strip()]
        if enclosing:
            loop = enclosing[-1]
            for argument in loop[3].split(','):
                argument = argument.strip()
                if argument not in values:
                    raise UnresolvedSourceMacro('Unresolved source macro guard argument: ' + argument)
                local = dict(values, **{loop[2]: values[argument]})
                if not _macro_condition(condition, local, helpers):
                    raise UnresolvedSourceMacro('Source macro profile rejected by admission guard')
        elif not _macro_condition(condition, local, helpers):
            raise UnresolvedSourceMacro('Source macro profile rejected by admission guard')
        visibility.pop()
    if len(guards) != len(re.findall(r'^\s*unless\b', code, re.M)):
        raise UnresolvedSourceMacro('Unsupported source macro admission guard')
    if len(loops) != len(re.findall(r'^\s*for\b', code, re.M)):
        raise UnresolvedSourceMacro('Unsupported source macro admission loop')
    rule = re.search(r'^\s*\|\s*`\([^`]+?\)\s*=>\s*do\b', code, re.M)
    # lean_first_preamble stops before the literal `triton {`, so a direct
    # final quotation can leave only its opening backtick/parenthesis here.
    quotation = re.search(r'^[ \t]*`\((?:fun\b|triton\b|(?=\s*$))', code, re.M)
    if rule is not None and quotation is not None:
        # No unparsed monadic effect or later mutation may be discarded merely
        # because it does not occur in the selected statement splice.
        remainder = list(code)
        spans = [(0, rule.end()), (quotation.start(), len(code))] + definition_ranges
        spans += [(guard.start(), guard.end()) for guard in guards]
        spans += [(loop.start(), loop.end()) for loop in loops]
        for start, end in spans:
            remainder[start:end] = ' ' * (end - start)
        extra = ''.join(remainder).strip()
        if extra:
            branch = re.fullmatch(r'if\s+(.+?)\s+then', extra, re.S)
            if branch is None or not _macro_condition(branch[1], values, helpers):
                raise UnresolvedSourceMacro('Unsupported source macro statements before quotation: ' + extra)
        for name in list(definitions):
            if name in definitions:
                resolve(name)
    return substitute(body)


def unresolved_source_splices(body: str) -> list[str]:
    """Unbound statement/array splices must never count as an empty program."""
    return sorted(set(re.findall(r'(?m)^\s*\$(\w+)(?::tritonStmt)?(?:\*)?\s*$', body)))


def strip_region_declarations(body: str) -> str:
    """Omit type declarations, including comma-continued lines, from statements."""
    out, continued = [], False
    for line in body.splitlines():
        if continued or line.strip().startswith('tl.region '):
            continued = line.rstrip().endswith(',')
            out.append('')
        else:
            out.append(line)
    return '\n'.join(out)


def quoted_kernel_signature(text: str) -> str:
    """The lambda that constructs the first macro-generated source kernel."""
    preamble = lean_first_preamble(strip_lean_comments(text))
    starts = list(re.finditer(r'`\(fun\s+', lean_code_mask(preamble)))
    if not starts:
        return ''
    signature = preamble[starts[-1].end():]
    return signature.split('=>', 1)[0] if '=>' in signature else ''


def declared_region_dtypes(lean_body: str) -> dict[str, set[str]]:
    regions = {}
    in_region = False
    for line in lean_code_mask(lean_body).splitlines():
        line = line.strip()
        if line.startswith('tl.region '):
            in_region = True
            line = line[len('tl.region '):]
        elif not in_region:
            continue
        elif not re.match(r'\w+\s*=\s*(?:tl\.\w+|[A-Z_]+|\$\w+)\s*(?:,|$)', line):
            in_region = False
            continue
        for name, dtype in re.findall(r'(\w+)\s*=\s*(tl\.\w+|[A-Z_]+|\$\w+)', line):
            regions.setdefault(name, set()).add(dtype)
    return regions


def resolve_source_dtype_attributes(python_body: str, lean_body: str) -> str:
    """Resolve Python pointer/tensor dtype attributes from explicit region types.

    Only unambiguous pointer aliases, block pointers and direct loads participate. Unknown
    attributes and literal dtypes stay unchanged, so mismatches remain errors.
    This normalizes names; it does not validate dtype inference or conversions.
    """
    regions = declared_region_dtypes(lean_body)
    types = dict(regions)
    bindings = {}
    writes = {}
    masked_body = lean_code_mask(lean_body)
    statements = strip_region_declarations(masked_body)
    write_ends = {}
    write_rhs_starts = {}
    for match in re.finditer(r'^\s*(\w+)\s*([+*/-]?=)(?!=)([^\n]*)', masked_body, re.M):
        write_ends.setdefault(match[1], []).append(match.end())
        write_rhs_starts.setdefault(match[1], []).append(match.start(3))
    for match in re.finditer(r'^\s*(\w+)\s*([+*/-]?=)(?!=)([^\n]*)', statements, re.M):
        name, op, rhs = match.groups()
        writes.setdefault(name, []).append((op, rhs))
        # Pointer addition/subtraction preserves its element type. Other
        # reassignments and updates still make the alias ambiguous.
        if op in ('+=', '-='):
            continue
        alias = re.fullmatch(r'\s*(\w+)\s*\+[^\n;]+', rhs) if op == '=' else None
        bindings.setdefault(name, []).append(alias[1] if alias else None)
    # Inline assignments and unsupported update operators must not disappear
    # from the line-oriented provenance inventory. Keyword arguments can make
    # this conservative, but can never justify an otherwise unknown dtype.
    assignment_counts = {}
    for match in re.finditer(r'(?<![\w.])(\w+)\s*'
                             r'(?:(?:\*\*|//|<<|>>|[+*/%&|^-])?=)(?!=)', statements):
        name = match[1]
        assignment_counts[name] = assignment_counts.get(name, 0) + 1
    unresolved_writes = {name for name, count in assignment_counts.items()
                         if count != len(writes.get(name, []))}
    for name in unresolved_writes:
        types.pop(name, None)
    block_pointers = {}
    for pointer, region in re.findall(
            r'(\w+)\s*=\s*tl\.make_block_ptr\s*\(\s*base\s*=\s*(\w+)\b', masked_body):
        block_pointers.setdefault(pointer, set()).update(regions.get(region, set()))

    def pointer_types(pointer, seen=frozenset()):
        if pointer in seen or pointer in unresolved_writes:
            return set()
        entries = bindings.get(pointer)
        if entries is None:
            return regions.get(pointer, set())
        if entries and all(entry == pointer for entry in entries):
            # An explicitly declared pointer advanced from itself keeps its
            # storage type. An undeclared self-cycle supplies no provenance.
            return regions.get(pointer, set())
        if len(entries) != 1:
            return set()
        if entries[0] is not None:
            return pointer_types(entries[0], seen | {pointer})
        return block_pointers.get(pointer, set())

    for pointer in bindings:
        types.pop(pointer, None)
        values = pointer_types(pointer)
        if values:
            types[pointer] = values
    def statement_boundary_after(offset):
        newline = masked_body.find('\n', offset)
        if newline < 0:
            return True
        following = masked_body[newline + 1:].lstrip()
        if not following or following.startswith('}'):
            return True
        line = following.split('\n', 1)[0]
        # A line break alone does not end a Lean DSL expression. Admit only a
        # recognizable next statement; arithmetic, method, call, indexing and
        # other unknown continuation tails cannot prove a direct-load dtype.
        return bool(re.match(r'\w+\s*(?:[+*/%&|^-]?=)(?!=)', line)
                    or re.match(r'tl\.(?:store|atomic_\w+)\s*\(', line)
                    or (re.match(r'(?:if|for|while|tl\.(?:if|for))\b', line) and '{' in line))

    for match in re.finditer(r'(\w+)\s*=\s*tl\.load\s*\(\s*(\w+)\b', masked_body):
        tensor, pointer = match.groups()
        assignments = writes.get(tensor, [])
        load_start = masked_body.index('tl.load', match.start(), match.end())
        calls = tl_calls(masked_body[load_start:], 'load')
        call = calls[0] if calls else ''
        load_end = load_start + len(call)
        line_end = masked_body.find('\n', load_end)
        if line_end < 0:
            line_end = len(masked_body)
        starts = write_rhs_starts.get(tensor, [])
        load_args = split_top_level_args(call, 'load') if call else []
        pointer_argument = load_args[0] if load_args else ''
        # Only a bare pointer or pointer-plus/minus-offset retains this root's
        # storage type. A method/cast/call on the pointer itself can change it;
        # conversions inside mask/other arguments do not change the load dtype.
        unchanged_pointer = bool(re.match(r'^' + re.escape(pointer) + r'\s*(?:$|[+-])',
                                          pointer_argument))
        # Balance the complete call before checking its suffix. A multiline
        # mask/other argument belongs to the load; arithmetic after the closing
        # parenthesis, including a continuation on the next line, does not.
        direct_load = (bool(starts) and bool(assignments) and assignments[0][0] == '='
                       and starts[0] <= load_start
                       and not masked_body[starts[0]:load_start].strip()
                       and bool(call) and unchanged_pointer and call.count('(') == call.count(')')
                       and not masked_body[load_end:line_end].strip()
                       and statement_boundary_after(load_end))
        # Native unary minus preserves these concrete floating tensor types.
        # A symbolic storage type may be integral or unsupported, so it cannot
        # justify the same inference. Every subsequent write must be a complete
        # self-negation; casts, arithmetic, duplicate loads and hidden writes
        # still make the tensor's dtype unresolved.
        pointer_dtypes = types.get(pointer, set())
        floating_negations = (
            len(pointer_dtypes) == 1
            and pointer_dtypes <= {'tl.float16', 'tl.bfloat16', 'tl.float32', 'tl.float64'}
            and len(write_ends.get(tensor, [])) == len(assignments)
            and all(op == '=' and re.fullmatch(r'\s*-\s*' + re.escape(tensor) + r'\s*', value)
                    and statement_boundary_after(end)
                    for (op, value), end in zip(assignments[1:], write_ends.get(tensor, [])[1:])))
        if (tensor in unresolved_writes or not direct_load
                or (len(assignments) != 1 and not floating_negations)):
            # Converted loads, arithmetic suffixes and rebindings do not
            # establish a direct storage-typed tensor binding.
            types.pop(tensor, None)
            continue
        types.setdefault(tensor, set()).update(types.get(pointer, set()))
    def resolve(match):
        values = types.get(match[1], set())
        return next(iter(values)) if len(values) == 1 else match[0]
    return re.sub(r'\b(\w+)\.(?:dtype(?:\.element_ty)?|type\.element_ty)\b', resolve, python_body)


def normalize_shape_ctor_call(call: str) -> str:
    """Normalize DSL quotation notation while retaining every dtype identifier."""
    call = re.sub(r'\$\(([^()]+)\)', r'\1', call)
    call = re.sub(r'(?<=\w)\s*:\s*tritonDType\b', '', call)
    call = call.replace('(', '[').replace(')', ']')
    call = call.replace('tl.float32', 'float32').replace('tl.float64', 'float64')
    call = call.replace('dtype=', '')
    call = re.sub(r'\s+', '', call)
    call = call.replace('_', '').lower()
    return re.sub(r'\[([^\[\],]+),\]', r'[\1]', call)


def cast_sequence(body: str) -> list[str]:
    """Keep literal and macro-selected cast dtypes visible to source checks."""
    casts = re.findall(r'\.to\(\s*(?:tl\.)?(\$?[A-Za-z0-9_.]+)'
                       r'(?:\s*:\s*tritonDType)?\s*\)', strip_comments(body))
    return [cast.replace('_', '').lower() for cast in casts]


def store_calls_with_explicit_destination_casts(python_body: str, lean_body: str) -> list[str]:
    """Match an explicit store cast only to the declared destination dtype.

    Triton converts a stored value to its pointer element type. A typed port
    may spell that conversion explicitly. Resolve only unique, simple pointer
    aliases in the Python body; unknown/reassigned aliases and casts to other
    precisions remain mismatches. All value expressions and masks stay visible.
    This is source normalization, not a proof of numerical cast identities.
    """
    regions = declared_region_dtypes(lean_body)
    bindings: dict[str, list[str | None]] = {}
    # Tokenization remains valid after dtype names become DSL antiquotations.
    # Mask strings/comments before collecting writes; they cannot establish
    # provenance or conceal an actual inline reassignment of a pointer alias.
    code = list(python_body)
    offsets, total = [], 0
    for line in python_body.splitlines(keepends=True):
        offsets.append(total)
        total += len(line)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(python_body).readline))
    except (IndentationError, tokenize.TokenError):
        return tl_calls(python_body, 'store')
    for token in tokens:
        if token.type in (tokenize.STRING, tokenize.COMMENT):
            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            for index in range(start, end):
                if code[index] != '\n':
                    code[index] = ' '
    code = ''.join(code)
    write_counts = {}
    for match in re.finditer(r'^\s*(\w+)\s*([+*/-]?=)(?!=)([^\n]*)',
                             code, re.M):
        name, op, rhs = match.groups()
        write_counts[name] = write_counts.get(name, 0) + 1
        # Pointer addition/subtraction preserves its element type. Other
        # reassignments and updates still make the alias ambiguous.
        if op in ('+=', '-='):
            continue
        alias = re.fullmatch(r'\s*(\w+)\s*\+[^\n;]+', rhs) if op == '=' else None
        bindings.setdefault(name, []).append(alias[1] if alias else None)
    assignment_counts = {}
    for match in re.finditer(r'(?<![\w.])(\w+)\s*'
                             r'(?:(?:\*\*|//|<<|>>|[+*/%&|^-])?=)(?!=)', code):
        assignment_counts[match[1]] = assignment_counts.get(match[1], 0) + 1
    unresolved_writes = {name for name, count in assignment_counts.items()
                         if count != write_counts.get(name, 0)}

    def dtype(pointer):
        # Pointer plus/minus an offset retains the known base element type.
        # Other operators and unknown roots remain unresolved.
        try:
            node = ast.parse(pointer, mode="eval").body
        except SyntaxError:
            return None
        while isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            node = node.left
        if not isinstance(node, ast.Name):
            return None
        pointer = node.id
        seen = set()
        while pointer not in seen:
            seen.add(pointer)
            if pointer in unresolved_writes:
                return None
            if pointer in bindings:
                aliases = bindings[pointer]
                if len(aliases) != 1 or aliases[0] is None:
                    return None
                if aliases[0] == pointer:
                    # A single self-advance preserves the declared pointer type.
                    values = regions.get(pointer, set())
                    return next(iter(values)) if len(values) == 1 else None
                pointer = aliases[0]
            else:
                values = regions.get(pointer, set())
                return next(iter(values)) if len(values) == 1 else None
        return None

    py_calls, lean_calls = tl_calls(python_body, 'store'), tl_calls(lean_body, 'store')
    if len(py_calls) != len(lean_calls):
        return py_calls
    result = []
    def compact(value):
        value = re.sub(r'\s+', '', value)
        value = re.sub(r'\$\((\w+)\)', r'\1', value)
        while value.startswith('(') and value.endswith(')'):
            depth = 0
            encloses = True
            for i, char in enumerate(value):
                depth += (char == '(') - (char == ')')
                if depth == 0 and i != len(value)-1:
                    encloses = False
                    break
            if not encloses or depth != 0:
                break
            value = value[1:-1]
        return value
    for py_call, lean_call in zip(py_calls, lean_calls):
        py_args = split_top_level_args(py_call, 'store')
        lean_args = split_top_level_args(lean_call, 'store')
        if len(py_args) < 2 or len(lean_args) < 2 or compact(py_args[0]) != compact(lean_args[0]):
            result.append(py_call)
            continue
        target = dtype(py_args[0].strip())
        cast = re.fullmatch(r'(.+)\.to\(\s*(tl\.\w+|\$\w+|[A-Z_]+)(?:\s*:\s*tritonDType)?\s*\)',
                            lean_args[1].strip(), re.S)
        if target is not None and cast and cast[2] == target and compact(cast[1]) == compact(py_args[1]):
            value = py_args[1].strip()
            if not re.fullmatch(r'\w+', value):
                value = '(' + value + ')'
            py_args[1] = value + '.to(' + target + ')'
            result.append('tl.store(' + ', '.join(py_args) + ')')
        else:
            result.append(py_call)
    return result


def lean_first_triton_body(text: str) -> str:
    code = strip_lean_comments(text)
    mask = lean_code_mask(text)
    idx = mask.find("triton {")
    if idx < 0:
        return ""
    start = mask.find("{", idx)
    depth = 0
    out = []
    for off, ch in enumerate(mask[start:], start):
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth >= 1:
            out.append(code[off])
    if depth != 0:
        raise ValueError('Unterminated Triton source body')
    body = original_statement_splices("".join(out), text[:idx])
    unknown = unresolved_source_splices(body)
    if unknown:
        raise UnresolvedSourceMacro('Unresolved statement splices: ' + ', '.join(unknown))
    from bench.source_finite_helpers import lean_body_with_finite_helper
    return lean_body_with_finite_helper(body, text)


def lean_triton_bodies(text: str) -> list[str]:
    code = strip_lean_comments(text)
    mask = lean_code_mask(text)
    bodies = []
    pos = 0
    while True:
        idx = mask.find("triton {", pos)
        if idx < 0:
            break
        start = mask.find("{", idx)
        depth = 0
        out = []
        end = start
        for off, ch in enumerate(mask[start:], start):
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = off + 1
                    break
            if depth >= 1:
                out.append(code[off])
        if depth != 0:
            raise ValueError('Unterminated Triton source body')
        body = original_statement_splices("".join(out), text[:idx])
        unknown = unresolved_source_splices(body)
        if unknown:
            raise UnresolvedSourceMacro('Unresolved statement splices: ' + ', '.join(unknown))
        bodies.append(body)
        pos = end
    return bodies


def auditable_source_file(directory):
    """Select only extractable views for rule-specific diagnostics.

    The shell's mandatory source-view extraction gate reports every excluded
    view as a hard failure. This is not an exemption: it avoids diagnosing an
    unresolved ``$store`` as a genuinely missing source store in later rules.
    """
    from bench.port_layout import source_file, source_text

    source = source_file(directory)
    if source is not None:
        try:
            view = source_text(source)
            lean_first_triton_body(view)
            if any(source_audit_metadata(view, field) is not None for field in ('params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes', 'materializations', 'dtype-constexpr')):
                python_files = sorted(source.parent.glob('*.py'))
                if len(python_files) != 1:
                    raise ValueError('Source-audit params requires one Python source file per port')
                from bench.source_materializations import python_source_with_materializations
                python_source_with_materializations(python_files[0].read_text(), view)
                from bench.source_pointer_dtypes import python_source_with_pointer_dtype_profile
                python_source_with_pointer_dtype_profile(python_files[0].read_text(), view)
                from bench.source_dtype_constexpr import python_source_with_dtype_constexpr_profile
                python_source_with_dtype_constexpr_profile(python_files[0].read_text(), view)
                python_source_with_runtime_scalar_profile(python_files[0].read_text(), view)
                source_parameter_map(python_files[0].read_text(), view)
                python_source_with_constexpr_profile(python_files[0].read_text(), view)
                from bench.source_helpers import python_source_with_helpers
                python_source_with_helpers(python_files[0].read_text(), view)
                from bench.source_extern import python_source_with_extern_orders
                python_source_with_extern_orders(python_files[0].read_text(), view)
        except ValueError:
            return None
    return source


def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def tl_calls(text: str, fn: str, *, arguments_only: bool = False,
             normalize_whitespace: bool = False) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(rf"tl\s*\.\s*{fn}\s*\(")
    calls = []
    pos = 0
    while True:
        match = pattern.search(text, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        call = text[match.end():k - 1] if arguments_only else text[match.start():k]
        if normalize_whitespace:
            call = re.sub(r"\s+", " ", call)
        calls.append(call)
        pos = k
    return calls


def split_top_level_args(args_text: str, fn: str | None = None) -> list[str]:
    """Split an argument list, or a complete `tl.<fn>(...)` call.

    Nested parentheses, brackets, and braces do not delimit arguments.
    """
    if fn is not None:
        match = re.search(rf"tl\s*\.\s*{fn}\s*\(", args_text)
        if match is None:
            raise ValueError(f"Expected a tl.{fn} call")
        args_text = args_text[match.end():-1]
    args = []
    cur = []
    depth = 0
    for ch in args_text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
            continue
        cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args


def bare_true_proof(line: str) -> bool:
    """Detect the old bare-True placeholder spelling, not a proved P = True.

    This is a lexical hint; elaboration/trust replay determines proof validity.
    """
    match = re.search(r"True\s*:=\s*by\b", line)
    return bool(match and not re.search(r"(?:=|↔)\s*$", line[:match.start()]))


def python_named_jit_body(text: str, name: str) -> str:
    """Body of the explicitly documented JIT, excluding other JITs and host code."""
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if (isinstance(node, ast.FunctionDef) and node.name == name
                and any(is_triton_jit(d) for d in node.decorator_list)):
            return "\n".join(lines[node.body[0].lineno - 1:node.end_lineno])
    raise ValueError(f"Missing documented JIT: {name}")
