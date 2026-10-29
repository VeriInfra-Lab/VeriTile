#!/usr/bin/env bash
set -uo pipefail
PORTS_ROOT="$PWD/bench/tritonbench_g"
failures=0
# A source view must be extractable before its call/statement inventory can be
# compared. Subsequent lexical gates skip these already-failed views rather
# than diagnosing an unexpanded $store as a missing source store. This gate is
# mandatory for all modes and does not exempt unsupported macro profiles.
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import sys
from bench.port_layout import source_files, source_text
from bench.audit_source import (lean_first_triton_body, source_audit_metadata, source_parameter_map,
                               python_source_with_constexpr_profile, python_source_with_runtime_scalar_profile)
from bench.source_helpers import python_source_with_helpers
from bench.source_extern import python_source_with_extern_orders
from bench.source_pointer_dtypes import python_source_with_pointer_dtype_profile
from bench.source_materializations import python_source_with_materializations
from bench.source_dtype_constexpr import python_source_with_dtype_constexpr_profile

failures = []
extracted = 0
for source in source_files(Path(sys.argv[1])):
    try:
        text = source_text(source)
        lean_first_triton_body(text)
        params = source_audit_metadata(text, 'params')
        constexpr = source_audit_metadata(text, 'constexpr')
        helpers = source_audit_metadata(text, 'helpers')
        runtime = source_audit_metadata(text, 'runtime')
        orders = source_audit_metadata(text, 'orders')
        extern = source_audit_metadata(text, 'extern')
        pointer_dtypes = source_audit_metadata(text, 'pointer-dtypes')
        materializations = source_audit_metadata(text, 'materializations')
        dtype_constexpr = source_audit_metadata(text, 'dtype-constexpr')
        if any(value is not None for value in (params, constexpr, helpers, runtime, orders, extern, pointer_dtypes, materializations, dtype_constexpr)):
            python_files = sorted(source.parent.glob('*.py'))
            if len(python_files) != 1:
                raise ValueError('Source-audit params requires one Python source file per port')
            python_source_with_materializations(python_files[0].read_text(), text)
            python_source_with_pointer_dtype_profile(python_files[0].read_text(), text)
            python_source_with_dtype_constexpr_profile(python_files[0].read_text(), text)
            python_source_with_runtime_scalar_profile(python_files[0].read_text(), text)
            source_parameter_map(python_files[0].read_text(), text)
            python_source_with_constexpr_profile(python_files[0].read_text(), text)
            python_source_with_helpers(python_files[0].read_text(), text)
            python_source_with_extern_orders(python_files[0].read_text(), text)
        extracted += 1
        selected = source_audit_metadata(source.read_text(), 'Lean')
        args = source_audit_metadata(source.read_text(), 'args')
        if selected is not None or args is not None:
            print(f"  {source.parent.name}: selected source view {selected or '(first body)'}"
                  + (f" [{args}]" if args else "")
                  + (f"; validated parameter map [{params}]" if params is not None else "")
                  + (f"; validated Python constexpr profile [{constexpr}]" if constexpr is not None else "")
                  + (f"; validated Python helper expansion [{helpers}]" if helpers is not None else "")
                  + (f"; validated Python runtime ABI profile [{runtime}]" if runtime is not None else "")
                  + (f"; validated Python rank-1 order profile [{orders}]" if orders is not None else "")
                  + (f"; validated recorded native extern profile [{extern}]" if extern is not None else "")
                  + (f"; validated native pointer dtype profile [{pointer_dtypes}]" if pointer_dtypes is not None else "")
                  + ("; validated identity materializations and complete computation AST" if materializations is not None else "")
                  + (f"; validated native dtype constexpr profile [{dtype_constexpr}]" if dtype_constexpr is not None else "")
                  + "; remaining source profiles are not certified by this view")
    except ValueError as error:
        failures.append(f"{source}: {error}")
print(f"  {extracted} extractable source views, {len(failures)} unresolved source views; "
      "extraction does not establish all-profile coverage")
if failures:
    print("\n".join(failures))
    sys.exit(1)
PY
then
  printf 'ok source-view extraction scan\n'
else
  printf 'FAIL source-view extraction scan (unresolved views are not source matches)\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text
import sys

from bench.audit_source import (
    lean_first_preamble,
    python_jit_names,
    target_kernel_candidates,
    target_kernel_name,
)

root = Path(sys.argv[1])

failures = []
missing_docs = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    jit_names = python_jit_names(py_file.read_text())
    preamble = lean_first_preamble(lean_file.read_text())
    candidates = target_kernel_candidates(preamble)
    if not candidates:
        missing_docs.append((py_file, lean_file))
        continue
    target = target_kernel_name(preamble, jit_names)
    if target is None:
        failures.append((py_file, lean_file, candidates[-1], sorted(jit_names)))

if missing_docs or failures:
    for py_file, lean_file in missing_docs:
        print(f"{lean_file}: missing preamble target JIT name for {py_file}")
    for py_file, lean_file, target, jit_names in failures:
        print(f"{lean_file}: target JIT `{target}` not found in {py_file}")
        print(f"  python JITs: {jit_names}")
    sys.exit(1)
PY
then
  printf 'ok preamble target JIT scan\n'
else
  printf 'FAIL preamble target JIT scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import ast
import re
import sys

from bench.audit_source import (
    is_triton_jit,
    quoted_kernel_signature,
    lean_first_preamble,
    python_jit_names,
    target_kernel_name,
    source_audit_metadata,
    python_source_parameter_names,
    lean_source_parameter_names,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def python_first_kernel_params(text: str, lean_text: str) -> list[str]:
    target = target_kernel_name(lean_first_preamble(lean_text), python_jit_names(text))
    tree = ast.parse(text)
    lines = text.splitlines()
    preferred = None
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(is_triton_jit(dec) for dec in node.decorator_list):
            continue
        if target is not None and node.name != target:
            continue
        body = "\n".join(lines[node.lineno - 1: node.end_lineno or node.lineno])
        if any(token in body for token in ("tl.program_id", "tl.load", "tl.store")):
            if node.name == "_attn_fwd":
                return [arg.arg for arg in node.args.args]
            if preferred is None:
                preferred = [arg.arg for arg in node.args.args]
    if preferred is not None:
        return preferred
    return []

def lean_first_kernel_signature(text: str) -> str:
    quoted = quoted_kernel_signature(text)
    if quoted:
        return quoted
    from bench.audit_source import lean_code_mask
    code = lean_code_mask(text)
    index = code.find("triton {")
    if index < 0:
        return ""
    definitions = list(re.finditer(r"^(?:(?:private|noncomputable)\s+)?def\s+", code[:index], re.M))
    if not definitions:
        return ""
    signature = code[definitions[-1].start():index]
    return signature if "ComputeKernel" in signature else ""


def lean_binder_groups(signature: str) -> list[str]:
    groups = []
    pairs = {"(": ")", "{": "}", "[": "]"}
    i = 0
    while i < len(signature):
        if signature[i] not in pairs:
            i += 1
            continue
        open_ch = signature[i]
        close_ch = pairs[open_ch]
        depth = 1
        j = i + 1
        while j < len(signature) and depth > 0:
            if signature[j] == open_ch:
                depth += 1
            elif signature[j] == close_ch:
                depth -= 1
            j += 1
        if depth == 0:
            groups.append(signature[i + 1:j - 1])
        i = j
    return groups

def lean_first_kernel_params(text: str) -> list[str]:
    params = []
    for group in lean_binder_groups(lean_first_kernel_signature(text)):
        if ":" not in group:
            continue
        left = group.split(":", 1)[0].strip()
        if not left or left.startswith("h"):
            continue
        params.extend(
            name.lstrip("_")
            for name in re.split(r"\s+", left)
            if name and name != "_"
        )
    return params

def normalize_param(name: str) -> str:
    name = name.lstrip("_").replace("_", "").lower()
    for suffix in ("pointer", "ptr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    if any(source_audit_metadata(lean_text, field) is not None for field in ('params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern', 'materializations', 'dtype-constexpr')):
        py_params = python_source_parameter_names(py_file.read_text(), lean_text)
        lean_params = lean_source_parameter_names(lean_text)
        missing = [param for param in py_params if param not in set(lean_params)]
    else:
        py_params = python_first_kernel_params(py_file.read_text(), lean_text)
        lean_params = lean_first_kernel_params(lean_text)
        lean_param_set = {normalize_param(param) for param in lean_params}
        missing = [param for param in py_params if normalize_param(param) not in lean_param_set]
    if missing:
        failures.append((py_file, lean_file, missing, py_params, lean_params))

if failures:
    for py_file, lean_file, missing, py_params, lean_params in failures:
        print(f"{py_file} -> {lean_file}: Python kernel params missing from Lean")
        print(f"  missing: {missing}")
        print(f"  python:  {py_params}")
        print(f"  lean:    {lean_params}")
    sys.exit(1)
PY
then
  printf 'ok kernel parameter presence scan\n'
else
  printf 'FAIL kernel parameter presence scan\n'
  failures=$((failures + 1))
fi

# Compare fp32 casts in the documented target, not unrelated helper/host code.
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys
from bench.port_layout import source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
from bench.audit_source import lean_first_preamble, lean_first_triton_body, python_jit_names, python_named_jit_body, target_kernel_name

failures = []
for py in sorted(Path(sys.argv[1]).glob('*/*.py')):
    lean = source_file(py.parent)
    if lean is None:
        continue
    text = source_text(lean)
    preamble = lean_first_preamble(text)
    python = py.read_text()
    target = target_kernel_name(preamble, python_jit_names(python))
    if target is None:
        # The preceding target-JIT gate reports this as an error.
        continue
    if not re.search(r"\.to\(tl\.float32\)", python_named_jit_body(python, target)):
        continue
    if re.search(r"\.to\(tl\.float32\)", lean_first_triton_body(text)):
        continue
    if 'Translation-surface blocker:' in preamble:
        continue
    if re.search(r'quant_policy = 0|unquantized path|quant_policy.*4/8', text):
        continue
    failures.append(f"  {py} -> {lean} (target {target})")
if failures:
    print("\n".join(failures))
    sys.exit(1)
PY
then
  printf 'ok documented float32 cast coverage scan\n'
else
  printf 'FAIL .to(tl.float32) missing from Lean without documented slice/scope\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    strip_comments,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

from bench.audit_source import cast_sequence, store_calls_with_explicit_destination_casts, tl_calls

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    from bench.audit_source import resolve_source_dtype_attributes
    lean_body = lean_first_triton_body(lean_text)
    py_body = resolve_source_dtype_attributes(python_audit_kernel_body(py_file.read_text(), lean_text), lean_body)
    for before, after in zip(tl_calls(py_body, 'store'),
                             store_calls_with_explicit_destination_casts(py_body, lean_body)):
        py_body = py_body.replace(before, after, 1)
    py_casts = cast_sequence(py_body)
    lean_casts = cast_sequence(resolve_source_dtype_attributes(lean_body, lean_body))
    if py_casts != lean_casts:
        failures.append((py_file, lean_file, py_casts, lean_casts))

if failures:
    for py_file, lean_file, py_casts, lean_casts in failures:
        print(f"{py_file} -> {lean_file}: .to(...) cast sequence mismatch")
        print(f"  python: {py_casts}")
        print(f"  lean:   {lean_casts}")
    sys.exit(1)
PY
then
  printf 'ok .to(...) cast sequence scan\n'
else
  printf 'FAIL .to(...) cast sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    split_top_level_args,
    tl_calls,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def mask_presence(body: str, fn: str) -> list[bool]:
    out = []
    for call in tl_calls(body, fn, normalize_whitespace=True):
        args = split_top_level_args(call, fn)
        has_kw = any(re.match(r"^mask\s*=", arg) for arg in args)
        positional = (
            (fn == "load" and len(args) >= 2 and "=" not in args[1])
            or (fn == "store" and len(args) >= 3 and "=" not in args[2])
        )
        out.append(has_kw or positional)
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_audit_kernel_body(py_file.read_text(), lean_text)
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_masks = mask_presence(py_body, fn)
        lean_masks = mask_presence(lean_body, fn)
        if py_masks != lean_masks:
            failures.append((py_file, lean_file, fn, py_masks, lean_masks))

if failures:
    for py_file, lean_file, fn, py_masks, lean_masks in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} mask presence mismatch")
        print(f"  python: {py_masks}")
        print(f"  lean:   {lean_masks}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store mask presence scan\n'
else
  printf 'FAIL tl.load/store mask presence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    normalize_shape_ctor_call,
    resolve_source_dtype_attributes,
    tl_calls,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    lean_body = lean_first_triton_body(lean_text)
    py_body = resolve_source_dtype_attributes(python_audit_kernel_body(py_file.read_text(), lean_text), lean_body)
    resolved_lean = resolve_source_dtype_attributes(lean_body, lean_body)
    for fn in ("zeros", "full"):
        py_calls = [normalize_shape_ctor_call(call) for call in tl_calls(py_body, fn)]
        lean_calls = [normalize_shape_ctor_call(call) for call in tl_calls(resolved_lean, fn)]
        if py_calls != lean_calls:
            failures.append((py_file, lean_file, fn, py_calls, lean_calls))

if failures:
    for py_file, lean_file, fn, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} shape/dtype call mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.zeros/full shape dtype scan\n'
else
  printf 'FAIL tl.zeros/full shape dtype scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    strip_comments,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def tl_arange_calls(text: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(r"tl\s*\.\s*arange\s*\(")
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
        call = re.sub(r"\$\(([^()]+)\)", r"\1", text[match.start():k])
        call = re.sub(r"\s+", "", call)
        calls.append(call.replace("_", "").lower())
        pos = k
    return calls

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_calls = tl_arange_calls(python_audit_kernel_body(py_file.read_text(), lean_text))
    lean_calls = tl_arange_calls(lean_first_triton_body(lean_text))
    if py_calls != lean_calls:
        failures.append((py_file, lean_file, py_calls, lean_calls))

if failures:
    for py_file, lean_file, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.arange call mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.arange call scan\n'
else
  printf 'FAIL tl.arange call scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    strip_comments,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def program_id_calls(text: str) -> list[str]:
    text = strip_comments(text)
    pattern = re.compile(r"tl\s*\.\s*program_id\s*\(")
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
        call = text[match.end():k - 1]
        call = re.sub(r"\$\(([^()]+)\)", r"\1", call)
        call = re.sub(r"\baxis\s*=", "", call)
        calls.append(re.sub(r"\s+", "", call))
        pos = k
    return calls

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_calls = program_id_calls(python_audit_kernel_body(py_file.read_text(), lean_text))
    lean_calls = program_id_calls(lean_first_triton_body(lean_text))
    if py_calls != lean_calls:
        failures.append((py_file, lean_file, py_calls, lean_calls))

if failures:
    for py_file, lean_file, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.program_id call mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.program_id call scan\n'
else
  printf 'FAIL tl.program_id call scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    split_top_level_args,
    tl_calls,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def kwarg_sequence(body: str, fn: str) -> list[list[str]]:
    out = []
    for call_args in tl_calls(body, fn, arguments_only=True):
        kwargs = []
        for arg in split_top_level_args(call_args):
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", arg)
            if match:
                kwargs.append(match.group(1))
        out.append(kwargs)
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_audit_kernel_body(py_file.read_text(), lean_text)
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_kwargs = kwarg_sequence(py_body, fn)
        lean_kwargs = kwarg_sequence(lean_body, fn)
        if py_kwargs != lean_kwargs:
            failures.append((py_file, lean_file, fn, py_kwargs, lean_kwargs))

if failures:
    for py_file, lean_file, fn, py_kwargs, lean_kwargs in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} kwarg sequence mismatch")
        print(f"  python: {py_kwargs}")
        print(f"  lean:   {lean_kwargs}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store kwarg sequence scan\n'
else
  printf 'FAIL tl.load/store kwarg sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    split_top_level_args,
    tl_calls,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_audit_kernel_body(py_file.read_text(), lean_text)
    lean_body = lean_first_triton_body(lean_text)
    for fn in ("load", "store"):
        py_arg_counts = [len(split_top_level_args(call)) for call in tl_calls(py_body, fn, arguments_only=True)]
        lean_arg_counts = [len(split_top_level_args(call)) for call in tl_calls(lean_body, fn, arguments_only=True)]
        if py_arg_counts != lean_arg_counts:
            failures.append((py_file, lean_file, fn, py_arg_counts, lean_arg_counts))

if failures:
    for py_file, lean_file, fn, py_arg_counts, lean_arg_counts in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} argument count mismatch")
        print(f"  python: {py_arg_counts}")
        print(f"  lean:   {lean_arg_counts}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store argument count scan\n'
else
  printf 'FAIL tl.load/store argument count scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    tl_calls,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def normalize_call(call: str) -> str:
    # Explicit Lean types on scalar parameter splices do not change their values.
    call = re.sub(r"\$\(\(([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:ℝ|Nat|Int)\)\)", r"\1", call)
    # A quotation category annotates the splice, not the selected dtype.
    call = re.sub(r"(\$[A-Za-z_][A-Za-z0-9_]*)\s*:\s*tritonDType\b", r"\1", call)
    call = call.replace("\\", "")
    call = re.sub(
        r"\$\(\(([^:()]+)\s*:\s*Region\s*(?:\.[A-Za-z0-9_]+)?\)\)",
        r"\1",
        call,
    )
    call = re.sub(r"\$\(([^()]+)\)", r"\1", call)
    call = call.replace("'", '"')
    call = re.sub(r"(?<![A-Za-z0-9_])0\.(?![0-9])", "0.0", call)
    call = re.sub(r"\(([^()]+)\)\.to\(", r"\1.to(", call)
    call = re.sub(r"mask=\(([^()]+)\)", r"mask=\1", call)
    for dtype in ("float16", "float32", "float64", "int32", "int64", "uint32", "uint64"):
        call = call.replace(f"tl.{dtype}", dtype)
    call = call.replace("True", "true").replace("False", "false")
    call = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", lambda m: m.group(0).replace("_", "").lower(), call)
    call = re.sub(r"\b([A-Za-z0-9]+)(?:ptr|pointer)\b", r"\1", call)
    call = re.sub(r",\s*\)", ")", call)
    call = re.sub(r"\s+", "", call)
    call = re.sub(r"boundarycheck=\(\[([0-9,]+)\]:listnat\)", r"boundarycheck=(\1)", call)
    # The source ports are hand-written Lean syntax, while the Python side
    # comes through AST-preserving text. Treat redundant wrapping parentheses
    # around a whole `mask=(lhs & rhs)` expression as formatting only.
    call = re.sub(r"mask=\(\(([^()]+)\)&(.+)\)\)", r"mask=(\1)&\2))", call)
    call = re.sub(r"mask=\(([^()]+)\)&", r"mask=\1&", call)
    call = re.sub(r"(\[none,:\]\)\))\)+", r"\1", call)
    return call

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_body = python_audit_kernel_body(py_file.read_text(), lean_text)
    lean_body = lean_first_triton_body(lean_text)
    from bench.audit_source import resolve_source_dtype_attributes
    py_body = resolve_source_dtype_attributes(py_body, lean_body)
    # The source may retain the same pointer-element spelling in both languages.
    # Normalize the checked Lean provenance on both sides, as the cast gate does.
    resolved_lean = resolve_source_dtype_attributes(lean_body, lean_body)
    for fn in ("load", "store"):
        from bench.audit_source import store_calls_with_explicit_destination_casts
        source_calls = (store_calls_with_explicit_destination_casts(py_body, lean_body)
                        if fn == "store" else tl_calls(py_body, fn))
        py_calls = [normalize_call(call) for call in source_calls]
        lean_calls = [normalize_call(call) for call in tl_calls(resolved_lean, fn)]
        if py_calls != lean_calls:
            failures.append((py_file, lean_file, fn, py_calls, lean_calls))

if failures:
    for py_file, lean_file, fn, py_calls, lean_calls in failures:
        print(f"{py_file} -> {lean_file}: tl.{fn} call expression mismatch")
        print(f"  python: {py_calls}")
        print(f"  lean:   {lean_calls}")
    sys.exit(1)
PY
then
  printf 'ok tl.load/store call expression scan\n'
else
  printf 'FAIL tl.load/store call expression scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    split_top_level_args,
    strip_comments,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def tl_load_calls(text: str) -> list[str]:
    text = strip_comments(text)
    calls = []
    i = 0
    needle = "tl.load("
    while True:
        start = text.find(needle, i)
        if start < 0:
            break
        k = start + len(needle)
        depth = 1
        while k < len(text) and depth > 0:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        calls.append(re.sub(r"\s+", " ", text[start:k]))
        i = k
    return calls

def normalize_other(value: str) -> str:
    value = re.sub(r"\$\(\(([^:()]+)\s*:[^)]+\)\)", r"\1", value.strip())
    value = re.sub(r"\$\(([^()]+)\)", r"\1", value)
    value = value.replace("'", '"').strip()
    if value in ('-inf', 'float("-inf"', '-float("inf"'):
        return "-inf"
    if value in ("0.", "0.0", "0.00"):
        return "0.0"
    return value

def other_values(body: str) -> list[str]:
    out = []
    for call in tl_load_calls(body):
        for arg in split_top_level_args(call, "load"):
            if re.match(r"^other\s*=", arg):
                out.append(normalize_other(arg.split("=", 1)[1]))
                break
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_others = other_values(python_audit_kernel_body(py_file.read_text(), lean_text))
    lean_others = other_values(lean_first_triton_body(lean_text))
    if py_others != lean_others:
        failures.append((py_file, lean_file, py_others, lean_others))

if failures:
    for py_file, lean_file, py_others, lean_others in failures:
        print(f"{py_file} -> {lean_file}: tl.load other= mismatch")
        print(f"  python: {py_others}")
        print(f"  lean:   {lean_others}")
    sys.exit(1)
PY
then
  printf 'ok tl.load other literal scan\n'
else
  printf 'FAIL tl.load other literal scan\n'
  failures=$((failures + 1))
fi

if rg -n 'tl\.load\([^\n]*dtype\s*=' "${PORTS_ROOT}" -g '*.lean'; then
  printf 'FAIL Lean tl.load dtype annotations found; review_criteria.md only permits them when present upstream\n'
  failures=$((failures + 1))
else
  printf 'ok no Lean-only tl.load dtype annotations\n'
fi

# `keep_dims`-style reductions are a review_criteria.md violation in the
# TRANSCRIBED KERNEL SURFACE only. The scan is scoped to `triton { … }` bodies:
# `Op.reduceSum` / `Op.reduceMax` carry a `keepDims : Bool` argument, so helper
# lemmas that quantify over that flag mention the identifier legitimately, and
# a whole-file grep reported them as violations.
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

def kernel_spans(text: str):
    """(start, end) offsets of every `triton { … }` body, by brace depth."""
    from bench.audit_source import lean_code_mask
    text = lean_code_mask(text)
    spans, i = [], 0
    while True:
        i = text.find("triton {", i)
        if i == -1:
            return spans
        open_i = text.index("{", i)
        depth, k = 0, open_i
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        spans.append((open_i, k))
        i = k + 1

pattern = re.compile(r"keep_dims\s*=\s*true|keepDims|keep_dims")
hits = []
for lean_file in source_files(Path(sys.argv[1])):
    if source_file(lean_file.parent) is None:
        continue
    text = source_text(lean_file)
    spans = kernel_spans(text)
    for m in pattern.finditer(text):
        if any(a <= m.start() <= b for a, b in spans):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{lean_file}:{line}: {m.group(0)}")

if hits:
    for hit in hits:
        print(hit)
    sys.exit(1)
PY
then
  printf 'ok no keep_dims reduction substitutions\n'
else
  printf 'FAIL keep_dims-style reduction found in a kernel body; review_criteria.md requires source-shape-preserving syntax\n'
  failures=$((failures + 1))
fi

# Python `+=` coverage uses the explicitly selected JIT and Lean source body,
# consistently with the other source-view gates. Other JITs need their own
# selected views; comments or sibling kernels cannot supply missing updates.
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import sys

from bench.audit_source import (
    python_audit_kernel_body,
    lean_first_triton_body,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    has_iadd = any(
        "+=" in line.split("#", 1)[0]
        for line in python_audit_kernel_body(py_file.read_text(), lean_text).splitlines()
    )
    if not has_iadd:
        continue
    if "+=" in lean_first_triton_body(lean_text):
        continue
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    failures.append(f"{py_file} -> {lean_file}")

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)
PY
then
  printf 'ok documented += coverage scan\n'
else
  printf 'FAIL Python += missing from Lean without documented slice/scope:\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def norm(name: str) -> str:
    name = name.lower()
    for suffix in ("_pointer", "_ptrs", "_ptr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    py_lhs = {
        norm(match.group(1))
        for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+=",
            python_audit_kernel_body(py_file.read_text(), lean_text, include_block_pointers=True), re.M)
    }
    if not py_lhs:
        continue
    lean_text = source_text(lean_file)
    lean_lhs = {
        norm(match.group(1))
        for match in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\+=",
            lean_first_triton_body(lean_text), re.M)
    }
    missing = sorted(py_lhs - lean_lhs)
    scope_text = lean_first_preamble(lean_text).lower()
    if missing and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, missing))

if failures:
    for py_file, lean_file, missing in failures:
        print(f"{py_file} -> {lean_file}: missing += lhs {', '.join(missing)}")
    sys.exit(1)
PY
then
  printf 'ok normalized += lhs coverage scan\n'
else
  printf 'FAIL normalized += lhs coverage scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
import re
import sys
from bench.port_layout import source_files, source_text
from bench.audit_source import auditable_source_file, lean_first_triton_body, python_audit_kernel_body

failures = []
for source in source_files(Path(sys.argv[1])):
    if auditable_source_file(source.parent) is None:
        continue
    python_files = sorted(source.parent.glob('*.py'))
    if not python_files:
        continue
    view = source_text(source)
    python_body = python_audit_kernel_body(python_files[0].read_text(), view)
    if not re.search(r'\brsqrt\b', python_body):
        continue
    if not re.search(r'\brsqrt\b', lean_first_triton_body(view)):
        failures.append(f"{python_files[0]} -> {source}: missing rsqrt in selected source view")
if failures:
    print("\n".join(failures))
    sys.exit(1)
PY
then
  printf 'ok rsqrt preservation scan\n'
else
  printf 'FAIL Python rsqrt missing from selected Lean source\n'
  failures=$((failures + 1))
fi

# Lean-only tl.where scan. Operates on comment-stripped Lean code so prose
# mentions of `tl.where` in docstrings cannot trip it (same lesson as the
# placeholder scan), and accepts the standard preamble
# `Translation-surface blocker:` marker exemption like every other textual
# py<->lean scan (a documented proof-slice guard is a declared deviation,
# not an unannotated extra statement).
if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import sys

from bench.audit_source import (
    strip_lean_comments,
)

root = Path(sys.argv[1])

blocker_marker = "translation-surface blocker:"

failures = []
for lean_file in source_files(root):
    if source_file(lean_file.parent) is None:
        continue
    lean_text = source_text(lean_file)
    code = strip_lean_comments(lean_text)
    if "tl.where" not in code:
        continue
    py_files = sorted(lean_file.parent.glob("*.py"))
    if not py_files:
        continue
    py_file = py_files[0]
    if "tl.where" in py_file.read_text():
        continue
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    failures.append(f"{lean_file} -> {py_file}")

if failures:
    print("FAIL Lean-only tl.where found:")
    for entry in failures:
        print(f"  {entry}")
    sys.exit(1)
PY
then
  printf 'ok no Lean-only tl.where statements\n'
else
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    strip_comments,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"
call_re = re.compile(r"\btl(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s*\(")
ignored = {
    "tl.constexpr",
    "tl.float32",
    "tl.float64",
    "tl.int32",
    "tl.int64",
    "tl.uint32",
    "tl.uint64",
    "tl.tensor",
    "tl.for",
}

def calls(text: str) -> set[str]:
    text = strip_comments(text)
    return {
        match.group(0).split("(", 1)[0].strip()
        for match in call_re.finditer(text)
        if match.group(0).split("(", 1)[0].strip() not in ignored
    }

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    py_calls = calls(python_audit_kernel_body(py_file.read_text(), lean_text, include_block_pointers=True))
    lean_text = source_text(lean_file)
    lean_calls = calls(lean_first_triton_body(lean_text))
    missing = sorted(py_calls - lean_calls)
    extra = sorted(lean_calls - py_calls)
    scope_text = lean_first_preamble(lean_text).lower()
    if (missing or extra) and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, missing, extra))

if failures:
    for py_file, lean_file, missing, extra in failures:
        if missing:
            print(f"{py_file} -> {lean_file}: missing tl calls {', '.join(missing)}")
        if extra:
            print(f"{lean_file} -> {py_file}: extra tl calls {', '.join(extra)}")
    sys.exit(1)
PY
then
  printf 'ok tl.* call surface scan\n'
else
  printf 'FAIL tl.* call surface scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_preamble,
    lean_first_triton_body,
    python_audit_kernel_body,
    python_jit_names_text as python_jit_names,
    target_kernel_name,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def strip_python_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

def python_control_counts(text: str, lean_text: str) -> tuple[int, int, int]:
    body = strip_python_comments(python_audit_kernel_body(text, lean_text))
    return (
        len(re.findall(r"^\s*for\s+", body, re.M)),
        len(re.findall(r"^\s*while\s+", body, re.M)),
        len(re.findall(r"^\s*if\s+", body, re.M)),
    )

def lean_control_counts(text: str) -> tuple[int, int, int]:
    body = lean_first_triton_body(text)
    return (
        len(re.findall(r"^\s*(for\s+|tl\.for\s+)", body, re.M)),
        len(re.findall(r"^\s*while\s+", body, re.M)),
        len(re.findall(r"^\s*(if\s+|tl\.if\s+)", body, re.M)),
    )

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    py_text = py_file.read_text()
    target = target_kernel_name(lean_first_preamble(lean_text), python_jit_names(py_text))
    py_counts = python_control_counts(py_text, lean_text)
    lean_counts = lean_control_counts(lean_text)
    scope_text = lean_first_preamble(lean_text).lower()
    if py_counts != lean_counts and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, py_counts, lean_counts))

if failures:
    for py_file, lean_file, py_counts, lean_counts in failures:
        print(
            f"{py_file} -> {lean_file}: control-flow count mismatch "
            f"python(for,while,if)={py_counts} lean(for,while,if)={lean_counts}"
        )
    sys.exit(1)
PY
then
  printf 'ok kernel control-flow surface scan\n'
else
  printf 'FAIL kernel control-flow surface scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    python_jit_kernel_bodies,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"
call_re = re.compile(r"\btl(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s*\(")
ignored = {
    "tl.constexpr",
    "tl.float32",
    "tl.float64",
    "tl.int32",
    "tl.int64",
    "tl.uint32",
    "tl.uint64",
    "tl.tensor",
    "tl.for",
}

def tl_call_sequence(text: str) -> list[str]:
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    out = []
    for match in call_re.finditer(text):
        call = match.group(0).split("(", 1)[0].strip()
        if call not in ignored:
            out.append(call)
    return out

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    py_bodies = [python_audit_kernel_body(py_file.read_text(), lean_text)]
    lean_bodies = [lean_first_triton_body(lean_text)]
    py_sequences = [tl_call_sequence(py_bodies[0])] if py_bodies else []
    lean_sequences = [tl_call_sequence(lean_bodies[0])] if lean_bodies else []
    scope_text = lean_first_preamble(lean_text).lower()
    if py_sequences != lean_sequences and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, py_sequences, lean_sequences))

if failures:
    for py_file, lean_file, py_sequences, lean_sequences in failures:
        print(f"{py_file} -> {lean_file}: tl call sequence mismatch")
        print(f"  python: {py_sequences}")
        print(f"  lean:   {lean_sequences}")
    sys.exit(1)
PY
then
  printf 'ok tl.* call sequence scan\n'
else
  printf 'FAIL tl.* call sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    python_jit_kernel_bodies,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"

def norm_name(name: str) -> str:
    name = name.strip().split(":", 1)[0].strip().lower()
    for suffix in ("_pointer", "_ptrs", "_ptr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

def split_lhs(lhs: str) -> list[str]:
    parts = [part.strip() for part in lhs.split(",") if part.strip()]
    if not parts:
        return []
    out = []
    for part in parts:
        if part == "_":
            continue
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\s*:[^=]+)?$", part):
            return []
        out.append(norm_name(part))
    return out

def top_assign_lhs(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith(("if ", "elif ")) and ":" in stripped and not stripped.endswith(":"):
        return top_assign_lhs(stripped.split(":", 1)[1])
    if not stripped or stripped.startswith(
        ("tl.store", "tl.atomic", "for ", "if ", "else", "tl.for", "tl.if")
    ):
        return []
    for op in ("+=", "-=", "*=", ":=", "="):
        idx = stripped.find(op)
        if idx <= 0:
            continue
        if op == "=" and (
            stripped[idx - 1 : idx + 1] in ("<=", ">=", "!=", "==") or
            stripped[idx : idx + 2] == "=="
        ):
            continue
        return split_lhs(stripped[:idx])
    return []

def lhs_sequence(text: str) -> list[str]:
    out = []
    depth = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        if depth == 0:
            out.extend(top_assign_lhs(line))
        depth += line.count("(") + line.count("[") - line.count(")") - line.count("]")
        depth = max(depth, 0)
    return out

def normalize_guard_lhs_sequence(port: str, seq: list[str]) -> list[str]:
    if port == "bmm_chunk_fwd":
        # Python returns before any side effects for inactive causal blocks.
        # The Lean surface uses the equivalent guard-normal form:
        #   active_block = true
        #   if IS_CAUSAL { active_block = ... }
        #   if active_block { body }
        return [name for name in seq if name != "active_block"]
    return seq

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    py_bodies = [python_audit_kernel_body(py_file.read_text(), lean_text)]
    lean_bodies = [lean_first_triton_body(lean_text)]
    port = py_file.parent.name
    py_sequences = [normalize_guard_lhs_sequence(port, lhs_sequence(py_bodies[0]))] if py_bodies else []
    from bench.audit_source import strip_region_declarations
    lean_sequences = [normalize_guard_lhs_sequence(port, lhs_sequence(strip_region_declarations(lean_bodies[0])))] if lean_bodies else []
    scope_text = lean_first_preamble(lean_text).lower()
    if py_sequences != lean_sequences and not blocker_marker in scope_text:
        failures.append((py_file, lean_file, py_sequences, lean_sequences))

if failures:
    for py_file, lean_file, py_sequences, lean_sequences in failures:
        print(f"{py_file} -> {lean_file}: statement lhs sequence mismatch")
        print(f"  python: {py_sequences}")
        print(f"  lean:   {lean_sequences}")
    sys.exit(1)
PY
then
  printf 'ok statement lhs sequence scan\n'
else
  printf 'FAIL statement lhs sequence scan\n'
  failures=$((failures + 1))
fi

if python3 - "${PORTS_ROOT}" <<'PY'
from pathlib import Path
from bench.port_layout import source_files, source_file, combined_text, source_text
from bench.audit_source import auditable_source_file as source_file, lean_first_preamble
import re
import sys

from bench.audit_source import (
    lean_first_triton_body,
    python_audit_kernel_body,
    python_jit_kernel_bodies,
    split_top_level_args,
)

root = Path(sys.argv[1])
blocker_marker = "translation-surface blocker:"
reduce_call_head_re = re.compile(r"\btl\.(sum|max)\s*\(")
axis_kw_re = re.compile(r"(?:^|,)\s*axis\s*=\s*([0-9]+)\s*(?:,|$)")

def logical_lines(text: str) -> list[str]:
    out = []
    cur = ""
    depth = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cur = (cur + " " + line).strip() if cur else line
        depth += (
            line.count("(") + line.count("[") + line.count("{")
            - line.count(")") - line.count("]") - line.count("}")
        )
        continues = bool(re.search(r"(\+|-|\*|/|&|\||,)$", line))
        if depth <= 0 and not continues:
            out.append(re.sub(r"\s+", " ", cur).strip())
            cur = ""
            depth = 0
    if cur:
        out.append(re.sub(r"\s+", " ", cur).strip())
    return out

def tl_reduce_calls(line: str) -> list[tuple[str, str]]:
    calls = []
    pos = 0
    while True:
        match = reduce_call_head_re.search(line, pos)
        if not match:
            break
        k = match.end()
        depth = 1
        while k < len(line) and depth > 0:
            if line[k] == "(":
                depth += 1
            elif line[k] == ")":
                depth -= 1
            k += 1
        if depth == 0:
            calls.append((match.group(1), line[match.end():k - 1]))
        pos = max(k, match.end())
    return calls

def reduce_axis_styles(text: str) -> list[tuple[str, str, str]]:
    styles = []
    for line in logical_lines(text):
        for fn, args in tl_reduce_calls(line):
            axis_kw = axis_kw_re.search(args)
            if axis_kw:
                styles.append((fn, axis_kw.group(1), "kw"))
                continue
            parts = split_top_level_args(args)
            if len(parts) >= 2 and re.fullmatch(r"[0-9]+", parts[1]):
                styles.append((fn, parts[1], "pos"))
    return styles

failures = []
for py_file in sorted(root.glob("*/*.py")):
    source = source_file(py_file.parent)
    lean_files = [source] if source else []
    if not lean_files:
        continue
    lean_file = lean_files[0]
    lean_text = source_text(lean_file)
    scope_text = lean_first_preamble(lean_text).lower()
    if blocker_marker in scope_text:
        continue
    py_bodies = [python_audit_kernel_body(py_file.read_text(), lean_text)]
    lean_bodies = [lean_first_triton_body(lean_text)]
    py_styles = reduce_axis_styles(py_bodies[0]) if py_bodies else []
    lean_styles = reduce_axis_styles(lean_bodies[0]) if lean_bodies else []
    if py_styles != lean_styles:
        failures.append((py_file, lean_file, py_styles, lean_styles))

if failures:
    for py_file, lean_file, py_styles, lean_styles in failures:
        print(f"{py_file} -> {lean_file}: reduce axis style mismatch")
        print(f"  python: {py_styles}")
        print(f"  lean:   {lean_styles}")
    sys.exit(1)
PY
then
  printf 'ok reduce axis style scan\n'
else
  printf 'FAIL reduce axis style scan\n'
  failures=$((failures + 1))
fi


if [ "$failures" -gt 0 ]; then exit 1; fi
printf "Explicit legacy source comparisons passed\n"
