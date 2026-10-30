"""Checked finite-classifier special form for ordinary source comparisons.

This is not a native admission, Python execution, or proof of a backend.
The complete pinned imports/helper/writer and the complete selected Lean
writer are checked before producing either comparison view. Fallback imports
remain in the checked source; only the primary CUDA identity is represented.
The original Python and Lean program files are never changed.
"""
import ast
import copy
import re

from bench import source_isfinite as strict

KIND = 'isfinite-dtype-dispatch'


def _profile(lean_text, profile=None):
    from bench import audit_source as source

    if profile is None:
        declared = source.source_audit_metadata(lean_text, 'helpers')
        if declared is None:
            return None
        profile = source._source_profile_json(declared, 'helpers')
    # Ordinary arithmetic helpers retain their existing implementation.
    if not isinstance(profile, dict) or 'kind' not in profile:
        return None
    if (set(profile) != {'kind', 'python', 'lean'} or profile['kind'] != KIND
            or profile['python'] != [strict.HELPER]):
        raise ValueError('Unsupported Source-audit helper special form')
    source._validate_source_profile_binding(lean_text, profile['lean'], 'finite helper')
    binding = profile['lean']
    if (binding['source'] != 'isfiniteSource%'
            or set(binding['args']) != {'storage'}
            or binding['args']['storage'] not in strict.STORAGES
            or source.source_audit_metadata(lean_text, 'target') != strict.KERNEL):
        raise ValueError('Finite helper requires the complete original source and a concrete floating dtype')
    # Earlier passes must never change the source that the strict checker sees,
    # nor may a later pass remove a branch or alter an already checked operand.
    for field in ('params', 'constexpr', 'runtime', 'orders', 'extern',
                  'pointer-dtypes', 'materializations'):
        if source.source_audit_metadata(lean_text, field) is not None:
            raise ValueError('Finite helper cannot overlap another source comparison profile: ' + field)
    return binding['args']['storage']


def _raw_selected(lean_text, storage):
    """Select the quotation without invoking the comparison view recursively."""
    from bench import audit_source as source
    # These are selectors only. The caller has already validated the real
    # metadata. Omitting helpers here preserves the raw helper declarations.
    selector = ('/-\nSource-audit Lean: isfiniteSource%\n'
                'Source-audit args: storage=' + storage + '\n-/\n')
    return source.select_lean_audit_source(lean_text, selector)


def _call_text(storage):
    if storage == 'tl.float64':
        return 'tl.extra.cuda.libdevice.isfinited(in0)'
    if storage in {'tl.float32', 'tl.bfloat16'}:
        return 'tl.extra.cuda.libdevice.finitef(in0.to(tl.float32))'
    raise ValueError('Unknown source dtype for finite helper dispatch')


def _checked_lean_body(lean_text, storage):
    from bench import audit_source as source
    # This verifies the original eleven source binders, both zero-order proof
    # binders, the constructor and its tail, all helper metadata, and every
    # writer statement/control/expression. It inspects the raw selected view.
    strict.lean_writer(lean_text, storage, source)
    body = source.lean_first_triton_body(_raw_selected(lean_text, storage))
    # The exact checked prefix contains two aliases then one helper. Its
    # removal is confined to this comparison representation. Region dtype
    # declarations and the complete actual writer remain visible to scans.
    region = body.index('tl.region')
    body = body[region:]
    body, count = re.subn(r'(?m)^(\s*out0\s*=\s*)isfinite_func\(in0\)(?=\s*(?:\n|$))',
                          lambda match: match[1] + _call_text(storage), body)
    if count != 2:
        raise ValueError('Finite helper requires one retained call in each writer branch')
    return body


def lean_body_with_finite_helper(body, lean_text):
    """Expand only an explicitly bound, independently checked Lean source."""
    from bench import audit_source as source
    storage = _profile(lean_text)
    if storage is None:
        return body
    raw = source.lean_first_triton_body(_raw_selected(lean_text, storage))
    if body != raw:
        raise ValueError('Finite helper comparison must use the selected original quotation')
    return _checked_lean_body(lean_text, storage)


def python_source_with_finite_helper(python_text, lean_text, profile):
    """Validate the complete source, then change only eight order sites and
    the two recorded helper calls in the *comparison* writer body.

    Both import chains and the original helper remain byte-for-byte intact.
    No branch is discarded and the eleven-argument signature is retained.
    """
    storage = _profile(lean_text, profile)
    if storage is None:
        raise ValueError('Missing finite helper special-form profile')
    _, _, kernel, _ = strict.python_pinned(python_text.encode())
    _checked_lean_body(lean_text, storage)

    # Notation canonicalizes all constructor arguments first; the following
    # order transformation permits only the exact four constructors and four
    # original boundary checks, under the checked input/output zero proofs.
    body = strict.Notation().visit(ast.Module(body=copy.deepcopy(kernel.body), type_ignores=[]))
    orders = strict.RankOneOrderNotation()
    body = orders.visit(body)
    if orders.changed != 4 or orders.boundaries != 4:
        raise ValueError('Finite helper comparison lost a block pointer or boundary check')

    class Calls(ast.NodeTransformer):
        changed = 0

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == strict.HELPER:
                expected = ast.parse('isfinite_func(in0)', mode='eval').body
                if strict.dump(node) != strict.dump(expected):
                    raise ValueError('Finite helper call argument changed')
                self.changed += 1
                return ast.parse(_call_text(storage), mode='eval').body
            return node

    calls = Calls()
    body = calls.visit(body)
    if calls.changed != 2:
        raise ValueError('Finite helper comparison requires both original call sites')
    printed = ast.unparse(ast.fix_missing_locations(body))
    lines = python_text.splitlines(keepends=True)
    replacement = ''.join('    ' + line + '\n' for line in printed.splitlines())
    return ''.join(lines[:kernel.body[0].lineno - 1]) + replacement + ''.join(lines[kernel.end_lineno:])
