"""Source-selection and failure-propagation regressions for structural audits."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source


class SourceSelectionTests(unittest.TestCase):
    def test_explicit_source_selector_avoids_legacy_and_other_python_jits(self):
        lean = '''/-!
Source-audit target: actual_jit
Source-audit Lean: typed%
-/
def legacy : ComputeKernel := triton { x = tl.load(X) }
syntax "typed%" ident : term
macro_rules
  | `(typed% $version:ident) => do
    let update : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| x = tl.load(X).to(tl.float32))
      else `(tritonStmt| x = tl.load(X).to(tl.float16))
    `(fun (X : RegionName) => ComputeKernel.mk [X] [] (triton { $update }).surfaceBody)
'''
        selected = source.select_lean_audit_source(lean)
        self.assertNotIn('def legacy', selected)
        body = source.lean_first_triton_body(selected)
        self.assertEqual(source.cast_sequence(body), ['float32'])
        python = '''@triton.jit
def helper(X):
    x = tl.load(X).to(tl.float16)
@triton.jit
def actual_jit(X):
    x = tl.load(X).to(tl.float32)
'''
        self.assertEqual(source.cast_sequence(source.python_audit_kernel_body(python, selected)),
                         ['float32'])
        with self.assertRaisesRegex(ValueError, 'not a Python JIT'):
            source.python_audit_kernel_body(python, selected.replace('target: actual_jit', 'target: typo'))

    def test_source_selectors_fail_closed(self):
        body = 'def actual : ComputeKernel := triton { x = tl.load(X) }\n'
        for header, diagnostic in [
            ('Source-audit Lean: typo', 'found 0'),
            ('Source-audit Lean: actual\nSource-audit Lean: actual', 'Duplicate'),
            ('Source-audit Lean:   ', 'Empty'),
        ]:
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, diagnostic):
                source.select_lean_audit_source('/-!\n' + header + '\n-/\n' + body)
        with self.assertRaisesRegex(ValueError, 'found 2'):
            source.select_lean_audit_source('/-!\nSource-audit Lean: actual\n-/\n' + body * 2)
        with self.assertRaisesRegex(ValueError, 'no literal Triton body'):
            source.select_lean_audit_source('/-!\nSource-audit Lean: wrapper\n-/\n'
                                            'def wrapper := actual\n' + body)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            source.target_kernel_candidates('Source-audit target: a\nSource-audit target: b\n')
        for key in ('target', 'args'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Empty'):
                source.source_audit_metadata('Source-audit ' + key + ':  \n', key)
        for command in ('specification later', 'example', 'opaque later', 'instance'):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'no literal Triton body'):
                source.select_lean_audit_source('/-!\nSource-audit Lean: wrapper\n-/\n'
                                                'def wrapper := actual\n'
                                                f'{command} := triton {{ tl.store(Y, wrong) }}\n')

    def test_macro_arguments_and_array_splices_keep_nested_stores_and_masks(self):
        template = '''/-!
Source-audit args: version=original, single=1, storage=tl.float32
-/
macro_rules
  | `(source%[$storage:tritonDType, $version:ident] $single:num) => do
    let original := version.getId == `original || version.getId == `actualOriginal
    let offset : TSyntax `tritonStmt ← if original then
      `(tritonStmt| i = pid * $(B)) else `(tritonStmt| i = $(B) * pid)
    let tail : Array (TSyntax `tritonStmt) := #[offset,
      ← `(tritonStmt| x = tl.load(X + i, mask=active, other=0)),
      ← `(tritonStmt| tl.store(Y + i, (x + f(a, b)).to($storage), mask=active))]
    let branch : Array (TSyntax `tritonStmt) ← if single.getNat = 1 then
      pure (#[← `(tritonStmt| pid = tl.program_id(0))] ++ tail)
      else pure #[← `(tritonStmt| for j in range(0, $(T)) { $tail:tritonStmt* })]
    `(fun (X Y : RegionName) (B T : Nat) => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = $storage, Y = $storage
      $branch:tritonStmt*
    }).surfaceBody)
'''
        body = source.lean_first_triton_body(template)
        self.assertIn('i = pid * $(B)', body)
        self.assertNotIn('$branch', body)
        self.assertNotIn('for j', body)
        self.assertEqual(source.cast_sequence(body), ['float32'])
        self.assertEqual(source.tl_calls(body, 'store'),
                         ['tl.store(Y + i, (x + f(a, b)).to(tl.float32), mask=active)'])
        multiple = source.lean_first_triton_body(template.replace('single=1', 'single=0'))
        self.assertIn('for j', multiple)
        self.assertEqual(source.tl_calls(multiple, 'store'), source.tl_calls(body, 'store'))
        for extra in ['unknown=true', 'original=true', 'tail=other', 'single=0, single=1']:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                source.lean_first_triton_body(template.replace('storage=tl.float32', 'storage=tl.float32, ' + extra))
        with self.assertRaises(ValueError):
            source.lean_first_triton_body(template.replace('single=1', 'single=original'))
        with self.assertRaises(source.UnresolvedSourceMacro):
            source.lean_first_triton_body(template.replace('single=1, ', ''))

    def test_unknown_helpers_and_statement_splices_are_not_empty_bodies(self):
        unknown = '''let same := promotedCode data cos == data
let store : TSyntax `tritonStmt ← if same then `(tritonStmt| tl.store(Y, x))
  else `(tritonStmt| tl.store(Y, x.to(tl.float32)))
`(fun (X Y : RegionName) => ComputeKernel.mk [X] [Y] (triton { $store }).surfaceBody)
'''
        with self.assertRaises(source.UnresolvedSourceMacro):
            source.lean_first_triton_body(unknown)
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'missing'):
            source.lean_first_triton_body('def k := triton {\n$missing\n}')
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'missing'):
            source.lean_first_triton_body('def k := triton { if active { $missing } }')
        dtype_match = '''macro_rules
  | `(typed% $input:tritonDType) => do
    let half := match input with
      | `(tritonDType| tl.float16) => Bool.true
      | _ => Bool.false
    let store : TSyntax `tritonStmt ← if half then `(tritonStmt| tl.store(Y, x))
      else `(tritonStmt| tl.store(Y, x.to(tl.float16)))
    `(triton { tl.region X = $input
      $store })
'''
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'match input with'):
            source.lean_first_triton_body(dtype_match)

    def test_comment_blanking_preserves_offsets_and_quoted_strings(self):
        text = '-- triton { decoy }\n/- nested /- triton { decoy } -/ -/\n' + \
               'def kernel := triton { x = float("--inf") }'
        self.assertEqual(len(source.strip_lean_comments(text)), len(text))
        self.assertEqual(source.lean_first_triton_body(text), ' x = float("--inf") ')
        self.assertNotIn('def kernel', source.lean_first_preamble(text).split(':=')[-1])

    def test_dtype_match_tuple_alternatives_and_monadic_boole_keep_source_stores(self):
        template = '''/-!
Source-audit args: input=tl.float32, weight=tl.float64, output=tl.float16
-/
macro_rules
  | `(typed%[$input:tritonDType, $weight:tritonDType,
      $output:tritonDType]) => do
    let wide : Bool ← match input, weight with
      | `(tritonDType| tl.float64), _
      | _, `(tritonDType| tl.float64) => pure Bool.true
      | _, _ => pure Bool.false
    let same : Bool ← match output with
      | `(tritonDType| tl.float64) => pure wide
      | `(tritonDType| tl.float32) => pure (!wide)
      | _ => pure Bool.false
    let store : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(Y + i, x, mask=i < $(N)))
      else `(tritonStmt| tl.store(Y + i, x.to($output), mask=i < $(N)))
    `(triton { $store })
'''
        body = source.lean_first_triton_body(template)
        self.assertIn('x.to(tl.float16), mask=i < $(N)', body)
        wide = source.lean_first_triton_body(template.replace('output=tl.float16', 'output=tl.float64'))
        self.assertEqual(source.tl_calls(wide, 'store'), ['tl.store(Y + i, x, mask=i < $(N))'])
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'match input, weight with'):
            source.lean_first_triton_body(template.replace('weight=tl.float64, ', ''))
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'Unsupported source dtype pattern'):
            source.lean_first_triton_body(template.replace('| _, _ =>', '| unexpected, _ =>'))
        with self.assertRaises(source.UnresolvedSourceMacro):
            source.lean_first_triton_body(template.replace('pure Bool.false', 'pure unknownHelper'))
        with self.assertRaisesRegex(ValueError, 'dtype literal'):
            source.lean_first_triton_body(template.replace('output=tl.float16', 'output=tl.flaot16'))

    def test_statement_dtype_matches_keep_both_arm_syntax_and_inline_masks(self):
        template = '''/-!\nSource-audit args: output=tl.float32\n-/
macro_rules
  | `(typed% $output:tritonDType) => do
    let store : TSyntax `tritonStmt ← match output with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Y, x, mask=active))
      | _ => `(tritonStmt| tl.store(Y, x.to($output), mask=active))
    `(triton { if enabled { $store } })
'''
        body = source.lean_first_triton_body(template)
        self.assertIn('if enabled', body)
        self.assertEqual(source.tl_calls(body, 'store'), ['tl.store(Y, x, mask=active)'])
        narrow = source.lean_first_triton_body(template.replace('output=tl.float32', 'output=tl.float16'))
        self.assertEqual(source.tl_calls(narrow, 'store'), ['tl.store(Y, x.to(tl.float16), mask=active)'])
        with self.assertRaises(source.UnresolvedSourceMacro):
            source.lean_first_triton_body(template.replace('| _ => `(tritonStmt|', '| _ => unknown `(tritonStmt|'))
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'unresolvedStmt'):
            source.lean_first_triton_body(template.replace(
                'tl.store(Y, x.to($output), mask=active)', '$unresolvedStmt'))

    def test_local_scalar_helpers_use_exact_reachable_definitions(self):
        template = '''/-!
Source-audit Lean: typed%
Source-audit args: data=tl.float32, other=tl.float16
-/
namespace N
private def dtypeCode (stx : TSyntax `tritonDType) : MacroM Nat :=
  match stx with
  | `(tritonDType| tl.float32) => pure 0
  | `(tritonDType| tl.float16) => pure 2
  | _ => Macro.throwError "unsupported dtype"
private def promotedCode (a b : Nat) : Nat :=
  if a=0 || b=0 then 0 else if a=b then a else 2
syntax "typed%" : term
macro_rules
  | `(typed%[$data:tritonDType, $other:tritonDType]) => do
    let d ← dtypeCode data
    let c ← dtypeCode other
    let same := promotedCode (promotedCode d c) d == d
    let store : TSyntax `tritonStmt ← if same then `(tritonStmt| tl.store(Y, x, mask=active))
      else `(tritonStmt| tl.store(Y, x.to($data), mask=active))
    `(triton { $store })
end N
'''
        extract = lambda text: source.lean_first_triton_body(source.select_lean_audit_source(text))
        self.assertEqual(source.tl_calls(extract(template), 'store'), ['tl.store(Y, x, mask=active)'])
        changed = template.replace('then 0 else', 'then 2 else')
        self.assertEqual(source.cast_sequence(extract(changed)), ['float32'])
        for bad, diagnostic in [
            (template.replace('data=tl.float32', 'data=tl.int32'), 'rejected by dtype helper'),
            (template.replace('let d ← dtypeCode data', 'let d ← undefinedHelper data'), 'Unresolved'),
            (template.replace('pure 0', 'pure undefinedValue'), 'Unresolved'),
            (template.replace('if a=0', 'if unknown a=0'), 'Unresolved'),
            (template.replace('dtypeCode data', 'dtypeCode data other'), 'arity'),
            (template.replace('pure 0', 'dtypeCode stx'), 'Circular'),
            (template.replace('let d ← dtypeCode data', 'let d ← promotedCode data other'), 'type differs'),
            (template.replace('let same :=', 'let d := Bool.false\n    let same :='), 'Shadowed'),
        ]:
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(source.UnresolvedSourceMacro, diagnostic):
                extract(bad)
        duplicate = template.replace('syntax "typed%" : term',
                                     'private def dtypeCode (stx : TSyntax `tritonDType) : MacroM Nat := pure 9\n'
                                     'syntax "typed%" : term')
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'Ambiguous'):
            extract(duplicate)
        inaccessible = template.replace('syntax "typed%" : term', 'end N\nnamespace Other\nsyntax "typed%" : term')
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'Inaccessible'):
            extract(inaccessible)
        # A section ends without ending its containing namespace.
        section = template.replace('namespace N\n', 'namespace N\nnoncomputable section\n').replace(
            'syntax "typed%" : term', 'end\nsyntax "typed%" : term')
        self.assertEqual(source.tl_calls(extract(section), 'store'), ['tl.store(Y, x, mask=active)'])
        local_function = template.replace('let d ← dtypeCode data',
            'let promotedCode (a b : Nat) : Nat := 9\n    let d ← dtypeCode data')
        with self.assertRaises(source.UnresolvedSourceMacro):
            extract(local_function)
        with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'missingArm'):
            extract(template.replace('tl.store(Y, x.to($data), mask=active)', '$missingArm'))

    def test_later_local_bindings_cannot_change_earlier_macro_expressions(self):
        for global_binding, earlier, later in [
            ('def flag : Bool := true', 'if flag then', 'let flag := Bool.false'),
            ('def initial : Nat := 0', 'if initial == 0 then', 'let initial := 1'),
        ]:
            template = f'''{global_binding}
macro_rules
  | `(typed% $version:ident) => do
    let write : TSyntax `tritonStmt ← {earlier}
      `(tritonStmt| tl.store(O, 1)) else `(tritonStmt| tl.store(O, 2))
    {later}
    `(triton {{ $write }})
'''
            with self.subTest(global_binding=global_binding), self.assertRaisesRegex(
                    source.UnresolvedSourceMacro, 'Forward reference'):
                source.lean_first_triton_body(template)

    def test_array_concatenation_does_not_mutate_an_earlier_binding(self):
        template = '''macro_rules
  | `(typed% $version:ident) => do
    let before : Array (TSyntax `tritonStmt) := #[← `(tritonStmt| tl.store(O, 1))]
    let combined : Array (TSyntax `tritonStmt) := before ++ #[← `(tritonStmt| tl.store(O, 2))]
    `(triton { $combined*\n$before* })
'''
        self.assertEqual(source.tl_calls(source.lean_first_triton_body(template), 'store'),
                         ['tl.store(O, 1)', 'tl.store(O, 2)', 'tl.store(O, 1)'])

    def test_quoted_identifiers_are_not_forward_local_references(self):
        template = '''/-!\nSource-audit args: weight=half\n-/
macro_rules
  | `(typed% $weight:ident) => do
    unless weight.getId == `half || weight.getId == `fp8 do
      Macro.throwError "expected half or fp8"
    let fp8 := weight.getId == `fp8
    let store : TSyntax `tritonStmt ← if fp8 then `(tritonStmt| tl.store(O, 1))
      else `(tritonStmt| tl.store(O, 2))
    `(triton { $store })
'''
        self.assertEqual(source.tl_calls(source.lean_first_triton_body(template), 'store'),
                         ['tl.store(O, 2)'])
        self.assertEqual(source.tl_calls(source.lean_first_triton_body(template.replace('weight=half', 'weight=fp8')), 'store'),
                         ['tl.store(O, 1)'])

    def test_mutable_statement_arrays_preserve_order_masks_and_admission(self):
        template = '''/-!\nSource-audit args: dm=no, dn=no, version=original\n-/
macro_rules
  | `(typed%[$dm:ident, $dn:ident, $version:ident]) => do
    for flag in [dm, dn] do
      unless flag.getId == `yes || flag.getId == `no do
        Macro.throwError "expected divisibility flag"
    unless version.getId == `original || version.getId == `candidate do
      Macro.throwError "expected version"
    let m := dm.getId == `yes
    let n := dn.getId == `yes
    let mut masks : Array (TSyntax `tritonStmt) := #[]
    if !m then masks := masks.push (← `(tritonStmt| mask_m = rows < $(M)))
    if !n then masks := masks.push (← `(tritonStmt| mask_n = cols < $(N)))
    if !m && !n then masks := masks.push (← `(tritonStmt| mask = mask_m[:, None] & mask_n[None, :]))
    else if !m then masks := masks.push (← `(tritonStmt| mask = mask_m[:, None]))
    else if !n then masks := masks.push (← `(tritonStmt| mask = mask_n[None, :]))
    let store : TSyntax `tritonStmt ← if m && n then `(tritonStmt| tl.store(Y, x))
      else `(tritonStmt| tl.store(Y, x, mask=mask))
    `(triton { $masks*\n$store })
'''
        for m in ('no', 'yes'):
            for n in ('no', 'yes'):
                text = template.replace('dm=no', 'dm=' + m).replace('dn=no', 'dn=' + n)
                body = source.lean_first_triton_body(text)
                self.assertEqual('mask_m = rows' in body, m == 'no')
                self.assertEqual('mask_n = cols' in body, n == 'no')
                self.assertEqual('mask=mask' in body, m == 'no' or n == 'no')
                self.assertEqual('mask_m[:, None] & mask_n[None, :]' in body, m == n == 'no')
                self.assertEqual(len(source.tl_calls(body, 'store')), 1)
        for bad in [template.replace('dm=no', 'dm=typo'),
                    template.replace('version=original', 'version=typo')]:
            with self.assertRaisesRegex(source.UnresolvedSourceMacro, 'admission guard'):
                source.lean_first_triton_body(bad)
        for bad in [template.replace('masks.push', 'masks.pop'),
                    template.replace('if !m then masks', 'if missing then masks'),
                    template.replace('if !m then masks', 'else if !m then masks'),
                    template.replace('let m :=', 'let dm :='),
                    template.replace('for flag in [dm, dn] do', 'for flag in computeFlags dm dn do'),
                    template.replace('Macro.throwError "expected version"', 'unknownGuardEffect')]:
            with self.assertRaises(source.UnresolvedSourceMacro):
                source.lean_first_triton_body(bad)
        delayed_mutation = template.replace('let store : TSyntax',
            'let later := Bool.true\n    if later then masks := masks.push (← `(tritonStmt| tl.store(Y, 99)))\n'
            '    let store : TSyntax')
        with self.assertRaises(source.UnresolvedSourceMacro):
            source.lean_first_triton_body(delayed_mutation)
        effect = template.replace('let m :=', 'unknownMacroEffect\n    let m :=')
        with self.assertRaises(source.UnresolvedSourceMacro):
            source.lean_first_triton_body(effect)
        continued_guard = template.replace('|| version.getId', '||\n        version.getId')
        self.assertEqual(source.tl_calls(source.lean_first_triton_body(continued_guard), 'store'),
                         ['tl.store(Y, x, mask=mask)'])

    def test_shape_dtype_attribute_normalization_retains_wrong_types(self):
        lean = 'tl.region out = $storage\nx = tl.full([$(B)], 0.0, dtype=$storage:tritonDType)'
        normalize = source.normalize_shape_ctor_call
        expected = normalize(source.tl_calls(lean, 'full')[0])
        for attribute in ('out.dtype.element_ty', 'out.type.element_ty'):
            py = 'x = tl.full([B], 0.0, dtype=' + attribute + ')'
            resolved = source.resolve_source_dtype_attributes(py, lean)
            self.assertEqual(normalize(source.tl_calls(resolved, 'full')[0]), expected)
        for dtype in ('missing.dtype.element_ty', 'tl.float16', '$wrong'):
            py = 'x = tl.full([B], 0.0, dtype=' + dtype + ')'
            resolved = source.resolve_source_dtype_attributes(py, lean)
            self.assertNotEqual(normalize(source.tl_calls(resolved, 'full')[0]), expected)

    def test_dtype_resolution_rejects_rebound_load_tensors_and_inline_aliases(self):
        lean = 'tl.region X = tl.float16, O = tl.float16\nv = tl.load(X)\n'
        for update in ['v = v.to(tl.float32)', 'v = other', 'v += other', 'v *= other']:
            self.assertEqual(source.resolve_source_dtype_attributes('x.to(v.dtype)', lean + update),
                             'x.to(v.dtype)')
        py = 'p = out + off; p = other + off\ntl.store(p, value)'
        lean = 'tl.region out = tl.float16, other = tl.float32\ntl.store(p, value.to(tl.float16))'
        self.assertEqual(source.store_calls_with_explicit_destination_casts(py, lean),
                         ['tl.store(p, value)'])
        self.assertEqual(source.resolve_source_dtype_attributes('x.to(p.dtype.element_ty)',
                         lean + '\np = out + off; p = other + off'), 'x.to(p.dtype.element_ty)')

    def test_implicit_store_cast_rejects_incomplete_alias_write_inventory(self):
        lean = 'tl.region out = tl.float16, other = tl.float32\ntl.store(p, value.to(tl.float16))'
        for update in ['if active: p = other + off', 'p //= stride',
                       'p = out + off; p = other + off', 'p = choose(out, other)']:
            python = 'p = out + off\n' + update + '\ntl.store(p, value)'
            with self.subTest(update=update):
                self.assertEqual(source.store_calls_with_explicit_destination_casts(python, lean),
                                 ['tl.store(p, value)'])
        for decoy in ['"p = out + off"', '"""\np = out + off\n"""']:
            python = 'message = ' + decoy + '\ntl.store(p, value)'
            self.assertEqual(source.store_calls_with_explicit_destination_casts(python, lean),
                             ['tl.store(p, value)'])
        python = 'p = out + off\nmessage = "p = other + off"\n# p = other + off\ntl.store(p, value)'
        self.assertEqual(source.store_calls_with_explicit_destination_casts(python, lean),
                         ['tl.store(p, value.to(tl.float16))'])

    def test_string_and_character_contents_are_not_triton_syntax(self):
        for literal in ['"triton { x = tl.load(wrong) }"',
                        'r"triton { x = tl.load(wrong) }"',
                        'r##"triton { \\" -- /- x = tl.load(wrong) }"##']:
            text = f'def doc := {literal}\ndef actual := triton {{ x = float("{{") }}\n'
            self.assertEqual(source.lean_first_triton_body(text), ' x = float("{") ')
            self.assertEqual(source.lean_triton_bodies(text), [' x = float("{") '])
            with self.assertRaisesRegex(ValueError, 'no literal Triton body'):
                source.select_lean_audit_source('/-!\nSource-audit Lean: doc\n-/\n' + text)
        text = "def brace := '{'\ndef quote := '\"'\ndef actual := triton { x = 1 }"
        self.assertEqual(source.lean_first_triton_body(text), ' x = 1 ')
        self.assertEqual(len(source.lean_code_mask(text)), len(text))
        for broken in ['def doc := "unterminated', 'def doc := r#"unterminated"']:
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                source.lean_first_triton_body(broken)


    def test_store_whitespace_and_nested_values(self):
        for py_ws in ['', ' ', '\n ']:
            for lean_ws in ['', ' ', '\n ']:
                py = 'tl.store' + py_ws + '(out + row, a * (b + c), mask=active)'
                lean = 'tl.region out = $storage\ntl.store' + lean_ws + '(out + row, (a * (b + c)).to($storage), mask=active)'
                self.assertEqual(source.store_calls_with_explicit_destination_casts(py, lean),
                                 ['tl.store(out + row, (a * (b + c)).to($storage), mask=active)'])

    def test_store_whitespace_preserves_pointer_value_and_dtype(self):
        py = 'tl.store (out + row, value, mask=active)'
        lean = 'tl.region out = $storage\ntl.store (out + row, value.to($storage), mask=active)'
        for changed in [lean.replace('out + row', 'out + other'),
                        lean.replace('value.to', 'other.to'),
                        lean.replace('to($storage)', 'to(tl.float32)')]:
            self.assertEqual(source.store_calls_with_explicit_destination_casts(py, changed), [py])


    def test_store_parameter_offsets_and_compound_values(self):
        normalize = source.store_calls_with_explicit_destination_casts
        py = 'tl.store(out + block * rows + row, a * b, mask=active)'
        lean = 'tl.region out = $storage\ntl.store(out + block * $(rows) + row, (a * b).to($storage), mask=active)'
        self.assertEqual(normalize(py, lean),
                         ['tl.store(out + block * rows + row, (a * b).to($storage), mask=active)'])
        for changed in [lean.replace('$(rows)', '$(other_rows)'),
                        lean.replace('(a * b)', '(a + b)'),
                        lean.replace('.to($storage)', '.to(tl.float32)')]:
            self.assertEqual(normalize(py, changed), [py])
        py2 = 'tl.store(out, a * (b + c))'
        lean2 = 'tl.region out = tl.float32\ntl.store(out, (a * b + c).to(tl.float32))'
        self.assertEqual(normalize(py2, lean2), [py2])

    def test_self_advanced_store_pointer_retains_only_declared_type(self):
        normalize = source.store_calls_with_explicit_destination_casts
        call = 'tl.store(out + off, value, mask=active)'
        py = 'out = out + row * stride\n' + call
        lean = 'tl.region out = $storage\ntl.store(out + off, value.to($storage), mask=active)'
        self.assertEqual(normalize(py, lean),
                         ['tl.store(out + off, value.to($storage), mask=active)'])
        self.assertEqual(normalize(py, lean.replace('tl.region out = $storage', '')), [call])
        self.assertEqual(normalize('out = other + off\n' + py, lean), [call])
        self.assertEqual(normalize('other = out + off\nout = other + off\n' + call, lean), [call])


    def test_store_offset_preserves_destination_cast(self):
        normalize = source.store_calls_with_explicit_destination_casts
        for ptr in ['b + offset', '(b + offset)', 'b + row * stride + offset', 'b - offset']:
            py = f'tl.store({ptr}, value, mask=active)'
            lean = f'tl.region b = $storage\ntl.store({ptr}, value.to($storage), mask=active)'
            self.assertEqual(normalize(py, lean),
                             [f'tl.store({ptr}, value.to($storage), mask=active)'])
            self.assertEqual(normalize(py, lean.replace('value.to($storage)', 'value.to(tl.float32)')),
                             [py])
            self.assertEqual(normalize(py, lean.replace('value.to($storage)', 'other.to($storage)')),
                             [py])

    def test_store_offset_does_not_hide_unknown_roots_or_address_changes(self):
        normalize = source.store_calls_with_explicit_destination_casts
        for ptr in ['missing + offset', 'b * offset', 'f(b) + offset']:
            py = f'tl.store({ptr}, value, mask=active)'
            lean = f'tl.region b = $storage\ntl.store({ptr}, value.to($storage), mask=active)'
            self.assertEqual(normalize(py, lean), [py])
        py = 'tl.store(b + offset, value, mask=active)'
        lean = 'tl.region b = $storage\ntl.store(b + other_offset, value.to($storage), mask=active)'
        self.assertEqual(normalize(py, lean), [py])

    def test_block_load_padding_distinguishes_omission_from_zero(self):
        text = '''import triton.language as lang
@triton.jit
def kernel(p, axes, padding):
    a = lang.load(p, boundary_check=(0, 1))
    b = lang.load(p, boundary_check=(0,), padding_option="zero")
    c = lang.load(p, boundary_check=(0,), padding_option="")
    d = lang.load(p, boundary_check=(0,), padding_option="nan")
    e = lang.load(p, boundary_check=axes, padding_option=padding)
    f = lang.load(p, boundary_check=())
    g = lang.load(p, mask=True, other=0)
    h = something.load(p, boundary_check=(0,))
def host(p):
    return lang.load(p, boundary_check=(0,))
'''
        sites = source.python_block_load_padding(text)
        self.assertEqual([s['padding'] for s in sites],
                         ['unspecified', 'zero', 'unspecified', 'nan', 'dynamic_or_invalid'])
        self.assertEqual([s['line'] for s in sites], [4, 5, 6, 7, 8])
        self.assertTrue(all(s['kernel'] == 'kernel' for s in sites))
        self.assertEqual(sites[-1]['boundary_check'], 'axes')

    def test_original_statement_macro_handles_nested_calls_and_both_arm_orders(self):
        for tag, yes, no in [
            ('sumLeft', 'b_z = tl.sum(b_s, 0) + b_z', 'b_z += tl.sum(b_s, 0)'),
            ('original', 'b_z += tl.sum(b_s, 0)', 'b_z = tl.sum(b_s, 0) + b_z'),
        ]:
            text = f'''let update : TSyntax `tritonStmt ←
  if version.getId == `{tag} then `(tritonStmt| {yes})
  else `(tritonStmt| {no})
`(fun (s z : RegionName) (T BT : Nat) =>
  ComputeKernel.mk [s] [z] (triton {{
    if i_t >= $(0) {{ $update }}
  }}).surfaceBody)'''
            body = source.lean_first_triton_body(text)
            self.assertIn('b_z += tl.sum(b_s, 0)', body)
            self.assertNotIn('$update', body)
            self.assertNotIn('b_z = tl.sum', body)
            self.assertEqual(source.lean_triton_bodies(text), [body])
            self.assertIn('(T BT : Nat)', source.quoted_kernel_signature(text))
        broken = text.replace('else `(tritonStmt|', 'else unsupported (')
        with self.assertRaises(ValueError):
            source.lean_first_triton_body(broken)

    def test_region_continuations_do_not_become_statements(self):
        body = "tl.region x = DTYPE,\n  y = OUT_DTYPE\nz = tl.load(x)\ny += z"
        stripped = source.strip_region_declarations(body)
        self.assertNotIn('y = OUT_DTYPE', stripped)
        self.assertIn('z = tl.load(x)', stripped)
        self.assertIn('y += z', stripped)

    def test_preamble_supports_explicit_source_headers(self):
        self.assertEqual(source.target_kernel_name(
            '/-! Source: vector.py, vector_kernel. -/', {'vector_kernel'}), 'vector_kernel')
        self.assertEqual(source.target_kernel_name(
            '/-! Complete typed source for `backward` in\n`backward.py`. -/', {'backward'}), 'backward')

    def test_dtype_resolution_preserves_literal_and_wrong_precision_mismatches(self):
        lean = '''tl.region x = DTYPE, y = OUT_DTYPE
p_x = tl.make_block_ptr(base=x+base, shape=(T), strides=(1))
p_y = tl.make_block_ptr(base=y+base, shape=(T), strides=(1))
b_x = tl.load(p_x)
promoted = tl.load(p_x).to(tl.float32)
'''
        resolve = lambda text: source.resolve_source_dtype_attributes(text, lean)
        self.assertEqual(resolve('x.to(p_y.dtype.element_ty)'), 'x.to(OUT_DTYPE)')
        self.assertEqual(resolve('x.to(b_x.dtype)'), 'x.to(DTYPE)')
        self.assertNotEqual(resolve('x.to(b_x.dtype)'), 'x.to(OUT_DTYPE)')
        self.assertEqual(resolve('x.to(tl.float16)'), 'x.to(tl.float16)')
        self.assertEqual(resolve('x.to(missing.dtype)'), 'x.to(missing.dtype)')
        self.assertEqual(resolve('x.to(promoted.dtype)'), 'x.to(promoted.dtype)')
        ambiguous = lean + '\np_x = tl.make_block_ptr(base=y+base)\n'
        self.assertEqual(source.resolve_source_dtype_attributes('x.to(p_x.dtype.element_ty)', ambiguous),
                         'x.to(p_x.dtype.element_ty)')


    def test_implicit_store_cast_requires_the_actual_destination_dtype(self):
        python = 'out_ptrs = Out + off_o\ntl.store(out_ptrs, acc, mask=active)'
        lean = 'tl.region Out = $storage\ntl.store(out_ptrs, acc.to($storage), mask=active)'
        normalize = source.store_calls_with_explicit_destination_casts
        self.assertEqual(normalize(python, lean),
                         ['tl.store(out_ptrs, acc.to($storage), mask=active)'])
        self.assertEqual(normalize(python, lean.replace('acc.to($storage)', '(acc).to($storage:tritonDType)')),
                         ['tl.store(out_ptrs, acc.to($storage), mask=active)'])
        for invalid in [lean.replace('$storage),', 'tl.float32),'),
                        lean.replace('acc.to', 'other.to'),
                        lean.replace('out_ptrs,', 'different,')]:
            self.assertEqual(normalize(python, invalid), ['tl.store(out_ptrs, acc, mask=active)'])
        for ambiguous in [python.replace('tl.store', 'out_ptrs = Other + off_o\ntl.store'),
                          python.replace('tl.store', 'out_ptrs *= stride\ntl.store'),
                          python.replace('Out + off_o', 'choose(Out, Other)')]:
            self.assertEqual(normalize(ambiguous, lean), ['tl.store(out_ptrs, acc, mask=active)'])
        for advance in ['+=', '-=']:
            advanced = python.replace('tl.store', f'out_ptrs {advance} stride\ntl.store')
            self.assertEqual(normalize(advanced, lean),
                             ['tl.store(out_ptrs, acc.to($storage), mask=active)'])
        masked = lean.replace('mask=active', 'mask=wrong')
        self.assertIn('mask=active', normalize(python, masked)[0])
        self.assertNotIn('mask=wrong', normalize(python, masked)[0])

    def test_loaded_pointer_alias_dtype_requires_unique_bindings(self):
        lean = '''tl.region V = $storage, W = tl.float32
v_ptrs = V + off_v
next_ptrs = v_ptrs + step
v = tl.load(next_ptrs + offset, mask=active, other=0.0)
'''
        resolve = lambda text: source.resolve_source_dtype_attributes('p.to(v.dtype)', text)
        self.assertEqual(resolve(lean), 'p.to($storage)')
        for extra in ['v_ptrs = W + other', 'v_ptrs *= stride',
                      'v_ptrs = choose(V, W)', 'v_ptrs = next_ptrs + stride']:
            self.assertEqual(resolve(lean + extra + '\n'), 'p.to(v.dtype)')
        for advance in ['+=', '-=']:
            self.assertEqual(resolve(lean + f'v_ptrs {advance} stride\n'), 'p.to($storage)')
        self.assertNotEqual(resolve(lean), 'p.to(tl.float32)')

    def test_cast_sequence_keeps_macro_types_and_missing_casts(self):
        self.assertEqual(source.cast_sequence('p.to($storage:tritonDType)'), ['$storage'])
        self.assertEqual(source.cast_sequence('p.to($storage)'), ['$storage'])
        self.assertNotEqual(source.cast_sequence('p.to($storage)'), source.cast_sequence('p'))
        self.assertNotEqual(source.cast_sequence('p.to($storage)'), source.cast_sequence('p.to($other)'))
        self.assertNotEqual(source.cast_sequence('p.to(tl.float16)'), source.cast_sequence('p.to(tl.float32)'))


    def test_true_equality_is_not_a_bare_placeholder(self):
        for text in ['theorem empty : True := by trivial', '    True := by']:
            self.assertTrue(source.bare_true_proof(text))
        for text in ['    p = True := by simp', '    p ↔ True := by simp',
                     '    p = True := by rw [safety_definition]']:
            self.assertFalse(source.bare_true_proof(text))

    def test_named_jit_cast_does_not_include_other_kernels_or_host(self):
        text = """@triton.jit
def helper(X):
    x = tl.load(X).to(tl.float32)

@triton.jit
def target(X):
    x = tl.load(X)

def host(X):
    return X.to(tl.float32)
"""
        self.assertNotIn('.to(tl.float32)', source.python_named_jit_body(text, 'target'))
        self.assertIn('.to(tl.float32)', source.python_named_jit_body(text, 'helper'))
        with self.assertRaises(ValueError):
            source.python_named_jit_body(text, 'host')

    def test_primary_kernel_and_helper_coverage_are_distinct(self):
        text = '''@triton.jit
def helper(x):
    return x + 1

@triton.jit
@triton.autotune(configs=[])
def other(
    X,
    N: tl.constexpr,
):
    x = tl.load(X)

@triton.jit
def _attn_fwd(X):
    preferred = tl.load(X)

def host():
    return None
'''
        selected = source.python_first_kernel_body(text)
        self.assertIn('preferred = tl.load(X)', selected)
        self.assertNotIn('def host', selected)
        self.assertEqual(source.python_jit_kernel_bodies(text), [selected])
        all_bodies = source.python_jit_kernel_bodies(text, all_jit=True)
        self.assertEqual(len(all_bodies), 3)
        self.assertIn('return x + 1', all_bodies[0])
        self.assertIn('x = tl.load(X)', all_bodies[1])

    def test_block_pointer_selection_is_explicit(self):
        text = '''@triton.jit
def pointer(X):
    p = tl.make_block_ptr(X, (16,), (1,), (0,), (16,), (0,))

@triton.jit
def memory(X):
    x = tl.load(X)
'''
        self.assertIn('tl.load', source.python_first_kernel_body(text))
        self.assertIn('tl.make_block_ptr', source.python_first_kernel_body(
            text, include_block_pointers=True))
        self.assertEqual(source.python_jit_names(text), {'pointer', 'memory'})
        self.assertEqual(source.python_jit_names_text(text), {'pointer', 'memory'})

    def test_empty_or_incomplete_sources_have_no_kernel(self):
        for text in ['', 'def host():\n    return 1\n', '@triton.jit\ndef kernel(\n']:
            with self.subTest(text=text):
                self.assertEqual(source.python_first_kernel_body(text), '')
                self.assertEqual(source.python_jit_kernel_bodies(text), [])
        self.assertEqual(source.lean_first_triton_body('def x := 1'), '')
        self.assertEqual(source.lean_triton_bodies('def x := 1'), [])

    def test_target_selection_ignores_later_non_kernel_prose(self):
        text = '''/- Port of `example.py`'s `helper`.
Headline: `example.py`'s `kernel`.
Translation note: `example.py`'s `X += offset`. -/
def kernel := triton { x = tl.load(X) }
'''
        preamble = source.lean_first_preamble(text)
        self.assertEqual(source.target_kernel_candidates(preamble),
                         ['helper', 'kernel', 'X += offset'])
        self.assertEqual(source.target_kernel_name(preamble, {'helper', 'kernel'}), 'kernel')
        self.assertIsNone(source.target_kernel_name(preamble, {'unrelated'}))

    def test_lean_bodies_keep_nested_braces_and_source_order(self):
        text = 'preamble\ndef a := triton { x = $({ value := 1 }) }\ndef b := triton { y = 2 }'
        self.assertEqual(source.lean_triton_bodies(text), [' x = $({ value := 1 }) ', ' y = 2 '])
        self.assertEqual(source.lean_first_triton_body(text), ' x = $({ value := 1 }) ')

    def test_call_views_preserve_nested_arguments(self):
        body = '''# tl.load(ignored)
x = tl.load(
    X + f(a, b), mask=g(i, [j, k]), other=0,
)
y = tl.load(Y)
'''
        whole = source.tl_calls(body, 'load')
        args = source.tl_calls(body, 'load', arguments_only=True)
        normalized = source.tl_calls(body, 'load', normalize_whitespace=True)
        self.assertEqual(len(whole), 2)
        self.assertEqual(whole[1], 'tl.load(Y)')
        self.assertEqual(args[1], 'Y')
        self.assertNotIn('\n', normalized[0])
        expected = ['X + f(a, b)', 'mask=g(i, [j, k])', 'other=0']
        self.assertEqual(source.split_top_level_args(args[0]), expected)
        self.assertEqual(source.split_top_level_args(whole[0], 'load'), expected)
        self.assertEqual(source.split_top_level_args(normalized[0], 'load'), expected)

    def test_nested_lean_comments_preserve_diagnostic_lines(self):
        text = 'theorem first := by\n/- outer\n/- nested -/\nstill comment -/\n-- line comment\ntheorem second := by\n'
        code = source.strip_lean_comments(text)
        self.assertEqual(code.count('\n'), text.count('\n'))
        self.assertEqual(code.splitlines()[5], 'theorem second := by')
        for hidden in ('outer', 'nested', 'still comment', 'line comment'):
            self.assertNotIn(hidden, code)


class SourceParameterMapTests(unittest.TestCase):
    python = '''@triton.jit
def kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr, enabled=True):
    cols = tl.arange(0, BLOCK_SIZE)
    if enabled:
        x = tl.load(input_ptr + cols, mask=cols < BLOCK_SIZE, other=0)
        tl.store(output_ptr + cols, x, mask=cols < BLOCK_SIZE)
'''
    lean = '''/-!
Source-audit target: kernel
Source-audit Lean: typed
Source-audit params: input_ptr=X, output_ptr=Y, BLOCK_SIZE=B, enabled=ACTIVE
-/
def legacy (Wrong : RegionName) : ComputeKernel := triton { tl.store(Wrong, 0) }
def typed (X Y : RegionName) (B : Nat) (ACTIVE : Bool) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  cols = tl.arange(0, $(B))
  if ACTIVE {
    x = tl.load(X + cols, mask=cols < $(B), other=0)
    tl.store(Y + cols, x, mask=cols < $(B))
  }
}
'''

    def test_exact_parameter_map_preserves_all_runtime_branches(self):
        body = source.python_audit_kernel_body(self.python, self.lean)
        self.assertIn('if ACTIVE:', body)
        self.assertIn('tl.arange(0, B)', body)
        self.assertIn('tl.load(X + cols, mask=cols < B, other=0)', body)
        self.assertIn('tl.store(Y + cols, x, mask=cols < B)', body)
        self.assertEqual(source.python_source_parameter_names(self.python, self.lean), ['X', 'Y', 'B', 'ACTIVE'])
        self.assertEqual(source.lean_source_parameter_names(self.lean), ['X', 'Y', 'B', 'ACTIVE'])
        partial = self.lean.replace(', enabled=ACTIVE', '')
        self.assertEqual(source.python_source_parameter_names(self.python, partial)[-1], 'enabled')
        self.assertNotIn('enabled', source.lean_source_parameter_names(partial))

    def test_mapping_changes_only_parameter_name_tokens_in_selected_body(self):
        prefix = '@triton.jit\ndef helper(input_ptr):\n    return input_ptr\n'
        python = self.python.replace('enabled=True', 'enabled=default_enabled').replace(
            '    cols =', '    note = "BLOCK_SIZE input_ptr enabled"  # input_ptr\n'
            '    other = obj.BLOCK_SIZE\n'
            '    tag = "文"; debug(BLOCK_SIZE=BLOCK_SIZE)\n    cols =')
        text = source.python_source_with_parameter_map(prefix + python, self.lean)
        self.assertTrue(text.startswith(prefix))
        self.assertIn('BLOCK_SIZE: tl.constexpr, enabled=default_enabled', text)
        self.assertIn('"BLOCK_SIZE input_ptr enabled"  # input_ptr', text)
        self.assertIn('obj.BLOCK_SIZE', text)
        self.assertIn('"文"; debug(BLOCK_SIZE=B)', text)
        self.assertIn('tl.load(X + cols', text)
        self.assertIn('@triton.jit\ndef kernel(input_ptr, output_ptr,', text)

    def test_parameter_permutations_are_simultaneous_and_capture_free(self):
        python = '@triton.jit\ndef kernel(X, Y):\n    tl.store(X, tl.load(Y))\n'
        lean = '''/-!\nSource-audit target: kernel\nSource-audit Lean: typed
Source-audit params: X=Y, Y=X
-/\ndef typed (X Y : RegionName) := triton { tl.store(Y, tl.load(X)) }
'''
        self.assertEqual(source.tl_calls(source.python_audit_kernel_body(python, lean), 'store'),
                         ['tl.store(Y, tl.load(X))'])
        with self.assertRaisesRegex(ValueError, 'not injective'):
            source.source_parameter_map(python, lean.replace(', Y=X', ''))
        for statement in ['B = 7', 'debug(B)', 'for B in range(2):\n        debug(B)']:
            captured = self.python.replace('    cols =', '    ' + statement + '\n    cols =')
            with self.subTest(statement=statement), self.assertRaisesRegex(ValueError, 'capture'):
                source.source_parameter_map(captured, self.lean)

    def test_invalid_maps_and_ambiguous_source_scopes_fail_closed(self):
        for old, new, diagnostic in [
            ('BLOCK_SIZE=B', 'missing=B', 'not Python parameters'),
            ('BLOCK_SIZE=B', 'BLOCK_SIZE=Wrong', 'not selected Lean binders'),
            ('BLOCK_SIZE=B', 'BLOCK_SIZE=B, BLOCK_SIZE=B', 'duplicate'),
            ('BLOCK_SIZE=B', 'BLOCK_SIZE=X', 'not injective'),
            ('BLOCK_SIZE=B', 'BLOCK_SIZE=16', 'Invalid'),
            ('enabled=ACTIVE', 'enabled=false', 'not selected Lean binders'),
            ('enabled=ACTIVE', 'enabled=False', 'identifier'),
            ('Source-audit target: kernel', 'Source-audit target: typo', 'unique Python JIT'),
            ('Source-audit target: kernel\n', '', 'explicit target'),
            ('Source-audit Lean: typed\n', '', 'explicit target'),
            ('(X Y : RegionName)', '(X X Y : RegionName)', 'Duplicate selected Lean'),
        ]:
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, diagnostic):
                source.source_parameter_map(self.python, self.lean.replace(old, new))
        for suffix in ['\n' + self.python, '\ndef kernel():\n    return 0\n']:
            with self.assertRaisesRegex(ValueError, 'unique Python JIT'):
                source.source_parameter_map(self.python + suffix, self.lean)
        with self.assertRaisesRegex(ValueError, 'Duplicate Source-audit params'):
            source.source_parameter_map(self.python, self.lean.replace('-/\n', 'Source-audit params: input_ptr=X\n-/\n', 1))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            source.source_parameter_map(self.python, re.sub(r'Source-audit params:[^\n]+', 'Source-audit params: ', self.lean))

    def test_indirect_python_bindings_and_nested_scopes_are_not_rewritten(self):
        for statement in [
            'f = lambda B: input_ptr + B',
            'def nested(B):\n        return input_ptr + B',
            'values = [input_ptr for B in range(2)]',
            'import other as B',
            'global B',
            'match unknown:\n        case B:\n            debug(input_ptr)',
        ]:
            python = self.python.replace('    cols =', '    ' + statement + '\n    cols =')
            with self.subTest(statement=statement), self.assertRaisesRegex(ValueError, 'nested scopes or indirect'):
                source.source_parameter_map(python, self.lean)
        with self.assertRaisesRegex(ValueError, 'variadic'):
            source.source_parameter_map(self.python.replace('enabled=True', '*extras, enabled=True'), self.lean)

    def test_selected_macro_binders_and_comments_do_not_create_fake_targets(self):
        lean = self.lean.replace('Source-audit Lean: typed', 'Source-audit Lean: typed%')
        lean = lean[:lean.index('def legacy')] + '''macro_rules
  | `(typed% $storage:tritonDType) => do
    `(fun (X Y : RegionName) (B : Nat) (ACTIVE : Bool) =>
      ComputeKernel.mk [X] [Y] (triton { tl.region X = $storage, Y = $storage
        tl.store(Y, tl.load(X), mask=ACTIVE) }).surfaceBody)
'''
        self.assertEqual(source.lean_source_parameter_names(lean), ['X', 'Y', 'B', 'ACTIVE'])
        self.assertEqual(len(source.source_parameter_map(self.python, lean)), 4)
        decoy = self.lean.replace(':= triton {\n', ':=\n  let message := "`(fun (Fake : Nat) =>"\n  triton {\n')
        self.assertEqual(source.lean_source_parameter_names(decoy), ['X', 'Y', 'B', 'ACTIVE'])
        with self.assertRaisesRegex(ValueError, 'not selected Lean binders'):
            source.source_parameter_map(self.python, decoy.replace('BLOCK_SIZE=B', 'BLOCK_SIZE=Fake'))







class PointerAdvanceTests(unittest.TestCase):
    def test_pointer_advances_keep_storage_dtype(self):
        lean='tl.region out = tl.float16\np = out + off\np += step\np -= step\ntl.store(p, value.to(tl.float16))'
        py='p = out + off\np += step\np -= step\ntl.store(p, value)'
        self.assertEqual(source.resolve_source_dtype_attributes('value.to(p.dtype.element_ty)',lean),'value.to(tl.float16)')
        self.assertEqual(source.store_calls_with_explicit_destination_casts(py,lean),['tl.store(p, value.to(tl.float16))'])
    def test_ambiguous_or_unknown_pointer_stays_unresolved(self):
        for bindings in ['p = out + off\np = other + off', 'p = out + off\np *= step', 'p = out + off\np /= step', 'p += step','p = missing + off\np -= step']:
            lean='tl.region out = tl.float16, other = tl.bfloat16\n'+bindings+'\np += step\ntl.store(p, value.to(tl.float16))'
            py=bindings+'\np += step\ntl.store(p, value)'
            self.assertEqual(source.resolve_source_dtype_attributes('value.to(p.dtype.element_ty)',lean),'value.to(p.dtype.element_ty)',bindings)
            self.assertEqual(source.store_calls_with_explicit_destination_casts(py,lean),['tl.store(p, value)'],bindings)
    def test_advances_do_not_hide_changed_store(self):
        py='p = out + off\np += step\ntl.store(p, value, mask=active)'
        for call in ['tl.store(p, value.to(tl.float32), mask=active)','tl.store(p, other.to(tl.float16), mask=active)','tl.store(p + different, value.to(tl.float16), mask=active)','tl.store(p, value.to(tl.float16), mask=other)']:
            lean='tl.region out = tl.float16\np = out + off\np += step\n'+call
            self.assertNotEqual(source.store_calls_with_explicit_destination_casts(py,lean),[call])

if __name__ == '__main__':
    unittest.main()
