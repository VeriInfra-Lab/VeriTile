"""Strict rules survive missing reports; empirical rewrites need their own admission."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('ScalarArithmetic', 'Reciprocal', 'Exponential', 'CountConversion', 'Maximum')
REPORT_POOLS = ('ReportedAdmission.all', 'SupplementalAdmission.all',
                'CountAdmission.all', 'ExpAdmission.all')
STRICT_IDS = {
    'ScalarArithmetic': {'ADD-COMMUTE', 'MUL-COMMUTE', 'MUL-ONE', 'DIV-ONE'},
    'Exponential': {'EXP-ZERO', 'EXP-ZERO-LIBDEVICE',
                    'EXP-NEG-INF-SUB', 'EXP-NEG-INF-SUB-LIBDEVICE'},
    'CountConversion': {'COUNT-ZERO', 'COUNT-SUCCESSOR'},
    'Maximum': {'MAX-COMMUTE', 'MAX-ASSOC', 'MAX-IDEM', 'MAX-NEG-INF'},
}


class PrimitiveCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build', *(f'VeriTile.Triton.Float.{f}' for f in FAMILIES)],
                                cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Candidates.lean'
            path.write_text(source)
            return subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                  text=True, capture_output=True, timeout=180)

    def check(self, source):
        result = self.lean(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def source(self, family):
        return (ROOT / f'VeriTile/Triton/Float/{family}.lean').read_text()

    def empty(self, source):
        for pool in REPORT_POOLS:
            source = source.replace(pool, '[]')
        return source

    def test_syntax_domains_precision_and_actual_dependency(self):
        output = self.check((ROOT / 'bench/tests/FPPrimitiveCandidates.lean').read_text())
        self.assertNotIn('unresolved FP proof', output)
        for theorem in ('selected_maximum', 'selected_exp_zero', 'selected_exp_zero_libdevice',
                        'selected_exp_neg_inf_sub', 'selected_exp_neg_inf_sub_libdevice'):
            self.assertIn(f'FP assumptions used by {theorem}:\n  none\n', output)
            self.assertIn(f'Strict model certificates used by {theorem}:\n', output)

    def test_empty_tables_keep_strict_rules_and_conditional_proofs(self):
        checks = {
            'ScalarArithmetic': '''
example (a : Atom) : a.Available ↔ a.isStrict = true := by cases a <;> decide
example (R : Rules) : R.assumptions = [] := rfl
example (a : Atom) (h : a.isStrict = true) :
    VeriTile.Spec.Derivation [] [lhs a] [rhs a] := strict_admitted [] a h
''',
            'Reciprocal': '''
example (a : Format) : ¬ a.Available := by cases a <;> decide
example (f : Format) (R : Rules f) : R.assumptions = [] := by cases f <;> rfl
''',
            'Exponential': '''
example (a : Atom) : a.Available ↔ a.isStrict = true := by
  rcases a with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _⟩ <;> decide
example (R : Rules) : R.assumptions = R.arithmetic.assumptions := by
  change R.arithmetic.assumptions ++ [] = R.arithmetic.assumptions
  exact List.append_nil _
example (a : Atom) (h : a.isStrict = true) :
    VeriTile.Spec.Derivation [] [a.lhs] [a.rhs] :=
  VeriTile.Triton.FP.Guarded.strict_derivation [] (strict_sound a h)
''',
            'CountConversion': '''
example (a : Atom) : a.Available := trivial
example (R : Rules) : R.assumptions = R.arithmetic.assumptions := rfl
example (a : Atom) : VeriTile.Spec.Derivation [] [lhs a] [rhs a] := strict [] a
''',
            'Maximum': '''
-- Available retains its historical report-lookup meaning in this family.
example (a : Atom) : ¬ a.Available := by cases a <;> decide
example (R : Rules) : R.assumptions = [] := rfl
example (a : Atom) :
    VeriTile.Spec.FloatingPoint ([] : VeriTile.Spec.Assumptions VeriTile.Triton.FP.GuardedFragment)
      [lhs a] [rhs a] := strict_rewrite a
''',
        }
        for family in FAMILIES:
            with self.subTest(family=family):
                self.check(self.empty(self.source(family)) +
                           f'\nopen VeriTile.Triton.FP.{family}\n' + checks[family])

    def test_removed_empirical_rules_cannot_be_used(self):
        for family, atom in (
            ('ScalarArithmetic', 'addAssociate'), ('Reciprocal', 'fp32'),
            ('Exponential', 'exp_sub .libdevice'),
        ):
            with self.subTest(family=family):
                rules = 'Rules .fp32' if family == 'Reciprocal' else 'Rules'
                args = 'R' if family == 'Reciprocal' else f'R (.{atom})'
                result = self.lean(self.empty(self.source(family)) + f'''
open VeriTile.Triton.FP.{family}
example (R : {rules}) := VeriTile.Triton.FP.{family}.rewrite {args} (by decide)
''')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('decide', result.stdout)

    def test_partial_table_enables_only_the_matching_empirical_rule(self):
        for family, pool, row, selected, absent in (
            ('ScalarArithmetic', 'ReportedAdmission.all', 'ReportedAdmission.fp32_add_assoc',
             'addAssociate', 'mulAssociate'),
            ('Exponential', 'ExpAdmission.all', 'ExpAdmission.fp32_exp_sub',
             'exp_sub .libdevice', 'exp_sub .tl'),
        ):
            with self.subTest(family=family):
                source = self.source(family)
                for other in REPORT_POOLS:
                    source = source.replace(other, f'[{row}]' if other == pool else '[]')
                self.check(source + f'''
open VeriTile.Triton.FP.{family}
example : (Atom.{selected}).Available := by decide
example : ¬ (Atom.{absent}).Available := by decide
example (a : Atom) (h : a.isStrict = true) : a.Available := by simp [Atom.Available, h]
example (a : Atom) (h : a.isStrict = true) : a.entry? = none := by simp [Atom.entry?, h]
example : (candidates.filterMap Atom.entry?).length = 1 := rfl
''')
        source = self.source('Reciprocal').replace('SupplementalAdmission.all',
                                                  '[SupplementalAdmission.fp32_div_mul_rcp]')
        self.check(source + '''
open VeriTile.Triton.FP.Reciprocal
example : Format.fp32.Available := by decide
example : ¬ Format.fp64_fp32.Available := by decide
''')

    def test_strict_maximum_does_not_consume_partial_historical_reports(self):
        source = self.source('Maximum').replace('SupplementalAdmission.all',
                                                '[SupplementalAdmission.fp32_max_commute]')
        self.check(source + '''
open VeriTile.Triton.FP.Maximum
example : Atom.max_commute.Available := by decide
example : ¬ Atom.max_assoc.Available := by decide
example : (candidates.filterMap Atom.entry?).length = 1 := rfl
example (R : Rules) : R.assumptions = [] := rfl
example (a : Atom) :
    VeriTile.Spec.FloatingPoint ([] : VeriTile.Spec.Assumptions VeriTile.Triton.FP.GuardedFragment)
      [lhs a] [rhs a] := strict_rewrite a
''')

    def test_count_domain_is_independent_of_the_historical_report_range(self):
        historical = self.source('CountAdmission')
        changed = historical.replace('def upperExclusive : Nat := 16777216',
                                     'def upperExclusive : Nat := 1024')
        self.assertNotEqual(historical, changed)
        self.check('import VeriTile.Triton.Float.CountConversion\n' + changed + '''
open VeriTile.Triton.FP.CountConversion
example : limit = 16777216 := rfl
example : limit ≠ VeriTile.Triton.FP.CountAdmission.upperExclusive := by decide
example (a : Atom) : a.Available := trivial
example (a : Atom) : VeriTile.Spec.Derivation [] [lhs a] [rhs a] := strict [] a
''')

    def test_count_strict_domain_cannot_be_extended(self):
        source = self.source('CountConversion')
        expanded = source.replace('def limit : Nat := 16777216', 'def limit : Nat := 16777217')
        self.assertNotEqual(source, expanded)
        result = self.lean(expanded)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('type mismatch', result.stdout.lower())

    def test_exp_backend_cannot_borrow_another_implementations_admission(self):
        for atom in ('exp_sub', 'exp_zero', 'exp_neg_inf_sub'):
            with self.subTest(atom=atom):
                row = 'fp32_' + atom + ('' if atom == 'exp_sub' else '_libdevice')
                source = self.source('Exponential').replace('ExpAdmission.all', f'[ExpAdmission.{row}]')
                selected_count = 1 if atom == 'exp_sub' else 0
                checks = '''
example : ¬ (Atom.exp_sub .tl).Available := by decide
''' if atom == 'exp_sub' else f'''
example : (Atom.{atom} .tl).Available := by decide
example (backend : Backend) :
    VeriTile.Triton.FP.Guarded.StrictFragmentSound (Atom.{atom} backend).lhs
      (Atom.{atom} backend).rhs := strict_sound _ rfl
'''
                self.check(source + f'''
open VeriTile.Triton.FP.Exponential
example : (Atom.{atom} .libdevice).Available := by decide
example : (Atom.{atom} .libdevice).report?.isSome = true := by decide
example : (Atom.{atom} .tl).report?.isSome = false := by decide
example : (Atom.{atom} .tl).matches VeriTile.Triton.FP.ExpAdmission.{row} = false := by decide
example : (candidates.filterMap Atom.entry?).length = {selected_count} := rfl
''' + checks)

    def test_historical_catalog_and_current_assumptions_are_separate(self):
        rows = []
        for path in ('report', 'supplement/report', 'primitives/report'):
            rows.extend(json.loads((ROOT / f'experiments/floating_point/{path}/summary.json').read_text())['rows'])
        accepted = {r['rule'] for r in rows if r['format'] in ('fp32', 'int32_fp32') and r['accept']}
        registry = json.loads((ROOT / 'experiments/floating_point/supplement/rules.json').read_text())['rules']
        for family, prefix in (('Exponential', 'EXP-'), ('CountConversion', 'COUNT-'), ('Maximum', 'MAX-'),
                               ('ScalarArithmetic', None)):
            with self.subTest(family=family):
                historical = ('VeriTile.Triton.FP.CountAdmission.all.map (·.ruleID)'
                              if family == 'CountConversion' else
                              '(candidates.filter fun a => a.report?.isSome).map Atom.ruleID')
                strict = ('candidates.map Atom.ruleID' if family in ('CountConversion', 'Maximum') else
                          '(candidates.filter Atom.isStrict).map Atom.ruleID')
                current = ('([] : List String)' if family in ('CountConversion', 'Maximum') else
                           '(candidates.filter fun a => a.entry?.isSome).map Atom.ruleID')
                assumptions = {
                    'Maximum': 'example (R : Rules) : R.assumptions = [] := rfl',
                    'CountConversion': 'example (R : Rules) : R.assumptions = R.arithmetic.assumptions := rfl',
                    'ScalarArithmetic': 'example (R : Rules) : R.assumptions = candidates.filterMap Atom.entry? := rfl',
                    'Exponential': 'example (R : Rules) : R.assumptions = R.arithmetic.assumptions ++ candidates.filterMap Atom.entry? := rfl',
                }[family]
                output = self.check(f'''
import VeriTile.Triton.Float.{family}
import VeriTile.Triton.Float.CountAdmission
open VeriTile.Triton.FP.{family}
{assumptions}
#eval Lean.Json.compress (Lean.toJson (candidates.map Atom.ruleID))
#eval Lean.Json.compress (Lean.toJson ({historical}))
#eval Lean.Json.compress (Lean.toJson ({strict}))
#eval Lean.Json.compress (Lean.toJson ({current}))
''')
                catalog, historical, strict, empirical = [json.loads(json.loads(line))
                                                          for line in output.splitlines()]
                if prefix:
                    self.assertEqual(set(catalog), {r['id'] for r in registry if r['id'].startswith(prefix)})
                self.assertEqual(len(catalog), len(set(catalog)))
                expected = accepted
                if family == 'Exponential':
                    current = json.loads((ROOT / 'experiments/floating_point/supplement/exp_report/summary.json').read_text())
                    expected = {r['rule'] for r in current['rows'] if r['format'] == 'fp32' and r['accept']}
                self.assertEqual(set(historical), expected & set(catalog))
                self.assertEqual(set(strict), STRICT_IDS[family])
                self.assertEqual(set(empirical), (expected & set(catalog)) - STRICT_IDS[family])
                self.assertTrue(set(strict).isdisjoint(empirical))


if __name__ == '__main__':
    unittest.main()
