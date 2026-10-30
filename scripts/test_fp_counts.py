"""Strict bounded counts and empirical arithmetic in Welford and LayerNorm."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FPCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build', 'bench.examples.Welford.FPEquiv',
                                 'bench.examples.FusedLayerNorm.FPEquiv'],
                                cwd=ROOT, text=True, capture_output=True, timeout=600)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check(self, source):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'CountCheck.lean'
            p.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(p)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_count_boundaries_and_independence_from_correct(self):
        self.check((ROOT / 'bench/tests/FPCountConversion.lean').read_text())

    def test_original_public_transforms_and_used_atoms_at_default_print_settings(self):
        expected = {'add_zero', 'cancel', 'add_assoc', 'mul_distrib', 'mul_assoc',
                    'mul_rcp_cancel', 'div_mul_rcp'}
        # Each recurrence proof gets the existing per-check time budget.
        for module, theorem in (('Welford', 'welford_equiv'),
                                ('FusedLayerNorm', 'layernorm_equiv')):
            with self.subTest(theorem=theorem):
                output = self.check(f'''
import bench.examples.{module}.FPEquiv
#print_fp_assumptions VeriTile.Bench.Examples.{module}FPEquiv.{theorem}
''')
                empirical, separator, strict = output.partition('Strict model certificates used by ')
                self.assertEqual(empirical.splitlines()[0], f'FP assumptions used by {theorem}:')
                lines = empirical.splitlines()[1:]
                self.assertEqual({s.strip() for s in lines}, expected)
                self.assertEqual(len(lines), len(expected))
                self.assertTrue(separator)
                for atom in ('addCommute', 'mulCommute', 'mulOne', 'divOne'):
                    self.assertIn(f'ScalarArithmetic.Atom.{atom}', strict)
                for atom in ('zero', 'successor'):
                    self.assertIn(f'CountConversion.Atom.{atom}', strict)

    def test_upper_bound_and_layernorm_empty_case_are_public(self):
        self.check('''
import bench.examples.Welford.FPEquiv
import bench.examples.FusedLayerNorm.FPEquiv
open VeriTile Triton FP.CountConversion
open VeriTile.Bench.Examples
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec
example (stride : Nat) (empty : ReductionPlan 0) (R : Rules) :
    WelfordFPContract.online "x" "mean" "variance" limit stride empty ≡[R]
      WelfordFPContract.twopass "x" "mean" "variance" limit stride empty :=
  WelfordFPEquiv.welford_equiv limit stride (by decide) le_rfl empty R
example (stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) (R : Rules) :
    LayerNormFPContract.fused "x" "gamma" "beta" "y" 0 stride ε empty ≡[R]
      LayerNormFPContract.twopass "x" "gamma" "beta" "y" 0 stride ε empty :=
  FusedLayerNormFPEquiv.layernorm_equiv 0 stride (by decide) ε empty R
''')


if __name__ == '__main__':
    unittest.main()
