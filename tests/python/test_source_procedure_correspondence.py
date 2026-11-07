import unittest

from scripts.bench.jit_correspondence import check, check_source_procedure


class SourceProcedureCorrespondenceTests(unittest.TestCase):
    python = """@triton.jit
def helper(X, Y, N, EPS: tl.constexpr, ROT: tl.constexpr):
    N = tl.cast(N, tl.int64)
    value = tl.load(X + 0) + EPS
    if ROT:
        value = value * 2.0
    tl.store(Y + 0, value)
"""
    lean = """def helper : SourceProcedure :=
  triton_source_procedure (X, Y, N, EPS, ROT)
    constexpr_args [EPS, ROT] fp32_constants [EPS] {
    N = tl.cast(N, tl.int64)
    value = tl.load(X + 0) + EPS
    if ROT:
        value = value * 2.0
    tl.store(Y + 0, value)
  }
"""

    def compare(self, python=None, lean=None, constants=None):
        return check_source_procedure(self.python if python is None else python,
                                      self.lean if lean is None else lean, 'helper', 'helper',
                                      ['EPS'] if constants is None else constants)

    def test_exact_original_parameters_body_and_constants(self):
        self.assertEqual(self.compare()['status'], 'PASS')

    def test_triton_params_with_and_without_fp32_constants(self):
        source = self.lean.replace('triton_source_procedure', 'with_triton_params')
        source = source.replace('fp32_constants [EPS] {', 'fp32_constants [EPS] in triton {')
        self.assertEqual(self.compare(lean=source)['status'], 'PASS')
        without_constants = source.replace(' fp32_constants [EPS]', '')
        self.assertEqual(self.compare(lean=without_constants, constants=[])['status'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'materialized constants differ'):
            self.compare(lean=without_constants)

    def test_source_cast_mask_and_store_are_not_skipped(self):
        for before, after in [('tl.int64', 'tl.int32'), ('X + 0', 'X + 1'),
                              ('tl.store(Y + 0, value)', 'tl.store(Y + 1, value)')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(lean=self.lean.replace(before, after))

    def test_exact_parameter_order_and_constexpr_subset(self):
        for before, after in [('X, Y, N, EPS, ROT', 'Y, X, N, EPS, ROT'),
                              ('[EPS, ROT]', '[ROT, EPS]'), ('fp32_constants [EPS]', 'fp32_constants [N]')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(lean=self.lean.replace(before, after))

    def test_rebinding_constexpr_is_rejected_even_for_equal_bodies(self):
        python = self.python.replace('    value =', '    EPS = 1.0\n    value =', 1)
        lean = self.lean.replace('    value =', '    EPS = 1.0\n    value =', 1)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.compare(python=python, lean=lean)

    def test_parent_checks_exact_helper_binding_and_literal_arguments(self):
        python = self.python + """\n@triton.jit
def parent(X, Y, N, EPS: tl.constexpr):
    helper(X=X, Y=Y, N=N, EPS=EPS, ROT=True)
"""
        lean = """def abi : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  scalars := [("N",.integer)]
  constexpr := ["EPS"]

def parent (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call abi call_args [N := N] fp32_constants [EPS]
    source_procedures [helper := helper] in triton {
    helper(X=X, Y=Y, N=N, EPS=EPS, ROT=True)
  }
"""
        profile = {'pointers': [['X', 'fp32'], ['Y', 'fp16']],
                   'scalars': [['N', 'integer']], 'constexpr': ['EPS']}
        def compare(source, constants):
            return check(python, source, 'parent', 'parent', 'abi', profile,
                         expected_call_bindings={'N': 'N'}, expected_fp32_constants=constants,
                         expected_source_procedures={'helper': 'helper'})
        self.assertEqual(compare(lean, ['EPS'])['status'], 'PASS')
        bound = lean.replace('(N : Nat) : ComputeKernel', '(N : Nat) (EPS : ℝ) : ComputeKernel')
        self.assertEqual(compare(bound.replace(' fp32_constants [EPS]', ''), None)['status'], 'PASS')
        self.assertEqual(compare(bound.replace('fp32_constants [EPS]', 'fp32_constants []'), [])['status'], 'PASS')
        for before, after in [('helper := helper', 'helper := other'), ('ROT=True', 'ROT=False')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                compare(lean.replace(before, after), ['EPS'])


if __name__ == '__main__':
    unittest.main()
