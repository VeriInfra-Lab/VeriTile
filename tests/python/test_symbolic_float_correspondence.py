import unittest

from scripts.bench.jit_correspondence import check


class SymbolicFloatCorrespondenceTests(unittest.TestCase):
    python = """@triton.jit
def kernel(X, N, EPS: tl.constexpr):
    N = tl.cast(N, tl.int64)
    value = tl.load(X + tl.arange(0, 1)) + EPS
    tl.store(X + tl.arange(0, 1), value)
"""
    lean = """def signature : CallSignature where
  pointers := [("X",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["EPS"]

def kernel (X : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
    N = tl.cast(N, tl.int64)
    value = tl.load(X + tl.arange(0, 1)) + EPS
    tl.store(X + tl.arange(0, 1), value)
  }
"""
    signature = {"pointers": [["X", "fp32"]], "scalars": [["N", "integer"]],
                 "constexpr": ["EPS"]}

    def compare(self, python=None, lean=None, constants=None):
        return check(self.python if python is None else python,
                     self.lean if lean is None else lean, "kernel", "kernel",
                     "signature", self.signature, expected_call_bindings={"N": "N"},
                     expected_fp32_constants=["EPS"] if constants is None else constants)

    def test_exact_body_and_original_constexpr_signature(self):
        self.assertEqual(self.compare()["status"], "PASS")

    def test_clause_cannot_be_changed_or_omitted(self):
        for clause in ["", "fp32_constants []", "fp32_constants [N]",
                       "fp32_constants [EPS, EPS]"]:
            with self.subTest(clause=clause), self.assertRaises(ValueError):
                self.compare(lean=self.lean.replace("fp32_constants [EPS]", clause))

    def test_constants_require_an_explicit_constexpr_review(self):
        for constants in [[], ["N"], ["EPS", "EPS"], ["EPS + N"], [7]]:
            with self.subTest(constants=constants), self.assertRaises(ValueError):
                self.compare(constants=constants)

    def test_scalar_materialization_cannot_hide_source_cast_changes(self):
        with self.assertRaises(ValueError):
            self.compare(lean=self.lean.replace("N = tl.cast(N, tl.int64)", "N = N"))

    def test_constexpr_rebinding_is_rejected_even_if_bodies_match(self):
        python = self.python.replace("    value =", "    EPS = 1.0\n    value =")
        lean = self.lean.replace("    value =", "    EPS = 1.0\n    value =")
        with self.assertRaisesRegex(ValueError, "must remain immutable"):
            self.compare(python=python, lean=lean)


if __name__ == "__main__":
    unittest.main()
