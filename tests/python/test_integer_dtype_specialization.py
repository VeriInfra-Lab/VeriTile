import unittest
from scripts.bench.jit_correspondence import check


class IntegerDTypeSpecializationTests(unittest.TestCase):
    python = '''@triton.jit
def kernel(output, B: tl.constexpr, ACC_TYPE: tl.constexpr = tl.int32):
    value = tl.zeros((B,), dtype=ACC_TYPE)
    tl.store(output + tl.arange(0, B), value)
'''
    lean = '''def profile : CallSignature where
  pointers := [("output",.i32)]
  scalars := []
  constexpr := ["B","ACC_TYPE"]
def body (output : RegionName) (B : Nat) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .i32
  with_triton_call profile call_args [] in triton {
    value = tl.zeros((B,), dtype=ACC_TYPE)
    tl.store(output + tl.arange(0, B), value)
  }
'''
    signature = {'pointers': [['output', 'i32']],
                 'scalars': [], 'constexpr': ['B', 'ACC_TYPE']}

    def verify(self, lean=None, binding='SourceDType.i32'):
        return check(self.python, self.lean if lean is None else lean, 'kernel', 'body',
                     'profile', self.signature, expected_call_bindings={},
                     expected_constexpr_bindings={'ACC_TYPE': binding})

    def test_static_integer_specialization(self):
        self.assertEqual(self.verify()['status'], 'PASS')

    def test_changed_or_rebound_constructor_is_rejected(self):
        for before, after in [(':= .i32', ':= .i64'),
                              ('SourceDType := .i32', 'FloatDType := .fp32'),
                              ('with_triton_call', 'let ACC_TYPE : SourceDType := .i64\n  with_triton_call')]:
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.verify(self.lean.replace(before, after))

    def test_noninteger_carrier_is_not_accepted_as_integer(self):
        with self.assertRaisesRegex(ValueError, 'unsupported closed constexpr'):
            self.verify(binding='SourceDType.fp32')
