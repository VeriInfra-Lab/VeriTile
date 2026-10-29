"""Fail-closed explicit boolean profiles; production shell gates are exercised."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source


class SourceConstexprProfileTests(unittest.TestCase):
    python = '''import triton
import triton.language as tl
@triton.jit
def kernel(X, value, other, runtime, FLAG: tl.constexpr = True):
    if FLAG:
        tl.store(X, value)
    else:
        tl.store(X, other)
    if runtime:
        value = value + 1
'''
    lean_body = '''def chosen (X : RegionName) (value other runtime : Nat) := triton {
  tl.region X = tl.float32
  tl.store(X, other)
  if runtime {
    value = value + 1
  }
}
'''

    def lean(self, values=None, binding=None, body=None, extra=''):
        profile = {'python': values if values is not None else {'FLAG': False},
                   'lean': binding if binding is not None else {'source': 'chosen', 'args': {}}}
        return ('/-\nSource-audit target: kernel\nSource-audit Lean: chosen\n'
                'Source-audit constexpr: ' + json.dumps(profile) + '\n' + extra + '-/\n' +
                (self.lean_body if body is None else body))

    def test_only_declared_boolean_guards_are_selected_and_runtime_branches_remain(self):
        for value, expected in [(False, 'tl.store(X, other)'), (True, 'tl.store(X, value)')]:
            with self.subTest(value=value):
                lean = self.lean({'FLAG': value})
                body = source.python_audit_kernel_body(self.python, lean)
                self.assertEqual(source.tl_calls(body, 'store'), [expected])
                self.assertIn('if runtime:', body)
                self.assertNotIn('if FLAG:', body)
                self.assertEqual(source.python_source_parameter_names(self.python, lean),
                                 ['X', 'value', 'other', 'runtime'])
        # An original default of True cannot override an explicit false profile.
        self.assertEqual(source.source_constexpr_profile(self.python, self.lean()), (('FLAG', False),))
        negated = self.python.replace('if FLAG:', 'if not FLAG:')
        self.assertEqual(source.tl_calls(source.python_audit_kernel_body(negated, self.lean()), 'store'),
                         ['tl.store(X, value)'])

    def test_full_original_jit_declaration_is_required(self):
        for changed in [self.python.replace('FLAG: tl.constexpr', 'FLAG'),
                        self.python.replace('FLAG: tl.constexpr', 'FLAG: int'),
                        self.python.replace('FLAG: tl.constexpr', 'FLAG: "tl.constexpr"'),
                        self.python.replace('FLAG: tl.constexpr', 'FLAG: other.constexpr'),
                        self.python.replace('@triton.jit\n', ''),
                        self.python + self.python,
                        self.python.replace('runtime, FLAG', 'runtime, *extra, FLAG')]:
            with self.subTest(python=changed), self.assertRaises(ValueError):
                source.source_constexpr_profile(changed, self.lean())
        for lean in [self.lean({'missing': False}), self.lean({'runtime': False}),
                     self.lean().replace('Source-audit target: kernel', 'Source-audit target: helper')]:
            with self.subTest(lean=lean), self.assertRaises(ValueError):
                source.source_constexpr_profile(self.python, lean)

    def test_profile_schema_and_exact_lean_binding_are_fail_closed(self):
        for values in [{}, {'FLAG': 0}, {'FLAG': 1}, {'FLAG': 'False'}, {'FLAG': None}, {'FLAG': []}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                source.source_constexpr_profile(self.python, self.lean(values))
        for binding in [{'source': 'other', 'args': {}}, {'source': 'chosen', 'args': {'unknown': 'half'}},
                        {'source': 'chosen'}, {'source': 'chosen', 'args': {}, 'extra': 0}]:
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                source.source_constexpr_profile(self.python, self.lean(binding=binding))
        good = self.lean()
        for bad in [good.replace('"FLAG": false', '"FLAG": false, "FLAG": true'),
                    good.replace('"args": {}', '"args": {}, "args": {}'),
                    good.replace('"python":', '"unknown":'),
                    good.replace('"FLAG": false', '"FLAG": False'),
                    good.replace('-/\n', 'Source-audit constexpr: {}\n-/\n', 1),
                    good.replace('Source-audit Lean: chosen\n', '')]:
            with self.subTest(lean=bad), self.assertRaises(ValueError):
                source.source_constexpr_profile(self.python, bad)

    def test_all_arms_are_validated_for_rebinding_and_non_guard_uses(self):
        for statement in ['FLAG = False', 'FLAG += 1', 'del FLAG', 'value = FLAG',
                          'if FLAG == False:\n            value = other',
                          'if FLAG and runtime:\n            value = other',
                          'for FLAG in values:\n            value = other',
                          'def helper():\n            return FLAG',
                          'value = [FLAG for x in values]']:
            changed = self.python.replace('        tl.store(X, value)',
                                          '        ' + statement + '\n        tl.store(X, value)')
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                source.source_constexpr_profile(changed, self.lean())
        with self.assertRaisesRegex(ValueError, 'no supported if guard'):
            source.source_constexpr_profile(self.python.replace('if FLAG:', 'if runtime:'), self.lean())

    def test_parameter_maps_cannot_rename_or_capture_profile_parameters(self):
        for params in ['FLAG=runtime', 'X=FLAG']:
            lean = self.lean(extra='Source-audit params: ' + params + '\n')
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, 'constexpr profile parameters'):
                source.source_parameter_map(self.python, lean)
        lean = self.lean(extra='Source-audit params: X=Y\n', body=self.lean_body.replace('X', 'Y'))
        body = source.python_audit_kernel_body(self.python, lean)
        self.assertEqual(source.tl_calls(body, 'store'), ['tl.store(Y, other)'])

    def test_docstrings_do_not_invent_statements_and_other_functions_stay_original(self):
        document = '    """Prose\n    c = invented\n    tl.store(X, decoy)\n    """\n'
        python = self.python.replace('    if FLAG:', document + '    if FLAG:', 1)
        helper = '\ndef helper(FLAG):\n    return "FLAG 字符串"\n'
        result = source.python_source_with_constexpr_profile(python + helper, self.lean())
        self.assertTrue(result.endswith(helper))
        self.assertIn('FLAG: tl.constexpr = True', result)
        self.assertNotIn('invented', result)
        self.assertNotIn('decoy', result)
        ast.parse(result)
        # A leading if arm may yield a string expression, but that expression
        # was not the original function docstring and must remain visible.
        python = self.python.replace('        tl.store(X, other)',
                                     '        "branch string remains"\n        tl.store(X, other)')
        result = source.python_source_with_constexpr_profile(python, self.lean())
        self.assertIn('branch string remains', result)
        self.assertIn('tl.store(X, other)', result)

    def macro(self):
        profile = {'python': {'FLAG': False}, 'lean': {'source': 'typed%',
                   'args': {'storage': 'tl.float32', 'mode': 'half'}}}
        return ('/-\nSource-audit target: kernel\nSource-audit Lean: typed%\n'
                'Source-audit args: storage=tl.float32, mode=half\n'
                'Source-audit constexpr: ' + json.dumps(profile) + '\n-/\n' + '''macro_rules
  | `(typed%[$storage:tritonDType, $mode:ident]) => do
    `(fun (X : RegionName) (value other runtime : Nat) => triton {
      tl.region X = $storage
      tl.store(X, other)
      if runtime { value = value + 1 }
    })
''')

    def test_macro_binding_pins_all_selected_arguments_and_types(self):
        good = self.macro()
        self.assertEqual(source.source_constexpr_profile(self.python, good), (('FLAG', False),))
        for bad in [good.replace('mode=half', 'mode=fp8'),
                    good.replace('storage=tl.float32', 'storage=tl.float16'),
                    good.replace(', mode=half', ''),
                    good.replace('"mode": "half"', '"mode": 0'),
                    good.replace('"mode": "half"', '"missing": "half"'),
                    good.replace('$mode:ident', '$different:ident')]:
            with self.subTest(lean=bad), self.assertRaises(ValueError):
                source.source_constexpr_profile(self.python, bad)





if __name__ == '__main__':
    unittest.main()
