"""Arithmetic helper expansion and selected-source augmented assignments."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench.source_helpers import python_source_with_helpers


class SourceHelperTests(unittest.TestCase):
    python = '''import triton
import triton.language as tl
@triton.jit
def previous(a, b):
    return tl.cdiv(a, b) * b - b
@triton.jit
def kernel(X, N, B):
    previous_multiple = previous(N, B)
    tl.store(X + previous_multiple, N)
'''
    body = '''def chosen (X : RegionName) (N B : Nat) := triton {
    tl.region X = tl.float32
    previous_multiple = tl.cdiv($(N), $(B)) * $(B) - $(B)
    tl.store(X + previous_multiple, $(N))
}
'''

    def lean(self, names=None, binding=None, body=None, extra=''):
        profile = {'python': ['previous'] if names is None else names,
                   'lean': {'source': 'chosen', 'args': {}} if binding is None else binding}
        return ('/-\nSource-audit target: kernel\nSource-audit Lean: chosen\n'
                'Source-audit helpers: ' + json.dumps(profile) + '\n' + extra + '-/\n'
                + (self.body if body is None else body))

    def production(self, python, lean):
        script = (ROOT/'bench/audit_tritonbench_g.sh').read_text()
        section = script[script.index('# A source view must be extractable'):
                         script.index('# ---- Machine-checkable trust audit')]
        command = 'failures=0\nPORTS_ROOT="$1"\n' + section + '\nexit "$failures"\n'
        with tempfile.TemporaryDirectory() as directory:
            port = Path(directory)/'fixture'
            port.mkdir()
            (port/'fixture.py').write_text(python)
            (port/'Fixture.lean').write_text(lean)
            return subprocess.run(['bash', '-c', command, 'helper-regression', directory],
                                  cwd=ROOT, text=True, capture_output=True, timeout=30)

    def test_exact_expression_is_inlined_without_changing_original_function(self):
        result = python_source_with_helpers(self.python, self.lean())
        self.assertIn('previous_multiple = tl.cdiv(N, B) * B - B', result)
        self.assertEqual(result[:result.index('def kernel')], self.python[:self.python.index('def kernel')])
        self.assertEqual(source.tl_calls(source.python_audit_kernel_body(self.python, self.lean()), 'cdiv'),
                         ['tl.cdiv(N, B)'])
        self.assertEqual(python_source_with_helpers(self.python, self.body), self.python)
        alternative = self.python.replace('import triton.language as tl', 'from triton import language as tl')
        self.assertIn('previous_multiple = tl.cdiv(N, B)', python_source_with_helpers(alternative, self.lean()))

    def test_exact_lean_expression_is_required_including_arithmetic_and_order(self):
        for rhs in ['tl.cdiv($(N), $(B)) * $(B) + $(B)',
                    'tl.cdiv($(N), $(B)) * ($(B) - $(B))',
                    'tl.cdiv($(B), $(N)) * $(B) - $(B)',
                    '$(B) * tl.cdiv($(N), $(B)) - $(B)']:
            lean = self.lean(body=self.body.replace('tl.cdiv($(N), $(B)) * $(B) - $(B)', rhs))
            with self.subTest(rhs=rhs), self.assertRaisesRegex(ValueError, 'expression mismatch'):
                python_source_with_helpers(self.python, lean)
        with self.assertRaisesRegex(ValueError, 'expression mismatch'):
            python_source_with_helpers(self.python.replace('* b - b', '* b + b'), self.lean())

    def test_schema_duplicates_and_exact_selector_binding_are_rejected(self):
        for names in [[], ['previous', 'previous'], ['missing'], ['tl'], [7], ['x.y'], ['class']]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, self.lean(names=names))
        for binding in [{'source': 'other', 'args': {}}, {'source': 'chosen', 'args': {'x': 1}},
                        {'source': 'chosen'}, {'source': 'chosen', 'args': {}, 'unknown': 0}]:
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, self.lean(binding=binding))
        for bad in [self.lean().replace('"python": ["previous"]', '"python": ["previous"], "python": []'),
                    self.lean().replace('"args": {}', '"args": {}, "args": {}'),
                    self.lean().replace('Source-audit target: kernel', 'Source-audit target: other'),
                    self.lean().replace('-/\n', 'Source-audit helpers: {}\n-/\n', 1)]:
            with self.subTest(lean=bad), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, bad)

    def test_only_single_return_arithmetic_helpers_are_supported(self):
        for changed in ['    tl.store(a, b)\n    return a',
                        '    temp = a + b\n    return temp',
                        '    if a:\n        return a\n    return b',
                        '    return tl.load(a)', '    return another(a, b)',
                        '    return tl.cdiv(a, b, 1)', '    return tl.cdiv(a, b=b)',
                        '    return a[0]', '    return a.to(tl.float32)',
                        '    return a / b', '    return a ** b', '    return True',
                        '    return captured + b', '    return [a, b]']:
            python = self.python.replace('    return tl.cdiv(a, b) * b - b', changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                python_source_with_helpers(python, self.lean())

    def test_signature_decorators_and_duplicate_definitions_are_rejected(self):
        for signature in ['a, b=1', 'a, *b', 'a, **b', 'a, /, b', 'a, *, b', 'a: int, b', 'a, a']:
            python = self.python.replace('def previous(a, b):', 'def previous(' + signature + '):')
            with self.subTest(signature=signature), self.assertRaises((ValueError, SyntaxError)):
                python_source_with_helpers(python, self.lean())
        for python in [self.python.replace('@triton.jit\ndef previous', '@triton.jit()\ndef previous'),
                       self.python.replace('@triton.jit\ndef previous', '@wrapper\n@triton.jit\ndef previous'),
                       self.python.replace('def previous(a, b):', 'def previous(a, b) -> int:'),
                       self.python.replace('def previous(a, b):', 'async def previous(a, b):'),
                       self.python + '\ndef previous(a, b):\n    return a\n']:
            with self.subTest(python=python), self.assertRaises(ValueError):
                python_source_with_helpers(python, self.lean())

    def test_module_and_parameter_rebindings_are_rejected(self):
        for suffix in ['\nprevious = replacement\n', '\ntl = replacement\n',
                       '\nimport foreign as tl\n', '\nfrom foreign import *\n',
                       '\ntl.cdiv = replacement\n', '\nsetattr(tl, "cdiv", replacement)\n',
                       '\nalias = tl\nalias.cdiv = replacement\n',
                       '\nalias = triton.language\nalias.cdiv = replacement\n',
                       '\nalias = tl.cdiv\n', '\nalias = previous\n',
                       '\ntriton.language.cdiv = replacement\n',
                       '\ntl.cdiv.__code__ = replacement\n',
                       '\nfrom triton import language as other\nother.cdiv = replacement\n',
                       '\nfrom triton.language import cdiv as other\nother.__code__ = replacement\n',
                       '\nimport triton.language as other\nother.cdiv = replacement\n',
                       '\nalias = tl.cdiv.__class__\nalias.__call__ = replacement\n',
                       '\n__import__("triton.language", fromlist=["cdiv"]).cdiv = replacement\n',
                       '\nimport sys\nsys.modules["triton.language"].cdiv = replacement\n',
                       '\nimport importlib\nimportlib.import_module("triton.language").cdiv = replacement\n',
                       '\n__builtins__["globals"]()["previous"] = replacement\n',
                       '\nglobals()["previous"] = replacement\n',
                       '\ntry:\n    pass\nexcept Exception as previous:\n    pass\n',
                       '\ndef host():\n    global previous\n    previous = replacement\n']:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                python_source_with_helpers(self.python + suffix, self.lean())
        for python in [self.python.replace('def kernel(X, N, B)', 'def kernel(X, N, B, previous)'),
                       self.python.replace('def previous(a, b)', 'def previous(a, tl)'),
                       self.python.replace('import triton.language as tl', 'import foreign as tl'),
                       self.python.replace('import triton\n', '')]:
            with self.subTest(python=python), self.assertRaises(ValueError):
                python_source_with_helpers(python, self.lean())

    def test_effectful_arguments_indirection_and_non_assignment_uses_are_rejected(self):
        for rhs in ['previous(tl.load(X), B)', 'previous(N + 1, B)', 'previous(N, 2)',
                    'previous(N, b=B)', 'previous(*args)', 'previous(N, B) + 1',
                    'previous', 'previous(N, previous(N, B))']:
            with self.subTest(rhs=rhs), self.assertRaises(ValueError):
                python_source_with_helpers(self.python.replace('previous(N, B)', rhs), self.lean())
        for statement in ['X = previous(N, B)', 'x, previous_multiple = previous(N, B)',
                          'previous_multiple: int = previous(N, B)', 'tl.store(X, previous(N, B))',
                          'previous_multiple = previous(N, B)\n    previous_multiple += 1',
                          'def nested():\n        return previous(N, B)']:
            python = self.python.replace('previous_multiple = previous(N, B)', statement)
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                python_source_with_helpers(python, self.lean())

    def test_all_original_branches_are_checked(self):
        python = self.python.replace('    previous_multiple =', '    if runtime:\n        previous = bad\n    previous_multiple =')
        with self.assertRaises(ValueError):
            python_source_with_helpers(python, self.lean())
        python = self.python.replace('    previous_multiple =', '    """helper docs"""\n    previous_multiple =')
        self.assertIn('previous_multiple = tl.cdiv', python_source_with_helpers(python, self.lean()))

    def test_macro_arguments_are_fixed_or_explicit_symbolic_dtypes(self):
        macro = '''macro_rules
  | `(chosen%[$dtype:tritonDType, $mode:ident]) => do
    `(fun (X : RegionName) (N B : Nat) => triton {
      tl.region X = $dtype
      previous_multiple = tl.cdiv($(N), $(B)) * $(B) - $(B)
      tl.store(X + previous_multiple, $(N))
    })
'''
        def view(binding, args):
            return self.lean(binding=binding, body=macro, extra='Source-audit args: '+args+'\n').replace(
                'Source-audit Lean: chosen\n', 'Source-audit Lean: chosen%\n')
        fixed = {'source': 'chosen%', 'args': {'dtype': 'tl.float32', 'mode': 'original'}}
        symbolic = {'source': 'chosen%', 'args': {'mode': 'original'}, 'symbolic': ['dtype']}
        for binding, args in [(fixed, 'dtype=tl.float32, mode=original'), (symbolic, 'mode=original')]:
            self.assertIn('previous_multiple = tl.cdiv', python_source_with_helpers(self.python, view(binding, args)))
        for binding, args in [({'source': 'chosen%', 'args': {'dtype': 'tl.float32'}}, 'dtype=tl.float32'),
                              (dict(fixed, symbolic=['dtype']), 'dtype=tl.float32, mode=original'),
                              (dict(symbolic, symbolic=['dtype', 'dtype']), 'mode=original'),
                              (dict(symbolic, symbolic=['missing']), 'mode=original'),
                              ({'source': 'chosen%', 'args': {}, 'symbolic': ['dtype', 'mode']}, ''),
                              (dict(symbolic, symbolic='dtype'), 'mode=original'),
                              (dict(symbolic, symbolic=[True]), 'mode=original')]:
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, view(binding, args))

    def test_parameter_map_is_applied_after_capture_free_inlining(self):
        lean = self.lean(body=self.body.replace('N', 'SIZE'), extra='Source-audit params: N=SIZE\n')
        body = source.python_audit_kernel_body(self.python, lean)
        self.assertIn('tl.cdiv(SIZE, B)', body)
        # The original helper reference must remain visible to map capture checks.
        bad = self.lean(body=self.body.replace('N', 'previous'), extra='Source-audit params: N=previous\n')
        with self.assertRaisesRegex(ValueError, 'capture'):
            python_source_with_helpers(self.python, bad)

    def test_unicode_and_comments_do_not_corrupt_byte_replacements(self):
        python = self.python.replace('previous_multiple = previous(N, B)',
                                     'message = "例子"; previous_multiple = previous(N, B) # previous(N, B)')
        result = python_source_with_helpers(python, self.lean())
        self.assertIn('message = "例子"; previous_multiple = tl.cdiv(N, B) * B - B # previous(N, B)', result)
        ast.parse(result)

    def test_production_baseline_and_helper_mutations(self):
        result = self.production(self.python, self.lean())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for python, lean in [(self.python.replace('* b - b', '* b + b'), self.lean()),
                             (self.python, self.lean(body=self.body.replace(' - $(B)', ' + $(B)'))),
                             (self.python, self.lean().replace('"previous"', '"missing"'))]:
            for marker in ['', 'Translation-surface blocker: existing fixture\n']:
                with self.subTest(python=python, lean=lean, marker=marker):
                    result = self.production(python, lean.replace('/-\n', '/-\n' + marker, 1))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('FAIL source-view extraction scan', result.stdout)

    def test_augmented_assignment_gate_uses_the_selected_jit(self):
        other = '\n@triton.jit\ndef other(X, N):\n    N += 1\n    tl.store(X, N)\n'
        result = self.production(self.python + other, self.lean())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        selected = self.python.replace('    tl.store(X + previous_multiple, N)',
                                       '    N += 1\n    tl.store(X + previous_multiple, N)')
        # Neither comments nor a different Lean body can supply the missing update.
        for decoy in ['-- N += 1\n', 'def sibling (N : Nat) := triton { N += 1 }\n']:
            with self.subTest(decoy=decoy):
                result = self.production(selected + other, self.lean() + decoy)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('FAIL Python += missing from Lean', result.stdout)
        good = self.lean(body=self.body.replace('    tl.store(', '    N += 1\n    tl.store('))
        result = self.production(selected + other, good)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
