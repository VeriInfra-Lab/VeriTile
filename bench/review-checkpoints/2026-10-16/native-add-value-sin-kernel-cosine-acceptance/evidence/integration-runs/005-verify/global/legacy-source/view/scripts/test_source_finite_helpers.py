"""Ordinary-audit finite-helper regressions. Root runs this draft.

No Triton import, native compilation, Lean build, or kernel execution occurs.
The Python fixture is the checked-in pinned source. The Lean fixture is the
complete canonical public Kernels.lean, with metadata supplied by this test.
"""
import ast
import copy
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
from bench import source_isfinite as strict
from bench.source_helpers import python_source_with_helpers


def mutate(text, before, after):
    if before not in text or before == after:
        raise AssertionError('Missing mutation anchor: ' + before)
    return text.replace(before, after, 1)


class SourceFiniteHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        port = ROOT / 'bench/tritonbench_g/isfinite_kernel'
        cls.python = (port / 'isfinite_kernel.py').read_text()
        cls.original_lean = (port / 'Kernels.lean').read_text()
        # Existing metadata is replaced only in an in-memory fixture, never
        # written to the canonical source or inferred from an arbitrary dtype.
        cls.original_lean = re.sub(r'^Source-audit helpers:[^\n]*\n', '', cls.original_lean, flags=re.M)

    def lean(self, storage='tl.float32', profile=None):
        if profile is None:
            profile = {'kind': 'isfinite-dtype-dispatch', 'python': ['isfinite_func'],
                       'lean': {'source': 'isfiniteSource%', 'args': {'storage': storage}}}
        value = re.sub(r'^Source-audit args: storage=[^\n]*$',
                       'Source-audit args: storage=' + storage, self.original_lean, count=1, flags=re.M)
        return mutate(value, 'Source-audit Lean: isfiniteSource%\n',
                      'Source-audit Lean: isfiniteSource%\nSource-audit helpers: ' + json.dumps(profile) + '\n')

    def view(self, lean):
        return source.select_lean_audit_source(lean)

    def normalized_lean(self, view):
        body = source.strip_region_declarations(source.lean_first_triton_body(view))
        body = re.sub(r'\$\((\w+)\)', r'\1', body)
        body = re.sub(r'\((\[\s*0\s*\])\s*:\s*List Nat\)', r'\1', body)
        self.assertNotIn('$', body)
        return strict.Notation().visit(strict.braces(body))

    def production(self, python, lean):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        section = script[script.index('# A source view must be extractable'):
                         script.index('# ---- Machine-checkable trust audit')]
        command = 'failures=0\nPORTS_ROOT="$1"\n' + section + '\nexit "$failures"\n'
        with tempfile.TemporaryDirectory() as directory:
            port = Path(directory) / 'fixture'
            port.mkdir()
            (port / 'fixture.py').write_text(python)
            # Legacy fixture filename avoids needing unrelated public proof
            # files. Local public imports use the original port's prefix and
            # therefore remain external to this source-view fixture.
            (port / 'Fixture.lean').write_text(lean)
            return subprocess.run(['bash', '-c', command, 'finite-helper-regression', directory],
                                  cwd=ROOT, text=True, capture_output=True, timeout=60)

    def test_every_profile_matches_the_entire_writer_ast(self):
        for storage in strict.STORAGES:
            with self.subTest(storage=storage):
                lean = self.lean(storage)
                view = self.view(lean)
                expanded = python_source_with_helpers(self.python, view)
                original_kernel = source._explicit_python_audit_kernel(self.python, view)
                kernel = source._explicit_python_audit_kernel(expanded, view)
                self.assertEqual(ast.dump(kernel.args), ast.dump(original_kernel.args))
                actual = strict.Notation().visit(ast.Module(body=kernel.body, type_ignores=[]))
                self.assertEqual(strict.dump(actual), strict.dump(self.normalized_lean(view)))
                self.assertEqual(len(source._python_parameter_names(kernel)), 11)
                branch = kernel.body[2]
                self.assertIsInstance(branch, ast.If)
                self.assertEqual(ast.unparse(branch.test), 'one_tile_per_cta')
                self.assertIsInstance(branch.orelse[1], ast.For)
                self.assertEqual(ast.unparse(branch.orelse[1].iter), 'range(0, tiles_per_cta)')

    def test_imports_and_helper_remain_exact_and_both_calls_are_visible(self):
        before = ast.parse(self.python)
        for storage in strict.STORAGES:
            view = self.view(self.lean(storage))
            expanded = python_source_with_helpers(self.python, view)
            after = ast.parse(expanded)
            self.assertEqual(len(before.body), len(after.body))
            for left, right in zip(before.body, after.body):
                if not isinstance(left, ast.FunctionDef) or left.name != strict.KERNEL:
                    self.assertEqual(ast.dump(left), ast.dump(right))
            for alias, symbol in strict.ALIASES.items():
                self.assertIn(strict.chain_text(alias, symbol), expanded)
            py_body = source.python_audit_kernel_body(self.python, view)
            lean_body = source.lean_first_triton_body(view)
            symbol = 'isfinited' if storage == 'tl.float64' else 'finitef'
            token = 'tl.extra.cuda.libdevice.' + symbol + '('
            self.assertEqual(py_body.count(token), 2)
            self.assertEqual(lean_body.count(token), 2)
            self.assertNotIn('source_helper', lean_body)
            self.assertNotIn('source_alias', lean_body)
            self.assertEqual(py_body.count('.to(tl.float32)'), 0 if storage == 'tl.float64' else 2)
            self.assertEqual(lean_body.count('.to(tl.float32)'), 0 if storage == 'tl.float64' else 2)
            call_pattern = r'\btl(?:\.[A-Za-z_][A-Za-z0-9_]*)+\s*\('
            self.assertEqual(re.findall(call_pattern, py_body), re.findall(call_pattern, lean_body))

    def test_no_metadata_means_no_inferred_helper_or_dtype(self):
        view = self.view(self.original_lean)
        self.assertEqual(python_source_with_helpers(self.python, view), self.python)
        body = source.lean_first_triton_body(view)
        self.assertIn('tl.source_helper isfinite_func(x)', body)
        self.assertIn('out0 = isfinite_func(in0)', body)
        self.assertNotIn('tl.extra.cuda.libdevice.finitef(in0', body)

    def test_exact_schema_binding_and_profile_are_required(self):
        valid = json.loads(source.source_audit_metadata(self.lean(), 'helpers'))
        profiles = []
        for key, value in [('kind', 'anything'), ('python', ['other']), ('extra', True)]:
            profile = copy.deepcopy(valid)
            profile[key] = value
            profiles.append(profile)
        for binding in [{'source': 'isfiniteCandidate%', 'args': {'storage': 'tl.float32'}},
                        {'source': 'isfiniteSource%', 'args': {'storage': 'tl.float64'}},
                        {'source': 'isfiniteSource%', 'args': {}},
                        {'source': 'isfiniteSource%', 'args': {'storage': 'tl.float32'}, 'symbolic': []}]:
            profile = copy.deepcopy(valid)
            profile['lean'] = binding
            profiles.append(profile)
        for profile in profiles:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, self.lean(profile=profile))
        for storage in ['tl.int1', 'tl.float16', 'OUT_DTYPE']:
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, self.lean(storage))
        lean = self.lean()
        for changed in [mutate(lean, '"kind": "isfinite-dtype-dispatch"',
                              '"kind": "isfinite-dtype-dispatch", "kind": "isfinite-dtype-dispatch"'),
                        mutate(lean, 'Source-audit target: ' + strict.KERNEL, 'Source-audit target: isfinite_func')]:
            with self.subTest(changed=changed[:200]), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, changed)
        for field in ['params', 'constexpr', 'runtime', 'orders', 'extern', 'pointer-dtypes', 'materializations']:
            changed = mutate(lean, 'Source-audit target:', 'Source-audit ' + field + ': {}\nSource-audit target:')
            with self.subTest(field=field), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, changed)

    def test_both_complete_import_chains_are_checked_structurally(self):
        # Test the structural validator itself, not just the upstream byte pin.
        for alias, symbol in strict.ALIASES.items():
            for module in ['extra.cuda.libdevice', 'math', 'libdevice']:
                before = 'from triton.language.' + module + ' import ' + symbol + ' as ' + alias
                changed = mutate(self.python, before, before.replace(symbol + ' as', 'other as'))
                with self.subTest(module=module, alias=alias), self.assertRaises(ValueError):
                    strict.python_structure(changed)
        changed = mutate(self.python, 'except ImportError:', 'except Exception:')
        with self.assertRaises(ValueError):
            strict.python_structure(changed)

    def test_dtype_branch_cast_and_all_writer_control_are_structurally_checked(self):
        pairs = [
            ('x.dtype.is_fp64()', 'x.dtype.is_fp32()'),
            ('_finitef(x.to(tl.float32))', '_finitef(x)'),
            ('_isfinited(x) if', '_isfinited(x.to(tl.float32)) if'),
            ('num_tiles0 = tl.cdiv(s0, tile_size0)', 'num_tiles0 = 0'),
            ('if one_tile_per_cta:', 'if True:'),
            ('range(0, tiles_per_cta)', 'range(0, tiles_per_cta + 1)'),
            ('tile_id = pid + j * num_ctas', 'tile_id = pid + num_ctas * j'),
            ('num_ctas = tl.num_programs(0)', 'num_ctas = 1'),
            ('out0 = isfinite_func(in0)', 'out0 = _finitef(in0)'),
            ('boundary_check=(in0_stride_order0,)', 'boundary_check=()'),
            ('.to(in0_ptr.type.element_ty)', ''),
            ('out0.to(out0_bptr.type.element_ty)', 'out0'),
        ]
        for before, after in pairs:
            with self.subTest(before=before), self.assertRaises(ValueError):
                strict.python_structure(mutate(self.python, before, after))

    def test_shadow_alias_injection_and_dynamic_binding_are_rejected(self):
        additions = [
            '_finitef = lambda x: True', 'del _isfinited', 'saved = _finitef',
            'saved = tl', 'tl.float32 = None', 'from unknown import _isfinited',
            'def shadow(_finitef):\n    return 1', 'def shadow(tl):\n    return 1',
            'def shadow():\n    global _finitef',
            'try:\n    pass\nexcept Exception as _finitef:\n    pass',
            'match 1:\n    case {"x": _finitef}:\n        pass',
            'value = getattr(tl, "float32")', 'from unknown import *',
            strict.EXPECTED_HELPER,
        ]
        for addition in additions:
            with self.subTest(addition=addition), self.assertRaises(ValueError):
                strict.python_structure(self.python + '\n' + addition + '\n')

    def test_lean_mutations_fail_before_normalization(self):
        pairs = [
            ('triton.language.extra.cuda.libdevice.isfinited', 'triton.language.math.isfinited'),
            ('tl.source_helper isfinite_func(x)', 'tl.source_helper isfinite_func(_finitef)'),
            ('x.dtype.is_fp64()', 'x.dtype.is_fp32()'),
            ('_finitef(x.to(tl.float32))', '_finitef(x)'),
            ('ComputeKernel.mk [in0_ptr] [out0_ptr]', 'ComputeKernel.mk [] []'),
            ('(s0 num_tasks tiles_per_cta tile_size0 : Nat)', '(s0 tiles_per_cta tile_size0 : Nat)'),
            ('(in_order_zero : in0_stride_order0 = 0)', '(in_order_zero : in0_stride_order0 = 1)'),
            ('out0_ptr = tl.int1', 'out0_ptr = tl.float32'),
            ('if one_tile_per_cta {', 'if true {'),
            ('range($(0), $(tiles_per_cta))', 'range($(0), $(s0))'),
            ('tile_id = pid + j * num_ctas', 'tile_id = pid + num_ctas * j'),
            ('strides=($(in0_stride0))', 'strides=($(out0_stride0))'),
            ('out0 = isfinite_func(in0)', 'out0 = tl.__finite_helper_resolved(in0)'),
            ('out0 = isfinite_func(in0)', 'out0 = isfinite_func(out0)'),
            ('boundary_check=([0] : List Nat)', 'boundary_check=([] : List Nat)'),
            ('order=(0)', 'order=(1)'),
            ('out0.to(out0_bptr.type.element_ty)', 'out0'),
        ]
        for before, after in pairs:
            changed = mutate(self.lean(), before, after)
            with self.subTest(before=before), self.assertRaises(ValueError):
                python_source_with_helpers(self.python, changed)
            with self.subTest(lean_reader=before), self.assertRaises(ValueError):
                source.lean_first_triton_body(self.view(changed))

    def test_production_extraction_rejects_changed_source_and_helper(self):
        result = self.production(self.python, self.lean())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for python, lean in [
            (mutate(self.python, '_finitef(x.to(tl.float32))', '_finitef(x)'), self.lean()),
            (self.python, mutate(self.lean(), '_finitef(x.to(tl.float32))', '_finitef(x)')),
            (self.python, mutate(self.lean(), 'boundary_check=([0] : List Nat)', 'boundary_check=([] : List Nat)')),
        ]:
            result = self.production(python, lean)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('FAIL source-view extraction scan', result.stdout)


if __name__ == '__main__':
    unittest.main()
