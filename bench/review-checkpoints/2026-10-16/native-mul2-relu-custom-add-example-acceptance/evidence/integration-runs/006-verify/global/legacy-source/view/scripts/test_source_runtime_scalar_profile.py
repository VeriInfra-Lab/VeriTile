"""Strict runtime ABI views, with real-source gates and synthetic AST negatives.

AST mutation tests deliberately update mocked receipt hashes so they exercise
validation past fingerprint checks. They are not native compilation evidence.
Production tests always use the reviewed on-disk native records unchanged.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench.port_layout import source_text

PORTS = ('kldiv_ops', 'l2_norm_bwd', 'l2_norm_triton1', 'l2_norm_triton2', 'rms_norm_triton')


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replace_profile(lean, profile):
    return re.sub(r'^Source-audit runtime: .*$', 'Source-audit runtime: ' + json.dumps(profile), lean, flags=re.M)


class SourceRuntimeScalarProfileTests(unittest.TestCase):
    def fixture(self, port='l2_norm_triton1'):
        directory = ROOT / 'bench/tritonbench_g' / port
        python = next(directory.glob('*.py')).read_text()
        lean = source_text(directory / 'Kernels.lean')
        profile = json.loads(source.source_audit_metadata(lean, 'runtime'))
        record = copy.deepcopy(source._runtime_scalar_registry()[profile['native']['profile']])
        return python, lean, profile, record

    def synthetic(self, python=None, lean=None, profile=None, record=None):
        """Bind an explicitly mocked record to test AST validation in isolation."""
        original_python, original_lean, original_profile, original_record = self.fixture()
        python = original_python if python is None else python
        lean = original_lean if lean is None else lean
        profile = original_profile if profile is None else profile
        record = original_record if record is None else record
        record['native']['source_sha256'] = hashlib.sha256(python.encode()).hexdigest()
        profile['native']['sha256'] = digest(record)
        return python, replace_profile(lean, profile), {profile['native']['profile']: record}

    def test_all_five_real_profiles_materialize_only_recorded_fp32_scalars(self):
        for port in PORTS:
            with self.subTest(port=port):
                python, lean, profile, record = self.fixture(port)
                self.assertEqual(source.source_runtime_scalar_profile(python, lean), (('eps', 'tl.float32'),))
                result = source.python_source_with_runtime_scalar_profile(python, lean)
                self.assertEqual(result.replace('eps.to(tl.float32)', 'eps'), python)
                self.assertIn('eps.to(tl.float32)', source.python_audit_kernel_body(python, lean))
                self.assertFalse(record['compiler']['kernel_launched'])
                self.assertEqual(record['specialization']['7'], ['i32', None])
                ast.parse(result)

    def test_missing_metadata_never_infers_type(self):
        python, lean, _, _ = self.fixture()
        lean = re.sub(r'^Source-audit runtime: .*\n', '', lean, flags=re.M)
        self.assertEqual(source.source_runtime_scalar_profile(python, lean), ())
        self.assertEqual(source.python_source_with_runtime_scalar_profile(python, lean), python)

    def test_schema_object_abi_and_duplicates_are_strict(self):
        python, lean, profile, _ = self.fixture()
        for declaration in [{}, {'object': 'int', 'abi': 'i32'}, {'object': 'float', 'abi': 'fp64'},
                            {'object': 'float', 'abi': 'fp32', 'default': 1e-6}, None, []]:
            bad = copy.deepcopy(profile)
            bad['python']['eps'] = declaration
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(python, replace_profile(lean, bad))
        for bad in [lean.replace('"python":', '"unknown":'),
                    lean.replace('"object":"float"', '"object":"float","object":"int"'),
                    lean.replace('Source-audit Lean: l2Source%\n', ''),
                    lean.replace('Source-audit runtime:', 'Source-audit runtime: {}\nSource-audit runtime:'),
                    replace_profile(lean, {}), replace_profile(lean, [])]:
            with self.subTest(lean=bad), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(python, bad)

    def test_native_record_and_source_fingerprints_are_mandatory(self):
        python, lean, profile, record = self.fixture()
        for changed in [python + '\n', python.replace('    eps,', '    epsilon,', 1),
                        python.replace('    eps,', '    eps=1e-6,', 1).replace('BLOCK_N: tl.constexpr,', 'BLOCK_N: tl.constexpr = 32,', 1),
                        python.replace('    eps,', '    eps: tl.constexpr,', 1),
                        python.replace('    eps,', '    eps: float,', 1),
                        python.replace('    eps,', '    eps: "float",', 1)]:
            with self.subTest(python=changed), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(changed, lean)
        for registry in [{}, {profile['native']['profile']: {}},
                         {profile['native']['profile']: dict(record, ttir=record['ttir']+'\n')}]:
            with self.subTest(registry=registry), patch.object(source, '_runtime_scalar_registry', return_value=registry), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(python, lean)
        # Refresh the metadata digest deliberately; internal evidence checks
        # must still reject changed TTIR and contradictory native signatures.
        for change in ['ttir', 'ttir_signature', 'runtime_signature', 'compiler', 'samples', 'constexprs']:
            bad = copy.deepcopy(record)
            if change == 'ttir':
                bad['ttir'] += '\n'
            elif change == 'ttir_signature':
                bad['native']['ttir_signature'] = 'tt.func public @wrong(%eps: f32) {'
            elif change == 'runtime_signature':
                bad['native']['signature']['eps'] = 'i32'
            elif change == 'compiler':
                bad['compiler']['target'] = 'cuda sm80 warp32'
            elif change == 'samples':
                bad['specialization']['7'] = ['fp32', None]
            else:
                bad['native']['constexprs'] = {}
            py, view, registry = self.synthetic(record=bad)
            with self.subTest(change=change), patch.object(source, '_runtime_scalar_registry', return_value=registry), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(py, view)

    def test_registry_schema_duplicates_and_boolean_compiler_values_are_strict(self):
        invalid = ['{}', '{"schema":true,"profiles":{}}',
                   '{"schema":1,"schema":1,"profiles":{}}',
                   '{"schema":1,"profiles":[],"path":"arbitrary"}']
        for value in invalid:
            with self.subTest(value=value), patch.object(Path, 'read_text', return_value=value), self.assertRaises(ValueError):
                source._runtime_scalar_registry()
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError), self.assertRaises(ValueError):
            source._runtime_scalar_registry()
        _, _, _, record = self.fixture()
        for field, value in [('kernel_launched', 0), ('options', {'num_warps': 4.0, 'num_stages': 3})]:
            bad = copy.deepcopy(record)
            bad['compiler'][field] = value
            py, lean, registry = self.synthetic(record=bad)
            with self.subTest(field=field), patch.object(source, '_runtime_scalar_registry', return_value=registry), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(py, lean)

    def test_exact_jit_lean_selector_macro_arguments_and_scalar_binder(self):
        python, lean, profile, _ = self.fixture()
        for bad in [lean.replace('Source-audit target: _l2_norm_fwd_1pass_kernel', 'Source-audit target: helper'),
                    lean.replace('Source-audit Lean: l2Source%', 'Source-audit Lean: wrong%'),
                    lean.replace('dtype=tl.float32', 'dtype=tl.float16'),
                    lean.replace(', version=original', ''),
                    lean.replace('(eps : ℝ)', '(eps : Nat)')]:
            with self.subTest(lean=bad), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(python, bad)
        for key, value in [('source', 'wrong%'), ('args', {'dtype': 'tl.float32'})]:
            bad = copy.deepcopy(profile)
            bad['lean'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(python, replace_profile(lean, bad))

    def test_rebindings_and_indirect_scopes_rejected_in_all_original_arms(self):
        python, _, _, _ = self.fixture()
        statements = ['eps = tl.load(X)', 'eps += 1', 'del eps', '(eps, value) = values',
                      'for eps in values:\n            value = 0',
                      'with manager() as eps:\n            value = 0',
                      'def helper(eps):\n            return eps',
                      'class helper:\n            eps = 0',
                      'helper = lambda eps: eps', 'value = [eps for eps in values]',
                      'from helper import eps', 'global eps',
                      'try:\n            value = 0\n        except Exception as eps:\n            value = 1',
                      'match value:\n            case eps:\n                value = 0']
        for statement in statements:
            changed = python.replace('    row =', '    if False:\n        '+statement+'\n    row =', 1)
            py, lean, registry = self.synthetic(python=changed)
            with self.subTest(statement=statement), patch.object(source, '_runtime_scalar_registry', return_value=registry), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(py, lean)

    def test_unsupported_expression_contexts_fail_closed(self):
        python, _, _, _ = self.fixture()
        statements = ['value = eps', 'value = tl.load(X + eps)', 'value = tl.full((eps,), 0, tl.float32)',
                      'value = tl.cast(x, eps)', 'value = eps(x)', 'value = eps.dtype',
                      'value = x[eps]', 'value = helper(eps)', 'value = tl.maximum(x, other=eps)',
                      'if eps:\n        value = 0', 'return eps', 'assert eps',
                      'value = eps ** 2', 'value = -eps', 'value = (eps := 3)',
                      'value = tl.sqrt(eps, x)', 'value = tl.maximum(eps)', 'tl.store(Y, eps)']
        for statement in statements:
            changed = python.replace('    row =', '    '+statement+'\n    row =', 1)
            py, lean, registry = self.synthetic(python=changed)
            with self.subTest(statement=statement), patch.object(source, '_runtime_scalar_registry', return_value=registry), self.assertRaises(ValueError):
                source.source_runtime_scalar_profile(py, lean)

    def test_utf8_labels_strings_comments_defaults_other_jits_are_preserved(self):
        python, _, _, _ = self.fixture()
        extra = ('    """eps 字符串"""\n    # eps unchanged\n'
                 '    文本 = "eps"; value = eps + 1\n'
                 '    value = holder.eps\n    value = helper(eps=1)\n')
        changed = python.replace('    row =', extra+'    row =', 1)
        helper = '\n@triton.jit\ndef other(eps=1e-6):\n    value = eps + 1\n'
        changed += helper
        py, lean, registry = self.synthetic(python=changed)
        with patch.object(source, '_runtime_scalar_registry', return_value=registry):
            result = source.python_source_with_runtime_scalar_profile(py, lean)
        self.assertEqual(result.replace('eps.to(tl.float32)', 'eps'), changed)
        self.assertIn('文本 = "eps"; value = eps.to(tl.float32) + 1', result)
        self.assertTrue(result.endswith(helper))
        ast.parse(result)

    def test_parameter_alias_and_boolean_profile_compose_after_original_source_validation(self):
        python, lean, _, _ = self.fixture()
        aliased = lean.replace('Source-audit Lean:', 'Source-audit params: eps=epsilon\nSource-audit Lean:', 1)
        # Preserve original metadata fields; rename only actual Lean binders/uses.
        aliased = aliased.replace('(eps : ℝ)', '(epsilon : ℝ)')
        body = source.python_audit_kernel_body(python, aliased)
        self.assertIn('epsilon.to(tl.float32)', body)
        self.assertNotIn('eps.to(tl.float32)', body)
        python, lean, _, _ = self.fixture('kldiv_ops')
        body = source.python_audit_kernel_body(python, lean)
        self.assertNotIn('if log_target:', body)
        self.assertIn('tl.maximum(y_true, eps.to(tl.float32))', body)
        self.assertIn('tl.load(Y + offsets', body)

    def production(self, mutations=None):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        section = script[script.index('# A source view must be extractable'):
                         script.index('# ---- Machine-checkable trust audit')]
        command = 'failures=0\nPORTS_ROOT="$1"\n'+section+'\nexit "$failures"\n'
        with tempfile.TemporaryDirectory() as directory:
            mutations = mutations or {port: None for port in PORTS}
            for port, mutation in mutations.items():
                dest = Path(directory) / port
                dest.mkdir()
                original = ROOT / 'bench/tritonbench_g' / port
                for path in original.iterdir():
                    if path.suffix in {'.lean', '.py'}:
                        shutil.copyfile(path, dest/path.name)
                if mutation is not None:
                    mutation(dest)
            return subprocess.run(['bash', '-c', command, 'runtime-regression', directory],
                                  cwd=ROOT, text=True, capture_output=True, timeout=40)

    def test_actual_production_five_port_baseline(self):
        result = self.production()
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('5 extractable source views, 0 unresolved', result.stdout)
        self.assertEqual(result.stdout.count('validated Python runtime ABI profile'), 5)

    def test_actual_production_preserves_scalar_and_tensor_cast_mutations(self):
        def mutate(old, new):
            def apply(directory):
                path = directory/'Kernels.lean'
                text = path.read_text()
                self.assertIn(old, text)
                prefix, selected = text.split('macro_rules', 1)
                self.assertIn(old, selected)
                path.write_text(prefix + 'macro_rules' + selected.replace(old, new, 1))
            return apply
        mutations = {
            'kldiv_ops': mutate('($((eps : ℝ))).to(tl.float32)', '($((eps : ℝ))).to(tl.float16)'),
            'l2_norm_triton1': mutate('other=0.0).to(tl.float32)', 'other=0.0)'),
            'l2_norm_triton2': mutate('y = x * rstd', 'y = x.to(tl.float32) * rstd'),
            'l2_norm_bwd': mutate('$((eps : ℝ)).to(tl.float32)', '$((eps : ℝ)).to(tl.float64)'),
            'rms_norm_triton': mutate('y = (x * rrms).to($dtype:tritonDType) * w',
                                      'y = (x * rrms) * w'),
        }
        result = self.production(mutations)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('FAIL source-view extraction scan', result.stdout)
        self.assertIn('FAIL .to(...) cast sequence scan', result.stdout)
        for port in mutations:
            self.assertRegex(result.stdout, re.escape(port)+r'/[^\n]+cast sequence mismatch')

    def test_actual_production_missing_metadata_and_changed_signature_still_fail(self):
        def remove_profile(directory):
            path = directory/'Kernels.lean'
            path.write_text(re.sub(r'^Source-audit runtime: .*\n', '', path.read_text(), flags=re.M))
        def signature(directory):
            path = next(directory.glob('*.py'))
            path.write_text(path.read_text().replace('    eps,', '    epsilon,', 1))
        result = self.production({'l2_norm_triton1': remove_profile, 'l2_norm_bwd': signature})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)
        self.assertIn('FAIL .to(...) cast sequence scan', result.stdout)
        self.assertIn('Source-audit runtime requires', result.stdout)

    def test_actual_production_rejects_wrong_jit_config_constexpr_rebinding_and_nested_scope(self):
        def lean_change(old, new):
            def apply(directory):
                path = directory/'Kernels.lean'
                self.assertIn(old, path.read_text())
                path.write_text(path.read_text().replace(old, new, 1))
            return apply
        def python_change(old, new):
            def apply(directory):
                path = next(directory.glob('*.py'))
                self.assertIn(old, path.read_text())
                path.write_text(path.read_text().replace(old, new, 1))
            return apply
        mutations = {
            'kldiv_ops': lean_change('Source-audit target: _kldiv_kernel_forward',
                                    'Source-audit target: _kldiv_kernel_backward'),
            'l2_norm_bwd': python_change('    eps,', '    eps: tl.constexpr,'),
            'l2_norm_triton1': lean_change('dtype=tl.float32', 'dtype=tl.float16'),
            'l2_norm_triton2': python_change('    row =', '    eps = tl.load(X)\n    row ='),
            'rms_norm_triton': python_change('    pid =', '    def helper(eps):\n        return eps\n    pid ='),
        }
        result = self.production(mutations)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('0 extractable source views, 5 unresolved', result.stdout)
        self.assertIn('FAIL source-view extraction scan', result.stdout)
        for port in PORTS:
            self.assertRegex(result.stdout, re.escape(port)+r'/Kernels.lean:')

    def test_actual_production_invalid_runtime_without_other_profiles_fails_even_with_blocker(self):
        def invalid(directory):
            path = directory/'Kernels.lean'
            text = path.read_text().replace('"object":"float"', '"object":"int"')
            text = text.replace('/-\n', '/-\nTranslation-surface blocker: fixture marker\n', 1)
            path.write_text(text)
        result = self.production({'l2_norm_triton1': invalid})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)
        self.assertIn('supports only explicit Python float / fp32', result.stdout)


if __name__ == '__main__':
    unittest.main()
