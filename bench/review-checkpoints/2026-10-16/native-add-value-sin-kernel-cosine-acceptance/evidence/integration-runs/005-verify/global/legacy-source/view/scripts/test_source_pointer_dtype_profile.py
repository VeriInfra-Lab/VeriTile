"""Actual recorded profiles and adversarial all-arm/source-evidence tests.

Synthetic AST tests mock source/record hashes to reach the structural guards;
they do not claim a new native compile. Production gate tests keep native
records unchanged and mutate copies of the actual persistent-matmul port.
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
from bench import source_pointer_dtypes as pointer
from bench.port_layout import source_text

PORT = 'matmul_persistent_triton'


def metadata(record, key):
    return {'python': record['python'], 'lean': record['lean'],
            'native': {'profile': key, 'sha256': pointer._digest(record)}}


def replace_profile(lean, profile):
    return re.sub(r'^Source-audit pointer-dtypes: .*$',
                  'Source-audit pointer-dtypes: ' + json.dumps(profile), lean, flags=re.M)


class PointerDtypeProfileTests(unittest.TestCase):
    def fixture(self, dtype='fp16'):
        directory = ROOT / 'bench/tritonbench_g' / PORT
        python = (directory / (PORT + '.py')).read_text()
        lean = source_text(directory / 'Kernels.lean')
        key = PORT + '.' + dtype
        record = copy.deepcopy(pointer._registry()[key])
        profile = metadata(record, key)
        lean = lean.replace('Source-audit args: dtype=tl.float16, version=original',
                            'Source-audit args: dtype=' + record['python']['c_ptr'] + ', version=original')
        return python, replace_profile(lean, profile), profile, record

    def synthetic(self, python=None, lean=None, record=None, profile=None):
        py0, lean0, profile0, record0 = self.fixture()
        python = py0 if python is None else python
        lean = lean0 if lean is None else lean
        profile = profile0 if profile is None else profile
        record = record0 if record is None else record
        record['receipt']['source_sha256'] = hashlib.sha256(python.encode()).hexdigest()
        record['receipt_sha256'] = pointer._digest(record['receipt'])
        profile['native']['sha256'] = pointer._digest(record)
        return python, replace_profile(lean, profile), {profile['native']['profile']: record}

    def assert_synthetic_rejects(self, **changes):
        py, lean, registry = self.synthetic(**changes)
        with patch.object(pointer, '_registry', return_value=registry), self.assertRaises(ValueError):
            pointer.source_pointer_dtype_profile(py, lean)

    def test_both_actual_native_profiles_select_only_the_dtype_guard(self):
        for dtype, expected in [('fp16', 'tl.float16'), ('fp8e4nv', 'tl.float8e4nv')]:
            with self.subTest(dtype=dtype):
                py, lean, _, _ = self.fixture(dtype)
                self.assertEqual(pointer.source_pointer_dtype_profile(py, lean), (('c_ptr', expected),))
                original = source._explicit_python_audit_kernel(py, lean)
                result = pointer.python_source_with_pointer_dtype_profile(py, lean)
                selected = source._explicit_python_audit_kernel(result, lean)
                self.assertEqual(sum(isinstance(n, ast.If) for n in ast.walk(original)), 4)
                self.assertEqual(sum(isinstance(n, ast.If) for n in ast.walk(selected)), 3)
                self.assertEqual(source.cast_sequence(source.python_audit_kernel_body(py, lean)), [expected[3:]])
                self.assertIn('if ki == k_tiles - 1:', result)
                self.assertEqual(ast.dump(original.args), ast.dump(selected.args))
                before = ast.parse(py).body
                after = ast.parse(result).body
                self.assertEqual([ast.dump(n) for n in before if n is not original and getattr(n, 'name', None) != original.name],
                                 [ast.dump(n) for n in after if getattr(n, 'name', None) != original.name])

    def test_absent_metadata_keeps_both_arms_without_inference(self):
        py, lean, _, _ = self.fixture()
        lean = re.sub(r'^Source-audit pointer-dtypes: .*\n', '', lean, flags=re.M)
        self.assertEqual(pointer.source_pointer_dtype_profile(py, lean), ())
        self.assertEqual(pointer.python_source_with_pointer_dtype_profile(py, lean), py)
        self.assertIn('c_ptr.dtype.element_ty', source.python_audit_kernel_body(py, lean))

    def test_source_fingerprint_includes_discarded_arm_and_wrappers(self):
        py, lean, _, _ = self.fixture()
        for changed in [py + '\n', py.replace('c = accumulator.to(tl.float8e4nv)', 'c = accumulator'),
                        py.replace('torch.randn((M, K)', 'torch.zeros((M, K)')]:
            with self.subTest(changed=changed[-60:]), self.assertRaises(ValueError):
                pointer.source_pointer_dtype_profile(changed, lean)

    def test_metadata_schema_duplicates_and_types_are_strict(self):
        py, lean, profile, _ = self.fixture()
        for bad in [{}, [], dict(profile, unknown=True), dict(profile, python={}),
                    dict(profile, python={'c_ptr': True}), dict(profile, python={'c_ptr': []}),
                    dict(profile, python={'c_ptr': 'tl.float32'}),
                    dict(profile, native={'profile': 'missing', 'sha256': '0'*64}),
                    dict(profile, native={'path': '/tmp/arbitrary'})]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pointer.source_pointer_dtype_profile(py, replace_profile(lean, bad))
        for altered in [lean.replace('"python":', '"python":{},"python":', 1),
                        lean.replace('Source-audit pointer-dtypes:',
                                     'Source-audit pointer-dtypes: {}\nSource-audit pointer-dtypes:', 1)]:
            with self.assertRaises(ValueError):
                pointer.source_pointer_dtype_profile(py, altered)

    def test_registry_schema_is_strict_and_not_cached(self):
        py, lean, _, _ = self.fixture()
        for registry in ['{}', '{"schema":true,"profiles":{}}',
                         '{"schema":1,"schema":1,"profiles":{}}',
                         '{"schema":1,"profiles":[],"path":"/tmp/other"}']:
            with patch.object(Path, 'read_text', return_value=registry), self.assertRaises(ValueError):
                pointer.source_pointer_dtype_profile(py, lean)
        with patch.object(pointer, '_registry', return_value={}), self.assertRaises(ValueError):
            pointer.source_pointer_dtype_profile(py, lean)

    def test_exact_lean_selector_all_macro_args_and_all_pointer_regions(self):
        py, lean, _, _ = self.fixture()
        for changed in [lean.replace('Source-audit Lean: persistentSource%', 'Source-audit Lean: missing%'),
                        lean.replace('version=original', 'version=commuted', 1),
                        lean.replace('dtype=tl.float16, version=original', 'dtype=tl.float16', 1),
                        lean.replace('dtype=tl.float16', 'dtype=tl.float8e4nv', 1),
                        lean.replace('c_ptr = $dtype', 'c_ptr = tl.float8e4nv'),
                        lean.replace('a_ptr = $dtype', 'a_ptr = tl.float8e4nv'),
                        lean.replace('a_ptr b_ptr c_ptr : RegionName', 'a_ptr b_ptr other_ptr : RegionName'),
                        lean.replace('tl.region a_ptr =', 'tl.region c_ptr = tl.float8e4nv\n        tl.region a_ptr =')]:
            with self.subTest(changed=changed[:40]), self.assertRaises(ValueError):
                pointer.source_pointer_dtype_profile(py, changed)

    def test_unrecorded_or_retyped_python_pointer_parameter_rejects(self):
        py, _, _, _ = self.fixture()
        for old, new in [('a_ptr, b_ptr, c_ptr,', 'a_ptr, b_ptr, other_ptr,'),
                         ('a_ptr, b_ptr, c_ptr,', 'a_ptr, b_ptr, c_ptr: tl.constexpr,'),
                         ('a_ptr, b_ptr, c_ptr,', 'a_ptr, b_ptr, c_ptr: int,')]:
            self.assertIn(old, py)
            self.assert_synthetic_rejects(python=py.replace(old, new, 1))

    def test_native_receipt_digest_and_ttir_digest_reject_staleness(self):
        py, lean, _, record = self.fixture()
        self.assert_synthetic_rejects(record=dict(record, ttir=record['ttir']+'\n'))
        changed = dict(record, receipt_sha256='0'*64)
        key = PORT+'.fp16'
        with patch.object(pointer, '_registry', return_value={key: changed}), self.assertRaises(ValueError):
            pointer.source_pointer_dtype_profile(py, replace_profile(lean, metadata(changed, key)))
        altered = copy.deepcopy(record)
        altered['receipt']['results'][0]['signature']['c_ptr'] = '*fp8e4nv'
        with patch.object(pointer, '_registry', return_value={PORT+'.fp16': altered}), self.assertRaises(ValueError):
            pointer.source_pointer_dtype_profile(py, lean)

    def test_native_signature_and_ttir_are_checked_after_mocked_record_hash(self):
        _, _, _, record = self.fixture()
        mutations = [lambda r: r['receipt']['results'][0]['signature'].__setitem__('c_ptr', '*fp8e4nv'),
                     lambda r: r['receipt']['results'][0]['signature'].__setitem__('M', 'fp32'),
                     lambda r: r['receipt']['results'][0]['signature'].__setitem__('extra', 'i32'),
                     lambda r: r['receipt']['results'][0]['constexprs'].__setitem__('NUM_SMS', True),
                     lambda r: r['receipt']['results'][0].__setitem__('num_warps', True),
                     lambda r: r['receipt']['results'][0].__setitem__('status', 'compile_failure'),
                     lambda r: r['receipt']['results'][0].__setitem__('loop_present', False),
                     lambda r: r['receipt']['results'][0].__setitem__('dot_lines', []),
                     lambda r: r['receipt'].__setitem__('triton_version', '3.6.0'),
                     lambda r: r['receipt']['target'].__setitem__('arch', 80),
                     lambda r: r.__setitem__('native_dtype', 'fp8e4nv')]
        for mutate in mutations:
            changed = copy.deepcopy(record)
            mutate(changed)
            self.assert_synthetic_rejects(record=changed)
        for old, new in [('%c_ptr: !tt.ptr<f16>', '%c_ptr: !tt.ptr<f8E4M3FN>'),
                         ('@matmul_kernel_persistent(', '@other_kernel('),
                         ('%c_ptr: !tt.ptr<f16>', '%c_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<f16>')]:
            changed = copy.deepcopy(record)
            self.assertIn(old, changed['ttir'])
            changed['ttir'] = changed['ttir'].replace(old, new, 1)
            changed['receipt']['results'][0]['ttir_sha256'] = hashlib.sha256(changed['ttir'].encode()).hexdigest()
            self.assert_synthetic_rejects(record=changed)

    def test_original_unselected_arm_rebinding_and_nested_scopes_reject(self):
        py, _, _, _ = self.fixture()
        anchor = '                c = accumulator.to(tl.float8e4nv)'
        self.assertIn(anchor, py)
        for inserted in ['c_ptr = a_ptr', 'c_ptr += 1', 'del c_ptr', 'tl = other',
                         'c_ptr.dtype = tl.float16', 'tl.float8e4nv = tl.float16',
                         'def hidden():\n                    return c_ptr',
                         'hidden = [c_ptr for c_ptr in things]', 'hidden = lambda: c_ptr']:
            self.assert_synthetic_rejects(python=py.replace(anchor, anchor+'\n                '+inserted, 1))

    def test_only_direct_equality_if_guards_are_supported(self):
        py, _, _, _ = self.fixture()
        for condition in ['c_ptr.dtype.element_ty != tl.float8e4nv',
                          'c_ptr.dtype.element_ty is tl.float8e4nv',
                          'c_ptr.dtype.element_ty == tl.float8e4nv == other',
                          '(c_ptr.dtype.element_ty == tl.float8e4nv) and enabled',
                          'not (c_ptr.dtype.element_ty == tl.float8e4nv)',
                          'tl.float8e4nv == c_ptr.dtype.element_ty',
                          'c_ptr.type.element_ty == tl.float8e4nv',
                          'c_ptr.dtype.element_ty == tl.float32']:
            self.assert_synthetic_rejects(python=py.replace('c_ptr.dtype.element_ty == tl.float8e4nv', condition))

    def test_unsupported_dtype_uses_in_original_arms_are_not_hidden(self):
        py, _, _, _ = self.fixture()
        anchor = '                c = accumulator.to(tl.float8e4nv)'
        for tail in ['value = c_ptr.dtype', 'value = c_ptr.dtype.element_ty',
                     'value = c_ptr.dtype.element_ty == tl.float8e4nv', 'value = c_ptr.cast(tl.float16)',
                     'value = helper(c_ptr)', 'setattr(c_ptr, "dtype", tl.float16)',
                     'alias = c_ptr', 'setattr(tl, "float8e4nv", other)']:
            self.assert_synthetic_rejects(python=py.replace(anchor, anchor+'\n                '+tail, 1))

    def test_combined_profile_is_rejected_before_original_source_changes(self):
        py, lean, _, _ = self.fixture()
        for field in ['params', 'constexpr', 'helpers', 'runtime', 'orders', 'extern']:
            changed = lean.replace('Source-audit target:', 'Source-audit '+field+': {}\nSource-audit target:', 1)
            with self.subTest(field=field), self.assertRaises(ValueError):
                pointer.source_pointer_dtype_profile(py, changed)

    def production(self, mutation=None, dtype='fp16'):
        script = (ROOT/'bench/audit_tritonbench_g.sh').read_text()
        section = script[script.index('# A source view must be extractable'):
                         script.index('# ---- Machine-checkable trust audit')]
        command = 'failures=0\nPORTS_ROOT="$1"\n'+section+'\nexit "$failures"\n'
        with tempfile.TemporaryDirectory() as temporary:
            dest = Path(temporary)/PORT
            shutil.copytree(ROOT/'bench/tritonbench_g'/PORT, dest)
            if dtype != 'fp16':
                record = pointer._registry()[PORT+'.'+dtype]
                entry = dest/'Kernels.lean'
                text = entry.read_text().replace('dtype=tl.float16, version=original',
                                                'dtype='+record['python']['c_ptr']+', version=original')
                entry.write_text(replace_profile(text, metadata(record, PORT+'.'+dtype)))
            if mutation:
                mutation(dest)
            return subprocess.run(['bash', '-c', command, 'pointer-dtype-regression', temporary],
                                  cwd=ROOT, capture_output=True, text=True, timeout=45)

    def test_actual_production_gates_pass_both_recorded_storage_views(self):
        for dtype in ['fp16', 'fp8e4nv']:
            result = self.production(dtype=dtype)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('validated native pointer dtype profile', result.stdout)
            self.assertNotIn('FAIL ', result.stdout)

    def test_actual_production_no_metadata_leaves_original_three_failures(self):
        def mutate(directory):
            p = directory/'Kernels.lean'
            p.write_text(re.sub(r'^Source-audit pointer-dtypes: .*\n', '', p.read_text(), flags=re.M))
        result = self.production(mutate)
        self.assertEqual(result.returncode, 3, result.stdout+result.stderr)
        for gate in ['.to(...) cast sequence', 'kernel control-flow surface', 'statement lhs sequence']:
            self.assertIn('FAIL '+gate+' scan', result.stdout)

    def test_actual_production_hidden_arm_mutation_is_hard_failure(self):
        def mutate(directory):
            p = directory/(PORT+'.py')
            p.write_text(p.read_text().replace('c = accumulator.to(tl.float8e4nv)', 'c = accumulator'))
        result = self.production(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)
        self.assertIn('original source/compiler/receipt', result.stdout)

    def test_actual_production_selected_lean_cast_mutation_remains_a_gate_failure(self):
        def mutate(directory):
            p = directory/'Proofs/Source.lean'
            p.write_text(p.read_text().replace('c = accumulator.to($dtype:tritonDType)',
                                              'c = accumulator.to(tl.float32)'))
        result = self.production(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL .to(...) cast sequence scan', result.stdout)

    def test_actual_production_invalid_profile_is_not_exempted_by_blocker(self):
        def mutate(directory):
            p = directory/'Kernels.lean'
            p.write_text(p.read_text().replace('Full persistent grouped',
                'translation-surface blocker: test\nFull persistent grouped').replace('version=original', 'version=commuted', 1))
        result = self.production(mutate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)


if __name__ == '__main__':
    unittest.main()
