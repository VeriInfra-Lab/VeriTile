"""Native fp16 specialization, rejected bf16, and production source mutations."""
import ast
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench import source_dtype_constexpr as dtype

PORT = ROOT / 'bench/tritonbench_g/attention_kernel_aligned'


class DtypeConstexprTests(unittest.TestCase):
    def setUp(self):
        self.python = (PORT / 'attention_kernel_aligned.py').read_text()
        self.lean = (PORT / 'AttentionKernelAligned.lean').read_text()
        self.profile = json.loads(source.source_audit_metadata(self.lean, dtype.FIELD))

    def changed_profile(self, profile):
        return re.sub(r'^Source-audit dtype-constexpr: .*$',
                      'Source-audit dtype-constexpr: ' + json.dumps(profile), self.lean, flags=re.M)

    def test_recorded_fp16_replaces_only_five_direct_dtype_operands(self):
        self.assertEqual(dtype.source_dtype_constexpr_profile(self.python, self.lean),
                         (('OUT_DTYPE', 'tl.float16'),))
        result = dtype.python_source_with_dtype_constexpr_profile(self.python, self.lean)
        before = source._explicit_python_audit_kernel(self.python, self.lean)
        after = source._explicit_python_audit_kernel(result, self.lean)
        self.assertEqual(ast.dump(before.args), ast.dump(after.args))
        self.assertEqual(sum(isinstance(n, ast.Name) and n.id == 'OUT_DTYPE'
                             for stmt in before.body for n in ast.walk(stmt)), 5)
        self.assertEqual(sum(isinstance(n, ast.Attribute) and ast.unparse(n) == 'tl.float16'
                             for stmt in after.body for n in ast.walk(stmt)), 5)
        self.assertEqual([ast.dump(n) for n in ast.parse(self.python).body if getattr(n, 'name', None) != before.name],
                         [ast.dump(n) for n in ast.parse(result).body if getattr(n, 'name', None) != after.name])
        self.assertNotIn('OUT_DTYPE', source.python_source_parameter_names(self.python, self.lean))
        body = source.python_audit_kernel_body(self.python, self.lean)
        self.assertIn('out_dtype=tl.float16', body)
        self.assertIn('qk += (b0 + b1)', body)
        self.assertIn('for start_n in range(lo, hi, BLOCK_N):', body)

    def test_missing_metadata_does_not_infer_fp16(self):
        lean = re.sub(r'^Source-audit dtype-constexpr: .*\n', '', self.lean, flags=re.M)
        self.assertEqual(dtype.source_dtype_constexpr_profile(self.python, lean), ())
        self.assertEqual(dtype.python_source_with_dtype_constexpr_profile(self.python, lean), self.python)
        self.assertIn('out_dtype=OUT_DTYPE', source.python_audit_kernel_body(self.python, lean))

    def test_bf16_failure_cannot_be_selected_as_a_success(self):
        record = copy.deepcopy(dtype._registry()['attention-aligned-fp16'])
        record['selected'] = 'bf16'
        record['python']['OUT_DTYPE'] = 'tl.bfloat16'
        profile = copy.deepcopy(self.profile)
        profile['python'] = record['python']
        profile['native']['sha256'] = dtype._digest(record)
        with patch.object(dtype, '_registry', return_value={'attention-aligned-fp16': record}):
            with self.assertRaisesRegex(ValueError, 'reviewed fp16 evidence'):
                dtype.source_dtype_constexpr_profile(self.python, self.changed_profile(profile))

    def test_source_hash_covers_original_wrapper_and_all_dtype_uses(self):
        for python in [self.python + '\n', self.python.replace('torch.randn(', 'torch.zeros(', 1),
                       self.python.replace('out_dtype=OUT_DTYPE', 'out_dtype=tl.float32')]:
            with self.subTest(python=python[-50:]), self.assertRaises(ValueError):
                dtype.source_dtype_constexpr_profile(python, self.lean)

    def test_wrong_selector_schema_dtype_and_fingerprint_reject(self):
        for bad in [{}, [], dict(self.profile, extra=True), dict(self.profile, python={}),
                    dict(self.profile, python={'OUT_DTYPE': 'tl.bfloat16'}),
                    dict(self.profile, native={'profile': 'unknown', 'sha256': '0'*64}),
                    dict(self.profile, native={'profile': 'attention-aligned-fp16', 'sha256': '0'*64}),
                    dict(self.profile, lean={'source': 'missing', 'args': {}}),
                    dict(self.profile, lean={'source': 'attention_kernel_aligned_fp16_surface', 'args': {'x': 0}})]:
            with self.subTest(profile=bad), self.assertRaises(ValueError):
                dtype.source_dtype_constexpr_profile(self.python, self.changed_profile(bad))
        for lean in [self.lean.replace('"python":', '"python":{},"python":', 1),
                     self.lean.replace('Source-audit dtype-constexpr:',
                                       'Source-audit dtype-constexpr: {}\nSource-audit dtype-constexpr:', 1)]:
            with self.assertRaises(ValueError):
                dtype.source_dtype_constexpr_profile(self.python, lean)

    def test_arithmetic_guard_alias_rebinding_and_nested_scope_uses_reject(self):
        for statement in ['OUT_DTYPE = tl.float32', 'del OUT_DTYPE', 'alias = OUT_DTYPE',
                          'q = q + OUT_DTYPE', 'if OUT_DTYPE:\n        q = q + 1',
                          'q = q.to(OUT_DTYPE, bitcast=True)',
                          'q = helper(q, dtype=OUT_DTYPE)',
                          'def hidden():\n        return OUT_DTYPE']:
            python = self.python.replace('    lo = 0', '    ' + statement + '\n    lo = 0', 1)
            with self.subTest(statement=statement), self.assertRaisesRegex(ValueError, 'non-dtype|nested scopes'):
                dtype.source_dtype_constexpr_profile(python, self.lean)

    def test_signature_retyping_and_parameter_capture_reject(self):
        for python in [self.python.replace('OUT_DTYPE: tl.constexpr', 'OUT_DTYPE: int'),
                       self.python.replace('OUT_DTYPE: tl.constexpr', 'OUT_DTYPE')]:
            with self.assertRaisesRegex(ValueError, 'exact tl.constexpr'):
                dtype.source_dtype_constexpr_profile(python, self.lean)
        for params in ['OUT_DTYPE=_Z', 'Z=OUT_DTYPE']:
            lean = self.lean.replace('Z=_Z, H=_H', params)
            with self.assertRaisesRegex(ValueError, 'constexpr profile parameters'):
                source.source_parameter_map(self.python, lean)

    def test_native_receipt_and_artifact_tampering_reject(self):
        original = Path.read_bytes
        for suffix in ['receipt.json', 'kernel.ttir', 'error.txt']:
            def tampered(path):
                result = original(path)
                return result + b'\n' if path.name == suffix else result
            with self.subTest(artifact=suffix), patch.object(Path, 'read_bytes', tampered):
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    dtype.source_dtype_constexpr_profile(self.python, self.lean)

    def test_unsupported_composition_and_registry_shapes_reject(self):
        for field in ['constexpr', 'helpers', 'runtime', 'orders', 'extern', 'pointer-dtypes', 'materializations']:
            lean = self.lean.replace('Source-audit target:', 'Source-audit '+field+': {}\nSource-audit target:', 1)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'untransformed original'):
                dtype.source_dtype_constexpr_profile(self.python, lean)
        for registry in ['[]', '{}', '{"schema":true,"profiles":{}}', '{"schema":1,"profiles":[]}']:
            with patch.object(Path, 'read_text', return_value=registry), self.assertRaises(ValueError):
                dtype._registry()

    def production(self, lean):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        section = script[script.index('# A source view must be extractable'):
                         script.index('# ---- Machine-checkable trust audit')]
        command = 'failures=0\nPORTS_ROOT="$1"\n' + section + '\nexit "$failures"\n'
        with tempfile.TemporaryDirectory() as directory:
            port = Path(directory) / 'attention_kernel_aligned'
            port.mkdir()
            (port/'attention_kernel_aligned.py').write_text(self.python)
            (port/'AttentionKernelAligned.lean').write_text(lean)
            return subprocess.run(['bash', '-c', command, 'dtype-profile-regression', directory],
                                  cwd=ROOT, text=True, capture_output=True, timeout=30)

    def test_production_keeps_wrong_dot_dtype_cast_and_bias_visible(self):
        result = self.production(self.lean)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for lean in [self.lean.replace('out_dtype=tl.float16)', 'out_dtype=tl.float32)', 1),
                     self.lean.replace('(q * qk_scale).to(tl.float16)', '(q * qk_scale).to(tl.bfloat16)', 1),
                     self.lean.replace('qk += (b0 + b1)', 'qk += (b0 - b1)', 1)]:
            with self.subTest(mutation=lean[:100]):
                result = self.production(lean)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('mismatch', result.stdout)
        bad = self.lean.replace('"OUT_DTYPE":"tl.float16"', '"OUT_DTYPE":"tl.bfloat16"', 1)
        result = self.production(bad)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FAIL source-view extraction scan', result.stdout)


if __name__ == '__main__':
    unittest.main()
