"""Production allocation-audit checks for dtype attributes and source mutations."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceShapeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = (ROOT / 'bench/audit_tritonbench_g.sh').read_text()
        bodies = re.findall(r"<<'PY'\n(.*?)\nPY\nthen", script, re.S)
        matching = [body for body in bodies if 'shape/dtype call mismatch' in body]
        if len(matching) != 1:
            raise AssertionError('Expected exactly one production allocation audit')
        cls.audit = matching[0]

    def run_audit(self, python_body, lean_body):
        with tempfile.TemporaryDirectory() as temp:
            port = Path(temp) / 'fixture'
            port.mkdir()
            (port / 'fixture.py').write_text(
                '@triton.jit\ndef kernel(A, B, N):\n' +
                '\n'.join('    ' + line for line in python_body.splitlines()) + '\n')
            (port / 'fixture.lean').write_text(
                '/-!\nSource-audit target: kernel\n-/\n'
                'def source (A B : RegionName) (N : Nat) := triton {\n' +
                lean_body + '\n}\n')
            return subprocess.run(
                [sys.executable, '-', temp], input=self.audit, text=True,
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_attributes_on_either_side_share_the_declared_pointer_dtype(self):
        for constructor, args in [('zeros', '(N,)'), ('full', '(N,), 0.0')]:
            py = f'value = tl.{constructor}({args}, dtype=A.dtype.element_ty)'
            for dtype in ['A.dtype.element_ty', 'A.type.element_ty', 'tl.float32']:
                lean = ('tl.region A = tl.float32, B = tl.float16\n'
                        f'value = tl.{constructor}({args}, dtype={dtype})')
                with self.subTest(constructor=constructor, dtype=dtype):
                    result = self.run_audit(py, lean)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dtype_shape_and_fill_mutations_remain_failures(self):
        py = 'value = tl.full((N,), 1.0, dtype=A.dtype.element_ty)'
        for args, dtype in [('(N,), 1.0', 'tl.float16'),
                            ('(N,), 1.0', 'B.dtype.element_ty'),
                            ('(N + 1,), 1.0', 'A.dtype.element_ty'),
                            ('(N,), 2.0', 'A.dtype.element_ty'),
                            ('(N,), 1.0', 'missing.dtype.element_ty')]:
            lean = ('tl.region A = tl.float32, B = tl.float16\n'
                    f'value = tl.full({args}, dtype={dtype})')
            with self.subTest(args=args, dtype=dtype):
                result = self.run_audit(py, lean)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('shape/dtype call mismatch', result.stdout)

    def test_reassigned_pointer_does_not_retain_its_original_type(self):
        py = 'value = tl.zeros((N,), dtype=tl.float32)'
        for update in ['A = B + offset', 'A *= stride', 'A = choose(A, B)']:
            lean = ('tl.region A = tl.float32, B = tl.float16\n' + update +
                    '\nvalue = tl.zeros((N,), dtype=A.dtype.element_ty)')
            with self.subTest(update=update):
                result = self.run_audit(py, lean)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('shape/dtype call mismatch', result.stdout)


if __name__ == '__main__':
    unittest.main()
