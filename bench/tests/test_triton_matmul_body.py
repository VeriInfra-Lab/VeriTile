"""Check the selected matmul bodies, fixed ABIs and full existing candidate."""
import ast
import json
from pathlib import Path
import runpy
import unittest

from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/triton_matmul'
MANIFEST = BASE / 'KernelOnly/jit-correspondence.json'


class TritonMatmulBody(unittest.TestCase):
    def setUp(self):
        self.pairs = json.loads(MANIFEST.read_text())

    def rejects(self, before, after):
        row = self.pairs[0]
        lean = (ROOT / row['lean']).read_text()
        start = lean.index('def halfSource ')
        end = lean.index('\ndef signature_f8e4', start)
        body = lean[start:end]
        self.assertIn(before, body)
        changed = lean[:start] + body.replace(before, after, 1) + lean[end:]
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), changed,
                  row['python_function'], row['lean_definition'],
                  signature_definition=row['signature_definition'],
                  expected_signature=row['expected_signature'],
                  expected_call_bindings=row['expected_call_bindings'])

    def test_all_four_profiles(self):
        results = check_manifest(ROOT, MANIFEST)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(x['status'] == 'PASS' for x in results))
        self.assertTrue(all(x['jit_options'] == {'launch_metadata': '_matmul_launch_metadata'}
                            for x in results))

    def test_pid_cast(self):
        self.rejects('tl.program_id(axis=0).to(tl.int64)', 'tl.program_id(axis=0)')

    def test_simultaneous_metadata_casts(self):
        self.rejects('tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)',
                     'M, N, K')

    def test_boolean_widening(self):
        self.rejects('(M % BLOCK_SIZE_M != 0).to(tl.int64)', '(M % BLOCK_SIZE_M != 0)')

    def test_dtype_branch(self):
        self.rejects('c_ptr.dtype.element_ty == tl.float8e4nv',
                     'c_ptr.dtype.element_ty == tl.float16')

    def test_signed_source_loop(self):
        self.rejects('range(0, blocks_k)', 'range(0, K)')

    def test_live_accumulator(self):
        self.rejects('tl.dot(a, b, accumulator)', 'tl.dot(a, b) + accumulator')

    def test_explicit_output_cast(self):
        self.rejects('c = accumulator.to(tl.float16)', 'c = accumulator')

    def test_full_candidate_module(self):
        runpy.run_path(str(BASE / 'KernelOnly/check_pair_source.py'))

    def check_changed_decorator(self, replacement):
        row = self.pairs[0]
        python = (ROOT / row['python']).read_text().replace(
            'launch_metadata=_matmul_launch_metadata', replacement, 1)
        with self.assertRaises(ValueError):
            check(python, (ROOT / row['lean']).read_text(), row['python_function'],
                  row['lean_definition'], signature_definition=row['signature_definition'],
                  expected_signature=row['expected_signature'],
                  expected_call_bindings=row['expected_call_bindings'])

    def test_computed_callback_rejected(self):
        self.check_changed_decorator('launch_metadata=make_callback()')

    def test_missing_callback_rejected(self):
        self.check_changed_decorator('launch_metadata=missing_callback')

    def test_other_jit_options_remain_rejected(self):
        self.check_changed_decorator('unknown_option=True')


if __name__ == '__main__':
    unittest.main()
