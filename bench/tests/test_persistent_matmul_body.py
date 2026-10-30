"""Guard literal persistent loops, masks, casts and the complete candidate pair."""
import json
from pathlib import Path
import runpy
import unittest

from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/matmul_persistent_triton_scalar_cast'


class PersistentMatmulBody(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def rejects(self, before, after, index=0):
        row = self.rows[index]
        text = (ROOT / row['lean']).read_text()
        start = text.index('def ' + row['lean_definition'] + ' (')
        end = text.find('\ndef ', start + 1)
        if end < 0:
            end = len(text)
        body = text[start:end]
        self.assertEqual(body.count(before), 1)
        mutated = text[:start] + body.replace(before, after, 1) + text[end:]
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), mutated, row['python_function'],
                  row['lean_definition'], signature_definition=row['signature_definition'],
                  expected_signature=row['expected_signature'],
                  expected_call_bindings=row['expected_call_bindings'])

    def test_four_profiles(self):
        checked = check_manifest(ROOT, BASE / 'jit-correspondence.json')
        self.assertEqual(len(checked), 4)
        self.assertTrue(all(r['status'] == 'PASS' for r in checked))

    def test_scalar_cast(self):
        self.rejects('tl.cast(M, tl.int64)', 'M')

    def test_integer_ceil(self):
        self.rejects('M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)',
                     'M // BLOCK_SIZE_M')

    def test_schedule_remainder(self):
        self.rejects('start_pid < num_tiles % NUM_SMS', 'start_pid <= num_tiles % NUM_SMS')

    def test_outer_loop(self):
        self.rejects('range(0, tiles_per_SM)', 'range(0, num_tiles)')

    def test_inner_loop(self):
        self.rejects('range(0, k_tiles)', 'range(0, K)')

    def test_pointer_clamp(self):
        self.rejects('tl.where(raw_k < K, raw_k, 0)', 'raw_k')

    def test_live_accumulator(self):
        self.rejects('tl.dot(a, b, accumulator)', 'tl.dot(a, b)')

    def test_output_dtype_branch(self):
        self.rejects('c_ptr.dtype.element_ty == tl.float8e4nv',
                     'c_ptr.dtype.element_ty == tl.float16', index=1)

    def test_store_mask(self):
        self.rejects('tl.store(c_ptrs, c, mask=c_mask)', 'tl.store(c_ptrs, c)')

    def test_full_pair(self):
        check_pair = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']
        self.assertEqual(check_pair()['status'], 'PASS')

    def test_candidate_extra_change(self):
        check_pair = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']
        candidate = (BASE / 'matmul_persistent_triton_reordered.py').read_text()
        with self.assertRaises(ValueError):
            check_pair(candidate_text=candidate.replace('other=0.0', 'other=1.0', 1))


if __name__ == '__main__':
    unittest.main()
