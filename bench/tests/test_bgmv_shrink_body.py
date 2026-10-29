"""Raw-body and complete-module guards for the selected BGMV shrink pair."""
import json
from pathlib import Path
import runpy
import unittest

from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/bgmv_shrink_kernel_physical'


class BgmvShrinkBody(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def rejects(self, before, after):
        row = self.rows[0]
        lean = (ROOT / row['lean']).read_text()
        start = lean.index('def original_fp16_i32 ')
        end = lean.index('\ndef scalarLeft_fp16_i32 ', start)
        body = lean[start:end]
        self.assertEqual(body.count(before), 1)
        mutated = lean[:start] + body.replace(before, after, 1) + lean[end:]
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), mutated, row['python_function'],
                  row['lean_definition'], signature_definition=row['signature_definition'],
                  expected_signature=row['expected_signature'],
                  expected_call_bindings=row['expected_call_bindings'])

    def test_all_eight_profiles(self):
        rows = check_manifest(ROOT, BASE / 'jit-correspondence.json')
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row['status'] == 'PASS' for row in rows))

    def test_program_id_cast(self):
        self.rejects('tl.program_id(axis=0).to(tl.int64)', 'tl.program_id(axis=0)')

    def test_metadata_cast(self):
        self.rejects('tl.load(lora_indices + cur_batch).to(tl.int64)',
                     'tl.load(lora_indices + cur_batch)')

    def test_sentinel_branch(self):
        self.rejects('if lora_index == -1:', 'if lora_index != -1:')

    def test_early_return(self):
        self.rejects('        return', '        pass')

    def test_signed_bound_cast(self):
        self.rejects('tl.cast(K, tl.int64)', 'K')

    def test_split_loop_stride(self):
        self.rejects('BLOCK_K * SPLIT_K', 'BLOCK_K')

    def test_reduction_axis(self):
        self.rejects('tl.sum(tiled_a * tiled_b, 1)', 'tl.sum(tiled_a * tiled_b, 0)')

    def test_implicit_output_conversion(self):
        self.rejects('tl.store(c_ptr, accumulator, mask=c_mask)',
                     'tl.store(c_ptr, accumulator.to(tl.float32), mask=c_mask)')

    def test_atomic_write(self):
        self.rejects('tl.atomic_add(c_ptr, accumulator, mask=c_mask)',
                     'tl.store(c_ptr, accumulator, mask=c_mask)')

    def test_original_augmented_assignment(self):
        self.rejects('accumulator *= scaling', 'accumulator = accumulator * scaling')

    def test_full_candidate_module(self):
        result = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']()
        self.assertEqual(result['status'], 'PASS')

    def test_candidate_host_change_rejected(self):
        checker = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']
        text = (BASE / 'bgmv_shrink_scalar_left.py').read_text()
        self.assertIn('256 * BLOCK_N > 2**20', text)
        with self.assertRaises(ValueError):
            checker(candidate_text=text.replace('256 * BLOCK_N > 2**20', '256 * BLOCK_N > 2**21'))

    def test_candidate_other_cast_change_rejected(self):
        checker = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']
        text = (BASE / 'bgmv_shrink_scalar_left.py').read_text()
        with self.assertRaises(ValueError):
            checker(candidate_text=text.replace('.to(tl.int64)', '.to(tl.int32)', 1))


if __name__ == '__main__':
    unittest.main()
