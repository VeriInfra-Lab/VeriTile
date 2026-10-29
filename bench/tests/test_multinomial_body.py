"""Preserve RNG, scan, label clamp and candidate loop semantics in pasted bodies."""
import json
from pathlib import Path
import runpy
import unittest

from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/multinomial_sampling'


class MultinomialBody(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def rejects(self, before, after, index=0):
        row = self.rows[index]
        text = (ROOT / row['lean']).read_text()
        start = text.index('def ' + row['lean_definition'] + '\n')
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

    def test_five_profiles(self):
        checked = check_manifest(ROOT, BASE / 'jit-correspondence.json')
        self.assertEqual(len(checked), 5)
        self.assertTrue(all(r['status'] == 'PASS' for r in checked))

    def test_offset_narrowing(self):
        self.rejects('tl.load(Offsets + off, mask=off_mask).to(tl.int32)',
                     'tl.load(Offsets + off, mask=off_mask).to(tl.int64)')

    def test_random_operands(self):
        self.rejects('tl.rand(seed, offset)', 'tl.rand(offset, seed)')

    def test_scan_axis(self):
        self.rejects('tl.cumsum(scores, 1)', 'tl.cumsum(scores, 0)')

    def test_threshold_endpoint(self):
        self.rejects('samp <= cum_scores', 'samp < cum_scores')

    def test_label_clamp(self):
        self.rejects('tl.minimum(valid_pos, num_tokens - 1)', 'valid_pos')

    def test_label_mask(self):
        self.rejects('mask=found_mask & off_mask', 'mask=found_mask')

    def test_static_loop(self):
        self.rejects('tl.static_range(tl.cdiv(num_tokens, BLOCK_N))', 'range(num_tokens)', index=4)

    def test_full_existing_candidate(self):
        check_pair = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']
        self.assertEqual(check_pair()['status'], 'PASS')

    def test_candidate_other_change_rejected(self):
        check_pair = runpy.run_path(str(BASE / 'check_pair_source.py'))['check_pair']
        text = (BASE / 'multinomial_sampling_repaired.py').read_text()
        start = text.index('def _multinomial_sampling_unrolled_kernel')
        mutated = text[:start] + text[start:].replace('tl.cumsum(scores, 1)', 'tl.cumsum(scores, 0)', 1)
        with self.assertRaises(ValueError):
            check_pair(mutated)


if __name__ == '__main__':
    unittest.main()
