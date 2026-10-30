"""Keep the full online loop and the existing address-only schedule change."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'bench/repairs/tritonbench_g/softmax_reducev_neginf_checked'

class SoftmaxReduceVBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lean = (BASE/'Kernels.lean').read_text()
        cls.rows = json.loads((BASE/'jit-correspondence.json').read_text())

    def compare(self, lean, row=0):
        record = self.rows[row]
        return check((ROOT/record['python']).read_text(), lean,
                     **{k:v for k,v in record.items() if k not in ('python','lean')})

    def test_complete_original_and_existing_candidate(self):
        results = check_manifest(ROOT, BASE/'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in results], [18,18])
        self.assertTrue(all(r['status']=='PASS' and r['mode']=='pasted_body' for r in results))

    def test_candidate_only_moves_the_two_address_assignments(self):
        fns = {n.name:n for n in ast.parse((BASE/'softmax_reducev_neginf_checked.py').read_text()).body
               if isinstance(n,ast.FunctionDef)}
        original = fns['_fwd_kernel']
        candidate = copy.deepcopy(fns['_fwd_kernel_early_address'])
        candidate.name = original.name
        # The final five statements contain address, normalization, store, return.
        self.assertEqual([ast.unparse(n.targets[0]) for n in candidate.body[-5:-2]],
                         ['off_o','out_ptrs','acc'])
        candidate.body[-5:-2] = [candidate.body[-3],candidate.body[-5],candidate.body[-4]]
        self.assertEqual(ast.dump(original),ast.dump(candidate))

    def test_no_extra_source_cast(self):
        changed = self.lean.replace('tl.store(out_ptrs, acc)', 'tl.store(out_ptrs, acc.to(tl.float32))', 1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_finite_clamp_cannot_disappear(self):
        changed = self.lean.replace('        n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)\n', '', 1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_value_padding_cannot_change(self):
        changed = self.lean.replace('[:, None], other=0.0)', '[:, None], other=1.0)', 1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_metadata_signed_width_is_checked(self):
        changed = self.lean.replace('("B_Loc",.i32)', '("B_Loc",.i64)', 1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_loop_extent_cannot_change(self):
        changed = self.lean.replace('range(0, cur_batch_seq_len, BLOCK_N)', 'range(0, max_input_len, BLOCK_N)', 1)
        with self.assertRaises(ValueError): self.compare(changed)

if __name__ == '__main__': unittest.main()
