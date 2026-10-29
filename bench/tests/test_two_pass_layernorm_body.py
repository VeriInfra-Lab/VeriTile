"""Guard exact two-pass bodies, invocation profiles and the whole candidate."""
import ast
import json
from pathlib import Path
import runpy
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/layer_norm_welfold'

class TwoPassLayernormBody(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def rejects(self, before, after):
        row = self.rows[0]
        text = (ROOT / row['lean']).read_text()
        self.assertIn(before, text)
        mutated = text.replace(before, after, 1)
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), mutated,
                  row['python_function'],row['lean_definition'],
                  signature_definition=row['signature_definition'],
                  expected_signature=row['expected_signature'],
                  expected_signature_arguments=row['expected_signature_arguments'],
                  expected_call_bindings=row['expected_call_bindings'],
                  expected_constexpr_bindings=row['expected_constexpr_bindings'])

    def test_both_literal_bodies(self):
        self.assertEqual(len(check_manifest(ROOT,BASE / 'jit-correspondence.json')),2)
    def test_original_intrinsic(self):
        self.rejects('libdevice.rsqrt(tmp17)','tl.rsqrt(tmp17)')
    def test_dynamic_loop(self):
        self.rejects('range(0, rnumel, RBLOCK)','range(0, RBLOCK, RBLOCK)')
    def test_cast(self):
        self.rejects('tmp1 = tmp0.to(tl.float32)','tmp1 = tmp0')
    def test_variance_tail(self):
        self.rejects('tl.where(rmask, tmp9 * tmp9, 0.0)','tmp9 * tmp9')
    def test_output_mask(self):
        self.rejects('tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp29, rmask)',
                     'tl.store(out_ptr0 + (r1 + (rnumel * x0)), tmp29)')
    def test_mean_store(self):
        self.rejects('tl.store(in_out_ptr0 + (x0), tmp6, None)',
                     'tl.store(in_out_ptr0 + (x0), tmp3, None)')
    def test_tree_argument(self):
        self.rejects('(rowSignature B tree)','(rowSignature B otherTree)')
    def test_tree_binding(self):
        self.rejects('("_tmp12",⟨B,tree⟩)','("_tmp12",⟨B,otherTree⟩)')
    def test_block_alias(self):
        self.rejects('let RBLOCK : Nat := B','let RBLOCK : Nat := D')
    def test_row_specialization(self):
        self.rejects('let XBLOCK : Nat := 1','let XBLOCK : Nat := 2')
    def test_full_pair(self):
        pair = runpy.run_path(str(BASE/'check_pair_source.py'))['check_pair']
        self.assertEqual(pair()['status'],'PASS')
    def test_unrelated_candidate_mutation(self):
        pair = runpy.run_path(str(BASE/'check_pair_source.py'))['check_pair']
        candidate = (BASE/'layer_norm_welfold_row_offset_reused.py').read_text()
        with self.assertRaises(ValueError):
            pair(candidate_text=candidate.replace('other=0.0','other=1.0',1))
    def test_candidate_definition_after_use(self):
        pair = runpy.run_path(str(BASE/'check_pair_source.py'))['check_pair']
        candidate = ast.parse((BASE/'layer_norm_welfold_row_offset_reused.py').read_text())
        entry = next(node for node in candidate.body if isinstance(node,ast.FunctionDef)
                     and node.name.endswith('_row_offset_reused'))
        loop = entry.body[-1]
        loop.body.append(loop.body.pop(3))
        with self.assertRaises(ValueError):
            pair(candidate_text=ast.unparse(candidate))
    def test_candidate_wrong_row_offset(self):
        pair = runpy.run_path(str(BASE/'check_pair_source.py'))['check_pair']
        candidate = (BASE/'layer_norm_welfold_row_offset_reused.py').read_text()
        with self.assertRaises(ValueError):
            pair(candidate_text=candidate.replace('row_offset = rnumel * x0',
                                                  'row_offset = xnumel * x0',1))

if __name__ == '__main__':
    unittest.main()
