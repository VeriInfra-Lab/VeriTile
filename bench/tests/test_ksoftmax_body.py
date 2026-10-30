"""Preserve complete forward/backward bodies, source widths and flag branches."""
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'bench/repairs/tritonbench_g/ksoftmax_triton_grid_y_checked'
SHARED = ROOT/'bench/repairs/tritonbench_g/ksoftmax_triton'

class KsoftmaxBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lean = (SHARED/'Kernels.lean').read_text()
        cls.rows = json.loads((BASE/'jit-correspondence.json').read_text())

    def compare(self, lean, row=0):
        record = self.rows[row]
        return check((ROOT/record['python']).read_text(), lean,
                     **{k:v for k,v in record.items() if k not in ('python','lean')})

    def test_complete_forward_and_backward(self):
        results = check_manifest(ROOT, BASE/'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in results], [17,15])
        self.assertTrue(all(r['status']=='PASS' and r['mode']=='pasted_body' for r in results))

    def test_all_explicit_widenings_remain(self):
        self.assertEqual(self.lean.count('.to(tl.int64)'),6)
        for row in range(2):
            with self.assertRaises(ValueError):
                self.compare(self.lean.replace('.to(tl.int64)',''),row)

    def test_backward_fp16_branch_cannot_disappear(self):
        changed = self.lean.replace('        if IS_FP16:\n            o = o.to(tl.float32)\n','')
        self.assertTrue(changed != self.lean)
        with self.assertRaises(ValueError): self.compare(changed,1)

    def test_optional_mask_presence_cannot_reverse(self):
        changed = self.lean.replace('MASK_TYPE is not None','MASK_TYPE is None',1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_qk_row_cannot_become_batch(self):
        changed = self.lean.replace('mask_ptrs = M + n * stride_m + k','mask_ptrs = M + m * stride_m + k',1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_causal_gradient_boundary_is_exact(self):
        changed = self.lean.replace('tl.where(k <= n, grad_in, 0.)','tl.where(k < n, grad_in, 0.)',1)
        with self.assertRaises(ValueError): self.compare(changed,1)

    def test_no_extra_store_cast(self):
        changed = self.lean.replace('tl.store(y_ptrs, y,','tl.store(y_ptrs, y.to(tl.float32),',1)
        with self.assertRaises(ValueError): self.compare(changed)

    def test_signed_stride_width_is_checked(self):
        changed = self.lean.replace('("stride_ym",.i64)','("stride_ym",.integer)',1)
        with self.assertRaises(ValueError): self.compare(changed)

if __name__ == '__main__': unittest.main()
