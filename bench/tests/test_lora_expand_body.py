"""Protect complete LoRA bodies, signed metadata ABIs and the existing candidate."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'bench/repairs/tritonbench_g/lora_expand_gemv'
SELECTED = BASE/'ByteSeparated'

class LoraExpandBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((SELECTED/'jit-correspondence.json').read_text())
        cls.lean = (SELECTED/'Kernels.lean').read_text()
        cls.candidate = (SELECTED/'CandidateKernels.lean').read_text()

    def compare(self, lean, row=0):
        record = self.rows[row]
        return check((ROOT/record['python']).read_text(), lean,
                     **{k:v for k,v in record.items() if k not in ('python','lean','profile_variant')})

    def reject(self, before, after, row=0):
        source = self.candidate if row%2 else self.lean
        changed = source.replace(before, after)
        self.assertNotEqual(source, changed)
        with self.assertRaises(ValueError): self.compare(changed,row)

    def test_complete_bodies_for_both_index_widths(self):
        results = check_manifest(ROOT, SELECTED/'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in results],[13]*4)
        self.assertTrue(all(r['mode']=='pasted_body' and r['status']=='PASS' for r in results))

    def test_early_return_cannot_be_replaced(self):
        self.reject('        return\n','        pass\n')

    def test_sentinel_cannot_reverse(self):
        self.reject('if lora_index == -1:', 'if lora_index != -1:')

    def test_cast_to_actual_weight_dtype_remains(self):
        self.reject('.to(lora_ptr.dtype.element_ty)', '.to(tl.float32)')

    def test_mask_boundary_is_exact(self):
        self.reject('mask=offset_k < K', 'mask=offset_k <= K')

    def test_add_inputs_branch_remains(self):
        self.reject(' + tiled_out\n', '\n')

    def test_loop_step_remains(self):
        self.reject('range(0, split_n_length, BLOCK_N)', 'range(0, split_n_length, 1)')

    def test_no_extra_output_cast(self):
        self.reject('accumulator, mask=c_mask)', 'accumulator.to(tl.float32), mask=c_mask)')

    def test_int32_index_profile_cannot_widen(self):
        self.reject('("lora_indices",.i32)', '("lora_indices",.i64)',2)

    def test_runtime_scalar_width_is_checked(self):
        self.reject('("K",.i32)', '("K",.integer)')

    def test_candidate_is_exact_existing_single_split_rewrite(self):
        def function(path,name):
            return next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name==name)
        original=function(BASE/'lora_expand_gemv_byte_separated.py','_bgmv_expand_kernel')
        candidate=function(BASE/'lora_expand_gemv_unit_split.py','_bgmv_expand_unit_split_kernel')
        expected=copy.deepcopy(original)
        sites=[n for n in ast.walk(expected) if isinstance(n,ast.Assign)
               and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
               and n.targets[0].id=='split_n_length']
        self.assertEqual(len(sites),1)
        self.assertEqual(ast.unparse(sites[0].value),'tl.cdiv(N, SPLIT_N)')
        sites[0].value=ast.Constant(value=1)
        expected.name=candidate.name
        self.assertEqual(ast.dump(expected),ast.dump(candidate))

    def test_candidate_split_value_cannot_change(self):
        self.reject('split_n_length = 1','split_n_length = 2',1)

if __name__ == '__main__': unittest.main()
