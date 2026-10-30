"""Full original rotary source, explicit caller variants and the existing xor-one candidate."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, validate_manifest_records

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/rotary_transform_ops'

class RotaryTransformOpsBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lean = (BASE / 'Kernels.lean').read_text()
        cls.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def run_row(self, row, lean=None):
        return check((ROOT / row['python']).read_text(), self.lean if lean is None else lean,
                     **{k:v for k,v in row.items() if k not in ('python','lean','profile_variant')})

    def test_all_distinct_call_profiles(self):
        self.assertEqual(len(self.rows),12)
        validate_manifest_records(self.rows)
        for row in self.rows:self.assertEqual(self.run_row(row)['status'],'PASS')

    def test_existing_candidate_only_changes_adjacent_index(self):
        functions={n.name:n for n in ast.parse((BASE/'rotary_transform_ops_repaired.py').read_text()).body
                   if isinstance(n,ast.FunctionDef)}
        original=functions['rotary_kernel'];candidate=copy.deepcopy(functions['rotary_kernel_xor'])
        candidate.name=original.name
        self.assertEqual(ast.unparse(candidate.body[-1].orelse[0]),'rk_swap = rk ^ 1')
        candidate.body[-1].orelse[0]=copy.deepcopy(original.body[-1].orelse[0])
        self.assertEqual(ast.dump(original),ast.dump(candidate))

    def test_explicit_cast_cannot_be_omitted(self):
        changed=self.lean.replace('.to(tl.float32)','',1)
        with self.assertRaises(ValueError):self.run_row(self.rows[0],changed)

    def test_seqlen_initializer_cannot_be_invented(self):
        changed=self.lean.replace('    pid_m = tl.program_id(axis=0)','    seqlen = 0\n    pid_m = tl.program_id(axis=0)',1)
        with self.assertRaises(ValueError):self.run_row(self.rows[0],changed)

    def test_scalar_pointer_choice_cannot_flip(self):
        changed=self.lean.replace('let IS_SEQLEN_OFFSETS_TENSOR : Bool := Bool.false',
                                  'let IS_SEQLEN_OFFSETS_TENSOR : Bool := Bool.true',1)
        with self.assertRaises(ValueError):self.run_row(self.rows[0],changed)

    def test_early_return_condition_cannot_change(self):
        changed=self.lean.replace('if pid_m * BLOCK_M >= seqlen:', 'if pid_m * BLOCK_M > seqlen:',1)
        with self.assertRaises(ValueError):self.run_row(self.rows[0],changed)

    def test_metadata_dtype_cannot_change(self):
        changed=self.lean.replace('("CU_SEQLENS",.i32)','("CU_SEQLENS",.i64)',1)
        with self.assertRaises(ValueError):self.run_row(self.rows[0],changed)

    def test_candidate_cannot_add_another_rewrite(self):
        row=next(r for r in self.rows if r['python_function']=='rotary_kernel_xor')
        start=self.lean.index('def '+row['lean_definition']+'\n')
        changed=self.lean[:start]+self.lean[start:].replace('o0 = x0 * cos - x1 * sin','o0 = x0 * cos + x1 * sin',1)
        with self.assertRaises(ValueError):self.run_row(row,changed)

if __name__=='__main__':unittest.main()
