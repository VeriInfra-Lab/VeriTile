"""Full original rotary source, explicit caller variants and the existing CSE."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, validate_manifest_records

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/rotary_transform_tile_checked'

class RotaryTransformBodyTests(unittest.TestCase):
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

    def test_existing_candidate_only_reuses_rk(self):
        original,candidate=[next(n for n in ast.parse((BASE/name).read_text()).body
                               if isinstance(n,ast.FunctionDef) and n.name==fn)
                            for name,fn in [('rotary_transform_tile_checked.py','rotary_kernel'),
                                            ('candidate_kernels.py','rotary_kernel_shared')]]
        candidate=copy.deepcopy(candidate);candidate.name=original.name
        self.assertEqual(ast.unparse(candidate.body[-1].orelse[1]),'rk_repeat = rk // 2')
        candidate.body[-1].orelse[1]=copy.deepcopy(original.body[-1].orelse[1])
        self.assertEqual(ast.dump(original),ast.dump(candidate))

    def test_unused_call_parameters_have_no_body_uses(self):
        function=next(n for n in ast.parse((BASE/'rotary_transform_tile_checked.py').read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='rotary_kernel')
        used={n.id for stmt in function.body for n in ast.walk(stmt) if isinstance(n,ast.Name)}
        self.assertTrue({'nheads','CACHE_KEY_SEQLEN'}.isdisjoint(used))

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
        row=next(r for r in self.rows if r['python_function']=='rotary_kernel_shared')
        start=self.lean.index('def '+row['lean_definition']+'\n')
        changed=self.lean[:start]+self.lean[start:].replace('o0 = x0 * cos - x1 * sin','o0 = x0 * cos + x1 * sin',1)
        with self.assertRaises(ValueError):self.run_row(row,changed)

if __name__=='__main__':unittest.main()
