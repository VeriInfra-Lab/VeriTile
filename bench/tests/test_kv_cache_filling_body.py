"""Exact helper/body spelling, real constexpr variants and the existing candidate."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest, validate_manifest_records

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'bench/repairs/tritonbench_g/kv_cache_filling'

class KvCacheFillingBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lean=(BASE/'Kernels.lean').read_text()
        cls.rows=json.loads((BASE/'jit-correspondence.json').read_text())

    def compare(self,row,lean):
        return check((ROOT/row['python']).read_text(),lean,
                     **{k:v for k,v in row.items() if k not in ('python','lean','profile_variant')})

    def test_complete_helper_and_six_body_profiles(self):
        results=check_manifest(ROOT,BASE/'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in results],[5,8,8,8,8,8,8])
        self.assertTrue(all(r['status']=='PASS' and r['mode']=='pasted_body' for r in results))

    def test_candidate_only_commutes_existing_vector_load_addresses(self):
        original=next(n for n in ast.parse((BASE/'kv_cache_filling_repaired.py').read_text()).body
                      if isinstance(n,ast.FunctionDef) and n.name=='_fill_rows')
        candidate=next(n for n in ast.parse((BASE/'candidate_kernels.py').read_text()).body
                       if isinstance(n,ast.FunctionDef) and n.name=='_fill_rows_commuted')
        candidate.name=original.name
        changed=[]
        for n in ast.walk(candidate):
            if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mult) and ast.unparse(n)=='DK * src':
                n.left,n.right=n.right,n.left;changed.append(n)
        self.assertEqual(len(changed),3)
        self.assertEqual(ast.dump(original),ast.dump(candidate))

    def test_helper_cast_cannot_disappear(self):
        changed=self.lean.replace('.to(tl.uint8)','',1)
        with self.assertRaises(ValueError):self.compare(self.rows[0],changed)

    def test_helper_constexpr_annotation_is_checked(self):
        changed=self.lean.replace('constexpr [LEVELS]','constexpr []',1)
        with self.assertRaisesRegex(ValueError,'constexpr'):self.compare(self.rows[0],changed)

    def test_helper_return_order_is_checked(self):
        changed=self.lean.replace('return code, scale, zero','return code, zero, scale',1)
        with self.assertRaises(ValueError):self.compare(self.rows[0],changed)

    def test_second_half_call_cannot_be_replaced_by_reuse(self):
        changed=self.lean.replace('qk2, _, _ = _quant_row(k2, klo, khi, 15)',
                                  'qk2 = (k2 / sk + zk + 0.5).to(tl.uint8)',1)
        with self.assertRaises(ValueError):self.compare(self.rows[1],changed)

    def test_source_int64_cast_is_required(self):
        changed=self.lean.replace('tl.program_id(0).to(tl.int64)','tl.program_id(0)',1)
        with self.assertRaises(ValueError):self.compare(self.rows[1],changed)

    def test_policy_value_is_checked_outside_body(self):
        changed=self.lean.replace('let POLICY : Nat := 8','let POLICY : Nat := 4',1)
        with self.assertRaisesRegex(ValueError,'constexpr caller'):self.compare(self.rows[3],changed)

    def test_fake_profile_tags_cannot_duplicate_the_same_policy(self):
        rows=copy.deepcopy(self.rows)
        for row in rows:
            if row.get('profile_variant')=='4':row['expected_constexpr_bindings']={'POLICY':8}
        with self.assertRaisesRegex(ValueError,'duplicate ABI'):validate_manifest_records(rows)

    def test_unsigned_byte_cache_dtype_is_checked(self):
        changed=self.lean.replace('("KC",.u8)','("KC",.u32)',1)
        with self.assertRaises(ValueError):self.compare(self.rows[3],changed)

if __name__=='__main__':unittest.main()
