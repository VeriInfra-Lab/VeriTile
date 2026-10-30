"""Protect full RWKV bodies, reverse pointers, optional state and all scalar ABIs."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'bench/repairs/tritonbench_g/fused_rwkv6_kernel_scalar_abi_checked'

class Rwkv6BodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((BASE/'jit-correspondence.json').read_text())
        cls.sources=[(ROOT/r['lean']).read_text() for r in cls.rows]

    def reject(self,before,after,row=0):
        source=self.sources[row]
        changed=source.replace(before,after)
        self.assertNotEqual(source,changed)
        record=self.rows[row]
        with self.assertRaises(ValueError):
            check((ROOT/record['python']).read_text(),changed,
                  **{k:v for k,v in record.items() if k not in ('python','lean','profile_variant')})

    def test_all_raw_scale_profiles_and_integer_one(self):
        result=check_manifest(ROOT,BASE/'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in result],[16]*10)
        self.assertTrue(all(r['status']=='PASS' and r['mode']=='pasted_body' for r in result))

    def test_reverse_backedge_keeps_negative_offset(self):
        self.reject('p_q += -K if REVERSE else K','p_q += K if REVERSE else K')

    def test_reverse_initial_offset_keeps_t_minus_one(self):
        self.reject('((T-1) * K if REVERSE else 0)','(T * K if REVERSE else 0)')

    def test_mask_excludes_boundary(self):
        self.reject('tl.arange(0, BK)) < K','tl.arange(0, BK)) <= K')

    def test_query_cast_is_not_removed(self):
        self.reject('tl.load(p_q, mask=mask_bk, other=0).to(tl.float32)',
                    'tl.load(p_q, mask=mask_bk, other=0)')

    def test_optional_initial_state_remains(self):
        self.reject('if USE_INITIAL_STATE:','if STORE_FINAL_STATE:')

    def test_optional_final_state_remains(self):
        self.reject('if STORE_FINAL_STATE:','if USE_INITIAL_STATE:')

    def test_output_cast_uses_actual_pointer_dtype(self):
        self.reject('b_o.to(p_o.dtype.element_ty)','b_o')

    def test_zero_trip_range_remains(self):
        self.reject('range(0, T)','range(1, T)')

    def test_scale_width_cannot_change(self):
        for row,dtype in [(2,'i32'),(4,'i64'),(6,'u64')]:
            with self.subTest(dtype=dtype):self.reject('("scale",.'+dtype+')','("scale",.fp32)',row)

    def test_candidate_only_hoists_three_bias_statements(self):
        original_module=ast.parse((BASE/'fused_rwkv6_kernel_scalar_abi_checked.py').read_text())
        candidate_module=ast.parse((BASE/'fused_rwkv6_kernel_u_early.py').read_text())
        def function(module):
            return next(n for n in module.body if isinstance(n,ast.FunctionDef)
                        and n.name=='fused_recurrent_rwkv6_fwd_kernel')
        original=function(original_module)
        candidate=function(candidate_module)
        expected=copy.deepcopy(original)
        names=['p_u','mask_bk','b_u']
        selected=[]
        for name in names:
            found=[n for n in expected.body if isinstance(n,ast.Assign) and
                   len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
            self.assertEqual(len(found),1)
            selected+=found
        expected.body=[n for n in expected.body if n not in selected]
        position=next(i for i,n in enumerate(expected.body) if isinstance(n,ast.Assign) and
                      isinstance(n.targets[0],ast.Name) and n.targets[0].id=='i_h')+1
        expected.body[position:position]=selected
        self.assertEqual(ast.dump(expected),ast.dump(candidate))
        expected_module=copy.deepcopy(original_module)
        position=next(i for i,n in enumerate(expected_module.body)
                      if isinstance(n,ast.FunctionDef) and n.name==original.name)
        expected_module.body[position]=expected
        self.assertEqual(ast.dump(expected_module),ast.dump(candidate_module))

    def compare_one(self,python=None,lean=None,record=None):
        row=self.rows[8] if record is None else record
        return check((ROOT/row['python']).read_text() if python is None else python,
                     self.sources[8] if lean is None else lean,
                     **{k:v for k,v in row.items() if k not in ('python','lean','profile_variant')})

    def test_integer_one_requires_exact_caller_value(self):
        self.reject('let scale : Nat := 1','let scale : Nat := 2',8)
        row=copy.deepcopy(self.rows[8])
        row.pop('expected_constexpr_bindings')
        with self.assertRaises(ValueError):self.compare_one(record=row)

    def test_integer_one_respects_specialization_opt_out(self):
        python=(ROOT/self.rows[8]['python']).read_text()
        fn=next(n for n in ast.parse(python).body if isinstance(n,ast.FunctionDef)
                and n.name=='fused_recurrent_rwkv6_fwd_kernel')
        index=next(i for i,a in enumerate(fn.args.args) if a.arg=='scale')
        for disabled in [['scale'],[index]]:
            with self.subTest(disabled=disabled),self.assertRaises(ValueError):
                self.compare_one(python=python.replace('@triton.jit',
                    '@triton.jit(do_not_specialize='+repr(disabled)+')',1))

    def test_annotated_fp32_cannot_become_integer_one(self):
        python=(ROOT/self.rows[8]['python']).read_text()
        fn=next(n for n in ast.parse(python).body if isinstance(n,ast.FunctionDef)
                and n.name=='fused_recurrent_rwkv6_fwd_kernel')
        next(a for a in fn.args.args if a.arg=='scale').annotation=ast.parse('tl.float32',mode='eval').body
        with self.assertRaises(ValueError):self.compare_one(python=ast.unparse(ast.Module(body=[fn],type_ignores=[])))

if __name__=='__main__':unittest.main()
