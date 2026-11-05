import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/layer_norm_triton'


class LayerNormTritonBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, text):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), text, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'], expected_integer_args=r['expected_integer_args'], expected_arange_args=r['expected_arange_args'])

    def test_six_complete_bodies(self):
        self.assertEqual([r['statements'] for r in check_pinned_pairs(ROOT, DIRECTORY)], [15,14,11,30,14,11])

    def test_cast_loop_mask_and_store_mutations_are_rejected(self):
        for before, after in [('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
                              ('stride = tl.cast(stride, tl.int64)', 'stride = stride'),
                              ('other=0.).to(tl.float32)', 'other=0.)'),
                              ('tl.store(Y + cols, y, mask=mask)', 'tl.store(Y + cols, y.to(tl.float32), mask=mask)')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError): self.verify(self.lean.replace(before, after))
        r=self.records[1]
        for before,after in [('while row < M:', 'for row in range(group, M, GROUP_SIZE_M):'),
                             ('row += GROUP_SIZE_M', 'row += 1'),
                             ('dtype=DW.dtype.element_ty', 'dtype=tl.float32')]:
            with self.subTest(before=before):
                with self.assertRaises(ValueError):
                    check((ROOT/r['python']).read_text(),self.lean.replace(before,after),r['python_function'],r['lean_definition'],r['signature_definition'],r['expected_signature'],expected_call_bindings=r['expected_call_bindings'],expected_integer_args={},expected_arange_args={})

    def test_dtype_and_call_binding_are_checked(self):
        for before,after in [('("X",.fp32)','("X",.fp16)'),('stride := stride','stride := 0')]:
            with self.subTest(before=before):
                with self.assertRaises(ValueError): self.verify(self.lean.replace(before,after))

    def test_existing_candidate_restores_entire_module(self):
        original_text = variant_source(DIRECTORY / 'layer_norm_triton.py')
        candidate_text = variant_source(DIRECTORY / 'layer_norm_triton_scheduled.py')
        record = variant_provenance(DIRECTORY)
        self.assertEqual(hashlib.sha256(original_text.encode()).hexdigest(), record['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate_text.encode()).hexdigest(), record['candidate_sha256'])
        original = ast.parse(original_text)
        fns = {n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
        fn=fns['_layer_norm_fwd_fused']; expanded=[]
        for statement in fn.body:
            if isinstance(statement,ast.For):
                expanded.extend([ast.Assign(targets=[copy.deepcopy(statement.target)],value=ast.Constant(0))]+statement.body)
            else: expanded.append(statement)
        fn.body=expanded
        body=next(n for n in fns['_layer_norm_bwd_dx_fused'].body if isinstance(n,ast.While)).body
        load=next(n for n in body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='w')
        body.remove(load)
        index=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='xhat')
        body.insert(index+1,load)
        body=fns['_layer_norm_bwd_dwdb'].body
        store=body.pop(-2);body.insert(-2,store)
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(ast.parse(candidate_text),include_attributes=False))
