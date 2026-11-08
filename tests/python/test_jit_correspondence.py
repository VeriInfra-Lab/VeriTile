import ast
import json
import textwrap
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_manifest, check_statement_function, quotation

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, split_swiglu_fixture, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "bench/tritonbench_g/triton_argmax"


class JITCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.python = (DIRECTORY / "triton_argmax.py").read_text()
        self.lean = (DIRECTORY / "Kernels.lean").read_text()
        self.record = pinned_pairs(DIRECTORY)[0]

    def compare(self, python, lean):
        return check(python, lean, "argmax_kernel_1_repaired", self.record["lean_definition"],
                     self.record["signature_definition"], self.record["expected_signature"])

    def test_all_original_and_candidate_bodies(self):
        self.assertEqual(len(check_pinned_pairs(ROOT, DIRECTORY)), 6)

    def test_cast_nan_clamp_store_and_parameter_changes_are_rejected(self):
        mutations = [
            ("pid.to(tl.int64)", "pid"),
            ("tl.maximum(stored_value,", "tl.minimum(stored_value,"),
            ("tl.where(max_val == max_val, max_val, 0.0)", "max_val"),
            ("tl.store(mid_value_ptr, stored_value)", "tl.store(mid_value_ptr, max_val)"),
            ('("M",.i64)', '("M",.fp32)'),
            ("offset < M", "offset <= M"),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.compare(self.python, self.lean.replace(before, after, 1))

    def test_jit_decorator_is_required(self):
        with self.assertRaisesRegex(ValueError, "expected one explicitly selected"):
            self.compare(self.python.replace("@triton.jit", "", 1), self.lean)

    def test_runtime_parameter_cannot_be_replaced_by_antiquote(self):
        with self.assertRaisesRegex(ValueError, "antiquot"):
            self.compare(self.python, self.lean.replace("offset < M", "offset < $(M)", 1))

    def test_unsupported_splice_cannot_hide_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "antiquot"):
            self.compare(self.python, self.lean.replace(" * BLOCK_SIZE", " * $(BLOCK_SIZE + 1)", 1))


class InlineAssemblyBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/tritonbench_g/int8_matmul_kernel"
        self.python = (self.directory / "int8_matmul_kernel.py").read_text()
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.record = pinned_pairs(self.directory)[0]

    def compare(self, lean):
        return check(self.python, lean, "matmul_kernel", "sourceKernel", "callSignature",
                     self.record["expected_signature"],
                     expected_call_bindings=self.record["expected_call_bindings"])

    def test_pure_assembly_placeholders_are_string_data(self):
        self.assertEqual(self.compare(self.lean)["status"], "PASS")

    def test_complete_assembly_abi_remains_compared(self):
        for before, after in [("add.u32 $0, $1, $2;", "sub.u32 $0, $1, $2;"),
                              ('constraints="=r,r,r"', 'constraints="=r,r,l"'),
                              ("is_pure=True", "is_pure=False"), ("pack=1", "pack=2")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))

    def test_antiquote_outside_string_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "antiquot"):
            self.compare(self.lean.replace("pid = tl.program_id(axis=0)", "pid = $(M)", 1))

    def test_source_comment_is_not_a_metadata_directive(self):
        self.assertEqual(self.compare(self.lean.replace("pid = tl.program_id(axis=0)",
            "# literal PTX $0 and tl.region appear only in this comment\n    pid = tl.program_id(axis=0)", 1))["status"], "PASS")


class PastedBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/tritonbench_g/dequantize_rowwise"
        self.python = (self.directory / "dequantize_rowwise.py").read_text()
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.record = pinned_pairs(self.directory)[0]

    def compare(self, lean):
        return check(self.python, lean, self.record["python_function"], self.record["lean_definition"],
                     self.record["signature_definition"], self.record["expected_signature"])

    def test_verbatim_body_and_external_signature(self):
        self.assertEqual(self.compare(self.lean)["statements"], 9)
        fn = next(n for n in ast.parse(self.python).body
                  if isinstance(n, ast.FunctionDef) and n.name == self.record["python_function"])
        python_body = "\n".join(self.python.splitlines()[fn.body[0].lineno-1:fn.body[-1].end_lineno])
        lean_body = quotation(self.lean, "bodyKernel")[1]
        self.assertEqual(textwrap.dedent(python_body).strip(), textwrap.dedent(lean_body).strip())

    def test_no_metadata_or_antiquotes_in_pasted_body(self):
        for statement in ["tl.region x_ptr = tl.int8", "tl.param inv_127 : tl.float32",
                          "tl.call_signature", "a = $(P2)"]:
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                self.compare(self.lean.replace("  pid =", "  " + statement + "\n  pid =", 1))

    def test_removed_cast_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(self.lean.replace(".to(tl.int64)", "", 1))

    def test_launch_types_and_unused_parameter_are_checked(self):
        for before, after in [('("output_ptr",.fp16)', '("output_ptr",.fp32)'),
                              ('("inv_127",.fp32)', '("inv_127",.fp64)'),
                              (',("n_elements",.integer)', '')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.compare(self.lean.replace(before, after, 1))

    def test_external_annotations_are_checked_without_in_body_parameters(self):
        python = self.python.replace("inv_127,", "inv_127: tl.float32,", 1)
        result = check(python, self.lean, self.record["python_function"], "bodyKernel",
                       "callSignature", self.record["expected_signature"])
        self.assertEqual(result["mode"], "pasted_body")
        for annotation in ("tl.int32", "tl.int64"):
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                check(self.python.replace("n_elements,", f"n_elements: {annotation},", 1),
                      self.lean, self.record["python_function"], "bodyKernel", "callSignature",
                      self.record["expected_signature"])

    def test_pointer_and_quoted_constexpr_annotations(self):
        python = self.python.replace("x_ptr,", "x_ptr: tl.tensor,", 1).replace(
            "BLOCK_SIZE: tl.constexpr", 'BLOCK_SIZE: "tl.constexpr"', 1)
        self.assertEqual(check(python, self.lean, self.record["python_function"], "bodyKernel",
                               "callSignature", self.record["expected_signature"])["status"], "PASS")

    def test_unreviewed_signature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed call signature"):
            check(self.python, self.lean, self.record["python_function"], "bodyKernel", "callSignature")

    def test_following_type_declaration_is_not_part_of_the_body(self):
        lean = self.lean[:self.lean.index("/-- Candidate:")]
        for declaration in ("inductive Config where\n  | first\n", "structure Config where\n  block : Nat\n"):
            with self.subTest(declaration=declaration):
                self.assertEqual(self.compare(lean + declaration)["mode"], "pasted_body")


class MatmulSourceTests(unittest.TestCase):
    def test_complete_standalone_dot_sources_are_checked(self):
        for task in ("matmul_triton1", "matmul_triton2"):
            directory = ROOT / "bench/tritonbench_g" / task
            with self.subTest(task=task):
                result = check_pinned_pairs(ROOT, directory)
                self.assertEqual(result[0]["mode"], "pasted_body")
                self.assertEqual(result[0]["status"], "PASS")

    def test_compiler_fusion_and_removed_uint64_cast_are_rejected(self):
        cases = {
            "matmul_triton1": ("z += tl.dot(x_sub, y_sub, allow_tf32=False)",
                               "z = tl.dot(x_sub, y_sub, z, allow_tf32=False)"),
            "matmul_triton2": ("accumulator += tl.dot(a, b)",
                               "accumulator = tl.dot(a, b, accumulator)")}
        for task, (before, after) in cases.items():
            directory = ROOT / "bench/tritonbench_g" / task
            python = (directory / (task + ".py")).read_text()
            lean = (directory / "Kernels.lean").read_text()
            record = pinned_pairs(directory)[0]
            declaration = "def " + record["lean_definition"]
            prefix, body = lean.split(declaration, 1)
            for original, changed in ((before, after), (".to(tl.uint64)", "")):
                with self.subTest(task=task, mutation=original):
                    self.assertIn(original, body)
                    with self.assertRaises(ValueError):
                        check(python, prefix + declaration + body.replace(original, changed, 1),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"])

    def test_zero_shape_dtype_and_spelling_are_retained(self):
        directory = ROOT / "bench/tritonbench_g/matmul_triton1"
        python = (directory / "matmul_triton1.py").read_text()
        lean = (directory / "Kernels.lean").read_text()
        record = pinned_pairs(directory)[0]
        declaration = "def " + record["lean_definition"]
        prefix, body = lean.split(declaration, 1)
        for before, after in (("(m_block_size, n_block_size)", "(n_block_size, m_block_size)"),
                              ("(m_block_size, n_block_size)", "[m_block_size, n_block_size]"),
                              ("dtype=tl.float32", "dtype=tl.float16")):
            with self.subTest(mutation=before):
                self.assertIn(before, body)
                with self.assertRaises(ValueError):
                    check(python, prefix + declaration + body.replace(before, after, 1),
                          record["python_function"], record["lean_definition"],
                          record["signature_definition"], record["expected_signature"])



class BlockSparseBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/block_sparse_attn'
        self.python = (self.directory / 'block_sparse_attn.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, lean):
        r = self.records[0]
        return check(self.python, lean, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_both_complete_programs(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [31, 31])
        self.assertTrue(all(row['mode'] == 'pasted_body' for row in rows))

    def test_csr_casts_channel_loop_dtype_masks_and_updates_are_preserved(self):
        for before, after in [
                ('end_l = tl.load(layout_ptr + 1).to(tl.int32)', 'end_l = tl.load(layout_ptr).to(tl.int32)'),
                ('tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)',
                 'tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx)'),
                ('tl.static_range(NUM_D_BLOCKS)', 'tl.static_range(1)'),
                ('qk += tl.dot(q, k)', 'qk = tl.dot(q, k)'),
                ('p = p.to(Q.dtype.element_ty)', 'p = p'),
                ('offs_n[None, :] + start_n < total_seq_len', 'offs_n[None, :] + start_n <= total_seq_len'),
                ('l_i = l_i_new', 'l_i = l_ij')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.verify(self.lean.replace(before, after, 1))

    def test_candidate_changes_only_the_existing_integer_head_expression(self):
        original_path = self.directory / 'block_sparse_attn.py'
        candidate_path = self.directory / 'block_sparse_attn_quotient_remainder.py'
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'block_sparse_attention_kernel')
        stmt = next(n for n in fn.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'off_h')
        self.assertEqual(ast.unparse(stmt.value), 'off_bh - off_bh // num_heads * num_heads')
        stmt.value = ast.parse('off_bh % num_heads', mode='eval').body
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        import hashlib
        provenance = variant_provenance(self.directory)
        self.assertEqual(hashlib.sha256(variant_source(original_path).encode()).hexdigest(), provenance['original_sha256'])
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance['candidate_sha256'])


class ChunkBwdDqkgBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/chunk_bwd_dqkg'
        self.python = (self.directory / 'chunk_bwd_dqkg.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, lean):
        r = self.records[0]
        return check(self.python, lean, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_original_and_existing_tail_bound_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [33, 33])
        self.assertTrue(all(row['mode'] == 'pasted_body' for row in rows))

    def test_tuple_ids_full_value_loop_casts_reductions_and_stores_are_preserved(self):
        for before, after in [
                ('tl.program_id(0), tl.program_id(1), tl.program_id(2)',
                 'tl.program_id(0), tl.program_id(2), tl.program_id(1)'),
                ('range(tl.cdiv(V, BV))', 'range(1)'),
                ('i_t * (K * s_h_t)', 'i_t * K'),
                ('tl.sum(b_h * b_dh)', 'tl.sum(b_h * b_dh, 1)'),
                ('b_h.to(b_do.dtype)', 'b_h'),
                ('b_ds = b_ds.to(b_k.dtype)', 'b_ds = b_ds'),
                ('b_dg_last *= tl.exp(b_g_last)', 'b_dg_last = tl.exp(b_g_last)'),
                ('b_dq.to(p_dq.dtype.element_ty)', 'b_dq'),
                ('min(BT, T-i_t*BT) - 1', 'BT - 1')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.verify(self.lean.replace(before, after, 1))

    def test_candidate_changes_only_the_existing_integer_tail_bound(self):
        original_path = self.directory / 'chunk_bwd_dqkg.py'
        candidate_path = self.directory / 'chunk_bwd_dqkg_tail_bound.py'
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'chunk_simple_gla_bwd_kernel_dqkg')
        stmt = next(n for n in fn.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == 'b_dg'
                    and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'tl.where')
        self.assertEqual(ast.unparse(stmt.value.args[0]),
                         'o_i < min(i_t * BT + BT, T) - i_t * BT - 1')
        stmt.value.args[0] = ast.parse('o_i < min(BT, T-i_t*BT) - 1', mode='eval').body
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        import hashlib
        provenance = variant_provenance(self.directory)
        self.assertEqual(hashlib.sha256(variant_source(original_path).encode()).hexdigest(), provenance['original_sha256'])
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance['candidate_sha256'])


class RebindingProcedureBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/parallel_retention_attention'
        self.python = (self.directory / 'parallel_retention_attention.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def test_all_seven_complete_rebinding_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [34, 32, 34, 17, 34, 32, 17])

    def test_rebinding_and_cast_width_are_not_erased(self):
        from scripts.bench.jit_correspondence import check_procedure_function
        record = self.records[1]
        start = self.lean.index('def queryProcedureSource')
        for before, after in [('T = tl.cast(T, tl.int64)', 'T = T'),
                              ('.to(tl.int32)', '.to(tl.int64)'),
                              ('s_qk_h = tl.cast(s_qk_h, tl.int64)', '')]:
            with self.subTest(mutation=before):
                self.assertIn(before, self.lean[start:])
                changed = self.lean[:start] + self.lean[start:].replace(before, after, 1)
                with self.assertRaises(ValueError):
                    check_procedure_function(self.python, changed, record['python_function'],
                        record['lean_definition'], record['signature_definition'],
                        record['expected_signature'], expected_parameter_rebinding=True)

    def test_rebinding_adapter_must_be_explicitly_reviewed(self):
        record = self.records[3]
        with self.assertRaises(ValueError):
            check(self.python, self.lean, record['python_function'], record['lean_definition'],
                  record['signature_definition'], record['expected_signature'],
                  expected_call_bindings=record['expected_call_bindings'],
                  expected_jit_procedures=record['expected_jit_procedures'])

if __name__ == "__main__":
    unittest.main()


class PastedLoopBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/matrix_vector_multip'
        self.python = (self.directory / 'matrix_vector_multip.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def compare(self, lean):
        return check(self.python, lean, self.record['python_function'], self.record['lean_definition'],
                     self.record['signature_definition'], self.record['expected_signature'])

    def test_both_complete_python_while_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([18, 18], [r['statements'] for r in results])
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))

    def test_cast_increment_and_arithmetic_changes_are_rejected(self):
        for before, after in [('.to(tl.uint64)', ''), ('        m += BLOCK_M', ''),
                              ('acc += a * b', 'acc += b * a')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.compare(self.lean.replace(before, after, 1))

    def test_braces_in_python_comments_and_strings_do_not_close_body(self):
        lean = '''def bodyKernel := with_triton_signature profile in triton {
    marker = 'False#} {'
    # } This comment contains a delimiter.
    result = marker
}
noncomputable def following := 0
'''
        body = quotation(lean, 'bodyKernel', raw_python=True)[1]
        parsed = ast.parse(textwrap.dedent(body))
        self.assertEqual(len(parsed.body), 2)
        self.assertEqual(parsed.body[0].value.value, 'False#} {')
        self.assertNotIn('following', body)

    def test_noncomputable_next_declaration_is_not_part_of_kernel(self):
        quoted = quotation(self.lean, 'fmaKernel', raw_python=True)[1]
        self.assertEqual(len(ast.parse(textwrap.dedent(quoted)).body), 18)
        self.assertNotIn('arguments', quoted)

    def test_scoped_options_before_next_definition_bound_the_quote(self):
        lean = "def bodyKernel := triton {\n x = 1\n}\nset_option maxHeartbeats 1000000 in\ndef next := 0\n"
        body = quotation(lean, "bodyKernel", raw_python=True)[1]
        self.assertEqual(len(ast.parse(textwrap.dedent(body)).body), 1)
        with self.assertRaisesRegex(ValueError, "unexpected code after quotation"):
            quotation(lean.replace("}\nset_option", "} + extra\nset_option"), "bodyKernel", raw_python=True)

    def test_executable_suffix_after_quote_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unexpected code after quotation'):
            quotation('def bodyKernel := triton {\n x = 1\n} + extra\nnoncomputable def next := 0\n',
                      'bodyKernel', raw_python=True)


class PolymorphicPastedBodyTests(unittest.TestCase):
    def setUp(self):
        self.python = '''@triton.jit
def copy(s, o, N: tl.constexpr):
    x = tl.load(s)
    tl.store(o, x.to(o.dtype.element_ty))
'''
        self.lean = '''def signature (dtype out_dtype : FloatDType) : CallSignature where
  floatingPointers := [("s",dtype),("o",out_dtype)]
  scalars := []
  constexpr := ["N"]

def body (dtype out_dtype : FloatDType) (N : Nat) :=
  with_triton_signature (signature dtype out_dtype) in triton {
    x = tl.load(s)
    tl.store(o, x.to(o.dtype.element_ty))
  }
'''
        self.profile = {"pointers": [], "scalars": [], "constexpr": ["N"],
                        "floatingPointers": [["s", "dtype"], ["o", "out_dtype"]],
                        "dtype_parameters": ["dtype", "out_dtype"]}

    def compare(self, lean):
        return check(self.python, lean, "copy", "body", "signature", self.profile)

    def test_external_dtype_parameters_and_exact_cast(self):
        self.assertEqual(self.compare(self.lean)["mode"], "pasted_body")

    def test_dtype_arguments_storage_mapping_and_cast_mutations_are_rejected(self):
        for before, after in [
            ("signature dtype out_dtype", "signature out_dtype dtype"),
            ('("o",out_dtype)', '("o",dtype)'),
            ('("o",out_dtype)', '("o",unknown)'),
            ("x.to(o.dtype.element_ty)", "x"),
            ("(dtype out_dtype : FloatDType)", "(dtype out_dtype : Nat)"),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))

    def test_duplicate_polymorphic_pointer_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(self.lean.replace('("o",out_dtype)', '("s",out_dtype)'))


class FinitePastedBodyTests(unittest.TestCase):
    def setUp(self):
        self.python = '''@triton.jit
def copy(s, o, N: tl.constexpr):
    x = tl.load(s)
    tl.store(o, x.to(o.dtype.element_ty))
'''
        self.lean = '''def signature (storage : Storage) (keeping : KeepStorage) : CallSignature where
  pointers := [("s",match storage with | .fp32 => .fp32 | .fp16 => .fp16),
    ("o",match keeping with | .integer => .i32 | .boolean => .boolean)]
  scalars := []
  constexpr := ["N"]

def body (storage : Storage) (keeping : KeepStorage) (N : Nat) :=
  with_triton_signature (signature storage keeping) in triton {
    x = tl.load(s)
    tl.store(o, x.to(o.dtype.element_ty))
  }
'''
        self.profile = {
            "pointers": [["s", {"on": "storage", "cases": [["fp32", "fp32"], ["fp16", "fp16"]]}],
                         ["o", {"on": "keeping", "cases": [["integer", "i32"], ["boolean", "boolean"]]}]],
            "scalars": [], "constexpr": ["N"],
            "dtype_parameters": ["storage", "keeping"],
            "dtype_parameter_types": {"storage": "Storage", "keeping": "KeepStorage"}}

    def compare(self, lean, profile=None):
        return check(self.python, lean, "copy", "body", "signature", profile or self.profile)

    def test_exact_finite_storage_maps_are_recorded(self):
        self.assertEqual(self.compare(self.lean)["mode"], "pasted_body")

    def test_fp8_storage_tags_in_fixed_and_finite_signatures(self):
        for tag in ["f8e4", "f8e5"]:
            for fixed in [False, True]:
                with self.subTest(tag=tag, fixed=fixed):
                    profile = json.loads(json.dumps(self.profile))
                    if fixed:
                        before = "match storage with | .fp32 => .fp32 | .fp16 => .fp16"
                        after = "." + tag
                        profile["pointers"][0][1] = tag
                    else:
                        before, after = ".fp16 => .fp16", ".fp16 => ." + tag
                        profile["pointers"][0][1]["cases"][1][1] = tag
                    self.assertEqual(self.compare(self.lean.replace(before, after), profile)["mode"], "pasted_body")

    def test_changed_storage_choices_are_rejected(self):
        for before, after in [
            (".fp16 => .fp16", ".fp16 => .bf16"),
            (".boolean => .boolean", ".boolean => .i32"),
            ("signature storage keeping", "signature keeping storage"),
            (" | .fp16 => .fp16", ""),
            (".fp16 => .fp16", ".fp32 => .fp16"),
            (".fp16 => .fp16", ".fp16 => .made_up_type"),
            ("match storage with", "match unknown with"),
            ("storage : Storage", "storage : Nat"),
            ("storage : Storage", "storage : OtherStorage"),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))

    def test_explicit_source_dtype_parameter(self):
        lean = self.lean.replace("(storage : Storage) (keeping : KeepStorage)", "(dtype : SourceDType)")
        lean = lean.replace('match storage with | .fp32 => .fp32 | .fp16 => .fp16', 'dtype')
        lean = lean.replace('match keeping with | .integer => .i32 | .boolean => .boolean', 'dtype')
        lean = lean.replace('signature storage keeping', 'signature dtype')
        profile = {"pointers": [["s", "dtype"], ["o", "dtype"]], "scalars": [], "constexpr": ["N"],
                   "dtype_parameters": ["dtype"], "dtype_parameter_types": {"dtype": "SourceDType"}}
        self.assertEqual(self.compare(lean, profile)["mode"], "pasted_body")
        with self.assertRaises(ValueError):
            self.compare(lean.replace("dtype : SourceDType", "dtype : Storage", 1), profile)

    def test_arbitrary_metadata_expressions_are_rejected(self):
        original = 'match storage with | .fp32 => .fp32 | .fp16 => .fp16'
        for expression in ["storage.toDType", "chooseType storage", "[.fp32]",
                           "{'on': 'storage', 'cases': [['fp32', 'fp32']]}"]:
            with self.subTest(expression=expression), self.assertRaises((ValueError, SyntaxError)):
                self.compare(self.lean.replace(original, expression))


class GatedCumulativeBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/tritonbench_g/chunk_gated_attention"
        self.python = (self.directory / "chunk_gated_attention.py").read_text()
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.record = pinned_pairs(self.directory)[0]

    def compare(self, lean):
        return check(self.python, lean, self.record["python_function"], self.record["lean_definition"],
                     self.record["signature_definition"], self.record["expected_signature"])

    def test_complete_repaired_body_with_runtime_strides_and_dtype_queries(self):
        self.assertEqual(self.compare(self.lean)["mode"], "pasted_body")

    def test_signed_cast_layout_dot_and_store_cast_changes_are_rejected(self):
        for before, after in [
            ("tl.cast(tl.program_id(0), tl.int64)", "tl.program_id(0)"),
            ("i_t * BT * s_s_t", "i_t * BT"),
            ("tl.dot(m_s, b_s,", "tl.dot(b_s, m_s,"),
            ("b_o.to(p_o.dtype.element_ty)", "b_o"),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.compare(self.lean.replace(before, after, 1))


class GatedRecurrentBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/tritonbench_g/chunk_gated_attention"
        self.python = (self.directory / "chunk_gated_attention.py").read_text()
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.record = pinned_pairs(self.directory)[1]

    def compare(self, lean):
        return check(self.python, lean, self.record["python_function"], self.record["lean_definition"],
                     self.record["signature_definition"], self.record["expected_signature"])

    def test_complete_repaired_recurrence(self):
        result = self.compare(self.lean)
        self.assertEqual(result["mode"], "pasted_body")
        self.assertEqual(result["statements"], 10)

    def test_seed_signed_loop_endpoint_and_dtype_casts_are_checked(self):
        body = quotation(self.lean, "hKernel", raw_python=True)[1]
        for before, after in [
            ("b_h += tl.load(p_h", "b_h = tl.load(p_h"),
            ("range(tl.cast(0, tl.int64), tl.cast(NT, tl.int64), tl.cast(1, tl.int64))", "range(NT)"),
            ("i_t * BT + BT - 1", "i_t * BT + BT + 1"),
            ("b_h.to(p_h.dtype.element_ty)", "b_h"),
            (".to(b_k.dtype)", ""),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, body)
                self.compare(self.lean.replace(body, body.replace(before, after, 1), 1))

class PastedGroupedMatmulTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/triton_matmul'
        self.python = (self.directory / 'triton_matmul.py').read_text()
        self.record = pinned_pairs(self.directory)[0]
        self.lean = (ROOT / self.record['lean']).read_text()

    def compare(self, python=None, lean=None):
        return check(self.python if python is None else python,
                     self.lean if lean is None else lean,
                     self.record['python_function'], self.record['lean_definition'],
                     self.record['signature_definition'], self.record['expected_signature'])

    def test_real_jit_configuration_and_bare_jit(self):
        self.assertEqual(self.compare()['status'], 'PASS')
        self.assertEqual(self.compare(python=self.python.replace(
            '@triton.jit(launch_metadata=_matmul_launch_metadata)', '@triton.jit'))['status'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'explicitly selected'):
            self.compare(python=self.python.replace('@triton.jit(', '@triton.other('))

    def test_signed_counts_loop_clamps_and_dtype_branch_are_preserved(self):
        for before, after in [('.to(tl.int64)', ''),
                              ('(K % BLOCK_SIZE_K != 0).to(tl.int64)', '0'),
                              ('for k in range(0, blocks_k):', 'for k in range(0, 1):'),
                              ('tl.where(mask_k, base_k, 0)', 'base_k'),
                              ('c_ptr.dtype.element_ty == tl.float8e4nv', 'False'),
                              ('accumulator.to(tl.float16)', 'accumulator.to(tl.float32)')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.compare(lean=self.lean.replace(before, after, 1))


class RuntimeCallBindingTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / 'bench/tritonbench_g/add_example'
        self.python = (directory / 'add_example.py').read_text()
        self.lean = (directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(directory)[0]

    def compare(self, lean, record=None):
        record = self.record if record is None else record
        return check(self.python, lean, record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     record.get('expected_call_bindings'))

    def test_runtime_int32_stays_runtime_in_exact_body(self):
        result = self.compare(self.lean)
        self.assertEqual(result['runtime_parameters'], {'n_elements': 'i32'})
        self.assertEqual(result['constexpr_parameters'], ['BLOCK_SIZE'])
        self.assertEqual(result['mode'], 'pasted_body')

    def test_changed_missing_or_computed_call_binding_rejected(self):
        for replacement in ('n_elements := BLOCK_SIZE', '', 'n_elements := n_elements + 1',
                            'n_elements := n_elements, n_elements := n_elements'):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.compare(self.lean.replace('n_elements := n_elements', replacement, 1))

    def test_empty_runtime_bindings_for_constexpr_only_signature(self):
        record = json.loads(json.dumps(self.record))
        record['expected_signature']['scalars'] = []
        record['expected_signature']['constexpr'].insert(0, 'n_elements')
        record['expected_call_bindings'] = {}
        python = self.python.replace('n_elements: tl.int32,', 'n_elements: tl.constexpr,', 1)
        lean = self.lean.replace('scalars := [("n_elements",.i32)]', 'scalars := []').replace(
            'constexpr := ["BLOCK_SIZE"]', 'constexpr := ["n_elements","BLOCK_SIZE"]').replace(
            'call_args [n_elements := n_elements]', 'call_args []')
        result = check(python, lean, record['python_function'], record['lean_definition'],
                       record['signature_definition'], record['expected_signature'], {})
        self.assertEqual(result['runtime_parameters'], {})
        for malformed in ('[,]', '[n_elements := n_elements]', '[, ,]'):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                check(python, lean.replace('call_args []', 'call_args ' + malformed),
                      record['python_function'], record['lean_definition'],
                      record['signature_definition'], record['expected_signature'], {})

    def test_empty_bindings_cannot_hide_required_runtime_input(self):
        record = json.loads(json.dumps(self.record))
        record['expected_call_bindings'] = {}
        with self.assertRaises(ValueError):
            self.compare(self.lean.replace('call_args [n_elements := n_elements]',
                                          'call_args []'), record)

    def test_unreviewed_call_cannot_pass_as_plain_signature(self):
        record = dict(self.record)
        del record['expected_call_bindings']
        with self.assertRaises(ValueError):
            self.compare(self.lean, record)

    def test_width_change_rejected_even_if_manifest_is_altered(self):
        record = json.loads(json.dumps(self.record))
        record['expected_signature']['scalars'] = [['n_elements', 'i64']]
        with self.assertRaisesRegex(ValueError, 'runtime scalar annotation'):
            self.compare(self.lean.replace('("n_elements",.i32)', '("n_elements",.i64)', 1), record)

    def test_body_changes_remain_rejected(self):
        for before, after in [('offsets < n_elements', 'offsets <= n_elements'),
                              ('output = x + y', 'output = y + x'),
                              ('mask=mask)', 'mask=mask, other=0.0)')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))

    def test_all_staged_call_sources_preserve_python_ast(self):
        cases = ('add_example add_value cosine_compute index_select_bwd kldiv_compute '
                 'l2_norm_triton1 matrix_transpose sin_computation sin_kernel square_matrix '
                 'swiglu_fwd vector_addition vector_addition_custom').split()
        for case in cases:
            with self.subTest(case=case):
                results = check_pinned_pairs(ROOT, ROOT / 'bench/tritonbench_g' / case)
                self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))


class ReductionBodyCopyTests(unittest.TestCase):
    def compare(self, case, lean):
        directory = ROOT / 'bench/tritonbench_g' / case
        record = pinned_pairs(directory)[0]
        return check((ROOT / record['python']).read_text(), lean,
                     record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     record.get('expected_call_bindings'))

    def test_all_selected_and_candidate_reduction_bodies(self):
        for case in ('triton_mul2', 'log_softmax', 'mean_reduction', 'max_reduction'):
            with self.subTest(case=case):
                results = check_pinned_pairs(ROOT, ROOT / 'bench/tritonbench_g' / case)
                self.assertTrue(all(x['mode'] == 'pasted_body' for x in results))

    def test_logsoftmax_retains_tuple_and_explicit_index_casts(self):
        case = 'log_softmax'
        source = (ROOT / 'bench/tritonbench_g' / case / 'Kernels.lean').read_text()
        for before, after in [('.to(tl.int64)', ''),
                              ('tl.cast(M, tl.int64)', 'M'),
                              ('safe_m[:, None] * n64 * k64', 'safe_m[:, None] * k64 * n64')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(case, source.replace(before, after, 1))

    def test_mean_positional_mask_and_native_conjunction_remain(self):
        source = (ROOT / 'bench/tritonbench_g/mean_reduction/Kernels.lean').read_text()
        for before, after in [('mask = row_mask and col_mask', 'mask = row_mask or col_mask'),
                              ('tl.store(Mean, mean, row_mask)', 'tl.store(Mean, mean)'),
                              ('.to(tl.float32)', '')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare('mean_reduction', source.replace(before, after, 1))


class FloatingCallBindingTests(unittest.TestCase):
    def test_all_full_matrix_and_layernorm_bodies_are_copied(self):
        from scripts.bench.jit_correspondence import check_manifest
        for name in ('AdamUpdateGridLaunch', 'FusedLayerNorm',
                     'HyperConnectionsDepth', 'HyperConnectionsWidth'):
            with self.subTest(example=name):
                results = check_pinned_pairs(ROOT, Path('bench/examples') / name)
                self.assertTrue(results)
                self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))

    def test_runtime_fp32_binding_value_and_abi_remain_reviewed(self):
        from scripts.bench.jit_correspondence import check
        import json
        records = pinned_pairs(ROOT / 'bench/examples/HyperConnectionsDepth')
        record = records[0]
        py = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        for changed in (lean.replace('tau := tau', 'tau := 0', 1),
                        lean.replace('("tau",.fp32)', '("tau",.fp64)', 1)):
            with self.assertRaises(ValueError):
                check(py, changed, record['python_function'], record['lean_definition'],
                      record['signature_definition'], record['expected_signature'],
                      record['expected_call_bindings'])


class PointerArgumentTests(unittest.TestCase):
    def setUp(self):
        self.python = '''@triton.jit
def k(X, Y, B: tl.constexpr):
    offsets = tl.arange(0, B)
    values = tl.load(X + offsets)
    tl.store(Y + offsets, values)
'''
        self.lean = '''def sig : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]
def kernel (X Y : RegionName) (x0 y0 B : Nat) : ComputeKernel :=
  with_triton_signature sig pointer_offsets [X := x0, Y := y0] in triton {
    offsets = tl.arange(0, B)
    values = tl.load(X + offsets)
    tl.store(Y + offsets, values)
  }
'''
        self.signature = {'pointers': [['X', 'fp32'], ['Y', 'fp32']],
                          'scalars': [], 'constexpr': ['B'],
                          'readPorts': ['X'], 'writePorts': ['Y']}
        self.offsets = {'X': 'x0', 'Y': 'y0'}

    def compare(self, lean=None, offsets=None):
        return check(self.python, lean or self.lean, 'k', 'kernel', 'sig',
                     self.signature, expected_pointer_offsets=offsets or self.offsets)

    def test_reviewed_pointer_arguments_preserve_the_original_body(self):
        self.assertEqual(self.compare()['mode'], 'pasted_body')

    def test_unreviewed_offsets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'declared call signature'):
            check(self.python, self.lean, 'k', 'kernel', 'sig', self.signature)

    def test_empty_reviewed_offset_bindings(self):
        lean = self.lean.replace('pointer_offsets [X := x0, Y := y0]', 'pointer_offsets []')
        self.assertEqual(check(self.python, lean, 'k', 'kernel', 'sig', self.signature,
                               expected_pointer_offsets={})['mode'], 'pasted_body')

    def test_changed_duplicate_and_unknown_offsets_are_rejected(self):
        for changed in (self.lean.replace('X := x0', 'X := y0'),
                        self.lean.replace('Y := y0', 'X := y0'),
                        self.lean.replace('Y := y0', 'Z := y0')):
            with self.subTest(lean=changed), self.assertRaisesRegex(ValueError, 'pointer offset'):
                self.compare(lean=changed)

    def test_offsets_cannot_be_added_to_the_pasted_body(self):
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.compare(lean=self.lean.replace('tl.load(X + offsets)', 'tl.load(X + x0 + offsets)'))

    def test_scalar_bindings_and_offsets_are_both_checked(self):
        python = self.python.replace('X, Y, B:', 'X, Y, stride: tl.int64, B:')
        lean = self.lean.replace('scalars := []', 'scalars := [("stride",.i64)]').replace(
            'with_triton_signature sig pointer_offsets',
            'with_triton_call sig call_args [stride := stride_value] pointer_offsets')
        signature = {**self.signature, 'scalars': [['stride', 'i64']]}
        result = check(python, lean, 'k', 'kernel', 'sig', signature,
                       {'stride': 'stride_value'}, self.offsets)
        self.assertEqual(result['mode'], 'pasted_body')


class PointerCellArgumentTests(PointerArgumentTests):
    def setUp(self):
        super().setUp()
        self.lean = self.lean.replace('pointer_offsets [X := x0, Y := y0]',
            'pointer_cells [X := (x0, width), Y := (y0, width)]')
        self.cells = {'X': ['x0', 'width'], 'Y': ['y0', 'width']}

    def compare(self, lean=None, offsets=None):
        return check(self.python, lean or self.lean, 'k', 'kernel', 'sig',
                     self.signature, expected_pointer_cells=self.cells)

    def test_cell_width_and_base_are_both_reviewed(self):
        for before, after in [('x0, width', 'y0, width'), ('y0, width', 'y0, 1'),
                              ('Y :=', 'X :='), ('Y :=', 'Z :=')]:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'pointer cell'):
                self.compare(self.lean.replace(before, after))

    def test_cell_expressions_require_named_caller_values(self):
        with self.assertRaisesRegex(ValueError, 'pointer cell'):
            self.compare(self.lean.replace('x0, width', 'x0 + 1, width'))

    def test_offset_and_cell_metadata_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, 'pointer cell views'):
            check(self.python, self.lean, 'k', 'kernel', 'sig', self.signature,
                  expected_pointer_offsets=self.offsets, expected_pointer_cells=self.cells)

    def test_empty_reviewed_offset_bindings(self):
        # This API deliberately rejects offset metadata for a cell-view invocation.
        with self.assertRaises(ValueError):
            check(self.python, self.lean, 'k', 'kernel', 'sig', self.signature,
                  expected_pointer_offsets={})

    def test_changed_duplicate_and_unknown_offsets_are_rejected(self):
        self.test_cell_width_and_base_are_both_reviewed()

    def test_scalar_bindings_and_offsets_are_both_checked(self):
        with self.assertRaisesRegex(ValueError, 'pointer cell views'):
            check(self.python, self.lean, 'k', 'kernel', 'sig', self.signature,
                  expected_call_bindings={}, expected_pointer_cells=self.cells)


class SourceModuleBindingTests(unittest.TestCase):
    def setUp(self):
        self.python = '''from triton.language.extra.libdevice import tanh
_ZERO = tl.constexpr(0)
@triton.jit
def k(x, y, MODE: tl.constexpr):
    value = tl.load(x)
    if MODE == _ZERO:
        value = tanh(value)
    tl.store(y, value)
'''
        self.lean = '''def sig : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  scalars := []
  constexpr := ["MODE"]
  sourceConstants := [("_ZERO","0")]
  unaryImports := [("tanh","triton.language.extra.libdevice.tanh")]

def kernel (x y : RegionName) (MODE : Nat) : ComputeKernel :=
  with_triton_signature sig in triton {
    value = tl.load(x)
    if MODE == _ZERO:
        value = tanh(value)
    tl.store(y, value)
  }
'''
        self.signature = {
            'pointers': [['x','fp32'],['y','fp32']], 'scalars': [], 'constexpr': ['MODE'],
            'sourceConstants': [['_ZERO','0']],
            'unaryImports': [['tanh','triton.language.extra.libdevice.tanh']]}

    def compare(self, python=None, lean=None, signature=None):
        return check(python or self.python, lean or self.lean, 'k', 'kernel',
                     'sig', signature or self.signature)

    def test_module_values_and_imports_are_checked(self):
        self.assertEqual(self.compare()['mode'], 'pasted_body')

    def test_original_libdevice_module_alias_is_checked(self):
        python = self.python.replace('from triton.language.extra.libdevice import tanh',
                                     'from triton.language.extra import libdevice\ntanh = libdevice.tanh')
        self.assertEqual(self.compare(python=python)['mode'], 'pasted_body')
        for changed in (
            python.replace('libdevice.tanh', 'libdevice.exp'),
            python.replace('libdevice.tanh', 'libdevice.tanh(0)'),
            python.replace('tanh = libdevice.tanh', 'libdevice = other\ntanh = libdevice.tanh'),
            python.replace('tanh = libdevice.tanh', 'del libdevice\ntanh = libdevice.tanh'),
            python.replace('tanh = libdevice.tanh', 'libdevice.tanh = other\ntanh = libdevice.tanh'),
            python.replace('tanh = libdevice.tanh', 'del libdevice.tanh\ntanh = libdevice.tanh'),
            python.replace('from triton.language.extra import libdevice',
                           'from another_module import libdevice'),
            python.replace('from triton.language.extra import libdevice\ntanh = libdevice.tanh',
                           'tanh = libdevice.tanh\nfrom triton.language.extra import libdevice'),
        ):
            with self.subTest(python=changed), self.assertRaisesRegex(ValueError, 'intrinsic import'):
                self.compare(python=changed)

    def float_case(self, declared='1.44269504', actual='1.44269504'):
        python = self.python.replace('tl.constexpr(0)', 'tl.constexpr(' + actual + ')')
        lean = self.lean.replace('("_ZERO","0")', '("_ZERO","' + declared + '")')
        signature = json.loads(json.dumps(self.signature))
        signature['sourceConstants'][0][1] = declared
        return python, lean, signature

    def test_floating_module_literal_is_checked(self):
        python, lean, signature = self.float_case()
        self.assertEqual(self.compare(python, lean, signature)['mode'], 'pasted_body')

    def test_floating_module_value_and_type_changes_are_rejected(self):
        for declared, actual in [('1.44269505', '1.44269504'), ('1', '1.0'), ('0.0', '-0.0'),
                                 ('-0.0', '0.0'), ('1e400', '1e400'), ('True', '1.0')]:
            with self.subTest(declared=declared, actual=actual), self.assertRaises(ValueError):
                self.compare(*self.float_case(declared, actual))

    def test_floating_module_expression_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(*self.float_case('1.44269504', '1.0 + 0.44269504'))

    def test_closed_module_math_source_expressions_are_checked(self):
        for value in ['math.sqrt(2.0)', 'math.sqrt(2.0 / math.pi)']:
            python, lean, signature = self.float_case(value, value)
            python = 'import math\n' + python
            self.assertEqual(self.compare(python, lean, signature)['mode'], 'pasted_body')
            for changed in [python.replace(value, 'math.sqrt(3.0)'),
                            python.replace('import math', 'import other as math'),
                            python.replace('import math', 'import math\nmath = other'),
                            python.replace('import math', 'import math\nmath.sqrt = other'),
                            python.replace('import math', 'import math\ndel math.pi')]:
                with self.subTest(value=value, python=changed), self.assertRaises(ValueError):
                    self.compare(changed, lean, signature)

    def test_module_math_cannot_call_unmodeled_or_free_expressions(self):
        for value in ['math.sqrt(caller_value)', 'math.exp(2.0)', 'math.sqrt(True)',
                      'math.sqrt(1e400)', 'math.sqrt(-1.0)', 'math.sqrt(1.0 / 0.0)',
                      'math.sqrt(2.0, other=1.0)', 'math.__dict__']:
            python, lean, signature = self.float_case(value, value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare('import math\n' + python, lean, signature)

    def test_unchanged_body_cannot_hide_changed_module_constant(self):
        with self.assertRaisesRegex(ValueError, 'Python module literal'):
            self.compare(python=self.python.replace('tl.constexpr(0)', 'tl.constexpr(1)'))

    def test_unchanged_body_cannot_hide_different_tanh_intrinsic(self):
        with self.assertRaisesRegex(ValueError, 'intrinsic import'):
            self.compare(python=self.python.replace('extra.libdevice', 'math'))

    def test_duplicate_module_assignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one static definition'):
            self.compare(python=self.python + '\n_ZERO = tl.constexpr(0)\n')

    def test_module_constant_expression_is_not_a_literal(self):
        with self.assertRaisesRegex(ValueError, 'literal integer'):
            self.compare(python=self.python.replace('tl.constexpr(0)', 'tl.constexpr(1-1)'))

    def test_reviewed_manifest_does_not_override_python_import(self):
        target = 'triton.language.extra.cuda.libdevice.tanh'
        signature = json.loads(json.dumps(self.signature))
        signature['unaryImports'][0][1] = target
        with self.assertRaisesRegex(ValueError, 'intrinsic import'):
            self.compare(lean=self.lean.replace('triton.language.extra.libdevice.tanh', target),
                         signature=signature)

    def test_alias_rebinding_after_import_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no rebinding'):
            self.compare(python=self.python.replace('_ZERO =', 'tanh = tl.tanh\n_ZERO =', 1))

    def test_augmented_constant_rebinding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no rebinding'):
            self.compare(python=self.python + '\n_ZERO += 1\n')

    def test_function_cannot_replace_an_imported_intrinsic(self):
        with self.assertRaisesRegex(ValueError, 'no rebinding'):
            self.compare(python=self.python + '\ndef tanh(x):\n    return x\n')

    def test_wildcard_import_cannot_silently_replace_a_module_binding(self):
        with self.assertRaisesRegex(ValueError, 'wildcard imports'):
            self.compare(python=self.python + '\nfrom another_module import *\n')


class OptionalPointerArgumentTests(unittest.TestCase):
    def setUp(self):
        self.python = '''@triton.jit
def k(X, Y, B: tl.constexpr):
    if X is not None:
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
'''
        self.lean = '''def sig : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]
def kernel (X Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature sig optional_pointers [X := present] in triton {
    if X is not None:
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
  }
'''
        self.signature = {'pointers': [['X', 'fp32'], ['Y', 'fp32']], 'scalars': [],
                          'constexpr': ['B'], 'readPorts': ['X'], 'writePorts': ['Y']}

    def compare(self, lean=None):
        return check(self.python, lean or self.lean, 'k', 'kernel', 'sig', self.signature,
                     expected_optional_pointers={'X': 'present'})

    def test_source_none_guard_is_preserved(self):
        self.assertEqual(self.compare()['mode'], 'pasted_body')

    def test_presence_binding_must_be_reviewed(self):
        for changed in ['Y := present', 'X := absent', 'X := present, X := present',
                        'X := present, Z := present']:
            with self.subTest(binding=changed), self.assertRaisesRegex(ValueError, 'optional pointer'):
                self.compare(self.lean.replace('X := present', changed))
        with self.assertRaisesRegex(ValueError, 'declared call signature'):
            check(self.python, self.lean, 'k', 'kernel', 'sig', self.signature)

    def test_guard_cannot_be_replaced_by_a_lean_condition(self):
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.compare(self.lean.replace('X is not None', 'present'))


class StaticSignedStrideTests(unittest.TestCase):
    def setUp(self):
        self.python = """@triton.jit
def k(X, Y, N, B: tl.constexpr):
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), B, tl.int64)
    for off in range(zero, N, step):
        value = tl.load(X + off)
        tl.store(Y + off, value)
"""
        self.signature = {'pointers': [['X','fp32'],['Y','fp32']],
                          'scalars': [['N','integer']], 'constexpr': ['B'],
                          'readPorts': ['X'], 'writePorts': ['Y']}
        self.lean = """def sig : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]
def kernel (X Y : RegionName) (stop : Int) (B : Nat) : ComputeKernel :=
  with_triton_call sig call_args [N := stop] static_strides in triton {
""" + self.python.split('B: tl.constexpr):\n')[1] + '  }\n'

    def compare(self, python=None, lean=None):
        return check(python or self.python, lean or self.lean, 'k', 'kernel', 'sig',
                     self.signature, {'N':'stop'}, expected_static_strides={'step':'B'})

    def test_original_signed_constructor_and_range_are_retained(self):
        self.assertEqual(self.compare()['mode'], 'pasted_body')

    def test_static_pass_requires_reviewed_invocation(self):
        with self.assertRaisesRegex(ValueError, 'reviewed runtime call bindings'):
            check(self.python, self.lean, 'k', 'kernel', 'sig', self.signature, {'N':'stop'})

    def test_reassigned_stride_or_constexpr_is_rejected_even_if_bodies_match(self):
        for inserted in ['    step += 1\n','    B = 2\n']:
            python = self.python.replace('    for off', inserted+'    for off')
            lean = self.lean.replace('    for off', inserted+'    for off')
            with self.subTest(inserted=inserted), self.assertRaisesRegex(ValueError, 'immutable'):
                self.compare(python, lean)

    def test_runtime_and_negative_constructors_are_rejected_even_if_bodies_match(self):
        for value in ['N','-1']:
            python = self.python.replace('(), B, tl.int64','(), '+value+', tl.int64')
            lean = self.lean.replace('(), B, tl.int64','(), '+value+', tl.int64')
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'constructors'):
                self.compare(python, lean)

class PureJITFunctionCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'bench/tritonbench_g/swiglu_triton'
        self.directory.mkdir(parents=True)
        original = ROOT / 'bench/tritonbench_g/swiglu_triton'
        for name in ['Kernels.lean', 'swiglu_triton.py']:
            (self.directory / name).write_text((original / name).read_text())
        self.manifest = self.directory / 'jit-correspondence.json'
        self.manifest.write_text(json.dumps(pinned_pairs(original)))

    def test_original_helper_both_kernels_and_candidates_are_checked(self):
        results = check_manifest(self.root, self.manifest)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]['kind'], 'pure_jit_function')
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))

    def test_changed_helper_operation_order_is_rejected(self):
        p = self.directory / 'Kernels.lean'
        p.write_text(p.read_text().replace('return x * tl.sigmoid(x)', 'return tl.sigmoid(x) * x'))
        with self.assertRaisesRegex(ValueError, 'return body differs'):
            check_manifest(self.root, self.manifest)

    def test_helper_parameter_must_match(self):
        p = self.directory / 'Kernels.lean'
        p.write_text(p.read_text().replace('with_triton_params (x)', 'with_triton_params (value)'))
        with self.assertRaisesRegex(ValueError, 'parameter differs'):
            check_manifest(self.root, self.manifest)

    def test_missing_helper_body_cannot_be_certified_by_a_binding(self):
        rows = json.loads(self.manifest.read_text())
        self.manifest.write_text(json.dumps(rows[1:]))
        with self.assertRaisesRegex(ValueError, 'separately checked helper body'):
            check_manifest(self.root, self.manifest)

    def test_import_of_a_different_helper_is_rejected(self):
        p = split_swiglu_fixture(self.directory, self.manifest)
        p.write_text(p.read_text().replace('from swiglu_triton import silu', 'from another_kernel import silu'))
        with self.assertRaisesRegex(ValueError, 'import differs'):
            check_manifest(self.root, self.manifest)

    def test_global_helper_rebinding_is_rejected(self):
        p = self.directory / 'swiglu_triton.py'
        p.write_text(p.read_text() + '\nsilu = lambda x: x\n')
        with self.assertRaisesRegex(ValueError, 'one static binding'):
            check_manifest(self.root, self.manifest)

class IntegerCallArgumentCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/seeded_dropout'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.python = (self.directory / 'seeded_dropout.py').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check(self.python, lean, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'],
                     expected_integer_args=r['expected_integer_args'], expected_arange_args=r['expected_arange_args'])

    def test_original_and_candidate_bodies_and_call_profiles(self):
        result = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(p['mode'] == 'pasted_body' for p in result))

    def test_unboxed_integer_caller_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'explicit IntegerArgument'):
            self.verify(self.lean.replace('(seed : IntegerArgument)', '(seed : Int)'))

    def test_unchecked_arange_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checked constexpr-size proof'):
            self.verify(self.lean.replace('(hBlock : Random.ArangeSizeDomain BLOCK_SIZE)', '(hBlock : True)'))

    def test_wrong_bound_integer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integer arguments differ'):
            self.verify(self.lean.replace('integer_args [seed := seed]', 'integer_args [seed := other]'))

    def test_probability_comparison_or_eager_division_cannot_change(self):
        for before, after in [('random > p', 'random >= p'),
                              ('tl.where(x_keep, x / (1 - p), 0.0)',
                               '(x / (1 - p) if x_keep else 0.0)')]:
            with self.subTest(after=after), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class ApplyPenaltyBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/apply_penalty'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.python = (self.directory / 'apply_penalty.py').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check(self.python, lean, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_both_complete_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['statements'] == 17 for r in results))

    def test_signed_metadata_profile_cannot_become_natural_or_unsigned(self):
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.verify(self.lean.replace('("p_token_counts",.i32)', '("p_token_counts",.u32)'))

    def test_original_widening_cannot_be_removed(self):
        mutations = [
            ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
            ('tl.arange(0, BLOCK_P).to(tl.int64)', 'tl.arange(0, BLOCK_P)'),
            ('tl.cast(stride_logit_b, tl.int64)', 'stride_logit_b'),
            ('tl.load(p_cumsum_seq_len + cur_batch).to(tl.int64)',
             'tl.load(p_cumsum_seq_len + cur_batch)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

    def test_count_cast_mask_and_pointer_grouping_are_preserved(self):
        mutations = [
            ('mask=cur_batch_id_offset < cur_batch_end_index, other=0)',
             'mask=cur_batch_id_offset <= cur_batch_end_index, other=0)'),
            ('batch_ids_count * cur_freqency', 'batch_ids_count.to(tl.float32) * cur_freqency'),
            ('cur_offset = row_start_ptr + batch_ids',
             'cur_offset = Logits + (cur_batch * tl.cast(stride_logit_b, tl.int64) + batch_ids)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class FastRmsCopiedBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fast_rms_layernorm'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index, lean):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_all_stages_and_existing_index_candidates(self):
        self.assertEqual(len(check_pinned_pairs(ROOT, self.directory)), 7)

    def test_docstring_text_and_rebased_indentation(self):
        self.verify(0, self.lean)
        start = self.lean.index('    """', self.lean.index('def regularRuntimeKernel'))
        stop = self.lean.index('\n  }', start)
        rebased = self.lean[:start] + textwrap.indent(self.lean[start:stop], '    ') + self.lean[stop:]
        self.verify(0, rebased)
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(0, self.lean.replace('Fast RMS Layernorm kernel', 'Different documentation', 1))

    def test_chained_casts_and_weight_dtype_query_are_preserved(self):
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(3, self.lean.replace('normed.to(dW.dtype.element_ty).to(tl.float32)',
                                           'normed.to(tl.float32)', 1))
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(0, self.lean.replace('normed.to(W_row.dtype)', 'normed.to(tl.float32)', 1))

    def test_backward_gemma_branch_and_signed_loop_cannot_change(self):
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(2, self.lean.replace('if GEMMA: dY_W = dY_row * (W_row + 1.0)',
                                           'if GEMMA: dY_W = dY_row * W_row', 1))
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(3, self.lean.replace('range(tl.cast(n_rows, tl.int64))', 'range(n_rows)', 1))

    def test_other_multiline_string_constants_keep_their_contents(self):
        r = self.records[0]
        old = '    row_idx = tl.program_id(0).to(tl.int64)'
        literal = '    label = """alpha\n      beta"""\n'
        py = (ROOT / r['python']).read_text().replace(old, literal + old, 1)
        lean = self.lean.replace(old, literal + old, 1)
        def verify(text):
            return check(py, text, r['python_function'], r['lean_definition'],
                         r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])
        verify(lean)
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            verify(lean.replace('      beta"""', '    beta"""', 1))

class TokenAttnReduceVCopiedBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/token_attn_reduceV'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_original_and_existing_affine_candidate(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['statements'] == 19 for r in results))

    def test_metadata_signed_storage_is_part_of_call_signature(self):
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.verify(self.lean.replace('("Req_to_tokens", .i32)', '("Req_to_tokens", .u32)', 1))

    def test_loop_hint_mask_and_output_cast_cannot_be_removed(self):
        mutations = [
            ('range(0, cur_batch_seq_len, BLOCK_N)', 'range(cur_batch_seq_len)'),
            ('start_n = tl.multiple_of(start_n, BLOCK_N)', 'start_n = start_n'),
            ('mask=(start_n + offs_n) < cur_batch_seq_len', 'mask=(start_n + offs_n) <= cur_batch_seq_len'),
            ('acc = acc.to(Out.dtype.element_ty)', 'acc = acc'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class KvCacheCopyBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/kv_cache_copy'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_both_complete_copy_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['statements'] == 11 for r in results))

    def test_static_count_and_original_nonzero_arange_cannot_change(self):
        mutations = [
            ('tl.static_range(HEAD_DIM // KCACHE_X)', 'range(HEAD_DIM // KCACHE_X)'),
            ('tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)',
             'split_x * KCACHE_X + tl.arange(0, KCACHE_X)'),
            ('tl.store(VCache + offsets_vcache, v)', 'tl.store(VCache + offsets_vcache, k)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class TokenAttnMistralBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/token_attn_mistral'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_both_complete_sliding_window_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['statements'] == 20 for r in results))

    def test_distinct_value_mask_must_not_be_replaced_by_probability_mask(self):
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(self.lean.replace(
                'mask=(start_n + offs_n + cur_batch_start_index) < cur_batch_seq_len',
                'mask=(start_n + offs_n) < cur_att_seq_len', 1))

    def test_sliding_origin_hint_and_output_cast_are_preserved(self):
        mutations = [
            ('tl.maximum(cur_batch_seq_len - sliding_window, 0)', 'cur_batch_seq_len - sliding_window'),
            ('start_n = tl.multiple_of(start_n, BLOCK_N)', 'start_n = start_n'),
            ('acc = acc.to(Out.dtype.element_ty)', 'acc = acc'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class SoftmaxReduceVBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/softmax_reducev'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_both_complete_original_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['statements'] == 18 for r in results))

    def test_signed_origin_masks_and_absence_of_extra_casts(self):
        mutations = [
            ('(max_input_len - cur_batch_seq_len)', 'tl.maximum(max_input_len - cur_batch_seq_len, 0)'),
            ('mask=start_n + offs_n < cur_batch_seq_len', 'mask=start_n + offs_n <= cur_batch_seq_len'),
            ('acc = acc / e_sum', 'acc = (acc / e_sum).to(Out.dtype.element_ty)'),
            ('e_max = n_e_max', 'e_max = n_e_max.to(tl.float32)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class TokenAttnLlama2BodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/token_attn_llama2'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, lean):
        r = self.record
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_both_complete_score_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['mode'] == 'pasted_body' and r['statements'] == 15 for r in results))

    def test_zero_one_loop_and_absence_of_extra_casts(self):
        mutations = [
            ('tl.where(block_stard_index < cur_batch_seq_len, 1, 0)', 'tl.where(block_stard_index <= cur_batch_seq_len, 1, 0)'),
            ('for start_mark in range(0, block_mask, 1):', 'for start_mark in range(1):'),
            ('q = tl.load(Q + off_q + start_mark)', 'q = tl.load(Q + off_q + start_mark).to(tl.float32)'),
            ('tl.store(Att_Out + off_o, att_value,', 'tl.store(Att_Out + off_o, att_value.to(Att_Out.dtype.element_ty),'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

class KsoftmaxBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/ksoftmax_triton'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, lean, backward=False):
        r = self.records[1 if backward else 0]
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'], r['expected_call_bindings'])

    def test_both_complete_forward_backward_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in results], [17, 15])
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))

    def test_widening_optional_mask_and_absence_of_output_cast(self):
        mutations = [
            ('m = tl.program_id(0).to(tl.int64)', 'm = tl.program_id(0)'),
            ('if MASK_TYPE is not None:', 'if True:'),
            ('tl.store(y_ptrs, y, mask=k < K)', 'tl.store(y_ptrs, y.to(tl.float32), mask=k < K)'),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, 'statement correspondence'):
                self.verify(self.lean.replace(before, after, 1))

    def test_backward_fp16_branch_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, 'statement correspondence'):
            self.verify(self.lean.replace('        if IS_FP16:\n            o = o.to(tl.float32)\n', '', 1), True)

class BinaryPureJITFunctionCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / 'bench/tritonbench_g/softmax_optimize'
        self.directory.mkdir(parents=True)
        original = ROOT / 'bench/tritonbench_g/softmax_optimize'
        for name in ['Kernels.lean', 'softmax_optimize.py']:
            (self.directory / name).write_text((original / name).read_text())
        self.manifest = self.directory / 'jit-correspondence.json'
        self.manifest.write_text(json.dumps(pinned_pairs(original)))

    def test_two_original_helpers_and_two_complete_callers(self):
        rows = check_manifest(self.root, self.manifest)
        self.assertEqual([r['statements'] for r in rows], [1, 12, 1, 12])
        self.assertEqual(rows[0]['kind'], 'binary_pure_jit_function')

    def test_helper_operand_order_and_parameter_order_are_checked(self):
        p = self.directory / 'Kernels.lean'
        source = p.read_text()
        for old, new in [('tl.cdiv(a, b) * b - b', 'b * tl.cdiv(a, b) - b'),
                         ('with_triton_params (a, b)', 'with_triton_params (b, a)')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                p.write_text(source.replace(old, new, 1))
                check_manifest(self.root, self.manifest)

    def test_helper_records_are_required_for_each_independent_python_file(self):
        rows = json.loads(self.manifest.read_text())
        for index in [0, 2]:
            self.manifest.write_text(json.dumps([r for i, r in enumerate(rows) if i != index]))
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'separately checked helper body'):
                check_manifest(self.root, self.manifest)

    def test_call_binding_casts_shapes_and_guard_cannot_be_rewritten(self):
        p = self.directory / 'Kernels.lean'
        source = p.read_text()
        changes = [('prev_multiple_of := previousMultipleSource', 'prev_multiple_of := commutedPreviousMultipleSource'),
                   ('prev_multiple_of(N, TILE_N)', 'tl.cdiv(N, TILE_N) * TILE_N - TILE_N'),
                   ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
                   ('tl.full((TILE_N,),', 'tl.full((TILE_N),'),
                   ('new_m == -float("inf")', 'inp == -float("inf")')]
        for old, new in changes:
            with self.subTest(new=new), self.assertRaises(ValueError):
                p.write_text(source.replace(old, new, 1))
                check_manifest(self.root, self.manifest)

class RotaryEmbNopadBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/rotary_emb_nopad'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def compare(self, record, lean):
        r = record
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'], r['signature_definition'],
                     r['expected_signature'], r['expected_call_bindings'])

    def test_all_four_complete_sources(self):
        self.assertEqual([r['statements'] for r in check_pinned_pairs(
            ROOT, self.directory)], [19, 19, 32, 32])

    def test_early_return_casts_metadata_and_store_order_are_preserved(self):
        cached = self.records[2]
        mutations = [('if block_head_index >= Q_HEAD_NUM:\n        return',
                      'if block_head_index > Q_HEAD_NUM:\n        return'),
                     ('block_head_index = tl.program_id(0).to(tl.int64)', 'block_head_index = tl.program_id(0)'),
                     ('past_kv_seq_len = tl.load(context_lengths + block_token_index) - 1',
                      'past_kv_seq_len = tl.load(context_lengths + block_token_index)'),
                     ('block_ids * cacheb_stride', 'block_ids * cacheh_stride'),
                     ('out_k0, mask=(block_head_index < K_HEAD_NUM)',
                      'out_k0.to(tl.float32), mask=(block_head_index < K_HEAD_NUM)')]
        for old, new in mutations:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.assertIn(old, self.lean)
                self.compare(cached, self.lean.replace(old, new, 1))

    def test_new_python_candidates_change_only_existing_q_address_reuse(self):
        original = ast.parse(variant_source(self.directory / 'rotary_emb_nopad.py'))
        candidate = ast.parse(variant_source(self.directory / 'rotary_emb_nopad_shared.py'))
        originals = [n for n in original.body if isinstance(n, ast.FunctionDef) and
                     any('jit' in ast.unparse(d) for d in n.decorator_list)]
        candidates = [n for n in candidate.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(len(originals), len(candidates))
        for source, changed in zip(originals, candidates):
            self.assertEqual(ast.dump(source.args), ast.dump(changed.args))
            self.assertEqual(len(source.body), len(changed.body))
            differences = [(a, b) for a, b in zip(source.body, changed.body) if ast.dump(a) != ast.dump(b)]
            self.assertEqual(len(differences), 1)
            before, after = differences[0]
            self.assertEqual([t.id for t in before.targets], ['off_q1'])
            self.assertEqual(ast.unparse(after), 'off_q1 = off_q0 + HEAD_DIM // 2 * head_dim_stride')

class FusedRotaryBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fused_rotary_embedding'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, lean):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), lean,
                     r['python_function'], r['lean_definition'], r['signature_definition'],
                     r['expected_signature'], r['expected_call_bindings'])

    def test_complete_q_k_v_sources(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [21, 21])

    def test_metadata_widenings_cache_pitches_and_grouped_v_branch_are_checked(self):
        mutations = [('tl.load(context_lengths + cur_token_idx * length_stride).to(tl.int64)',
                      'tl.load(context_lengths + cur_token_idx * length_stride)'),
                     ('dim_range0 // x', 'dim_range0 % x'),
                     ('dim_range1 % x', 'dim_range1 // x'),
                     ('handle_kv = cur_head_idx % KV_GROUP_NUM == 0',
                      'handle_kv = cur_head_idx % KV_GROUP_NUM != 0'),
                     ('v_cache + v_range, loaded_v', 'v_cache + v_range, loaded_v.to(tl.float32)')]
        for old, new in mutations:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.assertIn(old, self.lean)
                self.verify(self.lean.replace(old, new, 1))

    def test_derived_python_candidate_changes_only_q_address_reuse(self):
        def selected(name):
            return next(n for n in ast.parse(variant_source(self.directory / name)).body
                        if isinstance(n, ast.FunctionDef) and
                        n.name.startswith('decoding_fused_rotary_embedding_kernel'))
        original = selected('fused_rotary_embedding.py')
        candidate = selected('fused_rotary_embedding_shared.py')
        self.assertEqual(ast.dump(original.args), ast.dump(candidate.args))
        self.assertEqual(len(original.body), len(candidate.body))
        changes = [(a, b) for a, b in zip(original.body, candidate.body) if ast.dump(a) != ast.dump(b)]
        self.assertEqual(len(changes), 1)
        self.assertEqual([t.id for t in changes[0][0].targets], ['off_q1'])
        self.assertEqual(ast.unparse(changes[0][1]), 'off_q1 = off_q0 + HEAD_DIM // 2 * q_dim_stride')


class LayerNormLigerBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/layer_norm_liger'
        self.records = pinned_pairs(self.directory)

    def compare(self, record, lean):
        return check((ROOT / record['python']).read_text(), lean,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], record['expected_call_bindings'])

    def test_all_four_complete_sources(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [23, 22, 25, 24])

    def test_backward_signed_range_casts_and_scalar_divisions_are_preserved(self):
        r = self.records[3]
        source = (ROOT / r['lean']).read_text()
        changes = [('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
                   ('tl.full((), 1, tl.int64)', 'tl.full((), 1, tl.int32)'),
                   ('for _ in range(row_start, row_end, one):', 'for row in range(row_start, row_end, one):'),
                   (' / n_cols', ' / n_cols.to(tl.float32)'),
                   ('dx.to(dtype)', 'dx'),
                   ('tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)',
                    'tl.store(DB_ptr + row_block_id * stride_db + cols, db_row, mask=mask)')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, source)
                with self.assertRaises(ValueError):
                    self.compare(r, source.replace(old, new, 1))

    def test_forward_implicit_conversion_and_raw_storage_are_preserved(self):
        r = self.records[1]
        source = (ROOT / r['lean']).read_text()
        for old, new in [('col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)',
                          'col_offsets = tl.arange(0, BLOCK_SIZE)'),
                         ('mean = tl.sum(X_row, axis=0) / n_cols',
                          'mean = tl.sum(X_row, axis=0) / n_cols.to(tl.float32)'),
                         ('tl.store(Mean_ptr, mean)', 'tl.store(Mean_ptr, mean.to(tl.float32))')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.assertIn(old, source)
                self.compare(r, source.replace(old, new, 1))


class TritonArgmaxBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/triton_argmax'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'])

    def test_three_complete_original_kernels(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [20, 9, 12, 20, 12, 9])

    def test_nan_selection_original_address_reload_and_clamp_are_preserved(self):
        changes = [('pid = tl.program_id(0).to(tl.int64) + program_base',
                    'pid = tl.program_id(0) + program_base'),
                   ('max_val = tl.load(inp + max_index)', 'max_val = max_logits'),
                   ('width - 1 if signed else width', 'width if signed else width'),
                   ('(1 << (exponent - 24)) if exponent > 24 else 1',
                    '(1 << (exponent - 23)) if exponent > 23 else 1'),
                   ('stored_value = tl.where(max_val == max_val, max_val, 0.0)',
                    'stored_value = max_val')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_dimension_tie_order_and_metadata_types_are_checked(self):
        changes = [('return_indices_tie_break_left=True', 'return_indices_tie_break_left=False'),
                   ('local_max = tl.load(inp + selected_offset, mask=selected_mask, other=-float("inf"))',
                    'local_max = tl.load(inp + selected_offset, other=-float("inf"))'),
                   ('("K",.i64)', '("K",.integer)')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[2], self.source.replace(old, new, 1))


    def test_existing_candidates_only_replace_the_selected_integer_product(self):
        def functions(name):
            return {n.name: n for n in ast.parse(variant_source(self.directory / name)).body
                    if isinstance(n, ast.FunctionDef)}
        original = functions('triton_argmax.py')
        shifted = functions('triton_argmax_shifted.py')
        for before_name, after_name, target, block, value, shift in [
                ('argmax_kernel_1_repaired', 'argmax_kernel_1_shifted', 'offset', 'BLOCK_SIZE', 1024, 10),
                ('argmax_kernel_repaired', 'argmax_kernel_shifted', 'm_offset', 'BLOCK_M', 128, 7)]:
            before, after = original[before_name], shifted[after_name]
            self.assertEqual(ast.dump(before.args), ast.dump(after.args))
            self.assertEqual(len(before.body), len(after.body))
            changes = [(a, b) for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
            self.assertEqual(len(changes), 1)
            assignment, branch = changes[0]
            self.assertEqual([t.id for t in assignment.targets], [target])
            self.assertIsInstance(branch, ast.If)
            self.assertEqual(ast.unparse(branch.test), f'{block} == {value}')
            self.assertEqual(ast.dump(branch.orelse[0]), ast.dump(assignment))
            self.assertIn(f'<< {shift}', ast.unparse(branch.body[0]))
        self.assertEqual([ast.dump(n) for n in original['argmax_kernel_2_repaired'].body],
                         [ast.dump(n) for n in shifted['argmax_kernel_2_repaired'].body])


class CrossEntropy1BodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/cross_entropy1'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], expected_pointer_offsets=record['expected_pointer_offsets'])

    def test_complete_forward_backward_and_both_existing_candidates(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [12, 15, 12, 15])

    def test_caller_offsets_ignored_branch_and_original_scalar_division_are_preserved(self):
        changes = [('logits_ptr := p.logitsBase', 'logits_ptr := p.lseBase'),
                   ('logits_row_stride.to(tl.int64)', 'logits_row_stride'),
                   ('smoothing * sum_logits / total_classes', 'smoothing * sum_logits / n_cols'),
                   ('if label_idx == ignored_index:', 'if label_idx != ignored_index:'),
                   ('lse_square_scale * lse * lse', 'lse_square_scale * (lse * lse)')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_backward_safe_shift_and_final_zero_selection_are_preserved(self):
        changes = [('tl.where(is_ignored, class_start_idx, label_idx)', 'label_idx'),
                   ('tl.where(is_ignored, 0.0, dloss * probs)', 'dloss * probs'),
                   ('dlogits_ptr := p.gradientBase', 'dlogits_ptr := p.logitsBase')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[1], self.source.replace(old, new, 1))

    def test_derived_candidates_change_only_the_column_index_product(self):
        def selected(name):
            return [n for n in ast.parse(variant_source(self.directory / name)).body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith('cross_entropy_')]
        original = selected('cross_entropy1.py')[:2]
        candidates = selected('cross_entropy1_index_commuted.py')
        self.assertEqual(len(original), len(candidates))
        for before, after in zip(original, candidates):
            self.assertEqual(ast.dump(before.args), ast.dump(after.args))
            self.assertEqual(len(before.body), len(after.body))
            differences = [(a, b) for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
            self.assertEqual(len(differences), 1)
            assignment, replacement = differences[0]
            self.assertEqual([t.id for t in assignment.targets], ['col_offsets'])
            self.assertEqual(ast.unparse(replacement),
                             'col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)')


class CrossEntropy2BodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/cross_entropy2'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], expected_pointer_offsets=record['expected_pointer_offsets'])

    def test_all_four_complete_bodies_and_signed_label_width(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [15, 15, 17, 17])
        with self.assertRaises(ValueError):
            self.verify(self.records[0], self.source.replace('("labels_ptr",.i32)', '("labels_ptr",.i64)', 1))

    def test_class_ownership_padding_casts_and_z_loss_address_are_preserved(self):
        changes = [('label_idx < class_start_idx + n_cols', 'label_idx <= class_start_idx + n_cols'),
                   ('logits_row_stride.to(tl.int64)', 'logits_row_stride'),
                   ('other=0.0', 'other=-float("inf")'),
                   ('z_loss_ptr := p.zLossBase', 'z_loss_ptr := p.lossBase'),
                   ('lse_square_scale * lse * lse', 'lse_square_scale * (lse * lse)')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_backward_owner_selection_and_multiplication_order_are_preserved(self):
        changes = [('tl.where(owns_target, label_idx, class_start_idx)', 'label_idx'),
                   ('(dloss * logit_scale) * probs', 'dloss * (logit_scale * probs)'),
                   ('tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs)', '(dloss * logit_scale) * probs'),
                   ('dlogits_ptr := p.gradientBase', 'dlogits_ptr := p.logitsBase')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                # Only mutate the backward quotation, after the two forward bodies.
                prefix, backward = self.source.split('def backwardKernel', 1)
                with self.assertRaises(ValueError):
                    self.verify(self.records[2], prefix + 'def backwardKernel' + backward.replace(old, new, 1))

    def test_candidates_change_only_the_existing_integer_product(self):
        def functions(name):
            return [n for n in ast.parse(variant_source(self.directory / name)).body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith('cross_entropy_')][:2]
        original = functions('cross_entropy2.py')
        candidates = functions('cross_entropy2_index_commuted.py')
        self.assertEqual(len(original), len(candidates))
        for before, after in zip(original, candidates):
            self.assertEqual(ast.dump(before.args), ast.dump(after.args))
            self.assertEqual(len(before.body), len(after.body))
            differences = [(a, b) for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
            self.assertEqual(len(differences), 1)
            assignment, replacement = differences[0]
            self.assertEqual([t.id for t in assignment.targets], ['col_offsets'])
            self.assertEqual(ast.unparse(replacement),
                             'col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)')


class CrossEntropyOpsBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/cross_entropy_ops'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], expected_pointer_offsets=record['expected_pointer_offsets'])

    def test_both_resident_functions_and_existing_candidates(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [14, 14, 16, 16])
        with self.assertRaises(ValueError):
            self.verify(self.records[0], self.source.replace('("ignored_index",.integer)', '("ignored_index",.i64)', 1))

    def test_forward_conditional_z_store_padding_and_source_order(self):
        changes = [('if not SPLIT:\n        tl.store(z_loss_ptr', 'if SPLIT:\n        tl.store(z_loss_ptr'),
                   ('other=-float("inf")', 'other=0.0'),
                   ('z_loss_ptr := p.zlossBase', 'z_loss_ptr := p.lossBase'),
                   ('lse_square_scale * lse * lse', 'lse_square_scale * (lse * lse)')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_backward_unused_assignment_and_ignored_zero_are_retained(self):
        changes = [('        smooth_positive = 1.0 - smoothing\n', ''),
                   ('tl.where(is_ignored, class_start_idx, label_idx)', 'label_idx'),
                   ('(dloss * logit_scale) * probs', 'dloss * (logit_scale * probs)'),
                   ('tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs)', '(dloss * logit_scale) * probs')]
        for old, new in changes:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                prefix, backward = self.source.split('def backwardKernel', 1)
                with self.assertRaises(ValueError):
                    self.verify(self.records[2], prefix + 'def backwardKernel' + backward.replace(old, new, 1))

    def test_candidates_change_only_the_selected_integer_product(self):
        tree = ast.parse((self.directory / 'cross_entropy_ops.py').read_text())
        definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        original = [definitions[name] for name in ('cross_entropy_fwd_kernel', 'cross_entropy_bwd_kernel')]
        candidates = [definitions[name] for name in
                      ('cross_entropy_fwd_kernel_index_commuted', 'cross_entropy_bwd_kernel_index_commuted')]
        self.assertEqual(len(original), len(candidates))
        for before, after in zip(original, candidates):
            self.assertEqual(ast.dump(before.args), ast.dump(after.args))
            self.assertEqual(len(before.body), len(after.body))
            differences = [(a, b) for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
            self.assertEqual(len(differences), 1)
            assignment, replacement = differences[0]
            self.assertEqual([t.id for t in assignment.targets], ['col_offsets'])
            self.assertEqual(ast.unparse(replacement),
                             'col_offsets = BLOCK_SIZE * col_block_idx + tl.arange(0, BLOCK_SIZE).to(tl.int64)')


class FastLayernormBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fast_layernorm'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], expected_pointer_offsets=record['expected_pointer_offsets'])

    def test_six_complete_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [19, 19, 20, 20, 12, 12])

    def test_forward_casts_constructor_and_store_statements_are_exact(self):
        for old, new in [('n_cols = tl.cast(n_cols, tl.int64)', 'n_cols = n_cols'),
                         ('tl.cast(X_row_stride, tl.int64)', 'X_row_stride'),
                         ('tl.full((BLOCK_SIZE,), 0, tl.float32)', 'tl.zeros((BLOCK_SIZE,), tl.float32)'),
                         ('tl.store(r, inv_var)', 'tl.store(r, inv_var.to(tl.float32))')]:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_backward_padding_and_operation_order_are_retained(self):
        prefix, body = self.source.split('def backwardBodyPastedKernel', 1)
        for old, new in [('tl.where(mask, dY_W * normed, 0.0)', 'dY_W * normed'),
                         ('normed * tl.sum(product, axis=0) / n_cols', '(normed / n_cols) * tl.sum(product, axis=0)'),
                         ('tl.cast(dY_row_stride, tl.int64)', 'dY_row_stride')]:
            with self.subTest(new=new):
                self.assertIn(old, body)
                with self.assertRaises(ValueError):
                    self.verify(self.records[2], prefix + 'def backwardBodyPastedKernel' + body.replace(old, new, 1))

    def test_affine_original_while_counter_and_output_masks_are_retained(self):
        prefix, body = self.source.split('def affineBodyPastedKernel', 1)
        for old, new in [('while row < tl.cast(n_rows, tl.int64):', 'for row in range(n_rows):'),
                         ('row += tl.full((), 1, tl.int64)', 'row += 1'),
                         ('row = tl.full((), 0, tl.int64)', 'row = 0'),
                         ('mask=columns < weight_count', 'mask=columns < n_cols'),
                         ('tl.where(active, columns, 0)', 'columns')]:
            with self.subTest(new=new):
                self.assertIn(old, body)
                with self.assertRaises(ValueError):
                    self.verify(self.records[4], prefix + 'def affineBodyPastedKernel' + body.replace(old, new, 1))

    def test_derived_candidates_change_only_their_existing_integer_address(self):
        def selected(name):
            return [n for n in ast.parse(variant_source(self.directory / name)).body
                    if isinstance(n, ast.FunctionDef) and n.name.startswith('layernorm_')]
        original = selected('fast_layernorm.py')
        candidates = selected('fast_layernorm_candidates.py')
        self.assertEqual(len(original), len(candidates))
        for before, after in zip(original, candidates):
            self.assertEqual(ast.dump(before.args), ast.dump(after.args))
            self.assertEqual(len(before.body), len(after.body))
            if before.name == 'layernorm_affine_backward':
                left_loop = next(n for n in before.body if isinstance(n, ast.While))
                right_loop = next(n for n in after.body if isinstance(n, ast.While))
                self.assertEqual(ast.dump(left_loop.test), ast.dump(right_loop.test))
                self.assertEqual(len(left_loop.body), len(right_loop.body))
                changes = [(a, b) for a, b in zip(left_loop.body, right_loop.body) if ast.dump(a) != ast.dump(b)]
                self.assertEqual(len(changes), 1)
                self.assertEqual(ast.unparse(changes[0][1]), 'offsets = row * tl.cast(n_cols, tl.int64) + columns')
                left = [n for n in before.body if n is not left_loop]
                right = [n for n in after.body if n is not right_loop]
                self.assertEqual([ast.dump(n) for n in left], [ast.dump(n) for n in right])
            else:
                changes = [(a, b) for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
                self.assertEqual(len(changes), 1)
                self.assertEqual(ast.unparse(changes[0][1]), 'X += tl.cast(X_row_stride, tl.int64) * row_idx')


class MultinomialBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/multinomial_sampling'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], expected_call_bindings=record['expected_call_bindings'],
                     expected_integer_args=record['expected_integer_args'],
                     expected_arange_args=record['expected_arange_args'])

    def test_complete_original_and_static_unroll_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [12, 12])

    def test_seed_offset_and_score_conversions_are_retained(self):
        for old, new in [('tl.load(Offsets + off, mask=off_mask).to(tl.int32)',
                          'tl.load(Offsets + off, mask=off_mask)'),
                         ('other=0.0).to(tl.float32)', 'other=0.0)')]:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_dynamic_token_loop_is_not_replaced_by_static_loop(self):
        old = 'for b_idx in range(0, num_tokens, BLOCK_N):'
        self.assertIn(old, self.source)
        with self.assertRaises(ValueError):
            self.verify(self.records[0], self.source.replace(old,
                        'for b_idx in tl.static_range(tl.cdiv(num_tokens, BLOCK_N)):', 1))

    def test_label_clamp_and_combined_load_mask_are_retained(self):
        for old, new in [('tl.minimum(valid_pos, num_tokens - 1)', 'valid_pos'),
                         ('mask=found_mask & off_mask', 'mask=found_mask')]:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_static_candidate_retains_its_counter_remapping(self):
        before, after = self.source.split('def unrolledKernel', 1)
        old = 'b_idx = b_idx * BLOCK_N'
        self.assertIn(old, after)
        with self.assertRaises(ValueError):
            self.verify(self.records[1], before + 'def unrolledKernel' + after.replace(old, 'b_idx = b_idx', 1))


class KldivBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/kldiv_ops'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, record, source):
        return check((ROOT / record['python']).read_text(), source,
                     record['python_function'], record['lean_definition'], record['signature_definition'],
                     record['expected_signature'], expected_call_bindings=record['expected_call_bindings'],
                     expected_pointer_offsets=record['expected_pointer_offsets'])

    def test_complete_original_and_candidate_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [12, 12, 9, 9])

    def test_forward_int64_casts_are_not_erased(self):
        for name in ['n_cols', 'y_stride', 'gt_stride', 'loss_stride']:
            old = f'{name} = tl.cast({name}, tl.int64)'
            with self.subTest(name=name):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, f'{name} = {name}', 1))

    def test_epsilon_has_no_added_source_cast(self):
        old = 'tl.maximum(y_true, eps)'
        self.assertIn(old, self.source)
        with self.assertRaises(ValueError):
            self.verify(self.records[0], self.source.replace(old, 'tl.maximum(y_true, eps.to(tl.float32))', 1))

    def test_forward_padding_and_reduction_branch_are_exact(self):
        for old, new in [('loss = tl.where(mask, loss, 0.0)', 'loss = loss'),
                         ('if reduction != 0:', 'if reduction == 0:'),
                         ('loss_ptr += pid * loss_stride', 'loss_ptr += loss_stride * pid')]:
            with self.subTest(new=new):
                self.assertIn(old, self.source)
                with self.assertRaises(ValueError):
                    self.verify(self.records[0], self.source.replace(old, new, 1))

    def test_backward_retains_initial_unused_offsets_and_masks(self):
        before, after = self.source.split('def backwardKernel', 1)
        for old in ['    offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)\n',
                    '    mask = offsets < n_cols\n']:
            with self.subTest(statement=old):
                self.assertIn(old, after)
                with self.assertRaises(ValueError):
                    self.verify(self.records[2], before + 'def backwardKernel' + after.replace(old, '', 1))


class ImportedFiniteCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/isfinite_kernel'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.python = (self.directory / 'isfinite_kernel.py').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, source):
        r = self.records[1]
        return check(self.python, source, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_pointer_offsets=r['expected_pointer_offsets'],
                     expected_pure_functions=r['expected_pure_functions'])

    def helper(self, python=None, source=None):
        from scripts.bench.jit_correspondence import check_pure_function
        r = self.records[0]
        return check_pure_function(python or self.python, source or self.source,
                                   r['python_function'], r['lean_definition'],
                                   expected_imports=r['expected_imports'])

    def test_helper_and_both_complete_callers(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [1, 7, 7])

    def test_source_stride_and_pointer_element_casts_cannot_disappear(self):
        for old, new in [('indices = offsets.to(tl.int64)', 'indices = offsets'),
                         ('tl.cast(in0_stride, tl.int64)', 'in0_stride'),
                         ('tl.cast(out0_stride, tl.int64)', 'out0_stride'),
                         ('out0.to(out0_ptr.type.element_ty)', 'out0')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_signed_base_and_helper_bindings_are_checked(self):
        for old, new in [('in0_ptr := inputOffset', 'in0_ptr := outputOffset'),
                         ('signed_pointer_offsets', 'pointer_offsets'),
                         ('isfinite_func := isfiniteSource', 'isfinite_func := unreviewed')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_helper_dtype_dispatch_and_explicit_narrow_cast_are_exact(self):
        for old, new in [('x.dtype.is_fp64()', 'x.dtype.is_fp32()'),
                         ('_finitef(x.to(tl.float32))', '_finitef(x)'),
                         ('return _isfinited(x)', 'return _finitef(x)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.helper(source=self.source.replace(old, new, 1))

    def test_every_import_fallback_is_verified(self):
        for old, new in [('from triton.language.math import isfinited as _isfinited',
                          'from triton.language.math import finitef as _isfinited'),
                         ('from triton.language.libdevice import finitef as _finitef',
                          'from triton.language.libdevice import isfinited as _finitef'),
                         ('except ImportError:', 'except Exception:')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.helper(python=self.python.replace(old, new, 1))

    def test_import_rebinding_and_wrong_declared_target_are_rejected(self):
        with self.assertRaises(ValueError):
            self.helper(python=self.python + '\n_isfinited = _finitef\n')
        with self.assertRaises(ValueError):
            self.helper(source=self.source.replace('extra.cuda.libdevice.isfinited"',
                                                   'extra.cuda.libdevice.finitef"', 1))


class BmmOptimizedBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/bmm_optimized'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, source):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), source, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_complete_original_and_existing_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [20, 20])

    def test_original_tuple_assignment_and_counter_spelling_are_retained(self):
        for old, new in [('pid_m, pid_n = pidx, pidy', 'pid_m = pidx\n        pid_n = pidy'),
                         ('for _ in range(num_iters):', 'for _i in range(num_iters):')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_none_and_all_mask_branches_are_retained(self):
        for old, new in [('mask_a = None', 'mask_a = True'),
                         ('if not DIVISIBLE_M:', 'if DIVISIBLE_M:'),
                         ('elif DIVISIBLE_M and not DIVISIBLE_N:', 'elif DIVISIBLE_M:')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_positional_load_masks_and_padding_are_retained(self):
        for old, new in [('tl.load(a_ptrs, mask_a, other=0.0)', 'tl.load(a_ptrs)'),
                         ('tl.load(b_ptrs, mask_b, other=0.0)', 'tl.load(b_ptrs, mask_b)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_float_profile_and_pointer_advance_order_are_retained(self):
        for old, new in [('allow_tf32=False', 'allow_tf32=True'),
                         ('offs_k += TILE_K', 'offs_k += TILE_K * 2'),
                         ('b_ptrs += TILE_K * N', 'b_ptrs += TILE_K')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_original_group_remainder_is_not_replaced_by_candidate(self):
        old = 'inner_group_id = pid % num_CTA_per_group'
        with self.assertRaises(ValueError):
            self.verify(self.source.replace(old, 'inner_group_id = pid - (pid // num_CTA_per_group) * num_CTA_per_group', 1))


class BmmChunkBwdBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/bmm_chunk_bwd'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, source):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), source, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_original_and_existing_index_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [45, 45])

    def test_all_twenty_one_argument_casts_are_required(self):
        for name in self.records[0]['expected_call_bindings']:
            old = f'{name} = tl.cast({name}, tl.int64)'
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, f'{name} = {name}', 1))

    def test_explicit_dot_and_output_casts_are_required(self):
        for old, new in [('.to(dot_dtype)', '.to(tl.float32)'),
                         ('db = acc.to(db_ptr.dtype.element_ty)', 'db = acc')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_signed_limit_and_all_tail_masks_are_retained(self):
        for old, new in [('min(chunk_size, seqlen - pid_c * chunk_size)', 'chunk_size'),
                         ('offs_cs[None, :] < chunk_size_limit - cs * BLOCK_SIZE_CS',
                          'offs_cs[None, :] < chunk_size_limit'),
                         ('offs_m[:, None] < chunk_size_limit', 'offs_m[:, None] < chunk_size')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_residual_addition_order_and_pointer_advance_are_retained(self):
        for old, new in [('acc += res', 'acc = res + acc'),
                         ('dout_ptrs += BLOCK_SIZE_CS * stride_dout_csize_m',
                          'dout_ptrs += stride_dout_csize_m')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_original_index_and_caller_storage_profile_are_checked(self):
        for old, new in [('pid_h = pid_ch - pid_c * ngroups', 'pid_h = pid_ch - ngroups * pid_c'),
                         ('("res_ptr",dtype)', '("res_ptr",out_dtype)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))


class LayerNormWelfoldBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/layer_norm_welfold'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.record = pinned_pairs(self.directory)[0]
        self.python = (ROOT / self.record['python']).read_text()

    def verify(self, source=None, python=None):
        r = self.record
        return check(self.python if python is None else python,
                     self.source if source is None else source,
                     r['python_function'], r['lean_definition'], r['signature_definition'],
                     r['expected_signature'], expected_call_bindings=r['expected_call_bindings'],
                     expected_ordered_sum_trees=r['expected_ordered_sum_trees'],
                     expected_namespace_imports=r['expected_namespace_imports'],
                     expected_constexpr_bindings=r['expected_constexpr_bindings'],
                     expected_scalar_infinity_where=r.get('expected_scalar_infinity_where', False))

    def test_complete_original_and_existing_address_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [23, 23])

    def test_unused_source_assignments_and_dynamic_loops_are_retained(self):
        for old, new in [('xmask = xindex < xnumel', ''),
                         ('rbase = tl.arange(0, RBLOCK)[None, :]', ''),
                         ('for roffset in range(0, rnumel, RBLOCK):',
                          'for roffset in range(0, rnumel):')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_explicit_casts_and_original_cache_hints_are_retained(self):
        for old, new in [('tmp1 = tmp0.to(tl.float32)', 'tmp1 = tmp0'),
                         ('tmp20 = tmp19.to(tl.float32)', 'tmp20 = tmp19'),
                         ('eviction_policy="evict_first"', 'eviction_policy="evict_last"')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_source_sum_view_and_none_store_are_retained(self):
        for old, new in [('tmp3 = tl.sum(_tmp3, 1)[:, None]', 'tmp3 = tl.sum(_tmp3, 1)'),
                         ('tl.store(in_out_ptr0 + (x0), tmp6, None)',
                          'tl.store(in_out_ptr0 + (x0), tmp6)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_libdevice_is_not_replaced_by_the_tl_intrinsic(self):
        with self.assertRaises(ValueError):
            self.verify(self.source.replace('libdevice.rsqrt(tmp17)', 'tl.rsqrt(tmp17)', 1))

    def test_external_trees_and_closed_xblock_are_checked(self):
        for old, new in [('tmp12 := reductionTree', 'tmp12 := otherTree'),
                         ('let XBLOCK : Nat := 1', 'let XBLOCK : Nat := 2'),
                         ('let XBLOCK : Nat := 1', 'let XBLOCK : Nat := 1\n  let XBLOCK : Nat := 2')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_namespace_import_and_rebinding_are_checked(self):
        for python in [self.python.replace('from torch._inductor.runtime.triton_helpers import libdevice',
                                            'from somewhere_else import libdevice'),
                       self.python + '\nlibdevice = None\n',
                       self.python + '\nlibdevice.rsqrt = lambda x: x\n']:
            with self.subTest(python=python[-70:]), self.assertRaises(ValueError):
                self.verify(python=python)

    def test_no_tree_can_be_silently_omitted(self):
        with self.assertRaises(ValueError):
            self.verify(self.source.replace(', tmp12 := reductionTree', '', 1))


class UniformSamplingBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/uniform_sampling'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)
        self.python = (ROOT / self.records[1]['python']).read_text()

    def verify(self, source=None, python=None, arguments=None):
        r = self.records[1]
        return check(self.python if python is None else python,
                     self.source if source is None else source, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_philox_call=r['expected_philox_call'] if arguments is None else arguments)

    def test_complete_helper_original_and_index_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [3, 22, 22])

    def test_signed_casts_counter_widths_and_all_fields_are_retained(self):
        for old, new in [('philox_seed = philox_seed.to(tl.int64)', ''),
                         ('lane = tl.arange(0, BLOCK).to(tl.int64)', 'lane = tl.arange(0, BLOCK)'),
                         ('c0 += i4.to(tl.uint32)', 'c0 += i4'),
                         ('tl.philox(philox_seed, c0, c1, _O, _O)', 'tl.philox(philox_seed, c0, _O, _O, _O)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_fma_order_masks_and_source_store_values_are_retained(self):
        for old, new in [('tl.fma(uint_to_uniform_float(r0), to - from_, from_)',
                          'uint_to_uniform_float(r0) * (to - from_) + from_'),
                         ('tl.minimum(off_3, N - 1)', 'off_3'),
                         ('mask=off_2 < N', 'mask=off_2 <= N'),
                         ('r1, mask=off_1 < N', 'r1.to(tl.float32), mask=off_1 < N')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_word_bindings_cannot_be_swapped_or_retyped(self):
        for old, new in [('from_ := lower', 'from_ := upper'),
                         ('philox_offset := offset', 'philox_offset := seed'),
                         ('BLOCK := hBlock', 'BLOCK := anotherProof'),
                         ('lower upper : Float32Bits', 'lower upper : ℝ')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_builtin_fallback_selection_and_helper_rebinding_are_checked(self):
        for python in [self.python.replace('tl.uint_to_uniform_float\n', 'tl.rand\n', 1),
                       self.python.replace('except AttributeError:', 'except ImportError:', 1),
                       self.python + '\nuint_to_uniform_float = lambda x: x\n']:
            with self.subTest(python=python[-70:]), self.assertRaises(ValueError):
                self.verify(python=python)

    def test_helper_retains_both_dtype_branches_and_bitcast(self):
        from scripts.bench.jit_correspondence import check_uniform_function
        r = self.records[0]
        for old, new in [('bitcast=True', 'bitcast=False'),
                         ('4.6566127342e-10', '4.656612873e-10'),
                         ('tl.uint64', 'tl.uint32'),
                         ('-x - 1', '-x')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_uniform_function(self.python, self.source.replace(old, new, 1),
                                       r['python_function'], r['lean_definition'])


class AttentionFwdTriton1BodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/attention_fwd_triton1'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, source, candidate=False):
        r = self.records[1 if candidate else 0]
        return check((ROOT / r['python']).read_text(), source, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_constexpr_bindings=r.get('expected_constexpr_bindings'))

    def test_both_complete_sources_and_static_assertion(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [3, 4])

    def test_pointer_geometry_and_pre_update_history_remain(self):
        for old, new in [('(NT * BD, BD)', '(T, BD)'),
                         ('(s_ht, 1)', '(s_ht, s_qd)'),
                         ('tl.store(p_h, b_h.to(p_h.dtype.element_ty))', 'pass')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_all_alias_casts_and_head_int64_cast_are_retained(self):
        for old, new in [('i_bh = tl.program_id(0).to(tl.int64)', 'i_bh = tl.program_id(0)'),
                         ('(b_q * scale).to(b_q.dtype)', 'b_q * scale'),
                         ('b_s.to(b_q.dtype)', 'b_s'),
                         ('b_o.to(p_o.dtype.element_ty)', 'b_o')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_boundary_zero_padding_and_dot_options_are_retained(self):
        for old, new in [('boundary_check=(1,)', 'boundary_check=(0,)'),
                         ('padding_option="zero"', 'padding_option="nan"'),
                         ('allow_tf32=False', 'allow_tf32=True')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_initial_and_later_carry_branches_are_retained(self):
        for old, new in [('if i == 0:', 'if i <= 0:'),
                         ('b_h += tl.dot(b_k, b_v, allow_tf32=False)',
                          'b_h = tl.dot(b_k, b_v, allow_tf32=False)'),
                         ('if STORE:', 'if IFCOND:')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_candidate_assertion_and_fixed_block_are_required(self):
        for old, new in [('tl.static_assert(BT == 32)', ''),
                         ('let BT : Nat := 32', 'let BT : Nat := 16'),
                         ('(i << 5, 0)', '(i * BT, 0)')]:
            source = self.source
            if old == '(i << 5, 0)':
                at = source.index('def runtimeCandidateKernel')
                source = source[:at] + source[at:].replace(old, new, 1)
            else:
                source = source.replace(old, new, 1)
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source, candidate=True)


class PureJitDocstringCorrespondenceTests(unittest.TestCase):
    python = '''@triton.jit
def helper(x):
    """A $value uses no tl.region directive here."""
    return tl.where(x >= 0, x, 0.01 * x)
'''
    lean = '''def copied : PureJitFunction := triton_function (x) {
    """A $value uses no tl.region directive here."""
    return tl.where(x >= 0, x, 0.01 * x)
}
'''

    def verify(self, python=None, lean=None):
        from scripts.bench.jit_correspondence import check_pure_function
        return check_pure_function(self.python if python is None else python,
                                   self.lean if lean is None else lean, 'helper', 'copied')

    def test_complete_helper_includes_its_docstring(self):
        self.assertEqual(self.verify()['statements'], 2)

    def test_docstring_cannot_be_removed_or_changed(self):
        for changed in [self.lean.replace('    """A $value uses no tl.region directive here."""\n', ''),
                        self.lean.replace('A $value', 'A different value')]:
            with self.subTest(source=changed), self.assertRaises(ValueError):
                self.verify(lean=changed)

    def test_extra_executable_statement_is_still_refused(self):
        python = self.python.replace('    return ', '    extra = x\n    return ')
        lean = self.lean.replace('    return ', '    extra = x\n    return ')
        with self.assertRaisesRegex(ValueError, 'one return'):
            self.verify(python, lean)


class MatmulAutotuneBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/matmul_triton_autotune'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.python = (self.directory / 'matmul_triton_autotune.py').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, python=None, source=None):
        r = self.records[1]
        return check(self.python if python is None else python,
                     self.source if source is None else source,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_pure_functions=r['expected_pure_functions'])

    def test_both_full_bodies_and_shared_documented_helper(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [2, 28, 28])

    def test_real_autotune_decorator_and_builtin_min_are_not_helper_calls(self):
        self.assertIn('configs=get_autotune_config()', self.python)
        self.assertIn('min(num_pid_m - first_pid_m, GROUP_SIZE_M)', self.python)
        self.assertEqual(self.verify()['status'], 'PASS')

    def test_builtin_min_rebinding_and_unsupported_arity_are_refused(self):
        for changed in [self.python + '\nmin = lambda a, b: a\n',
                        'from arbitrary_module import min\n' + self.python]:
            with self.subTest(source=changed), self.assertRaisesRegex(ValueError, 'shadowed'):
                self.verify(python=changed)
        python = self.python.replace('min(num_pid_m - first_pid_m, GROUP_SIZE_M)',
                                     'min(num_pid_m - first_pid_m, GROUP_SIZE_M, 1)')
        source = self.source.replace('min(num_pid_m - first_pid_m, GROUP_SIZE_M)',
                                     'min(num_pid_m - first_pid_m, GROUP_SIZE_M, 1)', 1)
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            self.verify(python, source)

    def test_grouped_casts_helper_call_and_half_conversion_cannot_disappear(self):
        for old, new in [('M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)',
                          'M, N, K = M, N, K'),
                         ('accumulator = leaky_relu(accumulator)', 'accumulator = accumulator'),
                         ('c = accumulator.to(tl.float16)', 'c = accumulator')]:
            with self.subTest(source=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new, 1))

class ReluStridedBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.base = 'bench/tritonbench_g/relu_strided_buffer'
        self.directory = ROOT / self.base
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)
        self.python = (self.directory / 'relu_strided_buffer.py').read_text()

    def verify(self, python=None, source=None):
        r = self.records[1]
        return check(self.python if python is None else python,
                     self.source if source is None else source,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_pointer_offsets=r['expected_pointer_offsets'],
                     expected_pure_functions=r['expected_pure_functions'],
                     expected_constexpr_bindings=r['expected_constexpr_bindings'],
                     expected_scalar_infinity_where=r.get('expected_scalar_infinity_where', False))

    def test_both_complete_callers_and_copied_helpers(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [1, 3, 1, 3])
        self.assertEqual(self.records[0]['lean_definition'], self.records[2]['lean_definition'])

    def test_shared_python_helper_requires_a_checked_body(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / self.base
            directory.mkdir(parents=True)
            for name in ['Kernels.lean', 'relu_strided_buffer.py']:
                (directory / name).write_text((self.directory / name).read_text())
            manifest = directory / 'jit-correspondence.json'
            callers = [row for row in self.records if row.get('expected_pure_functions')]
            self.assertEqual(len(callers), 2)
            manifest.write_text(json.dumps(callers))
            with self.assertRaisesRegex(ValueError, 'separately checked helper body'):
                check_manifest(root, manifest)

    def test_cast_boundary_order_and_helper_call_are_preserved(self):
        for old, new in [('.to(in0_ptr.type.element_ty)', ''),
                         ('out0.to(out0_bptr.type.element_ty)', 'out0'),
                         ('boundary_check=(in0_stride_order0,)', 'boundary_check=(1,)'),
                         ('order=(out0_stride_order0,)', 'order=(1,)'),
                         ('relu_forward(in0)', 'tl.maximum(in0, 0)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new, 1))

    def test_signed_allocation_bases_and_storage_cannot_change(self):
        for old, new in [('in0_ptr := input_base', 'in0_ptr := output_base'),
                         ('out0_ptr := output_base', 'out0_ptr := 0'),
                         ('("in0_ptr",.fp32)', '("in0_ptr",.fp16)'),
                         ('relu_forward := reluForwardSource', 'relu_forward := otherHelper')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new, 1))

    def test_builtin_integer_annotation_does_not_accept_an_explicit_int64_abi(self):
        self.assertEqual(self.verify()['status'], 'PASS')
        for source in [self.python.replace('in0_stride0: int', 'in0_stride0: tl.int64', 1),
                       self.python + '\nint = custom_type\n']:
            with self.subTest(source=source[-40:]), self.assertRaises(ValueError):
                self.verify(python=source)

    def test_floating_storage_metadata_requires_a_real_dtype_constructor(self):
        for value in ['"fp32"', '.i64', 'unknown_dtype']:
            with self.subTest(value=value), self.assertRaises((ValueError, SyntaxError)):
                self.verify(source=self.source.replace('("in0_ptr",.fp32)', '("in0_ptr",' + value + ')', 1))

class RwkvBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fused_rwkv6_kernel'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.python = (self.directory / 'fused_rwkv6_kernel.py').read_text()
        self.record = pinned_pairs(self.directory)[0]

    def verify(self, source=None):
        r = self.record
        return check(self.python, self.source if source is None else source,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_both_complete_sources_keep_sixteen_top_level_statements(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [16, 16])

    def test_tuple_program_axes_and_reverse_pointer_updates_are_exact(self):
        for old, new in [('tl.program_id(0), tl.program_id(1)', 'tl.program_id(1), tl.program_id(0)'),
                         ('p_q += -K if REVERSE else K', 'p_q += K if REVERSE else -K'),
                         ('p_o += -V if REVERSE else V', 'p_o += -K if REVERSE else K')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_all_state_flags_and_output_casts_are_retained(self):
        for old, new in [('if USE_INITIAL_STATE:', 'if False:'),
                         ('if STORE_FINAL_STATE:', 'if False:'),
                         ('b_o.to(p_o.dtype.element_ty)', 'b_o'),
                         ('b_h.to(p_ht.dtype.element_ty)', 'b_h')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_scalar_abi_and_runtime_strides_cannot_be_rebound(self):
        for old, new in [('("scale",.fp32)', '("scale",.fp64)'),
                         ('s_k_h := s_k_h', 's_k_h := s_v_h')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.source.replace(old, new, 1))

    def test_repaired_partial_output_pitch_is_not_replaced_by_input_pitch(self):
        with self.assertRaises(ValueError):
            self.verify(self.source.replace('(i_bh + i_k * B * H) * (T * V)',
                                            '(i_bh + i_k * B * H) * s_v_h', 1))

class RecurrentRetentionBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fused_recurrent_retention'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.python = (self.directory / 'fused_recurrent_retention.py').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index, source):
        r = self.records[index]
        return check(self.python, source, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_all_four_complete_forward_backward_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [14, 23, 14, 23])

    def test_python_do_pointer_and_all_runtime_strides_are_preserved(self):
        for old, new in [('p_do = do +', 'p_do = q +'),
                         ('s_do_t := s_do_t', 's_do_t := s_vo_t'),
                         ('s_df_k := s_df_k', 's_df_k := s_df_v')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(1, self.source.replace(old, new))

    def test_reverse_counted_loop_and_barrier_cannot_be_changed(self):
        for old, new in [('for _ in range(T):', 'for _ in range(0, T):'),
                         ('tl.debug_barrier()', 'tl.debug_barrier(0)'),
                         ('d_h *= b_b', 'd_h += b_b')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(1, self.source.replace(old, new))

    def test_forward_seed_and_final_state_and_tuple_axes_are_exact(self):
        for old, new in [('if USE_INITIAL_STATE:', 'if False:'),
                         ('if STORE_FINAL_STATE:', 'if False:'),
                         ('tl.program_id(0), tl.program_id(1)', 'tl.program_id(1), tl.program_id(0)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(0, self.source.replace(old, new))

    def test_reverse_output_cast_and_scalar_abi_are_exact(self):
        for old, new in [('d_v.to(p_dv.dtype.element_ty)', 'd_v'),
                         ('("scale",.fp32)', '("scale",.fp64)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(1, self.source.replace(old, new))

class ChunkGateRecurrenceBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/chunk_gate_recurrence'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index, source):
        r = self.records[index]
        python = (ROOT / r['python']).read_text()
        return check(python, source, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_optional_pointers=r.get('expected_optional_pointers'))

    def test_four_complete_bodies_and_storage_profiles(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [10, 17, 10, 17])

    def test_four_seed_strides_and_presence_cannot_change(self):
        for old, new in [('if last_kv is not None:', 'if last_kv is None:'),
                         (' * SEED_H', ' * SEED_B'),
                         (' * SEED_V', ' * SEED_K'),
                         ('last_kv := hasSeed', 'last_kv := False')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(0, self.source.replace(old, new))

    def test_all_backward_stores_casts_and_counted_loop_remain(self):
        for old, new in [('DG_i.to(DG.dtype.element_ty)', 'DG_i'),
                         ('Dacc.to(DL.dtype.element_ty)', 'Dacc'),
                         ('for i in range(NUM_BLOCK - 1):', 'for i in range(NUM_BLOCK):'),
                         ('DG -= NUM_K * NUM_V', 'DG -= 1')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(1, self.source.replace(old, new))

    def test_original_and_sum_left_arithmetic_are_separately_checked(self):
        for index, old, new in [(0, 'acc = acc * d_i + S_i', 'acc = S_i + acc * d_i'),
                                 (2, 'acc = S_i + acc * d_i', 'acc = acc * d_i + S_i'),
                                 (3, 'Dacc = DS_i + Dacc * d_i', 'Dacc = Dacc * d_i + DS_i')]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(index, self.source.replace(old, new))

    def test_scalar_bindings_and_polymorphic_storage_are_checked(self):
        for index, old, new in [(0, 'NUM_BLOCK := NUM_BLOCK', 'NUM_BLOCK := NUM_HEAD'),
                                 (1, '("DL",source)', '("DL",incoming)'),
                                 (1, '| .fp16 => .fp16', '| .fp16 => .fp32')]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(index, self.source.replace(old, new))

class AttentionKernelBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/attention_kernel'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index, source):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), source, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_original_and_early_query_bodies_are_complete(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [27, 27])

    def test_output_dtype_and_probability_casts_are_retained(self):
        for old, new in [('(q * qk_scale).to(OUT_DTYPE)', '(q * qk_scale)'),
                         ('dtype=OUT_DTYPE', 'dtype=tl.float32'),
                         ('p.to(OUT_DTYPE)', 'p'),
                         ('acc.to(OUT_DTYPE)', 'acc')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(0, self.source.replace(old, new))

    def test_full_block_shapes_boundary_checks_and_bias_windows_are_retained(self):
        for old, new in [('N_CTX + P_SEQ', 'N_CTX'),
                         ('boundary_check=(0,)', 'boundary_check=(1,)'),
                         ('+ BIAS_LAST_SIZE', '+ BLOCK_N'),
                         ('off_hz = tl.program_id(1).to(tl.int64)', 'off_hz = tl.program_id(1)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(0, self.source.replace(old, new))

    def test_storage_branches_and_runtime_scale_binding_are_checked(self):
        for old, new in [('| .bf16 => .bf16', '| .bf16 => .fp16'),
                         ('sm_scale := sm_scale', 'sm_scale := 1'),
                         ('("sm_scale",.fp32)', '("sm_scale",.fp64)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(0, self.source.replace(old, new))

    def test_original_cannot_be_replaced_by_the_scheduled_candidate(self):
        from scripts.bench.jit_correspondence import quotation
        original, _ = quotation(self.source, 'originalKernel', raw_python=True)
        candidate, _ = quotation(self.source, 'queryEarlyKernel', raw_python=True)
        self.assertNotEqual(original, candidate)
        with self.assertRaises(ValueError):
            self.verify(0, self.source.replace(original, candidate, 1))

class AlignedAttentionBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/attention_kernel_aligned'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index, source):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), source, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'],
                     expected_constexpr_bindings=r['expected_constexpr_bindings'],
                     expected_scalar_infinity_where=r.get('expected_scalar_infinity_where', False))

    def test_all_four_complete_storage_specific_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [27, 27, 27, 27])

    def test_closed_output_dtype_matches_the_reviewed_branch(self):
        for index, old, new in [(0, 'FloatDType := .fp16', 'FloatDType := .bf16'),
                                 (1, 'FloatDType := .bf16', 'FloatDType := .fp16'),
                                 (0, 'FloatDType := .fp16', 'FloatDType := selected_type')]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(index, self.source.replace(old, new))

    def test_bf16_explicit_dot_narrowing_and_fp16_dot_result_are_separate(self):
        for index, old, new in [(0, 'out_dtype=OUT_DTYPE', 'out_dtype=tl.float32'),
                                 (1, 'out_dtype=tl.float32).to(OUT_DTYPE)', 'out_dtype=tl.float32)'),
                                 (1, 'out_dtype=tl.float32).to(OUT_DTYPE)', 'out_dtype=OUT_DTYPE)')]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(index, self.source.replace(old, new))

    def test_tail_mask_shape_boundaries_and_runtime_bindings_remain_exact(self):
        for old, new in [('boundary_check=(0, 1)', 'boundary_check=(0,)'),
                         ('N_CTX + P_SEQ', 'N_CTX'),
                         ('< N_CTX', '<= N_CTX'),
                         ('stride_kn := stride_kn', 'stride_kn := stride_kk'),
                         ('sm_scale := sm_scale', 'sm_scale := 1')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(0, self.source.replace(old, new))

    def test_scheduled_candidate_and_softmax_operations_are_not_erased(self):
        for index, old, new in [(0, 'l_i = l_i * alpha + tl.sum(p, 1)', 'l_i = tl.sum(p, 1)'),
                                 (3, 'acc = acc / l_i[:, None]', 'acc = acc * l_i[:, None]')]:
            self.assertTrue(old in self.source, old)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(index, self.source.replace(old, new))

class FusedLayernormBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fused_layernorm_triton'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index=0, source=None, python=None, **overrides):
        r = dict(self.records[index], **overrides)
        return check(python or (ROOT / r['python']).read_text(), source or self.source,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     **{k: v for k, v in r.items() if k.startswith('expected_') and k != 'expected_signature'})

    def test_complete_original_and_existing_hoisted_body(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [26, 29])

    def test_intrinsic_casts_unused_mask_and_original_shapes_are_preserved(self):
        for old, new in [('libdevice.rsqrt(tmp9)', 'tl.rsqrt(tmp9)'),
                         ('xnumel = tl.cast(xnumel, tl.int64)', 'xnumel = xnumel'),
                         ('xmask = xindex < xnumel', ''),
                         ('rindex = roffset + rbase', 'rindex = roffset + tl.arange(0, RBLOCK)'),
                         ('tmp1 = tmp0.to(tl.float32)', 'tmp1 = tmp0'),
                         ('range(zero, rnumel, step)', 'range(step, rnumel, step)')]:
            self.assertIn(old, self.source)
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))

    def test_tree_binding_width_axis_and_fixed_row_size_are_checked(self):
        for old, new in [('welford_tree tree', 'welford_tree other'),
                         ('DSL.WelfordSource.Tree RBLOCK', 'DSL.WelfordSource.Tree XBLOCK'),
                         ('let XBLOCK : Nat := 1', 'let XBLOCK : Nat := 2')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))
        with self.assertRaises(ValueError):
            self.verify(expected_welford_tree={'argument': 'tree', 'width': 'XBLOCK'})

    def test_intrinsic_namespaces_cannot_be_rebound_or_replaced(self):
        python = (ROOT / self.records[0]['python']).read_text()
        for old, new in [('from torch._inductor.runtime import triton_helpers',
                         'from unrelated import triton_helpers'),
                         ('from torch._inductor.runtime.triton_helpers import libdevice',
                          'from triton.language import libdevice')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(python=python.replace(old, new))
        with self.assertRaises(ValueError):
            self.verify(python=python + '\ntriton_helpers = replacement\n')


class ChunkGLASimpleBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/chunk_gla_simple'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index=0, source=None):
        r = self.records[index]
        return check((ROOT / r['python']).read_text(), source or self.source,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_complete_original_and_row_reuse_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in rows], [23, 23])

    def test_signed_addresses_loop_and_typed_dots_cannot_be_erased(self):
        for old, new in [('tl.program_id(2).to(tl.int64)', 'tl.program_id(2)'),
                         ('tl.cast(s_k_h, tl.int64)', 's_k_h'),
                         ('tl.cast(tl.cdiv(K, BK), tl.int64)', 'tl.cdiv(K, BK)'),
                         ('tl.cast(i_k, tl.int64)', 'i_k'),
                         ('b_s.to(b_v.dtype)', 'b_s'),
                         ('allow_tf32=False', 'allow_tf32=True'),
                         ('mask=rows < T, other=0', 'mask=rows < T, other=1')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))
        with self.assertRaises(ValueError):
            self.verify(1, self.source.replace('o_i.to(tl.int64)', 'o_i'))

    def test_runtime_stride_precision_and_output_dtype_are_bound(self):
        for old, new in [('("s_h_t",.integer)', '("s_h_t",.fp32)'),
                         ('("scale",.fp32)', '("scale",.integer)'),
                         ('("o",out_dtype)', '("o",.fp16)'),
                         ('s_v_h := s_v_h', 's_v_h := s_k_h')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))

    def test_materialized_candidate_only_reuses_the_row_lane_vector(self):
        original = (ROOT / self.records[0]['python']).read_text()
        candidate = (ROOT / self.records[1]['python']).read_text()
        def body(text, name):
            return next(n for n in ast.parse(text).body
                        if isinstance(n, ast.FunctionDef) and n.name == name).body
        lhs = body(original, self.records[0]['python_function'])
        rhs = body(candidate, self.records[1]['python_function'])
        changes = [(a, b) for a, b in zip(lhs, rhs)
                   if ast.dump(a) != ast.dump(b)]
        self.assertEqual(len(lhs), len(rhs))
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0].targets[0].id, 'rows')
        self.assertEqual(changes[0][1].targets[0].id, 'rows')


class ScalarTensorPowerBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/pow_scalar_tensor'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index=1, source=None, python=None):
        r = self.records[index]
        return check(python or (ROOT / r['python']).read_text(), source or self.source,
                     r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     **{k: v for k, v in r.items() if k.startswith('expected_') and k != 'expected_signature'})

    def helper(self, source=None, python=None):
        from scripts.bench.jit_correspondence import check_pure_function
        r = self.records[0]
        return check_pure_function(python or (ROOT / r['python']).read_text(),
                                   source or self.source, r['python_function'],
                                   r['lean_definition'], 2, r['expected_imports'])

    def test_both_complete_branches_and_their_copied_binary_helpers(self):
        checks = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in checks], [1, 3, 1, 3])

    def test_original_casts_block_boundaries_and_control_cannot_be_erased(self):
        for old, new in [('num_tiles0 = tl.cdiv(s0, tile_size0)', ''),
                         ('if one_tile_per_cta:', 'if True:'),
                         ('range(0, tiles_per_cta)', 'range(1, tiles_per_cta)'),
                         ('.to(in0_ptr.type.element_ty)', '.to(tl.float16)'),
                         ('boundary_check=(in0_stride_order0,)', 'boundary_check=()'),
                         ('out0.to(out0_bptr.type.element_ty)', 'out0')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))
        with self.assertRaises(ValueError):
            self.helper(source=self.source.replace('x.to(tl.float32)', 'x'))

    def test_all_three_import_fallbacks_are_checked(self):
        original = (ROOT / self.records[0]['python']).read_text()
        for old, new in [('triton.language.extra.cuda.libdevice', 'unreviewed.libdevice'),
                         ('triton.language.math', 'unreviewed.math'),
                         ('triton.language.libdevice', 'unreviewed.fallback'),
                         ('except ImportError:', 'except RuntimeError:')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.helper(python=original.replace(old, new))
        with self.assertRaises(ValueError):
            self.helper(python=original + '\n_pow = other\n')

    def test_signed_base_stride_and_fp32_scalar_metadata_are_bound(self):
        for old, new in [('("val0",.fp32)', '("val0",.fp64)'),
                         ('("in0_stride0",.integer)', '("in0_stride0",.fp32)'),
                         ('in0_ptr := input_base', 'in0_ptr := output_base'),
                         ('val0 := val0', 'val0 := other')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))

    def test_candidate_only_commutes_the_two_integer_offset_products(self):
        original = (ROOT / self.records[0]['python']).read_text()
        candidate = (ROOT / self.records[2]['python']).read_text()
        def body(text, name):
            return next(f for f in ast.parse(text).body
                        if isinstance(f, ast.FunctionDef) and f.name == name).body
        class Undo(ast.NodeTransformer):
            count = 0
            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'offset0':
                    self.assert_product(node.value)
                    node.value.left, node.value.right = node.value.right, node.value.left
                    self.count += 1
                return self.generic_visit(node)
            def assert_product(self, node):
                assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)
                assert isinstance(node.left, ast.Name) and node.left.id == 'tile_size0'
                assert isinstance(node.right, ast.Name) and node.right.id == 'tile_id0'
        undo = Undo()
        lhs = ast.Module(body(original, self.records[1]['python_function']), [])
        rhs = undo.visit(ast.Module(body(candidate, self.records[3]['python_function']), []))
        self.assertEqual(undo.count, 2)
        self.assertEqual(ast.dump(lhs), ast.dump(rhs))


class IvDependentMatmulBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/iv_dependent_matmul'
        self.source = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, index=0, source=None):
        record = self.records[index]
        return check((ROOT / record['python']).read_text(), source or self.source,
                     record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     expected_call_bindings=record['expected_call_bindings'])

    def test_both_copied_bodies_cover_all_five_schedules(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [29, 29])
        self.assertTrue(all(row['mode'] == 'pasted_body' for row in rows))

    def test_signed_casts_prefetch_branches_tail_mask_and_narrowing_are_required(self):
        for old, new in [('M = tl.cast(M, tl.int64)', ''),
                         ('tl.cast(tl.arange(0, BLOCK_SIZE_M), tl.int64)', 'tl.arange(0, BLOCK_SIZE_M)'),
                         ('k64 = tl.cast(k, tl.int64)', 'k64 = k'),
                         ('elif type == "post_load_three_iters":', 'elif type == "post_load":'),
                         ('a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak', ''),
                         ('K - k64 * BLOCK_SIZE_K', 'K'),
                         ('c = accumulator.to(tl.float16)', 'c = accumulator')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))
        with self.assertRaises(ValueError):
            self.verify(index=1, source=self.source.replace(
                'accumulator = tl.dot(a, b, accumulator)', 'accumulator += tl.dot(a, b)'))

    def test_runtime_stride_and_dtype_metadata_cannot_change(self):
        for old, new in [('("stride_am",.integer)', '("stride_am",.fp32)'),
                         ('stride_am := stride_am', 'stride_am := stride_bk'),
                         ('("c_ptr",dtype)', '("c_ptr",.fp16)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(source=self.source.replace(old, new))

    def test_retained_candidate_changes_only_the_live_dot_operand(self):
        original = variant_source(self.directory / 'iv_dependent_matmul.py')
        candidate = variant_source(self.directory / 'iv_dependent_matmul_dotacc.py')
        class Undo(ast.NodeTransformer):
            count = 0
            def visit_Assign(self, node):
                if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == 'accumulator'
                        and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == 'tl.dot'):
                    assert [ast.unparse(arg) for arg in node.value.args] == ['a', 'b', 'accumulator']
                    assert not node.value.keywords
                    self.count += 1
                    return ast.AugAssign(node.targets[0], ast.Add(),
                                         ast.Call(node.value.func, node.value.args[:2], []))
                return self.generic_visit(node)
        undo = Undo()
        restored = undo.visit(ast.parse(candidate))
        self.assertEqual(undo.count, 1)
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(restored))


class StatementHelperBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/rbe_triton_transform'
        self.python = (self.directory / 'rbe_triton_transform.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def caller(self, lean):
        record = self.records[1]
        return check(self.python, lean, record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     expected_call_bindings=record['expected_call_bindings'],
                     expected_jit_functions=record['expected_jit_functions'])

    def test_complete_caller_and_helper_bodies(self):
        results = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r['statements'] for r in results], [6, 31, 6, 31])
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))

    def test_caller_cast_mask_argument_and_helper_binding_erasure_rejected(self):
        for before, after in [('tl.program_id(axis=0).to(tl.int64)', 'tl.program_id(axis=0)'),
                              ('starting_idx=start_block', 'starting_idx=start_token_position'),
                              ('offs_cn=offs_n', 'offs_cn=offs_m'),
                              ('1 + offs_n[None, :] < K', 'offs_n[None, :] < K'),
                              ('get_freq_multi_tokens := frequencySource',
                               'get_freq_multi_tokens := anotherFunction')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.caller(self.lean.replace(before, after, 1))

    def test_helper_dimension_signed_cast_and_return_order_rejected(self):
        for before, after in [('DIM: tl.constexpr = 128', 'DIM: tl.constexpr = 256'),
                              ('freqs.to(tl.float32)', 'freqs'),
                              ('tl.cast(starting_idx, tl.int64)', 'starting_idx'),
                              ('return tl.cos(freqs), tl.sin(freqs)',
                               'return tl.sin(freqs), tl.cos(freqs)'),
                              ('constexpr_args [theta, NB_TOKENS]', 'constexpr_args [NB_TOKENS]')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                check_statement_function(self.python, self.lean.replace(before, after, 1),
                                         'get_freq_multi_tokens', 'frequencySource')

    def test_existing_candidate_changes_only_the_two_store_masks(self):
        original_source = variant_source(self.directory / 'rbe_triton_transform.py')
        original = ast.parse(original_source)
        candidate_path = self.directory / 'rbe_triton_transform_mask_reuse.py'
        candidate = ast.parse(variant_source(candidate_path))
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == 'rbe_triton')
        source_fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'rbe_triton')
        assignments = {n.targets[0].id: n for n in fn.body
                       if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        sources = {n.targets[0].id: n for n in source_fn.body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        for target, value in [('out_real_mask', 'x_real_mask'), ('out_imag_mask', 'x_imag_mask')]:
            self.assertEqual(ast.dump(assignments[target].value), ast.dump(ast.Name(id=value, ctx=ast.Load())))
            assignments[target].value = sources[target].value
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(self.directory)
        import hashlib
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance['candidate_sha256'])
        self.assertEqual(hashlib.sha256(original_source.encode()).hexdigest(), provenance['original_sha256'])


class FP4BodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/fp4_to_bf16'
        self.python = (self.directory / 'fp4_to_bf16.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def caller(self, lean):
        record = self.records[1]
        return check(self.python, lean, record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     expected_call_bindings=record['expected_call_bindings'],
                     expected_jit_functions=record['expected_jit_functions'],
                     expected_constexpr_bindings=record['expected_constexpr_bindings'])

    def test_complete_helper_and_caller(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [22, 13])
        self.assertTrue(all(row['mode'] == 'pasted_body' for row in rows))

    def test_bitcast_shift_unused_assignment_and_return_cannot_disappear(self):
        changes = [('result.to(tl.float32, bitcast=True)', 'result.to(tl.float32)'),
                   ('mantissa_f4.to(tl.int32)', 'mantissa_f4'),
                   ('output = mantissa_f32', 'output = mantissa_f4'),
                   ('mbits_f32 - mbits_f4_e2m1', 'mbits_f32 + mbits_f4_e2m1'),
                   ('return output', 'return result')]
        for before, after in changes:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                check_statement_function(self.python, self.lean.replace(before, after, 1),
                                         '_fp4_packed_to_bf16', 'conversionSource')

    def test_signed_addresses_masks_format_and_helper_binding_are_checked(self):
        changes = [('tl.program_id(axis=0).to(tl.int64)', 'tl.program_id(axis=0)'),
                   ('tl.arange(0, BLOCK_SIZE_OUT).to(tl.int64)', 'tl.arange(0, BLOCK_SIZE_OUT)'),
                   ('mask_out = offsets_out < n_elements_out', 'mask_out = offsets_out <= n_elements_out'),
                   ('let f32_exp_bias : Nat := 127', 'let f32_exp_bias : Nat := 128'),
                   ('_fp4_packed_to_bf16 := conversionSource', '_fp4_packed_to_bf16 := otherFunction')]
        for before, after in changes:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.caller(self.lean.replace(before, after, 1))

    def test_specialization_matches_the_actual_host_launch(self):
        module = ast.parse(self.python)
        environment = {}
        for node in module.body:
            if isinstance(node, ast.Assign) and all(isinstance(n, (ast.Constant, ast.Tuple))
                                                   for n in ast.walk(node.value) if isinstance(n, ast.expr)):
                # Evaluate only constant/tuple assignments, never imports or host/GPU code.
                values = ast.literal_eval(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        environment[target.id] = values
                    elif isinstance(target, ast.Tuple):
                        environment.update((part.id, value) for part, value in zip(target.elts, values))
        host = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'triton_f4_to_bf16_repaired')
        call = next(n for n in ast.walk(host) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Subscript)
                    and ast.unparse(n.func.value) == 'triton_f4_to_bf16_kernel_repaired')
        actual = {kw.arg: environment[kw.value.id] for kw in call.keywords
                  if isinstance(kw.value, ast.Name)}
        expected = {name: int(value) for name, value in self.records[1]['expected_constexpr_bindings'].items()}
        self.assertEqual(actual, expected)
        self.assertEqual(next(kw.value.value for kw in call.keywords if kw.arg == 'BLOCK_SIZE_IN'), 512)


class AttentionScoreBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/attention_score'
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, source):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), source,
                     r['python_function'], r['lean_definition'], r['signature_definition'],
                     r['expected_signature'], expected_call_bindings=r['expected_call_bindings'])

    def test_complete_original_and_existing_scalar_left_candidate(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [21, 21])
        self.assertTrue(all(row['mode'] == 'pasted_body' for row in rows))

    def test_signed_window_boundaries_block_pointer_offsets_and_precision_are_checked(self):
        for before, after in [
                ('(start_n * BLOCK_N).to(tl.int32)', 'start_n * BLOCK_N'),
                ('tl.cast(sliding_window_offset, tl.int64)', 'sliding_window_offset'),
                ('dist >= tl.cast(sliding_window_size, tl.int64)', 'dist < tl.cast(sliding_window_size, tl.int64)'),
                ('qk_scale *= 1.4426950408889634', 'qk_scale *= 1.44269504'),
                ('tl.multiple_of(start_m, BLOCK_M)', 'start_m'),
                ('tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))', 'tl.advance(Q_block_ptr, offsets=(0, BLOCK_M))'),
                ('o.to(Out.type.element_ty)', 'o')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.verify(self.lean.replace(before, after, 1))

    def test_materialized_candidate_changes_only_the_existing_multiplication(self):
        original_path = self.directory / 'attention_score.py'
        candidate_path = self.directory / 'attention_score_scalar_left.py'
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == '_score_kernel')
        loop = next(n for n in fn.body if isinstance(n, ast.For))
        stmt = next(n for n in loop.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.value) == 'qk_scale * qk')
        stmt.value.left, stmt.value.right = stmt.value.right, stmt.value.left
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        import hashlib
        provenance = variant_provenance(self.directory)
        self.assertEqual(hashlib.sha256(variant_source(original_path).encode()).hexdigest(), provenance['original_sha256'])
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance['candidate_sha256'])


class FlashAttentionBodyCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/flash_attn'
        self.python = (self.directory / 'flash_attn.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def verify(self, lean):
        r = self.records[0]
        return check(self.python, lean, r['python_function'], r['lean_definition'],
                     r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'])

    def test_both_complete_existing_sources(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [27, 27])
        self.assertTrue(all(row['mode'] == 'pasted_body' for row in rows))

    def test_tail_causal_bound_casts_log_output_and_advances_are_retained(self):
        for before, after in [
                ('tl.minimum((start_m + 1) * BLOCK_M, SEQLEN)', '(start_m + 1) * BLOCK_M'),
                ('start_n + off_n[None, :] < SEQLEN', 'start_n + off_n[None, :] <= SEQLEN'),
                ('q.to(tl.float32)', 'q'),
                ('nume.to(tl.float16).to(tl.float32)', 'nume.to(tl.float32)'),
                ('max + tl.math.log2(denom)', 'tl.math.log2(denom)'),
                ('tl.advance(V_block_ptr, (BLOCK_N, 0))', 'tl.advance(V_block_ptr, (0, BLOCK_N))'),
                ('out_buffer.to(tl.float16).to(tl.float32)', 'out_buffer.to(tl.float32)')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.verify(self.lean.replace(before, after, 1))

    def test_existing_candidate_only_commutes_four_head_products(self):
        functions = {n.name: n for n in ast.parse(self.python).body if isinstance(n, ast.FunctionDef)}
        a, b = functions['_fwd_kernel'], functions['_fwd_kernel_index_commuted']
        self.assertEqual(ast.dump(a.args), ast.dump(b.args))
        differences = [(x, y) for x, y in zip(a.body, b.body) if ast.dump(x) != ast.dump(y)]
        self.assertEqual(len(differences), 4)
        self.assertEqual([x.targets[0].id for x, _ in differences],
                         ['q_base_offset', 'k_base_offset', 'v_base_offset', 'o_base_offset'])
        for x, y in differences:
            self.assertIsInstance(x.value.right.op, ast.Mult)
            y.value.right.left, y.value.right.right = y.value.right.right, y.value.right.left
            self.assertEqual(ast.dump(x), ast.dump(y))


class ProcedureHelperBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/parallel_attention'
        self.python = (self.directory / 'parallel_attention.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.records = pinned_pairs(self.directory)

    def helper(self, lean):
        from scripts.bench.jit_correspondence import check_procedure_function
        record = self.records[2]
        return check_procedure_function(self.python, lean, record['python_function'], record['lean_definition'],
                                        record['signature_definition'], record['expected_signature'])

    def caller(self, lean):
        record = self.records[3]
        return check(self.python, lean, record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     expected_call_bindings=record['expected_call_bindings'],
                     expected_jit_procedures=record['expected_jit_procedures'])

    def test_all_seven_complete_original_and_fused_bodies(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([row['statements'] for row in rows], [22, 20, 13, 8, 22, 20, 8])
        self.assertEqual(sum(row.get('kind') == 'procedure_jit_function' for row in rows), 3)

    def test_reverse_loop_cast_mask_store_return_and_parameter_mutations_rejected(self):
        start = self.lean.index('def keyValueSource')
        for before, after in [
                ('(i_c + 1) * BTL - BTS, -BTS', '(i_c + 1) * BTL - BTS, BTS'),
                ('.to(b_q.dtype)  # [BV, BTS]', '  # [BV, BTS]'),
                ('mask=(i + tl.arange(0, BTS)) < T', 'mask=(i + tl.arange(0, BTS)) <= T'),
                ('tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty)', 'tl.store(p_dk, b_dv.to(p_dv.dtype.element_ty)'),
                ('    return\n  }\n\ndef backwardSignature', '\n  }\n\ndef backwardSignature'),
                ('with_triton_params (i_bh,i_c,i_k,i_v,i_h,',
                 'with_triton_params (i_bh,i_c,i_v,i_k,i_h,')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean[start:])
                with self.assertRaises(ValueError):
                    self.helper(self.lean[:start] + self.lean[start:].replace(before, after, 1))

    def test_helper_binding_call_order_argument_and_barrier_erasure_rejected(self):
        for before, after in [
                ('_parallel_rebased_bwd_dq := queryProcedureSource', '_parallel_rebased_bwd_dq := fusedQueryProcedureSource'),
                ('q, k, v, do, dz, dq, s_qk_h', 'q, k, v, do, dz, dk, s_qk_h'),
                ('    tl.debug_barrier()\n    _parallel_rebased_bwd_dkv(', '    _parallel_rebased_bwd_dkv(')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.caller(self.lean.replace(before, after, 1))


class ProcedureLiteralBodyTests(unittest.TestCase):
    def test_keyword_escape_adapter_never_rewrites_python_string_literals(self):
        from scripts.bench.jit_correspondence import check_procedure_function
        python = '@triton.jit\ndef helper(X):\n    tl.static_print("do")\n    return\n'
        lean = 'def signature : CallSignature where\n  pointers := [("X",.fp32)]\n  scalars := []\n  constexpr := []\n\ndef helper := triton_procedure signature (X) constexpr_args [] {\n    tl.static_print("do")\n    return\n}\n'
        signature = {'pointers': [['X', 'fp32']], 'scalars': [], 'constexpr': []}
        self.assertEqual(check_procedure_function(python, lean, 'helper', 'helper', 'signature', signature)['status'], 'PASS')
        with self.assertRaises(ValueError):
            check_procedure_function(python, lean.replace('"do"', '"«do»"'), 'helper', 'helper', 'signature', signature)
