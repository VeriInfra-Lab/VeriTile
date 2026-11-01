import ast
import json
import textwrap
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_manifest, quotation

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "bench/tritonbench_g/triton_argmax"


class JITCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.python = (DIRECTORY / "triton_argmax.py").read_text()
        self.lean = (DIRECTORY / "Kernels.lean").read_text()

    def test_all_original_and_candidate_bodies(self):
        self.assertEqual(len(check_manifest(ROOT, DIRECTORY / "jit-correspondence.json")), 6)

    def test_cast_nan_clamp_store_and_parameter_changes_are_rejected(self):
        mutations = [
            ("(pid).to(tl.int64)", "pid"),
            ("tl.maximum(stored_value,", "tl.minimum(stored_value,"),
            ("tl.where((max_val == max_val), max_val, 0.0)", "max_val"),
            ("tl.store(mid_value_ptr, stored_value)", "tl.store(mid_value_ptr, max_val)"),
            ("tl.param M : tl.int64", "tl.param M : tl.float32"),
            ("(offset < M)", "(offset <= M)"),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    check(self.python, self.lean.replace(before, after, 1),
                          "argmax_kernel_1_repaired", "stage1")

    def test_jit_decorator_is_required(self):
        with self.assertRaisesRegex(ValueError, "expected one explicitly selected"):
            check(self.python.replace("@triton.jit", "", 1), self.lean,
                  "argmax_kernel_1_repaired", "stage1")

    def test_runtime_parameter_cannot_be_replaced_by_antiquote(self):
        with self.assertRaisesRegex(ValueError, "not a constexpr"):
            check(self.python, self.lean.replace("(offset < M)", "(offset < $(M))", 1),
                  "argmax_kernel_1_repaired", "stage1")

    def test_unsupported_splice_cannot_hide_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "unsupported antiquotation"):
            check(self.python, self.lean.replace("$(BLOCK_SIZE)", "$(BLOCK_SIZE + 1)", 1),
                  "argmax_kernel_1_repaired", "stage1")


class PastedBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/tritonbench_g/dequantize_rowwise"
        self.python = (self.directory / "dequantize_rowwise.py").read_text()
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]

    def compare(self, lean):
        return check(self.python, lean, self.record["python_function"], self.record["lean_definition"],
                     self.record["signature_definition"], self.record["expected_signature"])

    def test_verbatim_body_and_external_signature(self):
        self.assertEqual(self.compare(self.lean)["statements"], 9)
        fn = next(n for n in ast.parse(self.python).body
                  if isinstance(n, ast.FunctionDef) and n.name == self.record["python_function"])
        python_body = "\n".join(self.python.splitlines()[fn.body[0].lineno-1:fn.body[-1].end_lineno])
        lean_body = quotation(self.lean, "body")[1]
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
        result = check(python, self.lean, self.record["python_function"], "body",
                       "callSignature", self.record["expected_signature"])
        self.assertEqual(result["mode"], "pasted_body")
        for annotation in ("tl.int32", "tl.int64"):
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                check(self.python.replace("n_elements,", f"n_elements: {annotation},", 1),
                      self.lean, self.record["python_function"], "body", "callSignature",
                      self.record["expected_signature"])

    def test_pointer_and_quoted_constexpr_annotations(self):
        python = self.python.replace("x_ptr,", "x_ptr: tl.tensor,", 1).replace(
            "BLOCK_SIZE: tl.constexpr", 'BLOCK_SIZE: "tl.constexpr"', 1)
        self.assertEqual(check(python, self.lean, self.record["python_function"], "body",
                               "callSignature", self.record["expected_signature"])["status"], "PASS")

    def test_unreviewed_signature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed call signature"):
            check(self.python, self.lean, self.record["python_function"], "body", "callSignature")

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
                result = check_manifest(ROOT, directory / "jit-correspondence.json")
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
            record = json.loads((directory / "jit-correspondence.json").read_text())[0]
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
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
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


if __name__ == "__main__":
    unittest.main()


class PastedLoopBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/tritonbench_g/matrix_vector_multip'
        self.python = (self.directory / 'matrix_vector_multip.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()
        self.record = json.loads((self.directory / 'jit-correspondence.json').read_text())[0]

    def compare(self, lean):
        return check(self.python, lean, self.record['python_function'], self.record['lean_definition'],
                     self.record['signature_definition'], self.record['expected_signature'])

    def test_both_complete_python_while_bodies(self):
        results = check_manifest(ROOT, self.directory / 'jit-correspondence.json')
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
        lean = '''def body := with_triton_signature profile in triton {
    marker = 'False#} {'
    # } This comment contains a delimiter.
    result = marker
}
noncomputable def following := 0
'''
        body = quotation(lean, 'body', raw_python=True)[1]
        parsed = ast.parse(textwrap.dedent(body))
        self.assertEqual(len(parsed.body), 2)
        self.assertEqual(parsed.body[0].value.value, 'False#} {')
        self.assertNotIn('following', body)

    def test_noncomputable_next_declaration_is_not_part_of_kernel(self):
        quoted = quotation(self.lean, 'fmaBody', raw_python=True)[1]
        self.assertEqual(len(ast.parse(textwrap.dedent(quoted)).body), 18)
        self.assertNotIn('arguments', quoted)

    def test_scoped_options_before_next_definition_bound_the_quote(self):
        lean = "def body := triton {\n x = 1\n}\nset_option maxHeartbeats 1000000 in\ndef next := 0\n"
        body = quotation(lean, "body", raw_python=True)[1]
        self.assertEqual(len(ast.parse(textwrap.dedent(body)).body), 1)
        with self.assertRaisesRegex(ValueError, "unexpected code after quotation"):
            quotation(lean.replace("}\nset_option", "} + extra\nset_option"), "body", raw_python=True)

    def test_executable_suffix_after_quote_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unexpected code after quotation'):
            quotation('def body := triton {\n x = 1\n} + extra\nnoncomputable def next := 0\n',
                      'body', raw_python=True)


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
        self.lean = (self.directory / "Sources/Kernels.lean").read_text()
        self.record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]

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
        self.lean = (self.directory / "Sources/Kernels.lean").read_text()
        self.record = json.loads((self.directory / "jit-correspondence.json").read_text())[1]

    def compare(self, lean):
        return check(self.python, lean, self.record["python_function"], self.record["lean_definition"],
                     self.record["signature_definition"], self.record["expected_signature"])

    def test_complete_repaired_recurrence(self):
        result = self.compare(self.lean)
        self.assertEqual(result["mode"], "pasted_body")
        self.assertEqual(result["statements"], 10)

    def test_seed_signed_loop_endpoint_and_dtype_casts_are_checked(self):
        body = quotation(self.lean, "hBody", raw_python=True)[1]
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
        self.record = json.loads((self.directory / 'jit-correspondence.json').read_text())[0]
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
        self.record = json.loads((directory / 'jit-correspondence.json').read_text())[0]

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
                results = check_manifest(ROOT, ROOT / 'bench/tritonbench_g' / case / 'jit-correspondence.json')
                self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))


class ReductionBodyCopyTests(unittest.TestCase):
    def compare(self, case, lean):
        directory = ROOT / 'bench/tritonbench_g' / case
        record = json.loads((directory / 'jit-correspondence.json').read_text())[0]
        return check((ROOT / record['python']).read_text(), lean,
                     record['python_function'], record['lean_definition'],
                     record['signature_definition'], record['expected_signature'],
                     record.get('expected_call_bindings'))

    def test_all_selected_and_candidate_reduction_bodies(self):
        for case in ('triton_mul2', 'log_softmax', 'mean_reduction', 'max_reduction'):
            with self.subTest(case=case):
                results = check_manifest(ROOT, ROOT / 'bench/tritonbench_g' / case / 'jit-correspondence.json')
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
                results = check_manifest(ROOT, Path('bench/examples') / name / 'jit-correspondence.json')
                self.assertTrue(results)
                self.assertTrue(all(r['mode'] == 'pasted_body' for r in results))

    def test_runtime_fp32_binding_value_and_abi_remain_reviewed(self):
        from scripts.bench.jit_correspondence import check
        import json
        records = json.loads((ROOT / 'bench/examples/HyperConnectionsDepth/jit-correspondence.json').read_text())
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
