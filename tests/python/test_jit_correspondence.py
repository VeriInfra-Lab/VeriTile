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
        for statement in ["tl.region x_ptr = tl.int8", "tl.param inv_127 : tl.float32", "a = $(P2)"]:
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
