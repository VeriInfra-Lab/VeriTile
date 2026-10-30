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
                self.assertEqual(result[0]["mode"], "legacy_ast")
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
            for original, changed in ((before, after), (".to(tl.uint64)", "")):
                with self.subTest(task=task, mutation=original):
                    self.assertIn(original, lean)
                    # Preserve the compiler graph and mutate only the new
                    # Python constructor, so the selected source must reject it.
                    prefix, body = lean.split("def pythonBody", 1)
                    with self.assertRaises(ValueError):
                        check(python, prefix + "def pythonBody" + body.replace(original, changed, 1),
                              "matmul_kernel", "pythonBody")

    def test_zero_shape_normalization_preserves_dtype_and_dimensions(self):
        directory = ROOT / "bench/tritonbench_g/matmul_triton1"
        python = (directory / "matmul_triton1.py").read_text()
        lean = (directory / "Kernels.lean").read_text()
        for before, after in (("[m_block_size, n_block_size]", "[n_block_size, m_block_size]"),
                              ("dtype=tl.float32", "dtype=tl.float16")):
            prefix, body = lean.split("def pythonBody", 1)
            with self.subTest(mutation=before), self.assertRaises(ValueError):
                check(python, prefix + "def pythonBody" + body.replace(before, after, 1),
                      "matmul_kernel", "pythonBody")


if __name__ == "__main__":
    unittest.main()
