import ast
import json
import textwrap
from pathlib import Path
import unittest

from bench.jit_correspondence import check, check_manifest, quotation, call_signature

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "bench/repairs/tritonbench_g/triton_argmax_nan_first"


class JITCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.python = (DIRECTORY / "triton_argmax_nan_first.py").read_text()
        self.lean = (DIRECTORY / "Kernels.lean").read_text()

    def test_all_original_and_candidate_bodies(self):
        self.assertEqual(len(check_manifest(ROOT, DIRECTORY / "jit-correspondence.json")), 5)

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
        self.directory = ROOT / "bench/repairs/tritonbench_g/dequantize_rowwise"
        self.python = (self.directory / "dequantize_rowwise_repaired.py").read_text()
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


class BodyMigrationBatchTests(unittest.TestCase):
    def test_all_original_and_candidate_pairs(self):
        directories = [ROOT / "bench/examples" / name for name in
                       ("VectorAdd", "FlatVectorAdd", "TritonBenchVectorAddition")]
        directories.append(ROOT / "bench/repairs/tritonbench_g/cache_transform_triton")
        for directory in directories:
            with self.subTest(directory=directory.name):
                pairs = check_manifest(ROOT, directory / "jit-correspondence.json")
                self.assertEqual(len(pairs), 2)
                self.assertTrue(all(pair["mode"] == "pasted_body" for pair in pairs))

    def test_byte_indices_casts_and_mask_stay_in_the_body(self):
        directory = ROOT / "bench/repairs/tritonbench_g/cache_transform_triton"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        python = (ROOT / record["python"]).read_text()
        lean = (ROOT / record["lean"]).read_text()
        for before, after in ((".to(tl.uint64)", ""), ("mask=active", "mask=True")):
            with self.subTest(before=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after, 1), record["python_function"], "body",
                      "callSignature", record["expected_signature"])

    def test_real_example_retains_its_runtime_integer_abi(self):
        directory = ROOT / "bench/examples/TritonBenchVectorAddition"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        python = (ROOT / record["python"]).read_text().replace("n_elements: tl.int32", "n_elements: tl.constexpr", 1)
        with self.assertRaisesRegex(ValueError, "parameters and call signature"):
            check(python, (ROOT / record["lean"]).read_text(), record["python_function"],
                  record["lean_definition"], "callSignature", record["expected_signature"],
                  record["expected_call_bindings"])


class IntegerCallBodyTests(unittest.TestCase):
    tasks = ("add_example", "add_value", "vector_addition", "vector_addition_custom",
             "cosine_compute", "sin_computation", "sin_kernel", "square_matrix",
             "matrix_transpose", "index_select_bwd", "swiglu_fwd", "kldiv_compute", "l2_norm_triton1",
             "rmsnorm_fused", "rmsnorm_fused_llama", "rmsnorm_triton", "rmsnorm_implementation")

    def test_every_migrated_original_and_candidate(self):
        for task in self.tasks:
            with self.subTest(task=task):
                records = check_manifest(ROOT, ROOT / "bench/repairs/tritonbench_g" / task / "jit-correspondence.json")
                self.assertTrue(all(r["mode"] == "pasted_body" for r in records))
                self.assertTrue(all(r["argument_mode"] in
                                    ("integer_per_call_specialization", "mixed_integer_calls_and_runtime_registers",
                         "typed_per_call_specialization")
                                    for r in records))

    def test_spec_derived_example_bodies_remain_exact(self):
        references = json.loads((ROOT / "bench/examples/source-references.json").read_text())
        for name in ("LogExp", "SoftmaxReciprocal", "FloatDTypeSoftmax", "RowWiseSum",
                     "RowWiseMax", "FloatDTypeAdd", "SoftmaxStable", "StableLogSumExp",
                     "Welford", "FusedLayerNorm", "OnlineSoftmax"):
            with self.subTest(example=name):
                self.assertEqual(references[name]["provenance"], "spec_derived_native_source")
                records = check_manifest(ROOT, ROOT / "bench/examples" / name / "jit-correspondence.json")
                self.assertTrue(all(r["mode"] == "pasted_body" for r in records))

    def test_wrong_missing_and_unreviewed_argument_binding_rejected(self):
        directory = ROOT / "bench/repairs/tritonbench_g/square_matrix"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        python, lean = ((ROOT / record[k]).read_text() for k in ("python", "lean"))
        for before, after in (("n_cols := cols", "n_cols := 1"),
                              (", n_cols := cols", ""),
                              ("input_row_stride := cols", "input_row_stride := cols, input_row_stride := cols"),
                              ('("n_cols",.i32)', '("n_cols",.integer)')):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])
        with self.assertRaisesRegex(ValueError, "reviewed invocation"):
            check(python, lean, record["python_function"], record["lean_definition"],
                  record["signature_definition"], record["expected_signature"])

    def test_float_argument_stays_a_typed_runtime_register(self):
        directory = ROOT / "bench/repairs/tritonbench_g/l2_norm_triton1"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        result = check_manifest(ROOT, directory / "jit-correspondence.json")[0]
        self.assertEqual(result["argument_mode"], "mixed_integer_calls_and_runtime_registers")
        self.assertEqual(result["runtime_parameters"]["eps"], "fp32")
        self.assertNotIn("eps", result["call_bindings"])
        python, lean = ((ROOT / record[k]).read_text() for k in ("python", "lean"))
        for before, after in (('readPorts := ["X"]', 'readPorts := ["Y"]'),
                              ("var + eps", "var + eps.to(tl.float32)"),
                              ("N := N", "N := N, eps := 1")):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])

    def test_reviewed_fp32_literal_call_keeps_its_runtime_abi(self):
        directory = ROOT / "bench/repairs/tritonbench_g/rmsnorm_fused_llama"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        python, lean = ((ROOT / record[k]).read_text() for k in ("python", "lean"))
        bindings = {**record["expected_call_bindings"], "eps": "eps"}
        result = check(python, lean, record["python_function"], "kernel",
                       record["signature_definition"], record["expected_signature"], bindings)
        self.assertEqual(result["argument_mode"], "typed_per_call_specialization")
        self.assertEqual(result["runtime_parameters"]["eps"], "fp32")
        with self.assertRaisesRegex(ValueError, "supported scalar kinds"):
            signature = {**record["expected_signature"], "scalars": [
                [n, "fp64" if n == "eps" else t] for n, t in record["expected_signature"]["scalars"]]}
            check(python, lean.replace('(\"eps\",.fp32)', '(\"eps\",.fp64)'),
                  record["python_function"], "kernel", record["signature_definition"], signature, bindings)

    def test_original_constexpr_epsilon_is_not_relabelled_runtime(self):
        directory = ROOT / "bench/repairs/tritonbench_g/rmsnorm_implementation"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        self.assertIn("eps", record["expected_signature"]["constexpr"])
        self.assertNotIn("eps", dict(record["expected_signature"]["scalars"]))
        python, lean = ((ROOT / record[k]).read_text() for k in ("python", "lean"))
        with self.assertRaisesRegex(ValueError, "parameters and call signature"):
            check(python.replace("eps: tl.constexpr", "eps"), lean,
                  record["python_function"], record["lean_definition"],
                  record["signature_definition"], record["expected_signature"],
                  record["expected_call_bindings"])

    def test_python_comments_and_docstring_cannot_terminate_or_inject_a_body(self):
        directory = ROOT / "bench/repairs/tritonbench_g/vector_addition_custom"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        python, lean = ((ROOT / record[k]).read_text() for k in ("python", "lean"))
        marker = '    """'
        self.assertIn(marker, python)
        # Put misleading Lean delimiters in Python metadata, in both sources.
        addition = '\n    } $ tl.region\ndef fake_definition:\nend fake_namespace\n    '
        python = python.replace(marker, marker + addition, 1)
        lean = lean.replace(marker, marker + addition, 1)
        result = check(python, lean, record["python_function"], record["lean_definition"],
                       record["signature_definition"], record["expected_signature"],
                       record["expected_call_bindings"])
        self.assertEqual(result["status"], "PASS")
        with self.assertRaises(ValueError):
            check(python, lean.replace("tl.store(C +", "tl.store(A +", 1),
                  record["python_function"], record["lean_definition"],
                  record["signature_definition"], record["expected_signature"],
                  record["expected_call_bindings"])



class SeededCallABI(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/repairs/tritonbench_g/seeded_dropout"
        self.python = (self.directory / "seeded_dropout_repaired.py").read_text()
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.records = json.loads((self.directory / "jit-correspondence.json").read_text())

    def compare(self, record, python=None, lean=None):
        return check(self.python if python is None else python,
                     self.lean if lean is None else lean,
                     record["python_function"], record["lean_definition"],
                     record["signature_definition"], record["expected_signature"],
                     record["expected_call_bindings"])

    def test_all_selected_seed_abis_retain_the_same_nine_statements(self):
        results = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([r["statements"] for r in results], [9]*6)
        self.assertEqual({r["runtime_parameters"]["seed"] for r in results},
                         {"i32", "i64", "u64"})

    def test_unsigned_annotation_and_reviewed_signedness_are_checked(self):
        record = next(r for r in self.records if r["lean_definition"] == "sourceU64")
        annotated = self.python.replace("    seed,", "    seed: tl.uint64,", 1)
        self.assertEqual(self.compare(record, python=annotated)["status"], "PASS")
        with self.assertRaises(ValueError):
            self.compare(record, python=annotated.replace("tl.uint64", "tl.int64", 1))
        with self.assertRaises(ValueError):
            self.compare(record, lean=self.lean.replace('("seed",.u64)', '("seed",.i64)', 1))

    def test_seed_binding_and_original_store_cannot_disappear(self):
        record = self.records[0]
        for before, after in [("seed := seedValue", ""),
                              ("output, mask=mask)", "(output).to(tl.float32), mask=mask)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(record, lean=self.lean.replace(before, after, 1))


class ScalarCallModeCorrespondence(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/repairs/tritonbench_g/rmsnorm_fused_llama"
        self.records = json.loads((self.directory / "jit-correspondence.json").read_text())

    def test_runtime_and_literal_argument_forms_check_every_body(self):
        results = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(results), 4)
        self.assertEqual({r["lean_definition"] for r in results},
                         {"kernel", "runtimeKernel", "indexCommutedKernel", "runtimeCandidate"})
        self.assertEqual({"eps" in r["call_bindings"] for r in results}, {True, False})

    def test_renaming_bindings_does_not_allow_duplicate_interfaces(self):
        from bench.jit_correspondence import validate_manifest_records
        record = next(r for r in self.records if r["lean_definition"] == "runtimeKernel")
        duplicate = json.loads(json.dumps(record))
        duplicate["lean_definition"] = "renamed_copy"
        duplicate["profile_variant"] = "renamed_alias"
        duplicate["expected_call_bindings"] = {k: "another_alias" for k in record["expected_call_bindings"]}
        with self.assertRaisesRegex(ValueError, "duplicate ABI"):
            validate_manifest_records([record, duplicate])

if __name__ == "__main__":
    unittest.main()


class CumsumRawBodyCorrespondence(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/chunk_cumsum_kernel"

    def test_both_complete_signed_loops_are_raw_bodies(self):
        results = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([r["statements"] for r in results], [3, 3])
        self.assertTrue(all(r["mode"] == "pasted_body" for r in results))

    def test_source_casts_and_output_pointer_attribute_cannot_be_removed(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        lean = (self.directory / "Kernels.lean").read_text()
        python = (self.directory / "chunk_cumsum_kernel_repaired.py").read_text()
        record = records[0]
        for before, after in [("tl.cast(0, tl.int64)", "0"),
                              ("tl.cast(tl.cdiv(T, BT), tl.int64)", "tl.cdiv(T, BT)"),
                              ("b_o.to(p_o.dtype.element_ty)", "b_o")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"], record["expected_signature"])

    def test_typed_storage_is_part_of_the_reviewed_call_metadata(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        lean = (self.directory / "Kernels.lean").read_text().replace(
            "typedFP32Storage := Bool.true", "typedFP32Storage := Bool.false", 1)
        with self.assertRaisesRegex(ValueError, "reviewed launch types"):
            check((self.directory / "chunk_cumsum_kernel_repaired.py").read_text(), lean,
                  record["python_function"], record["lean_definition"],
                  record["signature_definition"], record["expected_signature"])


class LogsumexpRawBodyCorrespondence(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/logsumexp_fwd"

    def test_both_existing_complete_functions(self):
        results = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([r["statements"] for r in results], [11, 10])
        self.assertTrue(all(r["mode"] == "pasted_body" for r in results))

    def test_literal_cast_and_mask_restoration_cannot_be_replaced(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        lean = (self.directory / "Kernels.lean").read_text()
        python = (self.directory / "logsumexp_fwd_repaired.py").read_text()
        for before, after in [("positive_inf = float('inf')", "positive_inf = tl.bitcast(tl.full([], 2139095040, dtype=tl.uint32), tl.float32)"),
                              ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                              ("    b_x = tl.where(m_d, b_x, -float('inf'))", "")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"], record["expected_signature"])


class HgrnRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/fused_recurrent_hgrn"

    def test_three_source_pairs_and_shared_candidate_backward(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        self.assertEqual(len(check_manifest(ROOT, self.directory / "jit-correspondence.json")), 3)
        record = records[1]
        result = check((self.directory / "fused_recurrent_hgrn_stride_folded.py").read_text(),
                       (self.directory / "Kernels.lean").read_text(),
                       record["python_function"], record["lean_definition"],
                       record["signature_definition"], record["expected_signature"])
        self.assertEqual(result["mode"], "pasted_body")

    def test_counter_cast_seed_and_reverse_branch_cannot_drift(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        lean = (self.directory / "Kernels.lean").read_text()
        for index, before, after in [
            (0, "for _ in range(0, T)", "for _i in range(0, T)"),
            (0, "b_h += tl.load(p_h0", "b_h = tl.load(p_h0"),
            (0, "b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)",
                "b_x = tl.load(p_x, mask=mask, other=0)"),
            (1, "elif USE_INITIAL_STATE:", "elif USE_FINAL_GRAD:"),
            (1, "p_do = do +", "p_do = o +"),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                record = records[index]
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"])


class SwiGluBackwardAndReluBodyTests(unittest.TestCase):
    def test_all_selected_bodies_and_existing_candidates(self):
        for name in ("swiglu_backward", "relu_triton_kernel"):
            with self.subTest(task=name):
                records = check_manifest(ROOT, ROOT / "bench/repairs/tritonbench_g" / name / "jit-correspondence.json")
                self.assertEqual(len(records), 2)
                self.assertTrue(all(r["mode"] == "pasted_body" for r in records))

    def test_optional_recomputation_order_and_tail_mask_stay_exact(self):
        directory = ROOT / "bench/repairs/tritonbench_g/swiglu_backward"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        python = (ROOT / record["python"]).read_text()
        lean = (ROOT / record["lean"]).read_text()
        for before, after in (("if RECOMPUTE_OUTPUT:", "if True:"),
                              ("x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout",
                               "dout * y * x_sigmoid * (1 + x * (1 - x_sigmoid))"),
                              ("mask=cols < ncols", "mask=cols <= ncols")):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])


class ChunkedCumsumRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/chunked_cumsum_fwd"

    def test_original_and_existing_scale_first_candidate(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["mode"] == "pasted_body" for pair in pairs))

    def test_mixed_storage_channels_and_unsigned_casts_are_pinned(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        python = (ROOT / record["python"]).read_text()
        lean = (ROOT / record["lean"]).read_text()
        for before, after in [('typedFP32Pointers := ["dt_ptr", "A_ptr"]',
                               'typedFP32Pointers := ["dt_ptr"]'),
                              ('seqlen = tl.cast(seqlen, tl.uint64)', 'seqlen = seqlen'),
                              ('dt_min), dt_max)', 'dt_max), dt_min)'),
                              ('if DT_SOFTPLUS:', 'if True:')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])

    def test_selective_storage_metadata_rejects_expressions_and_invalid_names(self):
        lean = (self.directory / "Kernels.lean").read_text()
        original = 'typedFP32Pointers := ["dt_ptr", "A_ptr"]'
        for replacement in ['typedFP32Pointers := selected',
                            'typedFP32Pointers := ["dt_ptr"] ++ ["A_ptr"]',
                            'typedFP32Pointers := ["missing"]',
                            'typedFP32Pointers := ["dt_ptr", "dt_ptr"]',
                            'typedFP32Pointers := [1]']:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                call_signature(lean.replace(original, replacement, 1), "callSignature")


class MatmulRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/matmul_kernel"

    def test_both_existing_complete_functions(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p["mode"] == "pasted_body" for p in pairs))

    def test_live_dot_tail_loop_and_narrowing_cannot_drift(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        lean = (self.directory / "Kernels.lean").read_text()
        for index, before, after in [
            (0, "tl.dot(a, b, accumulator)", "tl.dot(a, b) + accumulator"),
            (0, "tl.cdiv(K, BLOCK_SIZE_K)", "K // BLOCK_SIZE_K"),
            (0, "tl.cast(accumulator, tl.float16)", "accumulator"),
            (0, "tl.store(c_ptrs, c,", "tl.store(c_ptrs, tl.cast(c, tl.float32),"),
            (1, "offs_k[None, :] < K - k * BLOCK_SIZE_K", "offs_k[None, :] < K"),
        ]:
            record = records[index]
            with self.subTest(before=before), self.assertRaises(ValueError):
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"],
                      record["expected_call_bindings"])


class FusedActivationRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/fused_activation"

    def test_both_existing_functions_keep_their_string_comparisons(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p["mode"] == "pasted_body" for p in pairs))

    def test_string_branches_and_operation_order_cannot_be_replaced(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        lean = (self.directory / "Kernels.lean").read_text()
        for before, after in [('if activation == "sigmoid":', 'if useSigmoid:'),
                              ('elif activation == "relu":', 'else:'),
                              ('multiplier * tmp3 + tmp0 + tmp1', 'tmp1 + (multiplier * tmp3 + tmp0)'),
                              ('tl.arange(0, BLOCK_SIZE)[:]', 'tl.arange(0, BLOCK_SIZE)')]:
            record = records[0]
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"])
