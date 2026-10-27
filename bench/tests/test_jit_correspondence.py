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
        self.record = json.loads((DIRECTORY / "jit-correspondence.json").read_text())[0]

    def compare(self, lean, python=None, dtype=None):
        r = self.record
        return check(self.python if python is None else python, lean,
                     r["python_function"], r["lean_definition"],
                     r["signature_definition"], r["expected_signature"],
                     expected_source_dtype=r["expected_source_dtype"] if dtype is None else dtype)

    def test_all_original_and_candidate_bodies(self):
        pairs = check_manifest(ROOT, DIRECTORY / "jit-correspondence.json")
        self.assertEqual(len(pairs), 5)
        self.assertTrue(all(p["mode"] == "pasted_body" for p in pairs))
        self.assertEqual([p["statements"] for p in pairs], [20, 9, 12, 20, 12])

    def test_cast_nan_clamp_store_and_parameter_changes_are_rejected(self):
        mutations = [
            ("pid.to(tl.int64)", "pid"),
            ("tl.maximum(stored_value,", "tl.minimum(stored_value,"),
            ("tl.where(max_val == max_val, max_val, 0.0)", "max_val"),
            ("tl.store(mid_value_ptr, stored_value)", "tl.store(mid_value_ptr, max_val)"),
            ('("M",.i64)', '("M",.fp32)'),
            ("mask = offset < M", "mask = offset <= M"),
            ("8 if is_bool else", "16 if is_bool else"),
            ("width - 1 if signed else width", "width"),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.compare(self.lean.replace(before, after, 1))

    def test_jit_decorator_is_required(self):
        with self.assertRaisesRegex(ValueError, "expected one explicitly selected"):
            self.compare(self.lean, self.python.replace("@triton.jit", "", 1))

    def test_runtime_parameter_cannot_be_replaced_by_antiquote(self):
        with self.assertRaisesRegex(ValueError, "metadata directives or Lean antiquotes"):
            self.compare(self.lean.replace("offset < M", "offset < $(M)", 1))

    def test_unsupported_splice_cannot_hide_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "metadata directives or Lean antiquotes"):
            self.compare(self.lean.replace("* BLOCK_SIZE +", "* $(BLOCK_SIZE + 1) +", 1))

    def test_source_dtype_binding_and_pointer_declaration_are_checked(self):
        for before, after in [
            ("source_dtype mid_value := MID_DTYPE", "source_dtype inp := MID_DTYPE"),
            ("source_dtype mid_value := MID_DTYPE", "source_dtype mid_value := SourceDType.fp32"),
            ('sourceDTypePointers := ["mid_value"]', 'sourceDTypePointers := ["inp"]'),
            ('sourceDTypePointers := ["mid_value"]', 'sourceDTypePointers := ["mid_value","mid_value"]'),
            ('sourceDTypePointers := ["mid_value"]', 'sourceDTypePointers := []'),
        ]:
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))
        with self.assertRaises(ValueError):
            self.compare(self.lean, dtype={"pointer": "mid_value", "value": "OtherType"})

    def test_typed_constexpr_and_integer_runtime_annotations_stay_original(self):
        with self.assertRaisesRegex(ValueError, "annotation"):
            self.compare(self.lean, self.python.replace("M: tl.int64", "M: tl.int32", 1))
        self.assertIn("width: tl.constexpr = 8 if is_bool else", self.lean)
        self.assertNotIn("tl.region", quotation(self.lean, "stage1")[1])

    def test_bool_and_integer_scratch_branches_cannot_be_dropped(self):
        for before, after in [("mid_value.dtype.element_ty.is_int()", "mid_value.dtype.element_ty.is_int1()"),
                              ("is_bool or mid_value.dtype.element_ty.is_int_signed()", "False"),
                              ("(1 << exponent) - gap", "(1 << exponent)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))


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


class SourceFunctionBodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/repairs/tritonbench_g/relu_strided_buffer"
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.python = (self.directory / "relu_strided_buffer_repaired.py").read_text()
        self.manifest = self.directory / "jit-correspondence.json"

    def test_helper_and_both_complete_branches(self):
        pairs = check_manifest(ROOT, self.manifest)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["body_kind"], "helper")
        self.assertEqual(pairs[0]["helper_parameters"], ["x"])
        for pair in pairs[1:]:
            self.assertEqual(pair["source_functions"], ["reluFunction"])

    def test_helper_source_and_parameter_names_cannot_change(self):
        for before, after in [("relu_forward(x)", "relu_forward(y)"),
                              ("return tl.where(x > 0, x, 0)", "return tl.where(x >= 0, x, 0)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                check(self.python, self.lean.replace(before, after, 1),
                      "relu_forward", "reluFunction")

    def test_helper_pair_cannot_be_omitted_from_manifest(self):
        import tempfile
        records = json.loads(self.manifest.read_text())[1:]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as temporary:
            json.dump(records, temporary)
            temporary.flush()
            with self.assertRaisesRegex(ValueError, "helper"):
                check_manifest(ROOT, Path(temporary.name))

    def test_original_casts_tuples_and_boundary_checks_are_checked(self):
        record = json.loads(self.manifest.read_text())[1]
        for before, after in [(".to(in0_ptr.type.element_ty)", ""),
                              ("boundary_check=(in0_stride_order0,)", "boundary_check=(1,)"),
                              ("(in0_stride0,)", "(1,)"),
                              ("num_tasks := c.n", "num_tasks := c.tiles")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                check(self.python, self.lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"],
                      expected_call_bindings=record["expected_call_bindings"],
                      expected_pointer_offsets=record["expected_pointer_offsets"])


class SwigluSourceFunctionTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/repairs/tritonbench_g/swiglu_triton"
        self.lean = (self.directory / "Kernels.lean").read_text()
        self.python = (self.directory / "swiglu_triton_repaired.py").read_text()
        self.manifest = self.directory / "jit-correspondence.json"
        self.records = json.loads(self.manifest.read_text())

    def compare(self, lean, position):
        record = self.records[position]
        return check(self.python, lean, record["python_function"], record["lean_definition"],
                     record["signature_definition"], record["expected_signature"],
                     expected_call_bindings=record["expected_call_bindings"])

    def test_original_helper_and_all_four_complete_writers(self):
        pairs = check_manifest(ROOT, self.manifest)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(pairs[0]["body_kind"], "helper")
        self.assertEqual(pairs[1]["source_functions"], ["siluFunction"])
        self.assertEqual(pairs[3]["source_functions"], ["siluFunction"])

    def test_same_named_shorthand_cannot_hide_changed_helper_source(self):
        changed = self.lean.replace("return x * tl.sigmoid(x)", "return x", 1)
        with self.assertRaises(ValueError):
            check(self.python, changed, "silu", "siluFunction")

    def test_original_casts_and_integer_products_are_retained(self):
        for before, after in [(".to(tl.int64)", ""),
                              ("a_ptr += program_id * stride", "a_ptr += stride * program_id")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, self.lean)
                self.compare(self.lean.replace(before, after, 1), 1)

    def test_backward_floating_order_is_retained(self):
        before = "dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row"
        after = "(dc_row * b_row) * (silu_a * (1 - sig_a) + sig_a)"
        self.assertIn(before, self.lean)
        with self.assertRaises(ValueError):
            self.compare(self.lean.replace(before, after, 1), 2)

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


class EmbeddingRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/embedding_triton_kernel"

    def test_both_existing_signed_index_functions(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p["mode"] == "pasted_body" for p in pairs))

    def test_each_widening_and_mask_order_stays_in_the_source(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        lean = (self.directory / "Kernels.lean").read_text()
        for before, after in [
            ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
            ("tl.arange(0, BLOCK_NN).to(tl.int64)", "tl.arange(0, BLOCK_NN)"),
            ("tl.arange(0, BLOCK_DMODEL).to(tl.int64)", "tl.arange(0, BLOCK_DMODEL)"),
            ("other=0).to(tl.int64)", "other=0)"),
            ("tl.where(id_mask, token_ids, vob_start_id) - vob_start_id",
             "tl.where(id_mask, token_ids - vob_start_id, 0)"),
            ('("n_ctx",.i64)', '("n_ctx",.integer)'),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"],
                      record["expected_call_bindings"])


class MaxReductionRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/max_reduction"

    def test_three_originals_and_three_retained_candidates_are_raw_bodies(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(p["mode"] == "pasted_body" for p in pairs))
        self.assertEqual(sorted(p["statements"] for p in pairs), [6, 6, 8, 8, 15, 15])

    def test_widenings_paired_reduction_masks_and_abi_cannot_drift(self):
        records = {r["lean_definition"]: r for r in json.loads(
            (self.directory / "jit-correspondence.json").read_text())}
        lean = (self.directory / "Kernels.lean").read_text()
        mutations = [
            ("stage1", "tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
            ("stage1", "tl.arange(0, BLOCK_SIZE).to(tl.int64)", "tl.arange(0, BLOCK_SIZE)"),
            ("stage1", '("M",.i32)', '("M",.integer)'),
            ("stage2", "tl.arange(0, BLOCK_MID).to(tl.int64)", "tl.arange(0, BLOCK_MID)"),
            ("stage2Complement", "not (mid_size <= offset)", "mid_size < offset"),
            ("dimension", "tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
            ("dimension", "tl.arange(0, BLOCK_M).to(tl.int64)", "tl.arange(0, BLOCK_M)"),
            ("dimension", "tl.arange(0, BLOCK_N).to(tl.int64)", "tl.arange(0, BLOCK_N)"),
            ("dimension", "tl.cast(N, tl.int64)", "N"),
            ("dimension", "tl.cast(K, tl.int64)", "K"),
            ("dimension", "return_indices=True", "return_indices=False"),
            ("dimension", "tl.store(out_index_ptrs, result_index, mask=mask1)",
             "tl.store(out_index_ptrs, result_value, mask=mask1)"),
            ("dimension", '("out_index",.i64)', '("out_index",.i32)'),
        ]
        for definition, before, after in mutations:
            record = records[definition]
            with self.subTest(definition=definition, before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], definition, record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])


class MaskedSelectRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/masked_select"

    def test_complete_original_body_and_natural_prefix_storage_view(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["mode"], pairs[0]["statements"]), ("pasted_body", 7))
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        lean = (self.directory / "Kernels.lean").read_text()
        for before, after in [
            ('naturalIntegerPointers := ["prefix_sum_ptr"]', 'naturalIntegerPointers := []'),
            ('("prefix_sum_ptr",.i64)', '("prefix_sum_ptr",.u64)'),
            ('.to(tl.int1)', ''),
            ('other=0.0) - 1', 'other=0.0) - 0'),
            ('mask=(select_mask and mask)', 'mask=(mask and select_mask)'),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])

    def test_natural_integer_metadata_rejects_unreviewed_expressions_and_noninteger_names(self):
        lean = (self.directory / "Kernels.lean").read_text()
        before = 'naturalIntegerPointers := ["prefix_sum_ptr"]'
        for replacement in [
            'naturalIntegerPointers := names',
            'naturalIntegerPointers := ["prefix_sum_ptr"] ++ more',
            'naturalIntegerPointers := ["inp_ptr"]',
            'naturalIntegerPointers := ["missing"]',
            'naturalIntegerPointers := ["prefix_sum_ptr", "prefix_sum_ptr"]',
        ]:
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "natural integer"):
                call_signature(lean.replace(before, replacement), "callSignature")


class Int8AsmRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/int8_matmul_kernel_modular_checked"

    def test_original_body_keeps_assembly_and_its_complete_abi(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["mode"], pairs[0]["statements"]), ("pasted_body", 23))

    def test_assembly_cast_shape_and_call_metadata_cannot_drift(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        lean = (self.directory / "Kernels.lean").read_text()
        for before, after in [
            ('"add.u32 $0, $1, $2;"', '"sub.u32 $0, $1, $2;"'),
            ('constraints="=r,r,r"', 'constraints="=r,l,l"'),
            ('args=[accumulator, partial]', 'args=[partial, accumulator]'),
            ('is_pure=True', 'is_pure=False'),
            ('pack=1', 'pack=2'),
            ('.to(tl.int8)', ''),
            ('tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N)', 'tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N]'),
            ('partial = tl.dot', 'block_sum = tl.dot'),
            ('("b_ptr",.u8)', '("b_ptr",.i8)'),
            ('stride_am := stride_am', 'stride_am := stride_ak'),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])


class TokenSoftmaxRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/token_softmax_llama"

    def test_both_independent_originals_match_the_shared_source(self):
        for name in ["token_softmax_llama", "token_softmax_bloom"]:
            with self.subTest(task=name):
                pairs = check_manifest(ROOT, ROOT / "bench/repairs/tritonbench_g" / name / "jit-correspondence.json")
                self.assertEqual(len(pairs), 2)
                self.assertTrue(all((p["mode"], p["statements"]) == ("pasted_body", 11) for p in pairs))

    def test_all_source_casts_mask_and_floating_family_are_pinned(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        lean = (self.directory / "Kernels.lean").read_text()
        mutations = [
            ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
            ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
            ("tl.arange(0, BLOCK_SIZE).to(tl.int64)", "tl.arange(0, BLOCK_SIZE)"),
            ("cur_batch * stride_length).to(tl.int64)", "cur_batch * stride_length)"),
            ("cur_batch * stride_start).to(tl.int64)", "cur_batch * stride_start)"),
            ('("Logics",.inputFloat)', '("Logics",.fp32)'),
            ('("B_Seqlen",.i32)', '("B_Seqlen",.i64)'),
            ("mask=col_offsets < cur_batch_seq_len", "mask=col_offsets <= cur_batch_seq_len"),
            ("stride_prob_bs := stride_prob_bs", "stride_prob_bs := stride_logic_bs"),
        ]
        for before, after in mutations:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])


class ReversedCumsumRawBodyTests(unittest.TestCase):
    def test_both_full_selected_variants_and_all_widenings_are_preserved(self):
        for task, statements in [("reversed_cumsum_scalar", 3), ("reversed_cumsum", 5)]:
            directory = ROOT / "bench/repairs/tritonbench_g" / task
            records = json.loads((directory / "jit-correspondence.json").read_text())
            pairs = check_manifest(ROOT, directory / "jit-correspondence.json")
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all((p["mode"], p["statements"]) == ("pasted_body", statements) for p in pairs))
            lean = (directory / "Kernels.lean").read_text()
            for record in records:
                for before, after in [
                    ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                    ("tl.cast(tl.cdiv(T, BT), tl.int64)", "tl.cdiv(T, BT)"),
                    ("tl.cast(i_t, tl.int64)", "i_t"),
                    ("tl.arange(0, BT).to(tl.int64)", "tl.arange(0, BT)"),
                    ('("s",.inputFloat)', '("s",.fp32)'),
                ]:
                    with self.subTest(task=task, definition=record["lean_definition"], before=before), self.assertRaises(ValueError):
                        self.assertIn(before, lean)
                        # Mutate both quotations so the selected definition is changed too.
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"], record["signature_definition"],
                              record["expected_signature"])

    def test_vector_both_program_axes_and_feature_widening_remain_distinct(self):
        directory = ROOT / "bench/repairs/tritonbench_g/reversed_cumsum"
        record = json.loads((directory / "jit-correspondence.json").read_text())[0]
        lean = (directory / "Kernels.lean").read_text()
        for before, after in [
            ("tl.program_id(1).to(tl.int64)", "tl.program_id(0).to(tl.int64)"),
            ("tl.arange(0, BS).to(tl.int64)", "tl.arange(0, BS)"),
            ("b_z += tl.sum(b_s, 0)", "b_z = tl.sum(b_s, 0)"),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                      record["python_function"], record["lean_definition"], record["signature_definition"],
                      record["expected_signature"])


class ApplyPenaltyRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/apply_penalty"

    def test_both_original_and_reciprocal_keep_metadata_widenings_and_runtime_strides(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all((p["mode"], p["statements"]) == ("pasted_body", 17) for p in pairs))
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in [
                ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                ("tl.load(p_cumsum_seq_len + cur_batch).to(tl.int64)", "tl.load(p_cumsum_seq_len + cur_batch)"),
                ("tl.arange(0, BLOCK_P).to(tl.int64)", "tl.arange(0, BLOCK_P)"),
                ("other=0).to(tl.int64)", "other=0)"),
                ("tl.cast(stride_logit_b, tl.int64)", "stride_logit_b"),
                ('("p_token_counts",.i32)', '("p_token_counts",.i64)'),
                ("stride_logit_s := _stride_logit_s", "stride_logit_s := stride_logit_b"),
            ]:
                with self.subTest(definition=record["lean_definition"], before=before), self.assertRaises(ValueError):
                    self.assertIn(before, lean)
                    check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                          record["python_function"], record["lean_definition"], record["signature_definition"],
                          record["expected_signature"], record["expected_call_bindings"])


class ChunkCumsumVectorRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/chunk_cumsum_vector"

    def test_both_complete_bodies_preserve_signed_indices_and_fp_order(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all((p["mode"], p["statements"]) == ("pasted_body", 11) for p in pairs))
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in [
                ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
                ("tl.arange(0, BS).to(tl.int64)", "tl.arange(0, BS)"),
                ("tl.arange(0, BT).to(tl.int64)", "tl.arange(0, BT)"),
                ("tl.cast(tl.cdiv(T, BT), tl.int64)", "tl.cdiv(T, BT)"),
                ("tl.cast(i_t, tl.int64)", "i_t"),
                ("mask=active, other=0", "mask=active"),
                ("allow_tf32=False", "allow_tf32=True"),
                ("b_c.to(p_z.dtype.element_ty)", "b_c"),
                ('("s",.inputFloat)', '("s",.fp32)'),
            ]:
                with self.subTest(definition=record["lean_definition"], before=before), self.assertRaises(ValueError):
                    self.assertIn(before, lean)
                    check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                          record["python_function"], record["lean_definition"], record["signature_definition"],
                          record["expected_signature"])


class GatheredTokenAttentionRawBodyTests(unittest.TestCase):
    def test_complete_weighted_sum_bodies_bind_the_selected_fp32_calls(self):
        for task, statements in [("token_attn_reduceV", 19), ("token_attn_mistral", 20)]:
            directory = ROOT / "bench/repairs/tritonbench_g" / (task + "_grid_y_checked")
            records = json.loads((directory / "jit-correspondence.json").read_text())
            pairs = check_manifest(ROOT, directory / "jit-correspondence.json")
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all((p["mode"], p["statements"]) == ("pasted_body", statements) for p in pairs))
            self.assertEqual({r["lean_definition"] for r in records}, {"source", "affineSource"})
            for record in records:
                lean = (ROOT / record["lean"]).read_text()
                for before, after in [
                    ("start_n * stride_pbs", "start_n"),
                    ("acc += tl.sum(p_value[:, None] * v_value, 0)", "acc += tl.sum(p_value[:, None] * v_value, 1)"),
                    ("acc = acc.to(Out.dtype.element_ty)", "acc = acc"),
                    ('("Prob",.fp32)', '("Prob",.fp16)'),
                    ('("B_Seqlen",.i32)', '("B_Seqlen",.i64)'),
                    ("stride_vh := stride_vh", "stride_vh := stride_vbs"),
                ]:
                    with self.subTest(task=task, definition=record["lean_definition"], before=before), self.assertRaises(ValueError):
                        self.assertIn(before, lean)
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"], record["signature_definition"],
                              record["expected_signature"], record["expected_call_bindings"])

    def test_sliding_window_value_mask_retains_its_own_sequence_bound(self):
        directory = ROOT / "bench/repairs/tritonbench_g/token_attn_mistral_grid_y_checked"
        for record in json.loads((directory / "jit-correspondence.json").read_text()):
            lean = (ROOT / record["lean"]).read_text()
            before = "mask=(start_n + offs_n + cur_batch_start_index) < cur_batch_seq_len"
            self.assertIn(before, lean)
            with self.subTest(definition=record["lean_definition"]), self.assertRaises(ValueError):
                check((ROOT / record["python"]).read_text(),
                      lean.replace(before, "mask=(start_n + offs_n) < cur_att_seq_len"),
                      record["python_function"], record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])


class QuantizeGlobalPreciseBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/quantize_global"

    def test_full_body_keeps_precise_division_signed_casts_and_i64_arguments(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs], [("pasted_body", 13)])
        lean = (self.directory / "Kernels.lean").read_text()
        for before, after in [
            ("tl.program_id(axis=0).to(tl.int64)", "tl.program_id(axis=0)"),
            ("tl.arange(0, BLOCK_SIZE).to(tl.int64)", "tl.arange(0, BLOCK_SIZE)"),
            ("tl.div_rn(x, denominator)", "x / denominator"),
            ("scaled < -127.0", "scaled <= -127.0"),
            ('("n_elements",.i64)', '("n_elements",.integer)'),
            ("base := base", "base := N"),
            ("output, mask=mask", "output"),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                      record["python_function"], record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], record["expected_call_bindings"])


class RowwiseQuantizationRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/rowwise_quantization_triton"

    def test_both_bodies_keep_signed_coordinates_zero_type_and_store_order(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 12), ("pasted_body", 12)])
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in [
                ("tl.program_id(axis=0).to(tl.int64)", "tl.program_id(axis=0)"),
                ("tl.full((), BLOCK_SIZE, tl.int64)", "BLOCK_SIZE"),
                ("tl.where(row_mask, abs_x, 0)", "tl.where(row_mask, abs_x, 0.0)"),
                ("x / safe_max", "safe_max / x"),
                ("output, mask=row_mask", "output"),
                ("tl.store(output_maxs + pid, max_val)",
                 "tl.store(output_maxs + pid, max_val.to(tl.float16))"),
                ('("output_maxs",.fp16)', '("output_maxs",.fp32)'),
            ]:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"])


class QuantizedScatterRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/quantize_copy_kv"

    def test_both_bodies_preserve_half_scale_and_zero_maximum_branch(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 14), ("pasted_body", 14)])
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in [
                ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                ("(maximum / 127.0).to(tl.float16)", "(maximum / 127.0).to(tl.float32)"),
                ("data_scale.to(tl.float32)", "data_scale"),
                ("maximum[:, None] == 0.0", "data_scale == 0.0"),
                ("stride_dest := 1", "stride_dest := p.width"),
                ("stride_o_bs := denseStride", "stride_o_bs := p.width"),
                ("q_src_data, mask=offs_h[:, None] < head_num", "q_src_data"),
            ]:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"],
                              record["expected_call_bindings"])


class QuantizedCacheVariantsRawBodyTests(unittest.TestCase):
    def verify_variant(self, task, count, mutations):
        directory = ROOT / "bench/repairs/tritonbench_g" / task
        records = json.loads((directory / "jit-correspondence.json").read_text())
        pairs = check_manifest(ROOT, directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", count), ("pasted_body", count)])
        lean = (directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in mutations:
                with self.subTest(task=task, definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"],
                              record["expected_call_bindings"])

    def test_logical_width_mask_and_fp32_scale_dtype_access(self):
        self.verify_variant("quantize_kv_transform", 14, [
            ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
            ("tl.load(Dest_loc + cur_index * stride_dest).to(tl.int64)",
             "tl.load(Dest_loc + cur_index * stride_dest)"),
            ("(offs_d[None, :] < head_dim)", "(offs_d[None, :] < head_num)"),
            ("(maximum / 127.0).to(Out_scale.dtype.element_ty)", "(maximum / 127.0)"),
            ("data_scale.to(tl.float32)", "data_scale"),
            ("head_dim := p.columns", "head_dim := p.width"),
            ('("Dest_loc",.i32)', '("Dest_loc",.i64)'),
        ])

    def test_two_dimensional_group_geometry_and_scale_mask(self):
        self.verify_variant("quantize_kv_copy", 15, [
            ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
            ("tl.arange(0, BLOCK_GROUP_DIM).to(tl.int64)",
             "tl.arange(0, BLOCK_GROUP_DIM)"),
            ("(maximum / 127.0).to(Out_scale.dtype.element_ty)", "(maximum / 127.0)"),
            ("src_data / denominator[:, None]", "src_data / denominator"),
            ("data_scale, mask=offs_g < group_size", "data_scale"),
            ("stride_os_h := groups", "stride_os_h := heads"),
            ('("Dest_loc",.i32)', '("Dest_loc",.i64)'),
        ])


class LogSoftmaxRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/log_softmax"

    def test_forward_backward_and_existing_candidates_keep_signed_tuple_and_padding(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([p["mode"] for p in pairs], ["pasted_body"] * 4)
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            common = [
                ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
                ("tl.arange(0, BLOCK_N).to(tl.int64)", "tl.arange(0, BLOCK_N)"),
                ("tl.cast(M, tl.int64)", "M"),
                ("tl.cast(N, tl.int64)", "N"),
                ("tl.cast(K, tl.int64)", "K"),
                ("safe_n = tl.minimum(n_offset, n64 - 1)", "safe_n = n_offset"),
                ("N := N", "N := K"),
            ]
            if record["lean_definition"].startswith("backward"):
                common.append(("out_grad_ptr + offset, mask=mask, other=0.0",
                               "out_grad_ptr + offset, mask=mask"))
            for before, after in common:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"],
                              record["expected_call_bindings"])


class FusedDecodingRotaryRawBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/fused_rotary_embedding_tile_checked"

    def test_complete_q_k_cache_and_full_width_v_bodies(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 21), ("pasted_body", 21)])

    def test_cast_cache_mask_integer_channel_and_runtime_stride_drift_are_rejected(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in [
                ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
                ("tl.arange(0, BLOCK_HALF).to(tl.int64)", "tl.arange(0, BLOCK_HALF)"),
                ("handle_kv = cur_head_idx % KV_GROUP_NUM == 0", "handle_kv = cur_head_idx == 0"),
                ("dim_range < HEAD_DIM", "dim_range < BLOCK_HALF"),
                ("length_stride := length_stride", "length_stride := q_token_stride"),
                ('("context_lengths",.i32)', '("context_lengths",.i64)'),
            ]:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"],
                              record["expected_call_bindings"])


class RopeFunctionBodyTests(unittest.TestCase):
    def test_both_pitch_variants_keep_full_original_and_index_candidate(self):
        for task in ["rope_transform", "rope_backward_transform"]:
            directory = ROOT / "bench/repairs/tritonbench_g" / task
            pairs = check_manifest(ROOT, directory / "jit-correspondence.json")
            self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                             [("pasted_body", 23), ("pasted_body", 23)])

    def test_coefficient_casts_branches_padding_and_runtime_pitches_are_pinned(self):
        for task in ["rope_transform", "rope_backward_transform"]:
            directory = ROOT / "bench/repairs/tritonbench_g" / task
            records = json.loads((directory / "jit-correspondence.json").read_text())
            lean = (directory / "Kernels.lean").read_text()
            mutations = [
                (".to(sin_row.dtype)", ""),
                ("if not BACKWARD_PASS:", "if BACKWARD_PASS:"),
                ("tl.arange(0, pad_hd // 2)", "tl.arange(0, hd // 2)"),
                ("new_q_tile_2, mask=second_q_mask", "new_q_tile_2"),
                ('("q_ptr",.inputFloat)', '("q_ptr",.outputFloat)'),
                ("sl := sl", "sl := q_row_stride"),
            ]
            if task == "rope_backward_transform":
                mutations += [("cos_offsets * cos_col_stride", "cos_offsets"),
                              ("cos_offsets * sin_col_stride", "cos_offsets")]
            for record in records:
                for before, after in mutations:
                    with self.subTest(task=task, definition=record["lean_definition"], mutation=before):
                        self.assertIn(before, lean)
                        with self.assertRaises(ValueError):
                            check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                                  record["python_function"], record["lean_definition"],
                                  record["signature_definition"], record["expected_signature"],
                                  record["expected_call_bindings"])


class BatchedVecmatFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/batched_vecmat_mult"

    def test_complete_signed_index_source_and_existing_carry_candidate(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 9), ("pasted_body", 9)])

    def test_widening_broadcast_reduction_axis_and_actual_arguments_are_pinned(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        lean = (self.directory / "Kernels.lean").read_text()
        for record in records:
            for before, after in [
                ("dim_n = tl.cast(dim_n, tl.int64)", "dim_n = dim_n"),
                ("dim_k = tl.cast(dim_k, tl.int64)", "dim_k = dim_k"),
                ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
                ("expanded_a, _ = tl.broadcast(a, b)", "expanded_a = a"),
                ("tl.sum(expanded_a * b, axis=2)", "tl.sum(expanded_a * b, axis=1)"),
                ("dim_k := dim_k", "dim_k := dim_n"),
            ]:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        check((ROOT / record["python"]).read_text(), lean.replace(before, after),
                              record["python_function"], record["lean_definition"],
                              record["signature_definition"], record["expected_signature"],
                              record["expected_call_bindings"])


class FlashDecodeFunctionBodyTests(unittest.TestCase):
    tasks = ["flash_decode2_llama", "flash_decode2_phi_block_checked"]

    def compare(self, record, lean):
        return check((ROOT / record["python"]).read_text(), lean,
                     record["python_function"], record["lean_definition"],
                     record["signature_definition"], record["expected_signature"],
                     record["expected_call_bindings"])

    def test_complete_original_and_existing_cse_bodies(self):
        for task in self.tasks:
            directory = ROOT / "bench/repairs/tritonbench_g" / task
            results = check_manifest(ROOT, directory / "jit-correspondence.json")
            self.assertEqual([(x["mode"], x["statements"]) for x in results],
                             [("pasted_body", 13), ("pasted_body", 13)])

    def test_signed_casts_masks_and_canonical_runtime_stride_products_are_pinned(self):
        for task in self.tasks:
            directory = ROOT / "bench/repairs/tritonbench_g" / task
            records = json.loads((directory / "jit-correspondence.json").read_text())
            for record in records:
                lean = (ROOT / record["lean"]).read_text()
                mutations = [
                    ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                    ("tl.program_id(1).to(tl.int64)", "tl.program_id(1)"),
                    ("tl.load(B_Seqlen + cur_batch).to(tl.int64)", "tl.load(B_Seqlen + cur_batch)"),
                    ("stride_mid_ob := H * S * D", "stride_mid_ob := H * S"),
                    ("stride_mid_os := D", "stride_mid_os := H"),
                    ("stride_obs := H * D", "stride_obs := D"),
                    ("new_max_logic = tl.maximum(tlogic, max_logic)",
                     "new_max_logic = tl.minimum(tlogic, max_logic)"),
                ]
                if "phi" in task:
                    mutations += [("mask=offs_d < head_dim", "mask=offs_d <= head_dim"),
                                  ("head_dim := D", "head_dim := BD")]
                for before, after in mutations:
                    with self.subTest(task=task, definition=record["lean_definition"], mutation=before):
                        self.assertIn(before, lean)
                        with self.assertRaises(ValueError):
                            self.compare(record, lean.replace(before, after))

    def test_stride_products_accept_spacing_but_reject_calls_and_changed_grouping(self):
        task = ROOT / "bench/repairs/tritonbench_g/flash_decode2_llama"
        record = json.loads((task / "jit-correspondence.json").read_text())[0]
        lean = (ROOT / record["lean"]).read_text()
        self.compare(record, lean.replace("stride_mid_ob := H * S * D",
                                        "stride_mid_ob := (H*S) * D"))
        for value in ["H * (S * D)", "wrong(H, S, D)", "H[S] * D"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.compare(record, lean.replace("stride_mid_ob := H * S * D",
                                                 "stride_mid_ob := " + value))


class BgmvExpandFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/bgmv_expand_slice_physical_tile_checked"

    def compare(self, record, lean):
        return check((ROOT / record["python"]).read_text(), lean,
                     record["python_function"], record["lean_definition"],
                     record["signature_definition"], record["expected_signature"],
                     record["expected_call_bindings"])

    def test_complete_body_for_both_index_widths_and_candidates(self):
        results = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(x["mode"], x["statements"]) for x in results],
                         [("pasted_body", 13)] * 4)
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            source = (ROOT / record["python"]).read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                      and n.name == record["python_function"])
            body = "\n".join(source.splitlines()[fn.body[0].lineno-1:fn.end_lineno])
            lean = (ROOT / record["lean"]).read_text()
            self.assertEqual(textwrap.dedent(body).strip(),
                             textwrap.dedent(quotation(lean, record["lean_definition"])[1]).strip())

    def test_cast_early_return_mask_reduction_and_stride_changes_are_rejected(self):
        mutations = [
            ("tl.program_id(axis=0).to(tl.int64)", "tl.program_id(axis=0)"),
            ("tl.load(lora_indices + cur_batch).to(tl.int64)", "tl.load(lora_indices + cur_batch)"),
            ("tl.arange(0, BLOCK_N).to(tl.int64)", "tl.arange(0, BLOCK_N)"),
            ("if lora_index == -1:", "if lora_index == 0:"),
            ("tl.cast(N, tl.int64)", "N"),
            ("tiled_a.to(lora_ptr.dtype.element_ty)", "tiled_a"),
            ("current_n < split_n_length", "current_n <= split_n_length"),
            ("tl.sum(tiled_a * tiled_b, 1)", "tl.sum(tiled_a * tiled_b, 0)"),
            ("l0_stride := h.N * h.K", "l0_stride := h.N"),
            ("cn_stride := 1", "cn_stride := h.columns"),
            ('("out_ptr",.fp32)', '("out_ptr",.fp16)'),
        ]
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            lean = (ROOT / record["lean"]).read_text()
            for before, after in mutations:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        self.compare(record, lean.replace(before, after))


class AttentionFwdFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/attention_fwd_triton1_tile_checked"

    def compare(self, record, lean):
        return check((ROOT / record["python"]).read_text(), lean,
                     record["python_function"], record["lean_definition"],
                     record["signature_definition"], record["expected_signature"],
                     record["expected_call_bindings"])

    def test_complete_original_and_shift_candidate(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 3), ("pasted_body", 4)])
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            source = (ROOT / record["python"]).read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                      and n.name == record["python_function"])
            body = "\n".join(source.splitlines()[fn.body[0].lineno-1:fn.end_lineno])
            lean = (ROOT / record["lean"]).read_text()
            self.assertEqual(textwrap.dedent(body).strip(),
                             textwrap.dedent(quotation(lean, record["lean_definition"])[1]).strip())

    def test_cast_block_geometry_boundary_and_candidate_delta_are_preserved(self):
        mutations = [
            ("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
            ("b_h.to(p_h.dtype.element_ty)", "b_h"),
            ("(b_q * scale).to(b_q.dtype)", "(b_q * scale)"),
            ("b_s.to(b_q.dtype)", "b_s"),
            ("(NT * BD, BD)", "(T, BD)"),
            ("boundary_check=(0,)", "boundary_check=(1,)"),
            ('padding_option="zero"', 'padding_option=""'),
            ("allow_tf32=False", "allow_tf32=True"),
            ("if i == 0:", "if i == 1:"),
            ("b_h += tl.dot(b_k, b_v", "b_h = tl.dot(b_k, b_v"),
            ("s_hh := s_hh", "s_hh := s_qh"),
            ('("scale",.fp32)', '("scale",.fp64)'),
        ]
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            lean = (ROOT / record["lean"]).read_text()
            cases = mutations + ([("tl.static_assert(BT == 32)", "tl.static_assert(BT == 16)"),
                                  ("i << 5", "i << 4")]
                                 if record["lean_definition"] == "shiftedSource" else [])
            for before, after in cases:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        self.compare(record, lean.replace(before, after))


class IsfiniteFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/isfinite_kernel"

    def compare(self, record, lean, python=None):
        kwargs = {k: v for k, v in record.items() if k not in ("python", "lean", "profile_variant")}
        return check(python or (ROOT / record["python"]).read_text(), lean, **kwargs)

    def test_original_helper_and_both_precisions_of_each_writer(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 1)] + [("pasted_body", 7)] * 4)
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            source = (ROOT / record["python"]).read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                      and n.name == record["python_function"])
            raw = "\n".join(source.splitlines()[fn.body[0].lineno-1:fn.end_lineno])
            lean = (ROOT / record["lean"]).read_text()
            self.assertEqual(textwrap.dedent(raw).strip(),
                             textwrap.dedent(quotation(lean, record["lean_definition"])[1]).strip())

    def test_signed_casts_precision_imports_and_call_coordinates_are_checked(self):
        mutations = [
            ("offsets.to(tl.int64)", "offsets"),
            ("tl.cast(in0_stride, tl.int64)", "in0_stride"),
            ("tl.cast(out0_stride, tl.int64)", "out0_stride"),
            (".to(in0_ptr.type.element_ty)", ""),
            ("out0.to(out0_ptr.type.element_ty)", "out0"),
            ("offsets < n", "offsets <= n"),
            ("in0_stride := input.byteStride", "in0_stride := output.byteStride"),
            ("in0_ptr := input.pointer", "in0_ptr := output.pointer"),
            ('("_isfinited","triton.language.extra.cuda.libdevice.isfinited")',
             '("_isfinited","triton.language.extra.cuda.libdevice.finitef")'),
        ]
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        for record in records[1:]:
            lean = (ROOT / record["lean"]).read_text()
            for before, after in mutations:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        self.compare(record, lean.replace(before, after))
        helper = records[0]
        for before, after in [("x.dtype.is_fp64()", "x.dtype.is_fp32()"),
                              ("x.to(tl.float32)", "x"), ("_isfinited(x)", "_finitef(x)")]:
            with self.assertRaises(ValueError):
                self.compare(helper, lean.replace(before, after))

    def test_original_import_binding_must_match_profile(self):
        record = json.loads((self.directory / "jit-correspondence.json").read_text())[1]
        source = (ROOT / record["python"]).read_text()
        changed = source.replace("import isfinited as _isfinited", "import finitef as _isfinited")
        self.assertNotEqual(source, changed)
        with self.assertRaisesRegex(ValueError, "source import"):
            self.compare(record, (ROOT / record["lean"]).read_text(), changed)


class KldivOpsFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/kldiv_ops"

    def compare(self, record, lean):
        kwargs = {k: v for k, v in record.items() if k not in ("python", "lean")}
        return check((ROOT / record["python"]).read_text(), lean, **kwargs)

    def test_complete_forward_and_backward_bodies_are_verbatim(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 12)] * 2 + [("pasted_body", 9)] * 2)
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            source = (ROOT / record["python"]).read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                      and n.name == record["python_function"])
            raw = "\n".join(source.splitlines()[fn.body[0].lineno-1:fn.end_lineno])
            lean = (ROOT / record["lean"]).read_text()
            self.assertEqual(textwrap.dedent(raw).strip(),
                             textwrap.dedent(quotation(lean, record["lean_definition"])[1]).strip())

    def test_cast_branch_epsilon_mask_and_call_binding_drift_is_rejected(self):
        common = [("n_cols = tl.cast(n_cols, tl.int64)", "n_cols = n_cols"),
                  ("if not log_target:", "if log_target:"),
                  ("offsets < n_cols", "offsets <= n_cols"),
                  ("n_cols := p.cols", "n_cols := p.rows")]
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            lean = (ROOT / record["lean"]).read_text()
            mutations = common + ([("tl.maximum(y_true, eps)", "tl.maximum(y_true, eps.to(tl.float32))"),
                                   ("loss = tl.where(mask, loss, 0.0)", "loss = loss"),
                                   ("loss_stride := outputStride", "loss_stride := p.cols")]
                                  if record["lean_definition"].startswith("forward") else
                                  [("res = target * -1", "res = -target"),
                                   ("new_grads_stride := p.cols", "new_grads_stride := p.rows")])
            for before, after in mutations:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        self.compare(record, lean.replace(before, after))


class SoftmaxOptimizeFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/softmax_optimize"

    def compare(self, record, lean):
        kwargs = {k: v for k, v in record.items() if k not in ("python", "lean")}
        return check((ROOT / record["python"]).read_text(), lean, **kwargs)

    def test_helper_and_both_complete_four_loop_bodies_are_verbatim(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 1), ("pasted_body", 12), ("pasted_body", 12)])
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            source = (ROOT / record["python"]).read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                      and n.name == record["python_function"])
            raw = "\n".join(source.splitlines()[fn.body[0].lineno-1:fn.end_lineno])
            lean = (ROOT / record["lean"]).read_text()
            self.assertEqual(textwrap.dedent(raw).strip(),
                             textwrap.dedent(quotation(lean, record["lean_definition"])[1]).strip())

    def test_dtype_cast_guards_loop_bounds_and_masks_cannot_drift(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        helper = records[0]
        lean = (ROOT / helper["lean"]).read_text()
        with self.assertRaises(ValueError):
            self.compare(helper, lean.replace("tl.cdiv(a, b) * b - b", "tl.cdiv(a, b) * b"))
        mutations = [("tl.program_id(0).to(tl.int64)", "tl.program_id(0)"),
                     ("tl.arange(0, TILE_N).to(tl.int64)", "tl.arange(0, TILE_N)"),
                     (".to(output_ptr.dtype.element_ty)", ""),
                     ("dtype=output_ptr.dtype.element_ty", "dtype=tl.float32"),
                     ('new_m == -float("inf")', 'new_m == float("inf")'),
                     ('inp == -float("inf")', 'inp == float("inf")'),
                     ("range(0, prev_multiple, TILE_N)", "range(0, N, TILE_N)"),
                     ("range(prev_multiple, N, TILE_N)", "range(prev_multiple, N + 1, TILE_N)"),
                     ("n_offsets < N", "n_offsets <= N"),
                     ("tl.store(output_ptrs, out, mask=mask)", "tl.store(output_ptrs, out)"),
                     ('("N",.i64)', '("N",.i32)')]
        for record in records[1:]:
            for before, after in mutations:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        self.compare(record, lean.replace(before, after))


class ChunkDeltaFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/chunk_delta_fwd_static_dot"

    def compare(self, record, lean):
        kwargs = {k: v for k, v in record.items() if k not in ("python", "lean")}
        return check((ROOT / record["python"]).read_text(), lean, **kwargs)

    def test_complete_original_and_address_candidate_are_verbatim(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([(p["mode"], p["statements"]) for p in pairs],
                         [("pasted_body", 5), ("pasted_body", 5)])
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            source = (ROOT / record["python"]).read_text()
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                      and n.name == record["python_function"])
            raw = "\n".join(source.splitlines()[fn.body[0].lineno-1:fn.end_lineno])
            lean = (ROOT / record["lean"]).read_text()
            self.assertEqual(textwrap.dedent(raw).strip(),
                             textwrap.dedent(quotation(lean, record["lean_definition"])[1]).strip())

    def test_dtype_queries_tf32_flags_boundaries_loops_and_strides_cannot_drift(self):
        mutations = [("b_h.to(b_k.dtype)", "b_h"),
                     ("b_v.to(b_k.dtype)", "b_v.to(tl.float16)"),
                     ("b_h.to(p_h.dtype.element_ty)", "b_h.to(tl.float32)"),
                     ("b_v.to(p_v_new.dtype.element_ty)", "b_v"),
                     ("allow_tf32=False", "allow_tf32=True"),
                     ("boundary_check=(0, 1)", "boundary_check=(0,)"),
                     ("range(tl.cdiv(BT, BC))", "range(BT // BC)"),
                     ("range(NT)", "range(NT - 1)"),
                     ("if USE_INITIAL_STATE:", "if not USE_INITIAL_STATE:"),
                     ("if STORE_FINAL_STATE:", "if not STORE_FINAL_STATE:"),
                     ("s_w_t := p.wTime", "s_w_t := p.K"),
                     ('writePorts := ["v_new", "h", "final_state"]',
                      'writePorts := ["v_new", "h"]')]
        for record in json.loads((self.directory / "jit-correspondence.json").read_text()):
            lean = (ROOT / record["lean"]).read_text()
            for before, after in mutations:
                with self.subTest(definition=record["lean_definition"], mutation=before):
                    self.assertIn(before, lean)
                    with self.assertRaises(ValueError):
                        self.compare(record, lean.replace(before, after))

class SoftmaxTriton3BodyTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / 'bench/repairs/tritonbench_g/softmax_triton3'
        self.records = json.loads((self.directory / 'jit-correspondence.json').read_text())
        self.python = (self.directory / 'softmax_triton3_repaired.py').read_text()
        self.lean = (self.directory / 'Kernels.lean').read_text()

    def compare(self, lean=None, record=None):
        r = self.records[0] if record is None else record
        return check(self.python, self.lean if lean is None else lean,
                     r['python_function'], r['lean_definition'], r['signature_definition'],
                     r['expected_signature'], r['expected_call_bindings'],
                     expected_pointer_presence=r.get('expected_pointer_presence'))

    def test_both_complete_bodies(self):
        pairs = check_manifest(ROOT, self.directory / 'jit-correspondence.json')
        self.assertEqual([p['statements'] for p in pairs], [14, 17])
        self.assertTrue(all(p['mode'] == 'pasted_body' for p in pairs))
        self.assertTrue(all(p['pointer_presence'] == {'mask_ptr': 'HAS_MASK'} for p in pairs))

    def test_casts_none_test_and_caller_types_are_preserved(self):
        for before, after in [
            ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
            ('tl.arange(0, BLOCK_SIZE).to(tl.int64)', 'tl.arange(0, BLOCK_SIZE)'),
            ('if mask_ptr is not None:', 'if mask_ptr is None:'),
            ('pointer_presence [mask_ptr := HAS_MASK]', 'pointer_presence [mask_ptr := Bool.true]'),
            ('optionalPointers := ["mask_ptr"]', 'optionalPointers := []'),
            ('("row_stride",.i64)', '("row_stride",.i32)'),
        ]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after, 1))

    def test_presence_binding_requires_review(self):
        record = dict(self.records[0])
        record.pop('expected_pointer_presence')
        with self.assertRaises(ValueError):
            self.compare(record=record)

    def test_duplicate_and_unknown_presence_bindings_are_rejected(self):
        for binding in ['mask_ptr := HAS_MASK, mask_ptr := HAS_MASK',
                        'input_ptr := HAS_MASK', 'mask_ptr := (HAS_MASK)']:
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                self.compare(self.lean.replace('mask_ptr := HAS_MASK', binding, 1))

    def test_duplicate_optional_declarations_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compare(self.lean.replace('optionalPointers := ["mask_ptr"]',
                                          'optionalPointers := ["mask_ptr","mask_ptr"]'))

    def test_candidate_fallback_and_float_literal_are_preserved(self):
        for before, after in [('denominator >= 1.0', 'denominator >= 0.0'),
                              ('1.0 / denominator', '1.0 * denominator'),
                              ('softmax_output = numerator / denominator',
                               'softmax_output = numerator * denominator')]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.compare(self.lean.replace(before, after), self.records[1])

class BmmConstexprBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/bmm_optimized_grid_axes_checked"

    def test_two_complete_bodies(self):
        pairs = check_manifest(ROOT, self.directory / "jit-correspondence.json")
        self.assertEqual([p["statements"] for p in pairs], [20,20])
        self.assertTrue(all(p["mode"] == "pasted_body" for p in pairs))

    def test_masks_branches_tuple_loop_and_dot_cannot_drift(self):
        records = json.loads((self.directory / "jit-correspondence.json").read_text())
        for record in records:
            python = (ROOT / record["python"]).read_text()
            lean = (ROOT / record["lean"]).read_text()
            for before, after in [
                ("mask_a = None", "mask_a = True"),
                ("if DIVISIBLE_M and DIVISIBLE_K:", "if DIVISIBLE_M:"),
                ("pid_m, pid_n = pidx, pidy", "pid_m, pid_n = pidy, pidx"),
                ("for _ in range(num_iters):", "for _ in range(num_iters + 1):"),
                ("mask_b = mask_k[:, None] & mask_n[None, :]", "mask_b = mask_n[None, :]"),
                ("tl.dot(a, b, allow_tf32=False)", "tl.dot(a, b, allow_tf32=True)"),
                ("tl.store(o_ptrs, o, mask_c)", "tl.store(o_ptrs, o)"),
            ]:
                self.assertIn(before, lean)
                with self.subTest(definition=record["lean_definition"], change=before), self.assertRaises(ValueError):
                    check(python,lean.replace(before,after),**{k:v for k,v in record.items() if k not in ["python","lean"]})

    def test_runtime_sizes_and_constexpr_flags_keep_their_types(self):
        record=json.loads((self.directory / "jit-correspondence.json").read_text())[0]
        python=(ROOT/record["python"]).read_text();lean=(ROOT/record["lean"]).read_text()
        for before,after in [('("M",.i32)','("M",.i64)'),
                             ('"DIVISIBLE_K"]','"DIFFERENT_FLAG"]'),
                             ('M := p.M, N := p.N, K := p.K','M := p.N, N := p.N, K := p.K')]:
            self.assertIn(before,lean)
            with self.subTest(change=before),self.assertRaises(ValueError):
                check(python,lean.replace(before,after),**{k:v for k,v in record.items() if k not in ["python","lean"]})

    def test_candidate_changes_only_the_existing_integer_remainder(self):
        import copy
        original = ast.parse((self.directory / "bmm_optimized_grid_axes_checked.py").read_text())
        candidate = ast.parse((self.directory / "candidate_kernels.py").read_text())
        expected = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "bmm_kernel"))
        expected.name = "bmm_quotient_remainder"
        assignments = [n for n in ast.walk(expected) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "inner_group_id" for t in n.targets)]
        self.assertEqual(len(assignments),1)
        self.assertEqual(ast.unparse(assignments[0].value),"pid % num_CTA_per_group")
        assignments[0].value = ast.parse("pid - (pid // num_CTA_per_group) * num_CTA_per_group",mode="eval").body
        actual = next(n for n in candidate.body if isinstance(n,ast.FunctionDef) and n.name == expected.name)
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(actual,include_attributes=False))
        for name in ["heur_divisible_m","heur_divisible_n","heur_divisible_k"]:
            left=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name == name)
            right=next(n for n in candidate.body if isinstance(n,ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False))


class GegluFunctionBodyTests(unittest.TestCase):
    directory = ROOT / "bench/repairs/tritonbench_g/geglu_tanh_triton"

    def compare(self, record, lean=None, python=None):
        kwargs = {k: v for k, v in record.items() if k not in ("python", "lean")}
        return check((ROOT / record["python"]).read_text() if python is None else python,
                     (ROOT / record["lean"]).read_text() if lean is None else lean, **kwargs)

    def test_four_complete_bodies_preserve_comments_and_statement_order(self):
        records = json.loads((self.directory / 'jit-correspondence.json').read_text())
        result = check_manifest(ROOT, self.directory / 'jit-correspondence.json')
        self.assertEqual([p['statements'] for p in result], [12,12,18,18])
        self.assertTrue(all(p['mode'] == 'pasted_body' for p in result))
        for record in records:
            python = (ROOT / record['python']).read_text()
            function = next(f for f in ast.parse(python).body if isinstance(f, ast.FunctionDef)
                            and f.name == record['python_function'])
            raw = '\n'.join(python.splitlines()[function.body[0].lineno-1:function.end_lineno])
            self.assertEqual(textwrap.dedent(raw).strip(), textwrap.dedent(
                quotation((ROOT / record['lean']).read_text(),record['lean_definition'])[1]).strip())

    def test_original_import_is_checked(self):
        record=json.loads((self.directory/'jit-correspondence.json').read_text())[0]
        source=(ROOT/record['python']).read_text()
        for before,after in [('from triton.language.extra.libdevice import tanh',
                              'from triton.language import tanh'),
                             ('from triton.language.extra.libdevice import tanh',''),
                             ('from triton.language.extra.libdevice import tanh',
                              'from triton.language.extra.libdevice import tanh\ntanh = lambda x: x'),
                             ('from triton.language.extra.libdevice import tanh',
                              'from triton.language.extra.libdevice import tanh\nimport math as tanh'),
                             ('from triton.language.extra.libdevice import tanh',
                              'from triton.language.extra.libdevice import tanh\ndel tanh')]:
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.compare(record,python=source.replace(before,after))

    def test_missing_import_metadata_cannot_select_a_different_intrinsic(self):
        record=json.loads((self.directory/'jit-correspondence.json').read_text())[0]
        lean=(ROOT/record['lean']).read_text()
        changed=json.loads(json.dumps(record));changed['expected_signature'].pop('unaryImports')
        lean=lean.replace('  unaryImports := [("tanh", "triton.language.extra.libdevice.tanh")]\n','')
        with self.assertRaises(ValueError):self.compare(changed,lean=lean)

    def test_casts_intrinsic_and_runtime_order_cannot_drift(self):
        for record in json.loads((self.directory/'jit-correspondence.json').read_text()):
            lean=(ROOT/record['lean']).read_text()
            for before,after in [('tl.program_id(0).to(tl.int64)','tl.program_id(0)'),
                                 ('tl.arange(0, BLOCK_SIZE).to(tl.int64)','tl.arange(0, BLOCK_SIZE)'),
                                 ('tanh_result = tanh(tanh_arg)','tanh_result = tl.tanh(tanh_arg)'),
                                 ('a_cubed = a_row * a_row * a_row','a_cubed = a_row * (a_row * a_row)'),
                                 ('mask=mask, other=0).to(tl.float32)','mask=mask, other=0)'),
                                 ('3 * 0.044715 * a_row * a_row','0.134145 * a_row * a_row')]:
                if before not in quotation(lean,record['lean_definition'])[1]:continue
                with self.subTest(name=record['lean_definition'],change=before), self.assertRaises(ValueError):
                    self.compare(record,lean=lean.replace(before,after))

    def test_duplicate_or_wrong_unary_import_metadata_is_rejected(self):
        record=json.loads((self.directory/'jit-correspondence.json').read_text())[0]
        lean=(ROOT/record['lean']).read_text()
        for value in ['[("tanh", "triton.language.tanh")]',
                      '[("tanh", "triton.language.extra.libdevice.tanh"), ("tanh", "triton.language.extra.libdevice.tanh")]']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.compare(record,lean=lean.replace('[("tanh", "triton.language.extra.libdevice.tanh")]',value))
