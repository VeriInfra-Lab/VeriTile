import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from bench.check_body_corpus import scan


PYTHON = """import triton
import triton.language as tl
@triton.jit
def first(x_ptr, out_ptr, B: tl.constexpr):
    offsets = tl.arange(0, B)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)
"""
LEAN = """def callSignature : CallSignature where
  pointers := [("x_ptr",.fp32),("out_ptr",.fp32)]
  scalars := []
  constexpr := ["B"]
def body (x_ptr out_ptr : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    offsets = tl.arange(0, B)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)
  }
"""


class BodyCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "bench/repairs/sample"
        self.directory.mkdir(parents=True)
        self.write("bench/repairs/sample/source.py", PYTHON)
        self.write("bench/repairs/sample/Kernels.lean", LEAN)
        self.write("bench/tritonbench_g/task_progress.json", json.dumps({"tasks": [
            {"task": "sample", "kernel_revision": {"source": "bench/repairs/sample/source.py",
             "kernel_source": "bench/repairs/sample/Kernels.lean"}}]}))
        self.record = {"python": "bench/repairs/sample/source.py", "python_function": "first",
                       "lean": "bench/repairs/sample/Kernels.lean", "lean_definition": "body",
                       "signature_definition": "callSignature", "expected_signature": {
                           "pointers": [["x_ptr", "fp32"], ["out_ptr", "fp32"]],
                           "scalars": [], "constexpr": ["B"]}}
        self.manifest([self.record])

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def manifest(self, records):
        (self.directory / "jit-correspondence.json").write_text(json.dumps(records))

    def test_complete_single_stage_and_added_uncovered_stage(self):
        self.assertTrue(scan(self.root)["complete"])
        self.write("bench/repairs/sample/source.py", PYTHON + PYTHON.split("@triton.jit")[0]
                   + "@triton.jit" + PYTHON.split("@triton.jit")[1].replace("first(", "second(", 1))
        result = scan(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"]["uncovered_selected_jit_functions"], 1)
        self.assertEqual(result["tasks"][0]["uncovered_selected_jits"][0]["function"], "second")

    def test_broken_pair_is_an_error_even_when_other_stages_are_pending(self):
        self.write("bench/repairs/sample/Kernels.lean", LEAN.replace("tl.arange(0, B)", "tl.arange(1, B)"))
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            scan(self.root)

    def test_duplicate_mapping_cannot_cover_another_stage(self):
        self.manifest([self.record, self.record])
        with self.assertRaisesRegex(ValueError, "reused"):
            scan(self.root)

    def fp8_record(self):
        self.write("bench/repairs/sample/Kernels.lean", LEAN.replace(
            '("x_ptr",.fp32),("out_ptr",.fp32)',
            '("x_ptr",.f8e5),("out_ptr",.fp16)'))
        record = {**self.record, "expected_signature": {
            **self.record["expected_signature"],
            "pointers": [["x_ptr", "f8e5"], ["out_ptr", "fp16"]]}}
        self.manifest([record])
        return record

    def test_fp8_pointer_format_is_a_checked_call_dtype(self):
        self.fp8_record()
        result = scan(self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["tasks"][0]["checked_pairs"][0]["mode"], "pasted_body")

    def test_fp8_formats_cannot_reuse_each_others_reviewed_abi(self):
        self.fp8_record()
        p = self.directory / "Kernels.lean"
        p.write_text(p.read_text().replace(".f8e5", ".f8e4"))
        with self.assertRaisesRegex(ValueError, "reviewed launch types"):
            scan(self.root)

    def test_configured_jit_records_runtime_specialization_options(self):
        self.write("bench/repairs/sample/source.py", PYTHON.replace("@triton.jit",
            "@triton.jit(do_not_specialize=['x_ptr'], do_not_specialize_on_alignment=[1])"))
        result = scan(self.root)
        self.assertTrue(result["complete"])
        pair = result["tasks"][0]["checked_pairs"][0]
        self.assertEqual(pair["jit_options"], {"do_not_specialize": "['x_ptr']",
                                               "do_not_specialize_on_alignment": "[1]"})
        self.assertEqual(pair["constexpr_parameters"], ["B"])

    def test_custom_jit_frontend_is_not_a_standard_source_certificate(self):
        self.write("bench/repairs/sample/source.py", PYTHON.replace("@triton.jit",
            "@triton.jit(frontend=custom_frontend)"))
        with self.assertRaisesRegex(ValueError, "unsupported triton.jit decorator option"):
            scan(self.root)

    def test_jit_options_must_refer_to_actual_function_arguments(self):
        for value in ("['missing']", "[3]", "[True]", "dynamic_options"):
            self.write("bench/repairs/sample/source.py", PYTHON.replace("@triton.jit",
                "@triton.jit(do_not_specialize=" + value + ")"))
            with self.assertRaisesRegex(ValueError, "specialization option"):
                scan(self.root)

    def profiled_records(self):
        second = {**self.record, "lean_definition": "body64", "signature_definition": "signature64",
                  "profile_variant": "float64", "expected_signature": {
                      **self.record["expected_signature"],
                      "pointers": [["x_ptr", "fp64"], ["out_ptr", "fp64"]]}}
        lean64 = LEAN.replace("callSignature", "signature64").replace("def body (", "def body64 (")
        lean64 = lean64.replace(".fp32", ".fp64")
        self.write("bench/repairs/sample/Kernels.lean", LEAN + lean64)
        return [{**self.record, "profile_variant": "float32"}, second]

    def test_distinct_checked_abi_profiles_of_one_source_are_both_recorded(self):
        self.manifest(self.profiled_records())
        result = scan(self.root)
        self.assertTrue(result["complete"])
        pairs = result["tasks"][0]["checked_pairs"]
        self.assertEqual([p["profile_variant"] for p in pairs], ["float32", "float64"])

    def test_profile_tag_cannot_hide_duplicate_abi(self):
        records = self.profiled_records()
        records[1]["expected_signature"] = records[0]["expected_signature"]
        self.manifest(records)
        with self.assertRaisesRegex(ValueError, "duplicate ABI"):
            scan(self.root)

    def test_each_profile_body_is_checked(self):
        self.manifest(self.profiled_records())
        p = self.directory / "Kernels.lean"
        first, second = p.read_text().split("def signature64", 1)
        p.write_text(first + "def signature64" + second.replace("tl.arange(0, B)", "tl.arange(1, B)"))
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            scan(self.root)

    def offset_call(self, bindings="x_ptr := input_base, out_ptr := output_base"):
        self.write("bench/repairs/sample/Kernels.lean", LEAN.replace(
            "with_triton_signature callSignature in",
            "with_triton_call callSignature call_args [] pointer_offsets [" + bindings + "] in"))
        record = {**self.record, "expected_call_bindings": {},
                  "expected_pointer_offsets": {"x_ptr": "input_base", "out_ptr": "output_base"}}
        self.manifest([record])
        return record

    def test_external_pointer_offsets_are_checked_and_recorded(self):
        self.offset_call()
        result = scan(self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["tasks"][0]["checked_pairs"][0]["pointer_offsets"],
                         {"x_ptr": "input_base", "out_ptr": "output_base"})

    def test_unreviewed_or_changed_pointer_offset_is_rejected(self):
        record = self.offset_call()
        for expected in (None, {"x_ptr": "other_base", "out_ptr": "output_base"}):
            record = {**record, "expected_pointer_offsets": expected}
            self.manifest([record])
            with self.assertRaisesRegex(ValueError, "pointer offsets differ"):
                scan(self.root)

    def test_missing_or_duplicated_pointer_offset_is_rejected(self):
        for bindings in ("x_ptr := input_base", "x_ptr := input_base, x_ptr := input_base"):
            self.offset_call(bindings)
            with self.assertRaisesRegex(ValueError, "pointer offset"):
                scan(self.root)

    def test_offset_cannot_bind_a_nonpointer_argument(self):
        record = self.offset_call("B := input_base")
        record["expected_pointer_offsets"] = {"B": "input_base"}
        self.manifest([record])
        with self.assertRaisesRegex(ValueError, "not a pointer"):
            scan(self.root)

    def test_worked_example_without_reference_is_not_a_pass(self):
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        result = scan(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["examples"][0]["status"], "no_independent_python_reference")

    def test_explicit_example_reference_and_manifest_allow_progress(self):
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        self.write("bench/examples/source-references.json", json.dumps({"Sample": "bench/repairs/sample/source.py"}))
        record = {**self.record, "lean": "bench/examples/Sample/Kernels.lean"}
        self.write("bench/examples/Sample/jit-correspondence.json", json.dumps([record]))
        self.assertTrue(scan(self.root)["complete"])

    def test_authored_counterpart_does_not_claim_historical_source_correspondence(self):
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        self.write("bench/examples/source-references.json", json.dumps({"Sample": {
            "python": "bench/repairs/sample/source.py", "provenance": "authored_native_counterpart",
            "note": "New native counterpart; not an independent historical source."}}))
        record = {**self.record, "lean": "bench/examples/Sample/Kernels.lean"}
        self.write("bench/examples/Sample/jit-correspondence.json", json.dumps([record]))
        result = scan(self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["examples"][0]["status"], "authored_counterpart_body_copy_checked")
        self.assertEqual(result["examples"][0]["source_provenance"], "authored_native_counterpart")

    def test_missing_source_fails_closed(self):
        (self.directory / "source.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing selected Python"):
            scan(self.root)

    def test_spec_derived_source_requires_explicit_origin_and_pinned_mathematics(self):
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        self.write("bench/examples/Sample/Correct.lean", "-- Independent target: output[i] = input[i].\n")
        spec = "bench/examples/Sample/Correct.lean"
        reference = {"python": "bench/repairs/sample/source.py",
                     "provenance": "spec_derived_native_source",
                     "note": "New source from the identity specification, not a historical benchmark.",
                     "mathematical_spec_sha256": {spec: hashlib.sha256((self.root / spec).read_bytes()).hexdigest()}}
        self.write("bench/examples/source-references.json", json.dumps({"Sample": reference}))
        record = {**self.record, "lean": "bench/examples/Sample/Kernels.lean"}
        self.write("bench/examples/Sample/jit-correspondence.json", json.dumps([record]))
        result = scan(self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["examples"][0]["status"], "spec_derived_body_copy_checked")
        self.assertEqual(result["examples"][0]["source_provenance"], "spec_derived_native_source")
        self.assertIn(spec, result["examples"][0]["source_sha256"])
        self.write(spec, "-- Changed target.\n")
        with self.assertRaisesRegex(ValueError, "specification pin drift"):
            scan(self.root)
        reference.pop("mathematical_spec_sha256")
        self.write("bench/examples/source-references.json", json.dumps({"Sample": reference}))
        with self.assertRaisesRegex(ValueError, "pinned independent"):
            scan(self.root)

    def test_legacy_ast_and_nested_functions_cannot_pass_body_copy(self):
        record = {k: v for k, v in self.record.items()
                  if k not in ("signature_definition", "expected_signature")}
        self.manifest([record])
        self.assertEqual(scan(self.root)["tasks"][0]["status"], "legacy_ast_only")
        self.manifest([self.record])
        nested = "def factory():\n" + "\n".join("    " + line for line in PYTHON.splitlines()) + "\n"
        self.write("bench/repairs/sample/source.py", nested)
        with self.assertRaisesRegex(ValueError, "explicitly selected"):
            scan(self.root)

    def test_pasted_shape_spelling_is_not_silently_normalized(self):
        python = PYTHON.replace("x = tl.load(x_ptr + offsets)", "x = tl.full((B,), 0.0, tl.float32)")
        lean = LEAN.replace("x = tl.load(x_ptr + offsets)", "x = tl.full([B], 0.0, dtype=tl.float32)")
        self.write("bench/repairs/sample/source.py", python)
        self.write("bench/repairs/sample/Kernels.lean", lean)
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            scan(self.root)

    def test_braced_control_flow_is_not_a_pasted_python_body(self):
        self.write("bench/repairs/sample/source.py", PYTHON.replace(
            "    tl.store(out_ptr + offsets, x)",
            "    if B > 0:\n        tl.store(out_ptr + offsets, x)"))
        self.write("bench/repairs/sample/Kernels.lean", LEAN.replace(
            "    tl.store(out_ptr + offsets, x)",
            "    if B > 0 {\n      tl.store(out_ptr + offsets, x)\n    }"))
        with self.assertRaises(SyntaxError):
            scan(self.root)


if __name__ == "__main__":
    unittest.main()
