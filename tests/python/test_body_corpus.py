import json
from pathlib import Path
import tempfile
import unittest

from scripts.bench.check_body_corpus import scan


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
        self.directory = self.root / "bench/tritonbench_g/sample"
        self.directory.mkdir(parents=True)
        self.write("bench/tritonbench_g/sample/source.py", PYTHON)
        self.write("bench/tritonbench_g/sample/Kernels.lean", LEAN)
        self.write("validation/task_progress.json", json.dumps({"tasks": [
            {"task": "sample", "kernel_revision": {"source": "bench/tritonbench_g/sample/source.py",
             "kernel_source": "bench/tritonbench_g/sample/Kernels.lean"}}]}))
        self.record = {"python": "bench/tritonbench_g/sample/source.py", "python_function": "first",
                       "lean": "bench/tritonbench_g/sample/Kernels.lean", "lean_definition": "body",
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
        self.write("bench/tritonbench_g/sample/source.py", PYTHON + PYTHON.split("@triton.jit")[0]
                   + "@triton.jit" + PYTHON.split("@triton.jit")[1].replace("first(", "second(", 1))
        result = scan(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"]["uncovered_selected_jit_functions"], 1)
        self.assertEqual(result["tasks"][0]["uncovered_selected_jits"][0]["function"], "second")

    def test_broken_pair_is_an_error_even_when_other_stages_are_pending(self):
        self.write("bench/tritonbench_g/sample/Kernels.lean", LEAN.replace("tl.arange(0, B)", "tl.arange(1, B)"))
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            scan(self.root)

    def test_duplicate_mapping_cannot_cover_another_stage(self):
        self.manifest([self.record, self.record])
        with self.assertRaisesRegex(ValueError, "reused"):
            scan(self.root)

    def test_unchanged_candidate_stage_can_share_lean_source(self):
        self.write("bench/tritonbench_g/sample/candidate.py", PYTHON)
        self.manifest([self.record, {**self.record,
            "python": "bench/tritonbench_g/sample/candidate.py"}])
        self.assertTrue(scan(self.root)["complete"])
        self.write("bench/tritonbench_g/sample/candidate.py", PYTHON.replace(
            "tl.arange(0, B)", "tl.arange(1, B)"))
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            scan(self.root)

    def test_worked_example_without_reference_is_not_a_pass(self):
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        result = scan(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["examples"][0]["status"], "no_independent_python_reference")

    def test_explicit_example_reference_and_manifest_allow_progress(self):
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        self.write("bench/examples/source-references.json", json.dumps({"Sample": "bench/tritonbench_g/sample/source.py"}))
        record = {**self.record, "lean": "bench/examples/Sample/Kernels.lean"}
        self.write("bench/examples/Sample/jit-correspondence.json", json.dumps([record]))
        self.assertTrue(scan(self.root)["complete"])

    def test_missing_source_fails_closed(self):
        (self.directory / "source.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing selected Python"):
            scan(self.root)

    def test_legacy_ast_and_nested_functions_cannot_pass_body_copy(self):
        record = {k: v for k, v in self.record.items()
                  if k not in ("signature_definition", "expected_signature")}
        self.manifest([record])
        self.assertEqual(scan(self.root)["tasks"][0]["status"], "legacy_ast_only")
        self.manifest([self.record])
        nested = "def factory():\n" + "\n".join("    " + line for line in PYTHON.splitlines()) + "\n"
        self.write("bench/tritonbench_g/sample/source.py", nested)
        with self.assertRaisesRegex(ValueError, "explicitly selected"):
            scan(self.root)

    def test_pasted_shape_spelling_is_not_silently_normalized(self):
        python = PYTHON.replace("x = tl.load(x_ptr + offsets)", "x = tl.full((B,), 0.0, tl.float32)")
        lean = LEAN.replace("x = tl.load(x_ptr + offsets)", "x = tl.full([B], 0.0, dtype=tl.float32)")
        self.write("bench/tritonbench_g/sample/source.py", python)
        self.write("bench/tritonbench_g/sample/Kernels.lean", lean)
        with self.assertRaisesRegex(ValueError, "correspondence mismatch"):
            scan(self.root)

    def test_braced_control_flow_is_not_a_pasted_python_body(self):
        self.write("bench/tritonbench_g/sample/source.py", PYTHON.replace(
            "    tl.store(out_ptr + offsets, x)",
            "    if B > 0:\n        tl.store(out_ptr + offsets, x)"))
        self.write("bench/tritonbench_g/sample/Kernels.lean", LEAN.replace(
            "    tl.store(out_ptr + offsets, x)",
            "    if B > 0 {\n      tl.store(out_ptr + offsets, x)\n    }"))
        with self.assertRaises(SyntaxError):
            scan(self.root)

    def test_all_collections_and_unselected_tasks_are_discovered(self):
        for collection in ("bug_detection", "pretrain_kernels"):
            self.write(f"bench/{collection}/tasks/sample/Kernels.lean", LEAN)
            self.write(f"bench/{collection}/tasks/sample/task.json", '{"id":"sample"}')
        self.write("bench/examples/Sample/Kernels.lean", LEAN)
        self.write("bench/tritonbench_g/unselected/Kernels.lean", LEAN)
        self.write("bench/tritonbench_g/unselected/unselected.py", PYTHON)
        result = scan(self.root)
        self.assertEqual(result["counts"]["total"], 5)
        self.assertEqual(result["counts"]["collections"], {
            "bug_detection": 1, "examples": 1, "pretrain_kernels": 1, "tritonbench_g": 2})
        rows = result["tasks"] + result["examples"]
        self.assertEqual(len({r["id"] for r in rows}), 5)
        imported = [r for r in rows if r["collection"] in {"bug_detection", "pretrain_kernels"}]
        self.assertTrue(all(r["status"] == "no_independent_python_reference" for r in imported))
        self.assertFalse(result["complete"])

    def test_discovery_does_not_require_a_selection_ledger(self):
        (self.root / "validation/task_progress.json").unlink()
        self.assertEqual(scan(self.root)["counts"]["total"], 1)

    def test_missing_kernel_and_missing_selected_directory_are_errors(self):
        self.write("bench/pretrain_kernels/tasks/incomplete/task.json", '{}')
        with self.assertRaisesRegex(ValueError, "missing selected Lean"):
            scan(self.root)
        (self.root / "bench/pretrain_kernels/tasks/incomplete/task.json").unlink()
        (self.root / "bench/pretrain_kernels/tasks/incomplete").rmdir()
        self.write("validation/task_progress.json", json.dumps({"tasks": [
            {"task": "absent", "kernel_revision": {"source": "missing.py", "kernel_source": "missing.lean"}}]}))
        with self.assertRaisesRegex(ValueError, "selected tasks missing"):
            scan(self.root)

    def test_unregistered_candidate_file_is_not_omitted(self):
        self.write("bench/tritonbench_g/sample/candidate.py", PYTHON.replace("first(", "candidate(", 1))
        result = scan(self.root)
        self.assertFalse(result["complete"])
        self.assertEqual(result["counts"]["selected_jit_functions"], 1)
        self.assertEqual(result["counts"]["uncovered_selected_jit_functions"], 0)
        self.assertEqual(result["counts"]["inventoried_jit_functions"], 2)
        self.assertEqual(result["counts"]["uncovered_inventoried_jit_functions"], 1)
        self.assertEqual(result["tasks"][0]["uncovered_python_jits"][0]["function"], "candidate")

    def test_dynamic_wrapper_bodies_are_inventoried_without_executing_python(self):
        self.write("bench/tritonbench_g/sample/source.py", "raise RuntimeError('never execute this')\n")
        self.write("bench/tritonbench_g/sample/body.py", PYTHON)
        record = {**self.record, "python": "bench/tritonbench_g/sample/body.py"}
        self.manifest([record])
        result = scan(self.root)
        self.assertEqual(result["tasks"][0]["status"], "no_selected_jit_resolved")
        self.assertEqual(result["counts"]["inventoried_jit_functions"], 1)
        self.assertFalse(result["complete"])

    def test_imported_collection_can_register_an_independent_reference(self):
        self.write("bench/pretrain_kernels/tasks/sample/Kernels.lean", LEAN)
        self.write("bench/pretrain_kernels/source-references.json", json.dumps({"sample": "bench/tritonbench_g/sample/source.py"}))
        self.write("bench/pretrain_kernels/tasks/sample/jit-correspondence.json", json.dumps([
            {**self.record, "lean": "bench/pretrain_kernels/tasks/sample/Kernels.lean"}]))
        result = scan(self.root)
        self.assertEqual(result["counts"]["total"], 2)
        self.assertTrue(result["complete"])

    def test_unknown_reference_and_empty_inventory_do_not_pass(self):
        self.write("bench/tritonbench_g/source-references.json", '{"absent":"missing.py"}')
        with self.assertRaisesRegex(ValueError, "unknown"):
            scan(self.root)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "bench").mkdir()
            self.assertFalse(scan(root)["complete"])


if __name__ == "__main__":
    unittest.main()
