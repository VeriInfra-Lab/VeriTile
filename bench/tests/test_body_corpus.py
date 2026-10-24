import json
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
