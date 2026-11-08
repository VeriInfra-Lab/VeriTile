import ast
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "bench/tritonbench_g/matmul_dequantize_int4"

class Int4BodyMigrationTests(unittest.TestCase):
    def test_both_complete_bodies(self):
        rows = check_pinned_pairs(ROOT, DIR)
        self.assertEqual([r["statements"] for r in rows], [32, 32])
        self.assertTrue(all(r["mode"] == "pasted_body" for r in rows))

    def test_unpack_cast_branch_and_store_cannot_drift(self):
        record = pinned_pairs(DIR)[0]
        python = (ROOT / record["python"]).read_text()
        lean = (ROOT / record["lean"]).read_text()
        for before, after in [
                ("infearure_per_bits = 8", "infearure_per_bits = 4"),
                ("& 0xF", "& 0x7"),
                ("if not NO_GROUPS:", "if NO_GROUPS:"),
                ("b.to(a.dtype)", "b"),
                ("accumulator += tl.dot", "accumulator = tl.dot"),
                ("c = accumulator.to(c_ptr.dtype.element_ty)", "c = accumulator"),
                ("tl.store(c_ptrs, accumulator, mask=c_mask)", "tl.store(c_ptrs, c, mask=c_mask)")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check(python, lean.replace(before, after, 1),
                      record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"],
                      expected_call_bindings=record["expected_call_bindings"])

    def test_candidate_is_only_existing_host_extent_precomputation(self):
        original_path = DIR / "matmul_dequantize_int4.py"
        candidate_path = DIR / "matmul_dequantize_int4_group_extent.py"
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                  and n.name == "matmul4_kernel_group_extent")
        narg = next(a for a in fn.args.args if a.arg == "N")
        self.assertEqual(ast.unparse(narg.annotation), "tl.constexpr")
        narg.annotation = None
        index, stmt = next((i,n) for i,n in enumerate(fn.body) if isinstance(n, ast.AnnAssign)
                           and ast.unparse(n.target) == "num_pid_in_group")
        self.assertEqual(ast.unparse(stmt.annotation), "tl.constexpr")
        self.assertEqual(ast.unparse(stmt.value), "GROUP_SIZE_M * triton.cdiv(N, BLOCK_SIZE_N)")
        fn.body[index] = ast.Assign(targets=[stmt.target],
                                    value=ast.parse("GROUP_SIZE_M * num_pid_n", mode="eval").body)
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name in {
                    "matmul4_kernel_group_extent", "matmul_dequantize_int4_gptq_group_extent"}:
                node.name = node.name.removesuffix("_group_extent")
            if isinstance(node, ast.Name) and node.id in {
                    "matmul4_kernel_group_extent", "matmul_dequantize_int4_gptq_group_extent"}:
                node.id = node.id.removesuffix("_group_extent")
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(DIR)
        self.assertEqual(hashlib.sha256(variant_source(original_path).encode()).hexdigest(), provenance["original_sha256"])
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance["candidate_sha256"])


class StandaloneInt4BodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = ROOT / "bench/tritonbench_g/matmul_dequant_int4"

    def test_both_stages_and_both_existing_files(self):
        rows = check_pinned_pairs(ROOT, self.directory)
        self.assertEqual([r["statements"] for r in rows], [32, 19, 32, 19])
        records = pinned_pairs(self.directory)
        self.assertEqual(records[0]["lean"], records[2]["lean"])
        self.assertEqual(records[0]["lean_definition"], "originalKernel")

    def test_signed_dequantization_and_complete_memory_contract_cannot_drift(self):
        record = pinned_pairs(self.directory)[1]
        python = (ROOT / record["python"]).read_text()
        lean = (ROOT / record["lean"]).read_text()
        for before, after in [
                ("((zp_b >> bzp_shift) & 0xF)", "zp_b"),
                ("offs_k[:, None] // group_size", "offs_k[:, None] // 8"),
                ("mask = n_mask & k_mask", "mask = n_mask | k_mask"),
                ("other=0.0", "other=1.0"),
                ("stride_fpbn", "stride_fpbk")]:
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertIn(before, lean)
                check(python, lean.replace(before, after, 1), record["python_function"],
                      record["lean_definition"], record["signature_definition"],
                      record["expected_signature"], expected_call_bindings=record["expected_call_bindings"])
        with self.assertRaises(ValueError):
            check(python, lean.replace('("b_ptr",.i32)', '("b_ptr",.fp32)', 1),
                  record["python_function"], record["lean_definition"], record["signature_definition"],
                  record["expected_signature"], expected_call_bindings=record["expected_call_bindings"])

    def test_existing_candidate_only_removes_unit_column_multiply(self):
        original_path = self.directory / "matmul_dequant_int4.py"
        candidate_path = self.directory / "matmul_dequant_int4_unit_stride.py"
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_unit_stride"):
                node.name = node.name.removesuffix("_unit_stride")
            if isinstance(node, ast.Name) and node.id.endswith("_unit_stride"):
                node.id = node.id.removesuffix("_unit_stride")
        fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                  and n.name == "dequantize_kernel")
        stmt = next(n for n in fn.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "fpb_offs")
        self.assertEqual(ast.unparse(stmt.value), "offs_k[:, None] * stride_fpbk + offs_n[None, :]")
        stmt.value = ast.parse("offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn", mode="eval").body
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(self.directory)
        self.assertEqual(hashlib.sha256(variant_source(original_path).encode()).hexdigest(), provenance["original_sha256"])
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance["candidate_sha256"])
