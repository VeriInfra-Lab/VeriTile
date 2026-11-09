import ast
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "bench/tritonbench_g/matmul_dequantize"

class MatmulDequantizeBodyMigrationTests(unittest.TestCase):
    def test_all_three_original_and_candidate_stages(self):
        rows = check_pinned_pairs(ROOT, DIR)
        self.assertEqual([r["statements"] for r in rows], [32,25,19,32,25,19])
        self.assertTrue(all(r["mode"] == "pasted_body" for r in rows))
        records = pinned_pairs(DIR)
        self.assertEqual(records[2]["lean"], "bench/tritonbench_g/matmul_dequant_int4/Kernels.lean")

    def test_split_factor_both_write_branches_and_explicit_casts_are_required(self):
        record = pinned_pairs(DIR)[1]
        python = (ROOT / record["python"]).read_text()
        lean = (ROOT / record["lean"]).read_text()
        for before, after in [
                ("range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K))", "range(0, 1)"),
                ("k * BLOCK_SIZE_K * SPLIT_K", "k * BLOCK_SIZE_K"),
                ("if SPLIT_K == 1:", "if True:"),
                ("tl.atomic_add(c_ptrs, c, mask=c_mask)", "tl.store(c_ptrs, c, mask=c_mask)"),
                ("a.to(tl.float16)", "a"),
                ("b.to(tl.float16)", "b"),
                ("b = ((int_b - int_bzp) * bs).to(tl.float16)", "b = (int_b - int_bzp) * bs"),
                ("other=0.0", "other=1.0")]:
            # Restrict replacement to the selected definition; the GPTQ body
            # precedes it and also contains masked loads.
            index = lean.index("def " + record["lean_definition"] + " (")
            changed = lean[:index] + lean[index:].replace(before, after, 1)
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.assertNotEqual(changed, lean)
                check(python, changed, record["python_function"], record["lean_definition"],
                      record["signature_definition"], record["expected_signature"],
                      expected_call_bindings=record["expected_call_bindings"])

    def test_existing_candidate_contains_only_the_reviewed_scalar_address_changes(self):
        original_path = DIR / "matmul_dequantize.py"
        candidate_path = DIR / "matmul_dequantize_scalar_address.py"
        original = ast.parse(variant_source(original_path))
        candidate = ast.parse(variant_source(candidate_path))
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_scalar_address"):
                node.name = node.name.removesuffix("_scalar_address")
            if isinstance(node, ast.Name) and node.id.endswith("_scalar_address"):
                node.id = node.id.removesuffix("_scalar_address")
        for fn in candidate.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name in {"matmul4_kernel", "matmul_kernel"}:
                arg = next(a for a in fn.args.args if a.arg == "N")
                self.assertEqual(ast.unparse(arg.annotation), "tl.constexpr")
                arg.annotation = None
                index, stmt = next((i,n) for i,n in enumerate(fn.body) if isinstance(n, ast.AnnAssign)
                                   and ast.unparse(n.target) == "num_pid_in_group")
                self.assertEqual(ast.unparse(stmt.annotation), "tl.constexpr")
                self.assertEqual(ast.unparse(stmt.value), "GROUP_SIZE_M * triton.cdiv(N, BLOCK_SIZE_N)")
                fn.body[index] = ast.Assign(targets=[stmt.target],
                                            value=ast.parse("GROUP_SIZE_M * num_pid_n", mode="eval").body)
            if fn.name == "dequantize_kernel":
                stmt = next(n for n in fn.body if isinstance(n, ast.Assign)
                            and ast.unparse(n.targets[0]) == "fpb_offs")
                self.assertEqual(ast.unparse(stmt.value),
                                 "k_block_idx * (BLOCK_SIZE_K * stride_fpbk) + n_block_idx * (BLOCK_SIZE_N * stride_fpbn) + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_fpbk + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_fpbn")
                stmt.value = ast.parse("offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn", mode="eval").body
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(DIR)
        self.assertEqual(hashlib.sha256(variant_source(original_path).encode()).hexdigest(), provenance["original_sha256"])
        self.assertEqual(hashlib.sha256(variant_source(candidate_path).encode()).hexdigest(), provenance["candidate_sha256"])
