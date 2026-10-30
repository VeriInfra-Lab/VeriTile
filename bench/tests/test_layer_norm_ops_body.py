"""Literal bodies, invocation metadata and the two existing candidate edits."""
import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("bench/repairs/tritonbench_g/layer_norm_ops")


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)}


class LayerNormOpsBody(unittest.TestCase):
    def test_all_four_bodies(self):
        self.assertEqual(len(check_manifest(ROOT, ROOT / BASE / "jit-correspondence.json")), 4)

    def reject_lean_mutation(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / BASE
            destination.mkdir(parents=True)
            for name in ["Kernels.lean", "layer_norm_ops_repaired.py",
                         "layer_norm_ops_candidates.py", "jit-correspondence.json"]:
                shutil.copy2(ROOT / BASE / name, destination / name)
            path = destination / "Kernels.lean"
            text = path.read_text()
            self.assertIn(before, text)
            path.write_text(text.replace(before, after, 1))
            with self.assertRaises(ValueError):
                check_manifest(root, destination / "jit-correspondence.json")

    def test_none_branch_cannot_disappear(self):
        self.reject_lean_mutation(
            "tl.load(Mean + row * stride_mean) if Mean is not None else 0.0",
            "tl.load(Mean + row * stride_mean)")

    def test_input_cast_is_preserved(self):
        self.reject_lean_mutation(
            "other=0.0).to(tl.float32)", "other=0.0).to(tl.float16)")

    def test_loop_bounds_are_preserved(self):
        self.reject_lean_mutation("range(row_start, row_end):", "range(0, row_end):")

    def test_store_mask_is_preserved(self):
        self.reject_lean_mutation("tl.store(DX + cols, dx, mask=mask)",
                                  "tl.store(DX + cols, dx)")

    def test_storage_profile_is_checked(self):
        self.reject_lean_mutation('(\"X\",.fp32)', '(\"X\",.fp16)')

    def test_mean_presence_binding_is_checked(self):
        self.reject_lean_mutation("pointer_presence [Mean := meanPresent]",
                                  "pointer_presence [Mean := Bool.true]")

    def test_mean_presence_follows_existing_rms_contract(self):
        text = (ROOT / BASE / "Kernels.lean").read_text()
        self.assertEqual(text.count("let meanPresent : Bool := !IS_RMS_NORM"), 2)

    def test_only_existing_candidate_rewrites(self):
        original = functions(ROOT / BASE / "layer_norm_ops_repaired.py")
        candidate = functions(ROOT / BASE / "layer_norm_ops_candidates.py")
        fwd = original["_layer_norm_fwd_1pass_kernel"]
        restored = copy.deepcopy(candidate["_layer_norm_fwd_1pass_kernel_mask_reused"])
        is_mask = lambda n: isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "mask"
        old_position = next(i for i, n in enumerate(fwd.body) if is_mask(n))
        new_position = next(i for i, n in enumerate(restored.body) if is_mask(n))
        mask = restored.body.pop(new_position)
        self.assertLess(new_position, old_position)
        self.assertEqual(ast.dump(mask), ast.dump(fwd.body[old_position]))

        class RestoreEarlyComparisons(ast.NodeTransformer):
            def visit_Name(self, node):
                return copy.deepcopy(mask.value) if node.id == "mask" else node

        for i in range(old_position):
            restored.body[i] = RestoreEarlyComparisons().visit(restored.body[i])
        restored.body.insert(old_position, mask)
        self.assertEqual(ast.dump(restored.args), ast.dump(fwd.args))
        self.assertEqual([ast.dump(n) for n in restored.body], [ast.dump(n) for n in fwd.body])

        bwd = original["_layer_norm_bwd_kernel"]
        restored = copy.deepcopy(candidate["_layer_norm_bwd_kernel_row_start_reused"])
        position = next(i for i, n in enumerate(restored.body)
                        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == "row_end")
        self.assertEqual(ast.dump(restored.body[position].value),
                         ast.dump(ast.parse("min(row_start + rows_per_program, M)", mode="eval").body))
        restored.body[position] = copy.deepcopy(bwd.body[position])
        self.assertEqual(ast.dump(restored.args), ast.dump(bwd.args))
        self.assertEqual([ast.dump(n) for n in restored.body], [ast.dump(n) for n in bwd.body])


if __name__ == "__main__":
    unittest.main()
