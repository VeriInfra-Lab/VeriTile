"""Source casts, loop capture, dtype argument and existing candidate edits."""
import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("bench/repairs/tritonbench_g/layer_norm_liger")


class LayerNormLigerBody(unittest.TestCase):
    def test_all_four_bodies(self):
        self.assertEqual(len(check_manifest(ROOT, ROOT / BASE / "jit-correspondence.json")), 4)

    def reject_mutation(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / BASE
            dest.mkdir(parents=True)
            for name in ["Kernels.lean", "layer_norm_liger_repaired.py",
                         "layer_norm_liger_optimized.py", "jit-correspondence.json"]:
                shutil.copy2(ROOT / BASE / name, dest / name)
            path = dest / "Kernels.lean"
            text = path.read_text()
            self.assertIn(before, text)
            path.write_text(text.replace(before, after, 1))
            with self.assertRaises(ValueError):
                check_manifest(root, dest / "jit-correspondence.json")

    def test_all_index_casts_retained(self):
        self.reject_mutation("row_idx = tl.program_id(0).to(tl.int64)",
                             "row_idx = tl.program_id(0)")

    def test_original_denominator_not_prewritten(self):
        self.reject_mutation("mean = tl.sum(X_row, axis=0) / n_cols",
                             "mean = tl.sum(X_row, axis=0) / n_cols.to(tl.float32)")

    def test_original_step_tensor_retained(self):
        self.reject_mutation("range(row_start, row_end, one):",
                             "range(row_start, row_end, 1):")

    def test_store_dtype_retained(self):
        self.reject_mutation("dx.to(dtype)", "dx")

    def test_caller_dtype_checked(self):
        self.reject_mutation("let dtype : FloatDType := .fp32",
                             "let dtype : FloatDType := .fp16")

    def test_caller_dtype_cannot_be_omitted(self):
        self.reject_mutation("  let dtype : FloatDType := .fp32\n", "")

    def test_original_loop_pointer_updates_retained(self):
        self.reject_mutation("Mean_ptr += 1", "Mean_ptr += 2")

    def test_masked_variance_retained(self):
        self.reject_mutation("tl.where(mask, (X_row - mean) * (X_row - mean), 0.0)",
                             "(X_row - mean) * (X_row - mean)")

    def test_only_existing_candidate_rewrites(self):
        def functions(name):
            return {n.name: n for n in ast.parse((ROOT / BASE / name).read_text()).body
                    if isinstance(n, ast.FunctionDef)}

        original = functions("layer_norm_liger_repaired.py")
        candidates = functions("layer_norm_liger_optimized.py")
        forward = copy.deepcopy(candidates["_layer_norm_forward_kernel"])
        centered = next(n for n in forward.body if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "centered")
        forward.body.remove(centered)

        class Restore(ast.NodeTransformer):
            def __init__(self, name, value):
                self.name, self.value = name, value

            def visit_Name(self, node):
                return copy.deepcopy(self.value) if node.id == self.name else node

        forward = Restore("centered", centered.value).visit(forward)
        backward = copy.deepcopy(candidates["_layer_norm_backward_kernel"])
        denominator = next(n for n in backward.body if isinstance(n, ast.Assign)
                           and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "denominator")
        self.assertEqual(ast.dump(denominator.value),
                         ast.dump(ast.parse("n_cols.to(tl.float32)", mode="eval").body))
        backward.body.remove(denominator)
        backward = Restore("denominator", ast.Name(id="n_cols", ctx=ast.Load())).visit(backward)
        for candidate, name in [(forward, "_layer_norm_forward_kernel"),
                                (backward, "_layer_norm_backward_kernel")]:
            self.assertEqual(ast.dump(candidate.args), ast.dump(original[name].args))
            self.assertEqual([ast.dump(n) for n in candidate.body],
                             [ast.dump(n) for n in original[name].body])


if __name__ == "__main__":
    unittest.main()
