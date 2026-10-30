"""Regression checks for deferred evidence, declaration binding and omissions."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import deferred_numerical_experiments as queue


class DeferredNumericalExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = queue.collect()

    def changed_json(self, filename, mutate):
        original = queue.Inputs.read

        def read(inputs, name, expected=None):
            data = original(inputs, name, expected)
            if name == filename:
                data = copy.deepcopy(data)
                mutate(data)
            return data

        return patch.object(queue.Inputs, "read", read)

    def test_all_five_pending_profiles_remain_pending(self):
        pending = {r["id"] for r in self.current["rules"]
                   if r["queue_status"] == "pending_exact_binding"}
        self.assertEqual(pending, {"fp32_mul_commute_typed", "BGMV-ADD-fp64-real",
                                  "BGMV-ADD-fp32-fp16", "BGMV-ADD-fp32-bf16", "BGMV-ADD-fp32-fp32"})

    def test_report_and_counterexample_are_not_native_acceptance(self):
        for row in self.current["rules"]:
            self.assertFalse(row["native_whole_kernel_validated"])
        refuted = [r for r in self.current["rules"]
                   if r["queue_status"] == "refuted_unrestricted_assumption"]
        self.assertEqual([r["id"] for r in refuted], ["ROUND-FP16-ID"])
        reports = [r for r in self.current["rules"] if r["queue_status"] == "reported_profile_only"]
        self.assertTrue(reports)
        self.assertTrue(all(r.get("existing_report") for r in reports))

    def test_new_raw_copy_tasks_are_not_omitted_from_old_native_plan(self):
        rows = {r["task"]: r for r in self.current["tasks"]}
        self.assertEqual(len(rows), 184)
        for name in ("destindex_copy", "matrix_transpose"):
            row = rows[name]
            self.assertEqual(row["native_execution"], "NOT_RUN")
            self.assertTrue(row["binding_relations"])
            self.assertTrue(row["declarations"])
            self.assertFalse(row["native_whole_kernel_validated"])

    def test_historical_job_label_is_preserved(self):
        row = next(r for r in self.current["tasks"] if r["task"] == "mixed_sparse_attention")
        self.assertEqual(row["existing_native_job"]["source_job_label"],
                         "mixed_sparse_attention (maximum revision)")

    def test_new_execution_cannot_be_silently_relabelled_not_run(self):
        with self.changed_json(queue.BASE + "fp_value_extensions.json",
                               lambda d: d.update(native_gpu_execution=True)):
            with self.assertRaisesRegex(ValueError, "New execution evidence"):
                queue.collect()

    def test_unknown_rule_status_is_rejected(self):
        with self.changed_json(queue.BASE + "numeric_experiment_tasks.json",
                               lambda d: d[0].update(status="compiled_so_assumed_validated")):
            with self.assertRaisesRegex(ValueError, "Unknown numerical rule status"):
                queue.collect()

    def test_supplement_cannot_turn_compilation_into_experimental_success(self):
        with self.changed_json(queue.SUPPLEMENT,
                               lambda d: d["obligations"][0].update(status="experiment_passed")):
            with self.assertRaisesRegex(ValueError, "Unreviewed supplemental evidence status"):
                queue.collect()

    def test_changed_bound_native_recipe_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "recipe.json").write_text('{"status":"NOT_RUN"}')
            digest = hashlib.sha256((root / "recipe.json").read_bytes()).hexdigest()
            (root / "recipe.json").write_text('{"status":"PASS"}')
            with self.assertRaisesRegex(ValueError, "Changed bound input"):
                queue.Inputs(root).read("recipe.json", digest)

    def test_changed_statement_is_rejected_even_when_file_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Contract.lean").write_text("specification contract : False := by contradiction\n")
            row = {"name": "contract", "statement_sha256": hashlib.sha256(b"contract : True").hexdigest()}
            with self.assertRaisesRegex(ValueError, "Changed numerical declaration"):
                queue.Inputs(root).declaration("Contract.lean", row)

    def test_duplicate_task_cannot_hide_missing_task(self):
        with self.changed_json(queue.BASE + "fp_value_current_status.json",
                               lambda d: d["tasks"].append(d["tasks"][0])):
            with self.assertRaisesRegex(ValueError, "Duplicate task"):
                queue.collect()


if __name__ == "__main__":
    unittest.main()
