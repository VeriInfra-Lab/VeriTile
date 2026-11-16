"""Fixed-budget, restoration, uncertainty and evidence-tampering regressions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import mpmath
import numpy as np

from scripts import check_two_gates as gates
from scripts import two_gates_sampling as sampling, two_gates_statistics as statistics


def args(**changes):
    values = dict(config=None, profile=None, samples=None, workers=None, smoke=False,
        catalog="all", rules=None, formats=None, iid=False)
    return SimpleNamespace(**{**values, **changes})


def row(n=2, mean=0., m2=0., p=1., q=1., selected=None):
    selected = n if selected is None else selected
    return [n, mean, m2, p, q, n, n, 0., selected, p if selected else 0., q if selected else 0., selected]


class PlanTests(unittest.TestCase):
    def test_canonical_json_roundtrip_keeps_multi_catalogue_plan_order(self):
        plan = gates.make_plan(args())
        saved = json.loads(gates.primary.registry.canonical_json(plan))
        self.assertEqual(saved["items"], gates.build_items(saved["profiles"], saved["settings"], saved["iid_control"]))

    def test_defaults_inventory_and_full_peak_budget(self):
        p = gates.make_plan(args())
        self.assertEqual(len(p["items"]), 168)
        supported = [i for i in p["items"] if i["unsupported"] is None]
        self.assertEqual(len(supported), 115)
        weighted = [i for i in supported if i["method"] == "weighted"]
        self.assertEqual({i["rule"] for i in weighted}, sampling.WEIGHTED_RULES)
        self.assertTrue(all(i["source_count"] == 2**32+2**25 for i in weighted))
        self.assertEqual(sum(i["source_count"] for i in supported), 115*2**32+3*2**25)
        self.assertTrue(all("replicates" not in v and "batch" not in v for v in p["profiles"].values()))

    def test_filter_and_explicit_iid_control(self):
        a = args(); a.catalog = "primary"; a.formats = "fp32"; a.rules = "CANCEL"; a.iid = True
        p = gates.make_plan(a)
        self.assertEqual(len(p["items"]), 1)
        self.assertEqual(p["items"][0]["method"], "iid")
        self.assertEqual(p["items"][0]["source_count"], 2**32)
        self.assertIsNone(p["items"][0]["target_near_count"])
        a.rules = "missing-rule"
        with self.assertRaisesRegex(ValueError, "unknown --rules"):
            gates.make_plan(a)

    def test_weighting_is_restricted_to_the_validated_target_and_precision(self):
        p = gates.make_plan(args()); i = next(i for i in p["items"] if i["method"] == "weighted")
        normal = {"family": "normal", "mean": 1., "std": 1.}
        self.assertTrue(sampling.weighted("primary", "CANCEL", i["format"], normal))
        self.assertFalse(sampling.weighted("primary", "CANCEL", i["format"], {**normal, "std": 2.}))
        self.assertFalse(sampling.weighted("supplement", "CANCEL", i["format"], normal))
        self.assertFalse(sampling.weighted("primary", "CANCEL", {**i["format"], "input": "bf16"}, normal))

    def test_settings_reject_implicit_boolean_counts_and_unvalidated_envelopes(self):
        p = gates.make_plan(args())
        for key, value in [("samples", True), ("workers", True), ("samples", 1),
                           ("near_half_width", .1), ("rejection_envelope", 1.),
                           ("execution_shape", [0, 3]), ("bias_tolerance_ulp", float("inf"))]:
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                gates.validate_settings({**p["settings"], key: value})

    def test_execution_partition_never_changes_the_sample_budget(self):
        p = gates.make_plan(args()); i = next(i for i in p["items"] if i["method"] == "weighted")
        s = {**p["settings"], "samples": 19, "near_samples": 5, "execution_shape": [2, 3]}
        parts = gates.part_plan(i, s)
        self.assertEqual(sum(n for st, n in parts if st == "near"), 5)
        self.assertEqual(sum(n for st, n in parts if st == "rest"), 19)
        self.assertEqual(parts[-1], ("rest", 1))


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.settings = gates.make_plan(args())["settings"]

    def test_weighted_mean_and_design_se_use_every_sample_without_normalization(self):
        x = np.array([-6., 2.]); y = np.array([3., 4., 5.])
        a, b = statistics.moments(x), statistics.moments(y)
        rows = np.array([row(2, a["mean"], a["m2"]), row(3, b["mean"], b["m2"])])
        result = statistics.evaluate_parts(rows, [0, 1], [2, 3], [.1, .9], self.settings)
        self.assertAlmostEqual(result["bias"]["mean"], .1*x.mean()+.9*y.mean())
        self.assertAlmostEqual(result["bias"]["standard_error"]**2, .1**2*x.var(ddof=1)/2+.9**2*y.var(ddof=1)/3)

    def test_biased_full_bank_reference_outlier_cannot_hide_restored_peak_failure(self):
        rows = np.array([row(p=100., q=1., selected=0), row(p=1., q=20.)])
        result = statistics.evaluate_parts(rows, [0, 1], [2, 2], [.01, .99], self.settings)
        self.assertEqual(result["bank_peaks"]["reference_peak"], 100.)
        self.assertEqual(result["magnitude"]["reference_peak"], 1.)
        self.assertEqual(result["magnitude"]["ratio"], 20.)
        self.assertEqual(result["bias"]["status"], "PASS")
        self.assertEqual(result["decision"], "REJECT")

    def test_insufficient_supply_is_inconclusive_and_smoke_is_never_admitted(self):
        r = sampling.Restoration.create(10, 2)
        self.assertEqual(r.finish()["state"], "INCONCLUSIVE")
        rows = np.array([row(), row()])
        result = statistics.evaluate_parts(rows, [0, 1], [2, 2], [.1, .9], self.settings, restoration=r.finish())
        self.assertEqual(result["decision"], "INCONCLUSIVE")
        result = statistics.evaluate_parts(rows, [0, 1], [2, 2], [.1, .9], self.settings, smoke=True)
        self.assertEqual(result["decision"], "SMOKE_ONLY")

    def test_integer_and_zero_reference_peak_cases(self):
        for p, q, verdict in [(0., 0., "ACCEPT"), (0., 1., "REJECT"), (2., 20., "ACCEPT"), (2., 20.01, "REJECT")]:
            result = statistics.evaluate_parts(np.array([row(p=p, q=q)]), [0], [2], [1.], self.settings)
            self.assertEqual(result["decision"], verdict)

    def test_empty_selected_subset_cannot_report_a_positive_peak(self):
        data = np.array([row(selected=0)])
        data[0, 9] = 1.
        with self.assertRaisesRegex(ValueError, "without a selected valid sample"):
            statistics.evaluate_parts(data, [0], [2], [1.], self.settings)

    def test_active_rejection_requires_complete_mask_and_randomness_evidence(self):
        metadata = {"stratum": "near", "drawn": 2, "before": {"near_remaining": 1}}
        with self.assertRaisesRegex(ValueError, "complete restoration-mask"):
            statistics.replay_audit({}, np.array(row()), metadata, "CANCEL", sampling)


class SamplingTests(unittest.TestCase):
    def test_binomial_counts_support_full_budget_and_endpoints(self):
        for p, expected in [(0., 0), (1., 2**32)]:
            self.assertEqual(sampling.mixture_count(2**32, p, 1), expected)
        for rule in sampling.WEIGHTED_RULES:
            p = sampling.constants(rule)["p0"]
            k = np.random.default_rng(1928).binomial(2**32, p, size=10000)
            se = np.sqrt(2**32*p*(1-p)/len(k))
            self.assertLess(abs(k.mean()-2**32*p), 6*se)

    def test_gaussian_probabilities_weights_and_envelopes(self):
        with mpmath.workdps(100):
            for rule in sampling.WEIGHTED_RULES:
                c = sampling.constants(rule)
                self.assertLess(float(c["ratio_upper_bound_100digits"]), sampling.ENVELOPE)
                t = np.linspace(-sampling.H, sampling.H, 257)
                latent = t-1 if rule == "CANCEL" else t
                self.assertTrue(np.all(sampling.weights(latent, rule, c) > 0))
                if rule != "CANCEL":
                    mu, var, h = mpmath.mpf(c["sum_mean"]), mpmath.mpf(c["sum_variance"]), mpmath.mpf(2)**-12
                    p = mpmath.quad(lambda x: mpmath.exp(-(x-mu)**2/(2*var))/mpmath.sqrt(2*mpmath.pi*var), [-h, h])
                    self.assertLess(abs(p-mpmath.mpf(c["p0_100digits"])), mpmath.mpf('1e-98'))

    def test_nontrivial_rejection_and_mixture_restore_joint_maxima(self):
        # Reference outliers and candidate faults are correlated via the input.
        target = np.array([.005, .015, .08, .1, .2, .25, .345, .005])
        q = np.array([.8, .2]); p0 = target[:2].sum(); ratio = target[:2]/p0/q
        rng = np.random.default_rng(202611149117); trials, n, bank = 100000, 64, 128
        k = rng.binomial(n, p0, size=trials)
        near = rng.choice(2, size=(trials, bank), p=q)
        accept = rng.random(near.shape) < ratio[near]/4.
        self.assertTrue((accept.sum(axis=1) >= k).all())
        choose = accept & (np.cumsum(accept, axis=1) <= k[:, None])
        far = rng.choice(np.arange(2, 8), size=(trials, n), p=target[2:]/(1-p0))
        far_choose = np.arange(n)[None, :] < (n-k)[:, None]
        ep, eq = np.array([100., .1, 1, 1, 1, 1, 1, 1]), np.array([1., .1, 1, 1, 1, 1, 1, 20])
        p = np.maximum(np.where(choose, ep[near], 0).max(axis=1), np.where(far_choose, ep[far], 0).max(axis=1))
        qmax = np.maximum(np.where(choose, eq[near], 0).max(axis=1), np.where(far_choose, eq[far], 0).max(axis=1))
        expected = 1-(.995**n-.99**n)
        self.assertLess(abs(np.mean(qmax <= 10*p)-expected), 6*np.sqrt(expected*(1-expected)/trials))


class ReplayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        a = args(); a.catalog = "primary"; a.rules = "ADD-COMMUTE"; a.formats = "fp32"; a.samples = 4
        self.plan = gates.make_plan(a); self.plan["settings"]["execution_shape"] = [1, 2]
        self.manifest = {"format": gates.BUNDLE_FORMAT, "schema": 1, **self.plan,
            "backend": {"numpy": np.__version__, "mpmath": mpmath.__version__},
            "sources": {n: gates.file_digest(p) for n, p in gates.source_paths().items()}}
        for n, p in gates.source_paths().items():
            dst = self.root/"sources"/n; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(p.read_bytes())
        self.write_fixture()

    def write_fixture(self):
        gates.primary.write_json(self.root/"manifest.json", self.manifest)
        item = self.manifest["items"][0]; self.directory = self.root/item["id"]; self.directory.mkdir(exist_ok=True)
        rows = np.array([row(), row()]); strata = np.array([0, 0], np.int8); drawn = np.array([2, 2], np.int64)
        gates.save_npz(self.directory/"statistics.npz", {"rows": rows, "strata": strata, "drawn": drawn})
        gates.primary.write_json(self.directory/"fingerprints.json", ["synthetic-a", "synthetic-b"])
        lowerings = {}
        for name in gates.ptx_files(item["format"]):
            key = f"ptx/2/{name}"; path = self.directory/key; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// Synthetic unit-test fixture, not GPU evidence.\n"); lowerings[key] = gates.file_digest(path)
        audits = []
        for j in range(2):
            name = f"audit-{j:05d}.npz"
            gates.save_npz(self.directory/name, {"delta_values": np.array([0.]), "delta_counts": np.array([2], np.int64),
                "peaks": np.ones(2), "selected_peaks": np.ones(2), "valid_count": np.array(2, np.int64)})
            audits.append({"part": j, "stratum": "iid", "drawn": 2, "before": None, "file": name, "sha256": gates.file_digest(self.directory/name)})
        self.record = {"state": "COMPLETE", "item": item, "manifest_sha256": gates.file_digest(self.root/"manifest.json"),
            "result": statistics.evaluate_parts(rows, strata, drawn, [1.], self.manifest["settings"], self.manifest["smoke"]),
            "restoration": None, "lowerings": lowerings, "audits": audits,
            "statistics_sha256": gates.file_digest(self.directory/"statistics.npz"),
            "fingerprints_sha256": gates.file_digest(self.directory/"fingerprints.json")}
        gates.primary.write_json(self.directory/"record.json", self.record)

    def test_replay_all_parts_and_exclude_smoke(self):
        report = gates.replay(self.root)
        self.assertTrue(report["complete"]); self.assertEqual(len(report["accepted"]), 1)
        self.assertEqual(report["part_statistics_verified"], 2); self.assertEqual(report["raw_audits_verified"], 2)
        self.manifest["smoke"] = True; self.write_fixture()
        self.assertEqual(gates.replay(self.root)["accepted"], [])

    def test_modified_decision_is_recomputed(self):
        self.record["result"]["decision"] = "REJECT"
        gates.primary.write_json(self.directory/"record.json", self.record)
        with self.assertRaisesRegex(ValueError, "disagree with replay"):
            gates.replay(self.root)

    def test_missing_and_repeated_audits_are_rejected(self):
        self.record["audits"] = [self.record["audits"][0]]*2
        gates.primary.write_json(self.directory/"record.json", self.record)
        with self.assertRaisesRegex(ValueError, "required raw audit"):
            gates.replay(self.root)

    def test_modified_observations_or_ptx_are_rejected(self):
        with (self.directory/"statistics.npz").open("ab") as f:
            f.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "observation hash mismatch"):
            gates.replay(self.root)
        self.write_fixture()
        next((self.directory/"ptx/2").glob("*.ptx")).write_text("tamper")
        with self.assertRaisesRegex(ValueError, "PTX hash mismatch"):
            gates.replay(self.root)

    def test_modified_seed_and_cross_manifest_record_are_rejected(self):
        self.manifest["items"][0]["seeds"]["iid"] += 1
        gates.primary.write_json(self.root/"manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            gates.replay(self.root)


if __name__ == "__main__":
    unittest.main()
