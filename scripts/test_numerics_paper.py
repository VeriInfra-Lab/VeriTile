"""Fixed-batch statistics and CPU replay regressions."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import check_numerics_paper as paper


class FixedBatchTests(unittest.TestCase):
    def test_common_mask_sample_variance_and_separate_peaks(self):
        reference = np.array([0, 4, np.nan, 4, 5, 100])
        candidate = np.array([1, 2, 0, np.inf, 2, -100])
        oracle = np.array([0, 2.5, 0, 3, np.nan, 0])
        obs = paper.observe(reference, candidate, oracle, np.array([1, 2, 1, 1, 1, 1]),
                            np.array([True, True, True, True, True, False]))
        np.testing.assert_array_equal(obs["delta"], [1., -1.])
        result = paper.evaluate(obs, [2, 3], 5.0)
        self.assertEqual(result["sampling"], {"drawn": 6, "in_domain": 5, "valid": 2,
                                               "excluded_domain": 1, "excluded_nonfinite": 3})
        self.assertAlmostEqual(result["bias"]["sample_std"], np.sqrt(2))
        self.assertEqual(result["bias"]["upper"], 5.)
        self.assertEqual(result["magnitude"]["reference_peak"], 1.5)
        self.assertEqual(result["magnitude"]["candidate_peak"], 1.)
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertEqual(paper.evaluate(obs, [2, 3], 4.99)["decision"], "REJECT")

    def test_peak_threshold_and_zero_reference(self):
        for peaks, ratio, decision in (
            ([2., 20.], 10., "ACCEPT"), ([2., 20.01], 10.005, "REJECT"),
            ([0., 0.], 0., "ACCEPT"), ([0., 1.], "+inf", "REJECT"),
            ([1., 0.], 0., "ACCEPT"),
        ):
            with self.subTest(peaks=peaks):
                obs = dict(delta=np.zeros(2), peaks=np.array(peaks), in_domain=np.array(2, np.int64))
                result = paper.evaluate(obs, [1, 2], .05)
                self.assertEqual(result["magnitude"]["ratio"], ratio)
                self.assertEqual(result["decision"], decision)

    def test_insufficient_samples_and_residual_oracle(self):
        reference, candidate = np.array([0., np.inf]), np.array([0., 2.])
        obs = paper.observe(reference, candidate, np.zeros(2), np.ones(2), np.ones(2, dtype=bool),
                            (np.array([.125, np.inf]), np.array([.25, np.nan])))
        np.testing.assert_array_equal(obs["peaks"], [.125, .25])
        self.assertEqual(paper.evaluate(obs, [1, 2], .05)["decision"], "INCONCLUSIVE")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name)
        primary = paper.primary
        self.profile = deepcopy(primary.load_module(primary.DEFAULT_PROFILE).PROFILE)
        self.profile.update(shape=[1, 2], rules=["ADD-ASSOC"], formats=self.profile["formats"][-1:])
        self.rule = self.profile["rules"][0]
        self.fmt = self.profile["formats"][0]
        sources = {}
        for name, path in paper.source_paths("primary").items():
            saved = self.bundle / "sources" / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(path.read_bytes())
            sources[name] = paper.file_digest(saved)
        self.manifest = {"bundle_version": paper.BUNDLE_VERSION, "catalog": "primary",
                         "profile": self.profile, "protocol": paper.protocol(self.profile),
                         "smoke": False, "sources": sources, "backend": {"kind": "synthetic"}}
        self.directory = self.bundle / f"{self.fmt['name']}__{self.rule}"
        self.directory.mkdir()
        self.obs = dict(delta=np.zeros(2), peaks=np.zeros(2), in_domain=np.array(2, np.int64))
        self.write_bundle()

    def write_bundle(self):
        primary = paper.primary
        manifest_path = self.bundle / "manifest.json"
        primary.write_json(manifest_path, self.manifest)
        ptx_hashes = {}
        for name in paper.ptx_files(self.fmt):
            path = self.directory / name
            path.write_text("// Synthetic test fixture\n")
            ptx_hashes[name] = paper.file_digest(path)
        observations = self.directory / "observations.npz"
        np.savez_compressed(observations, **self.obs)
        self.record = {"rule": self.rule, "format": self.fmt["name"], "state": "COMPLETE",
                       "manifest_sha256": paper.file_digest(manifest_path),
                       "seed": primary.seed_for(self.profile, self.fmt, self.rule),
                       "lowerings": ptx_hashes, "observations_sha256": paper.file_digest(observations),
                       "result": paper.evaluate(self.obs, self.profile["shape"], .05,
                                                self.manifest["smoke"])}
        primary.write_json(self.directory / "record.json", self.record)

    def test_replay_recomputes_gates_and_excludes_smoke(self):
        report = paper.replay(self.bundle)
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["accepted"]), 1)
        self.record["result"]["decision"] = "REJECT"
        paper.primary.write_json(self.directory / "record.json", self.record)
        with self.assertRaisesRegex(ValueError, "disagree with replay"):
            paper.replay(self.bundle)
        self.manifest["smoke"] = True
        self.write_bundle()
        report = paper.replay(self.bundle)
        self.assertEqual(report["rows"][0]["decision"], "SMOKE_ONLY")
        self.assertEqual(report["accepted"], [])

    def test_replay_rejects_a_record_from_another_profile(self):
        self.profile["shape"] = [2, 1]
        self.manifest["protocol"] = paper.protocol(self.profile)
        paper.primary.write_json(self.bundle / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "record identity mismatch"):
            paper.replay(self.bundle)

    def test_replay_rejects_a_modified_observation_file(self):
        path = self.directory / "observations.npz"
        with path.open("ab") as output:
            output.write(b"modified")
        with self.assertRaisesRegex(ValueError, "observation hash mismatch"):
            paper.replay(self.bundle)


if __name__ == "__main__":
    unittest.main()
