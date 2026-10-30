"""CPU probes for removing/inserting one FP32 rounding before addition.

Uses the repository's numerical gates without adding a Lean admission. Binary64
inputs represent values before the proposed FP32 rounding. TwoSum retains the
sum residual, including the cases where binary64 addition lands on a binary32
midpoint. Raw observations support a separate statistical replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import numerical_gates as gates

GATES = {
    "bias": {"tau": 0.05, "se_multiplier": 5.0},
    "vars": {"quantile": 0.9, "horizon": 625000, "alpha": 0.00135,
             "bootstrap": 1000, "min_exceedances": 40, "warn": 10.0, "fail": 100.0},
    "warning_policy": "pass_only",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def two_sum(a, b):
    """Error-free binary64 addition for the finite, nonoverflowing probes."""
    high = a + b
    recovered_b = high - a
    low = (a - (high - recovered_b)) + (b - recovered_b)
    return high, low


def round_sum32(a, b):
    """Round the exact sum to binary32, correcting binary64 double rounding."""
    high, low = two_sum(a, b)
    rounded = high.astype(np.float32)
    if not np.isfinite(rounded).all():
        raise FloatingPointError("nonfinite output on an admitted input; do not skip it")
    # A binary32 midpoint has bit 28 set and bits 0..27 clear in a normal
    # binary64 significand. Only an exact midpoint can be affected by `low`.
    normal = np.abs(high) >= float(np.finfo(np.float32).tiny)
    midpoint = normal & ((high.view(np.uint64) & ((1 << 29) - 1)) == (1 << 28))
    if (~normal).any():
        scaled = np.ldexp(np.abs(high[~normal]), 149)
        midpoint[~normal] = scaled - np.floor(scaled) == 0.5
    indices = np.flatnonzero(midpoint & (low != 0))
    if len(indices):
        q = rounded[indices]
        above = q.astype(np.float64) > high[indices]
        move_up = ~above & (low[indices] > 0)
        move_down = above & (low[indices] < 0)
        rounded[indices[move_up]] = np.nextafter(q[move_up], np.float32(np.inf))
        rounded[indices[move_down]] = np.nextafter(q[move_down], np.float32(-np.inf))
        corrections = int(move_up.sum() + move_down.sum())
    else:
        corrections = 0
    return rounded, high, low, corrections


def rational_round32(value):
    """Independent exact-rational nearest-neighbour oracle for spot checks."""
    centre = np.float32(float(value))
    neighbours = [np.nextafter(centre, np.float32(-np.inf)), centre,
                  np.nextafter(centre, np.float32(np.inf))]
    finite = [x for x in neighbours if np.isfinite(x)]
    return min(finite, key=lambda x: (
        abs(Fraction.from_float(float(x)) - value), int(x.view(np.uint32)) & 1))


def check_rounding():
    rng = np.random.default_rng(714033)
    pairs = list(zip(rng.normal(1, 1, 2048), rng.normal(1, 1, 2048)))
    for midpoint, tiny in [(1 + 2**-24, 2**-70), (1 + 3 * 2**-24, 2**-70),
                           (2 - 2**-24, 2**-70), (2**-150, 2**-200)]:
        for sign in [-1, 1]:
            for offset in [-tiny, 0.0, tiny]:
                pairs.append((sign * midpoint, offset))
    pairs += [((1 + 2**-13) * (1 - 2**-13), -1.0), (0.0, 0.0)]
    a, b = np.asarray(pairs, dtype=np.float64).T.copy()
    actual, _, _, corrections = round_sum32(a, b)
    for x, y, observed in zip(a, b, actual):
        exact = Fraction.from_float(float(x)) + Fraction.from_float(float(y))
        expected = rational_round32(exact)
        if observed != expected:
            raise AssertionError((x.hex(), y.hex(), float(observed).hex(), float(expected).hex()))
    if not corrections:
        raise AssertionError("midpoint fixtures did not exercise double-rounding correction")
    return {"exact_rational_checks": len(pairs), "midpoint_corrections": corrections,
            "status": "PASS", "signed_zero": "numeric comparison, not bitwise zero-sign equivalence"}


def errors(output, high, low):
    difference, residual = two_sum(output.astype(np.float64), -high)
    return np.abs(difference + (residual - low))


def evaluate(arrays, config):
    forward = gates.evaluate(arrays, config["gates"], config["seed"], config["replicates"])
    reverse = gates.evaluate({"delta": -arrays["delta"],
                              "reference_error": arrays["candidate_error"],
                              "candidate_error": arrays["reference_error"]},
                             config["gates"], config["seed"], config["replicates"])
    return {"remove_rounding": forward, "insert_rounding": reverse}


def run(args):
    if args.replicates < 2 or args.samples < 1:
        raise ValueError("need at least two replicates and one sample per replicate")
    checks = check_rounding()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"relation": "R32(R32(u) + v) -> R32(u + v)",
              "rounding": "nearest, ties to even; gradual underflow",
              "arithmetic": "exact sum represented with binary64 TwoSum before R32",
              "oracle_error": "compensated binary64 distance to the TwoSum expansion",
              "domain": "finite inputs; nonfinite outputs are errors, never filtered",
              "distribution": {"family": "independent normal", "mean": 1.0, "std": 1.0},
              "u": "binary64 draw; no preliminary binary32 quantization",
              "v_profiles": ["fp32", "fp64"], "output": "fp32",
              "seed": args.seed, "replicates": args.replicates,
              "samples_per_replicate": args.samples,
              "stopping": "fixed budget chosen before sampling; no retry-until-PASS",
              "gates": GATES, "gpu_tested": False, "lean_admission": "NOT_IMPORTED",
              "scope": "CPU numerical probe; not the catalogue's 4096x4096 GPU profile",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "backend": {"python": platform.python_version(), "numpy": np.__version__,
                          "machine": platform.machine(), "platform": platform.platform()},
              "sources": {"run.py": sha(__file__), "numerical_gates.py": sha(gates.__file__)}}
    write_json(args.output / "config.json", config)
    write_json(args.output / "rounding_checks.json", checks)
    shutil.copyfile(__file__, args.output / "run.py")
    shutil.copyfile(gates.__file__, args.output / "numerical_gates.py")
    summary = {"config": config, "rounding_checks": checks, "results": {}}
    for v_profile in config["v_profiles"]:
        rng = np.random.default_rng(args.seed)
        collected = {name: [] for name in gates.OBSERVATIONS}
        diagnostics = {"intermediate_rounding_changed": 0, "outputs_differ": 0,
                       "binary64_midpoint_corrections": 0, "max_abs_local_ulp_delta": 0.0}
        started = time.perf_counter()
        for replicate in range(args.replicates):
            u = rng.normal(1, 1, args.samples)
            v = rng.normal(1, 1, args.samples)
            if v_profile == "fp32":
                v = v.astype(np.float32).astype(np.float64)
            rounded_u = u.astype(np.float32).astype(np.float64)
            reference, _, _, fixes1 = round_sum32(rounded_u, v)
            candidate, high, low, fixes2 = round_sum32(u, v)
            magnitude = np.abs(candidate)
            scale = (np.nextafter(magnitude, np.float32(np.inf)).astype(np.float64)
                     - magnitude.astype(np.float64))
            delta = (candidate.astype(np.float64) - reference.astype(np.float64)) / scale
            collected["delta"].append([float(delta.mean())])
            collected["reference_error"].append(float(errors(reference, high, low).max()))
            collected["candidate_error"].append(float(errors(candidate, high, low).max()))
            diagnostics["intermediate_rounding_changed"] += int(np.count_nonzero(rounded_u != u))
            diagnostics["outputs_differ"] += int(np.count_nonzero(reference != candidate))
            diagnostics["binary64_midpoint_corrections"] += fixes1 + fixes2
            index = int(np.argmax(np.abs(delta)))
            if abs(delta[index]) > diagnostics["max_abs_local_ulp_delta"]:
                diagnostics["max_abs_local_ulp_delta"] = float(abs(delta[index]))
                diagnostics["worst_local_ulp_sample"] = {
                    k: float(value[index]).hex() for k, value in
                    {"u": u, "v": v, "reference": reference, "candidate": candidate,
                     "exact_sum_high": high, "exact_sum_low": low}.items()}
            if (replicate + 1) % 512 == 0 or replicate + 1 == args.replicates:
                print(f"v={v_profile}: {replicate + 1}/{args.replicates} replicates; "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in collected.items()}
        observations = args.output / f"v_{v_profile}.npz"
        np.savez_compressed(observations, **arrays)
        result = {"v_profile": v_profile, "samples": args.samples * args.replicates,
                  "elapsed_seconds": time.perf_counter() - started, "diagnostics": diagnostics,
                  "observations_sha256": sha(observations), **evaluate(arrays, config)}
        write_json(args.output / f"v_{v_profile}.json", result)
        summary["results"][v_profile] = result
        write_json(args.output / "summary.json", summary)
        for direction in ["remove_rounding", "insert_rounding"]:
            row = result[direction]
            print(f"v={v_profile} {direction}: {row['decision']}; "
                  f"bias={row['bias']['status']} B={row['bias'].get('upper')}; "
                  f"vars={row['vars']['status']} U={row['vars'].get('upper')}", flush=True)


def replay(args):
    summary = json.loads((args.bundle / "summary.json").read_text())
    config = summary["config"]
    for name, expected in config["sources"].items():
        if sha(args.bundle / name) != expected:
            raise ValueError(f"frozen source hash mismatch: {name}")
    if sha(gates.__file__) != config["sources"]["numerical_gates.py"]:
        raise ValueError("current gate implementation differs from the frozen run")
    for profile in config["v_profiles"]:
        stored = summary["results"][profile]
        path = args.bundle / f"v_{profile}.npz"
        if sha(path) != stored["observations_sha256"]:
            raise ValueError(f"observation hash mismatch: {profile}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {k: loaded[k] for k in loaded.files}
        if arrays["delta"].shape != (config["replicates"], 1):
            raise ValueError("wrong replicate dimensions")
        recomputed = evaluate(arrays, config)
        for direction, value in recomputed.items():
            if value != stored[direction]:
                raise ValueError(f"gate replay mismatch: {profile}/{direction}")
    result = {"status": "PASS", "profiles_replayed": config["v_profiles"],
              "kind": "independent process statistical replay; not a GPU or raw-output replay",
              "summary_sha256": sha(args.bundle / "summary.json")}
    write_json(args.bundle / "replay.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--replicates", type=int, default=4096)
    run_parser.add_argument("--samples", type=int, default=65536)
    run_parser.add_argument("--seed", type=int, default=20261021)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(check_rounding(), indent=2))
    elif args.command == "run":
        run(args)
    else:
        replay(args)
