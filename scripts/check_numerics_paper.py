#!/usr/bin/env python3
"""Run one fixed batch per scalar rule/format and replay the paper's two gates."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

if __package__:
    from . import check_numerics as primary, supplement_numerics as supplement
else:
    import check_numerics as primary
    import supplement_numerics as supplement

BUNDLE_VERSION = "paper-two-gates-v1"
BIAS_MULTIPLIER = 5.0
PEAK_RATIO_LIMIT = 10.0
ROOT = primary.ROOT


def implementation(catalog):
    if catalog == "primary":
        return primary
    if catalog == "supplement":
        return supplement
    raise ValueError(f"unknown catalogue: {catalog}")


def source_paths(catalog):
    paths = [*primary.SOURCES, Path(supplement.__file__), Path(__file__).resolve()]
    if catalog == "supplement":
        paths.extend((supplement.KERNELS, supplement.CATALOG))
    return {str(path.relative_to(ROOT)): path for path in paths}


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol(profile):
    return {
        "version": BUNDLE_VERSION,
        "batch_shape": profile["shape"],
        "bias_tolerance_ulp": profile["gates"]["bias"]["tau"],
        "bias_se_multiplier": BIAS_MULTIPLIER,
        "peak_ratio_limit": PEAK_RATIO_LIMIT,
        "selection": "in-domain quantized operands and finite reference, candidate and oracle",
        "ulp": "output-format spacing at rounded abs(oracle); inward at max finite; minimum subnormal at zero",
    }


def instances(profile):
    return {f"{fmt['name']}__{rule}": (rule, fmt)
            for fmt in profile["formats"] for rule in profile["rules"]}


def load_profile(args):
    engine = implementation(args.catalog)
    profile = engine.validate_profile(deepcopy(
        primary.load_module(args.profile or engine.DEFAULT_PROFILE).PROFILE))
    if args.rules is not None:
        profile["rules"] = args.rules.split(",")
    if args.formats is not None:
        requested = set(args.formats.split(","))
        if not requested <= {fmt["name"] for fmt in profile["formats"]}:
            raise ValueError("unknown --formats entry")
        profile["formats"] = [fmt for fmt in profile["formats"] if fmt["name"] in requested]
    if args.smoke:
        profile["shape"] = [32, 33]
    return engine.validate_profile(profile)


def unsupported(catalog, rule, fmt):
    if catalog == "supplement":
        return supplement.unsupported(rule, fmt)
    if rule == "BF16-WIDEN-RETURN" and (fmt["input"], fmt["output"]) != ("bf16", "bf16"):
        return "requires bf16 input and output"
    return None


def observe(reference, candidate, oracle, ulps, in_domain, errors=None):
    """Collect individual signed ULP shifts and the two peaks on the same lanes."""
    if in_domain.dtype != np.bool_ or any(
            a.shape != in_domain.shape for a in (reference, candidate, oracle, ulps)):
        raise ValueError("outputs, ULPs and domain mask must share the batch shape")
    valid = in_domain & np.isfinite(reference) & np.isfinite(candidate) & np.isfinite(oracle)
    p, q, o = (np.asarray(x[valid], dtype=np.float64) for x in (reference, candidate, oracle))
    scale = ulps[valid]
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid ULP spacing on retained samples")
    if errors is None:
        errors = (np.abs(p - o), np.abs(q - o))
    else:
        if len(errors) != 2 or any(error.shape != valid.shape for error in errors):
            raise ValueError("residual oracle errors must match the batch shape")
        errors = tuple(error[valid] for error in errors)
    if any(not np.isfinite(error).all() or (error < 0).any() for error in errors):
        raise ValueError("invalid oracle error on retained samples")
    return {
        "delta": (q - p) / scale,
        "peaks": np.array([error.max(initial=0.0) for error in errors], dtype=np.float64),
        "in_domain": np.array(np.count_nonzero(in_domain), dtype=np.int64),
    }


def evaluate(observations, shape, tolerance, smoke=False):
    if set(observations) != {"delta", "peaks", "in_domain"}:
        raise ValueError("unexpected fixed-batch observation fields")
    delta, peaks, in_domain = (observations[k] for k in ("delta", "peaks", "in_domain"))
    if (delta.dtype != np.float64 or delta.ndim != 1 or not np.isfinite(delta).all()
            or peaks.dtype != np.float64 or peaks.shape != (2,)
            or not np.isfinite(peaks).all() or (peaks < 0).any()
            or in_domain.dtype != np.int64 or in_domain.shape != ()):
        raise ValueError("invalid fixed-batch observation shape, dtype or values")
    drawn = math.prod(shape)
    count = len(delta)
    in_domain = int(in_domain)
    if not count <= in_domain <= drawn or (count == 0 and peaks.any()):
        raise ValueError("sample counts or empty-batch peaks do not match the fixed batch")

    sampling = {"drawn": drawn, "in_domain": in_domain, "valid": count,
                "excluded_domain": drawn - in_domain, "excluded_nonfinite": in_domain - count}
    if count < 2:
        return {"decision": "SMOKE_ONLY" if smoke else "INCONCLUSIVE", "sampling": sampling,
                "bias": None, "magnitude": None, "reason": "fewer than two valid samples"}

    mean = float(delta.mean())
    std = float(delta.std(ddof=1))
    se = std / math.sqrt(count)
    bound = abs(mean) + BIAS_MULTIPLIER * se
    if not math.isfinite(bound):
        raise ValueError("nonfinite bias statistic on retained samples")
    p, q = map(float, peaks)
    ratio = 0.0 if q == 0 else q / p if p > 0 else math.inf
    bias_pass = bound <= tolerance
    magnitude_pass = q <= PEAK_RATIO_LIMIT * p
    decision = "ACCEPT" if bias_pass and magnitude_pass else "REJECT"
    return {
        "decision": "SMOKE_ONLY" if smoke else decision,
        "sampling": sampling,
        "bias": {"status": "PASS" if bias_pass else "FAIL", "mean": mean,
                 "sample_std": std, "standard_error": se, "upper": bound, "tolerance": tolerance},
        "magnitude": {"status": "PASS" if magnitude_pass else "FAIL", "reference_peak": p,
                      "candidate_peak": q, "ratio": ratio if math.isfinite(ratio) else "+inf",
                      "limit": PEAK_RATIO_LIMIT},
    }


def ptx_files(fmt):
    files = {"reference.0.ptx", "candidate.0.ptx"}
    if fmt["compute"] == "fp64":
        files.add("oracle.0.ptx")
    return files


def run_instance(torch, triton, kernels, manifest, manifest_digest, rule, fmt, directory):
    directory.mkdir(exist_ok=True)
    record = {"rule": rule, "format": fmt["name"], "state": "RUNNING",
              "manifest_sha256": manifest_digest}
    record_path = directory / "record.json"
    primary.write_json(record_path, record)
    catalog, profile = manifest["catalog"], manifest["profile"]
    if reason := unsupported(catalog, rule, fmt):
        record.update(state="UNSUPPORTED", reason=reason)
    else:
        try:
            engine = implementation(catalog)
            seed = engine.seed_for(profile, fmt, rule)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            inputs = supplement.sample_inputs(torch, profile, fmt, rule, generator)
            in_domain = primary.domains.mask(torch, rule, inputs)
            oracle = engine.oracle(torch, rule, inputs)
            if catalog == "supplement":
                (p, q), compiled, errors = engine.launch_pair(
                    torch, triton, kernels, rule, inputs, profile, fmt)
            else:
                (p, q), compiled = engine.launch_pair(
                    torch, triton, kernels, rule, inputs, profile, fmt)
                errors = None
            # Clamp only the spacing lookup when a finite oracle lies outside
            # the output format. The oracle values used for errors stay intact.
            dtype = torch.bfloat16 if fmt["output"] == "bf16" else torch.float32
            ulps = primary.ulp(torch, oracle.abs().clamp(max=torch.finfo(dtype).max), fmt["output"])
            obs = observe(p.cpu().float().numpy(), q.cpu().float().numpy(),
                          oracle.cpu().numpy(), ulps.cpu().numpy(), in_domain.cpu().numpy(),
                          None if errors is None else tuple(e.cpu().numpy() for e in errors))
            result = evaluate(obs, profile["shape"], manifest["protocol"]["bias_tolerance_ulp"],
                              manifest["smoke"])
            ptx = {f"{side}.{i}.ptx": program.encode()
                   for side, programs in compiled.items() for i, program in enumerate(programs)}
            if set(ptx) != ptx_files(fmt):
                raise ValueError("unexpected compiled PTX files")
            for name, data in ptx.items():
                (directory / name).write_bytes(data)
            path = directory / "observations.npz"
            with (directory / "observations.tmp").open("wb") as output:
                np.savez_compressed(output, **obs)
            (directory / "observations.tmp").replace(path)
            record.update(state="COMPLETE", result=result, seed=seed,
                          lowerings={name: primary.sha(data) for name, data in ptx.items()},
                          observations_sha256=file_digest(path))
        except Exception as error:
            record.update(state="ERROR", reason=f"{type(error).__name__}: {error}")
    primary.write_json(record_path, record)
    return record


def replay(bundle):
    """Recalculate the two gates on CPU from the saved shifts and peaks."""
    manifest_path = bundle / "manifest.json"
    manifest = primary.read_json(manifest_path)
    manifest_digest = file_digest(manifest_path)
    if manifest.get("bundle_version") != BUNDLE_VERSION or type(manifest.get("smoke")) is not bool:
        raise ValueError("expected a paper-two-gates-v1 bundle")
    catalog, profile = manifest["catalog"], manifest["profile"]
    engine = implementation(catalog)
    if engine.validate_profile(deepcopy(profile)) != profile or manifest["protocol"] != protocol(profile):
        raise ValueError("invalid frozen paper profile or protocol")
    paths = source_paths(catalog)
    expected_hashes = {name: primary.sha(path.read_bytes()) for name, path in paths.items()}
    if manifest["sources"] != expected_hashes:
        raise ValueError("source hashes differ: use the exact experiment revision for replay")
    for name, digest in expected_hashes.items():
        if primary.sha((bundle / "sources" / name).read_bytes()) != digest:
            raise ValueError(f"saved source hash mismatch: {name}")

    rows = []
    for name, (rule, fmt) in instances(profile).items():
        directory = bundle / name
        record_path = directory / "record.json"
        row = {"rule": rule, "format": fmt["name"]}
        if not record_path.exists():
            rows.append({**row, "state": "MISSING"})
            continue
        record = primary.read_json(record_path)
        if (record.get("manifest_sha256") != manifest_digest
                or any(record[k] != v for k, v in row.items())):
            raise ValueError(f"record identity mismatch: {name}")
        state = record["state"]
        reason = unsupported(catalog, rule, fmt)
        if state == "UNSUPPORTED" and reason == record.get("reason") and reason:
            rows.append({**row, "state": state, "reason": reason})
            continue
        if state in {"RUNNING", "ERROR"}:
            rows.append({**row, "state": state, "reason": record.get("reason")})
            continue
        if state != "COMPLETE" or reason or record["seed"] != engine.seed_for(profile, fmt, rule):
            raise ValueError(f"invalid result state, precision or seed: {name}")
        lowerings = record["lowerings"]
        if set(lowerings) != ptx_files(fmt) or any(
                primary.sha((directory / path).read_bytes()) != digest
                for path, digest in lowerings.items()):
            raise ValueError(f"compiled PTX hash mismatch: {name}")
        path = directory / "observations.npz"
        if file_digest(path) != record["observations_sha256"]:
            raise ValueError(f"observation hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as data:
            result = evaluate(dict(data), profile["shape"], manifest["protocol"]["bias_tolerance_ulp"],
                              manifest["smoke"])
        if result != record["result"]:
            raise ValueError(f"stored decision or statistics disagree with replay: {name}")
        rows.append({**row, "state": state, "seed": record["seed"], **result,
                     "observations_sha256": record["observations_sha256"], "lowerings": lowerings})

    return {
        "bundle_version": BUNDLE_VERSION,
        "manifest_sha256": manifest_digest,
        "manifest": manifest,
        "complete": all(row["state"] in {"COMPLETE", "UNSUPPORTED"} for row in rows),
        "rows": rows,
        "accepted": [row for row in rows if row.get("decision") == "ACCEPT"],
    }


def run(args, profile):
    try:
        import torch
        import triton
    except ImportError as error:
        raise ValueError("GPU run requires torch and triton in the CUDA environment") from error
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError("GPU run requires an NVIDIA CUDA device")

    catalog = args.catalog
    engine = implementation(catalog)
    kernels = primary.load_module(engine.KERNELS)
    rules = supplement.load_catalog() if catalog == "supplement" else primary.registry.load_catalog()
    if kernels.SUPPORTED != set(rules):
        raise ValueError("candidate catalogue and executable templates disagree")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        driver = ["unavailable"]
    paths = source_paths(catalog)
    sources = {name: path.read_bytes() for name, path in paths.items()}
    manifest = {
        "bundle_version": BUNDLE_VERSION, "catalog": catalog, "profile": profile,
        "protocol": protocol(profile), "smoke": args.smoke,
        "sources": {name: primary.sha(data) for name, data in sources.items()},
        "backend": {"device": torch.cuda.get_device_name(),
                    "capability": list(torch.cuda.get_device_capability()), "drivers": driver,
                    "torch": str(torch.__version__), "triton": triton.__version__,
                    "cuda": torch.version.cuda, "numpy": np.__version__,
                    "enable_fp_fusion": False},
    }

    output = args.output
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        kept = set()
        if args.resume:
            if primary.read_json(output / "manifest.json") != manifest:
                raise ValueError("resume requires identical profile, sources, device and software")
            kept = {f"{row['format']}__{row['rule']}" for row in replay(output)["rows"]
                    if row["state"] in {"COMPLETE", "UNSUPPORTED"}}
        else:
            for name, data in sources.items():
                path = output / "sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            primary.write_json(output / "manifest.json", manifest)
        manifest_digest = file_digest(output / "manifest.json")
        errors = 0
        for name, (rule, fmt) in instances(profile).items():
            if name in kept:
                print(f"[KEEP] {name}", flush=True)
                continue
            print(f"[RUN] {name}", flush=True)
            record = run_instance(torch, triton, kernels, manifest, manifest_digest,
                                  rule, fmt, output / name)
            status = record["result"]["decision"] if record["state"] == "COMPLETE" else record["state"]
            print(f"[{status}] {name} {record.get('reason', '')}".rstrip(), flush=True)
            errors += record["state"] == "ERROR"
            torch.cuda.empty_cache()
        print(f"Bundle saved to {output}", flush=True)
        return int(errors > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        command = sub.add_parser(name)
        command.add_argument("--catalog", choices=("primary", "supplement"), default="primary")
        command.add_argument("--profile", type=Path)
        command.add_argument("--rules", help="comma-separated rule IDs")
        command.add_argument("--formats", help="comma-separated format names")
        command.add_argument("--smoke", action="store_true", help="32x33, measurements labeled SMOKE_ONLY")
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
    importer = sub.add_parser("import", help="replay statistics from a fixed-batch bundle on CPU")
    importer.add_argument("bundle", type=Path)
    importer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "import":
            report = replay(args.bundle)
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(f"Replayed {len(report['rows'])} rows, {len(report['accepted'])} accepted. "
                  f"Saved {args.output}")
            return int(not report["complete"])
        profile = load_profile(args)
        if args.command == "check":
            print(json.dumps({"protocol": protocol(profile), "profile": profile,
                              "smoke": args.smoke}, indent=2))
            return 0
        return run(args, profile)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
