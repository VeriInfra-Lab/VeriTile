#!/usr/bin/env python3
"""Run fixed-budget two-gates experiments and independently replay their evidence."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import mpmath
import numpy as np

if __package__:
    from . import check_numerics as primary, supplement_numerics as supplement
    from . import two_gates_sampling as sampling, two_gates_statistics as statistics
else:
    import check_numerics as primary
    import supplement_numerics as supplement
    import two_gates_sampling as sampling
    import two_gates_statistics as statistics

ROOT = primary.ROOT
DEFAULT_CONFIG = ROOT / "experiments/floating_point/two_gates_config.py"
PROFILES = {"primary": primary.DEFAULT_PROFILE, "supplement": supplement.DEFAULT_PROFILE,
            "counts": ROOT / "experiments/floating_point/primitives/count_config.py"}
BUNDLE_FORMAT = "veritile-two-gates"
# CPU observation and gate utilities.
observe, evaluate = statistics.observe, statistics.evaluate


def implementation(catalog):
    if catalog == "primary":
        return primary
    if catalog in {"supplement", "counts"}:
        return supplement
    raise ValueError("unknown catalogue")


def unsupported(catalog, rule, fmt):
    if catalog != "primary":
        return supplement.unsupported(rule, fmt)
    if rule == "BF16-WIDEN-RETURN" and (fmt["input"], fmt["output"]) != ("bf16", "bf16"):
        return "requires bf16 input and output"
    return None


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_npz(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def source_paths(catalog=None):
    paths = [*primary.SOURCES, Path(supplement.__file__), supplement.KERNELS, supplement.CATALOG,
             Path(__file__), Path(sampling.__file__), Path(statistics.__file__), DEFAULT_CONFIG,
             ROOT / "scripts/check_numerics_paper.py", *PROFILES.values()]
    return {str(path.resolve().relative_to(ROOT)): path.resolve() for path in paths}


def validate_settings(settings):
    expected = set(primary.load_module(DEFAULT_CONFIG).SETTINGS)
    if type(settings) is not dict or set(settings) != expected:
        raise ValueError(f"settings require exactly {sorted(expected)}")
    primary.registry.canonical_json(settings)
    for key in ("samples", "near_samples", "workers", "audit_parts"):
        if type(settings[key]) is not int or settings[key] < (1 if key == "workers" else 2):
            raise ValueError(f"{key} must be a positive sample/worker/audit count")
    if settings["samples"] > 2**53 or settings["near_samples"] > 2**53:
        raise ValueError("sample counts must be exactly representable in the statistics")
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0,2^63)")
    shape = settings["execution_shape"]
    if type(shape) is not list or len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("execution_shape requires two positive integers")
    if math.prod(shape) > 2**24:
        raise ValueError("execution_shape is too large for bounded-memory replay")
    for key in ("bias_tolerance_ulp", "bias_se_multiplier", "peak_ratio_limit"):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if settings["near_half_width"] != sampling.H or settings["rejection_envelope"] != sampling.ENVELOPE:
        raise ValueError("the validated conditional design requires half-width 2^-12 and envelope 1.001")
    return settings


def csv_values(value):
    values = value.split(",") if value is not None else None
    if values is not None and (any(not v for v in values) or len(set(values)) != len(values)):
        raise ValueError("selection requires distinct, nonempty comma-separated names")
    return set(values) if values is not None else None


def build_items(profiles, settings, iid=False):
    items = []
    for catalog in sorted(profiles):
        profile = profiles[catalog]
        for fmt in profile["formats"]:
            for rule in profile["rules"]:
                identity = f"{catalog}__{fmt['name']}__{rule}"
                seed = implementation(catalog).seed_for({"seed": settings["seed"]}, fmt, rule)
                seeds = sampling.role_seeds(seed, identity)
                seeds["iid"] = seed  # Preserve intentional paired-input streams in the catalogue.
                weighted = not iid and sampling.weighted(catalog, rule, fmt, profile["distribution"])
                k = sampling.mixture_count(settings["samples"], sampling.constants(rule)["p0"], seeds["mixture"]) if weighted else None
                items.append({"id": identity, "catalog": catalog, "rule": rule, "format": fmt,
                    "unsupported": unsupported(catalog, rule, fmt), "method": "weighted" if weighted else "iid",
                    "seeds": seeds, "target_near_count": k,
                    "source_count": settings["samples"]+(settings["near_samples"] if weighted else 0)})
    return items


def make_plan(args):
    settings = deepcopy(primary.load_module(args.config or DEFAULT_CONFIG).SETTINGS)
    if args.samples is not None:
        settings["samples"] = args.samples
    if args.workers is not None:
        settings["workers"] = args.workers
    if args.smoke:
        settings.update(samples=1056, near_samples=8, execution_shape=[32, 33])
    validate_settings(settings)
    if args.profile and args.catalog == "all":
        raise ValueError("--profile requires one explicit --catalog")
    requested_rules, requested_formats = csv_values(args.rules), csv_values(args.formats)
    profiles, known_rules, known_formats = {}, set(), set()
    for catalog in PROFILES if args.catalog == "all" else [args.catalog]:
        p = implementation(catalog).validate_profile(deepcopy(primary.load_module(args.profile or PROFILES[catalog]).PROFILE))
        known_rules.update(p["rules"]); known_formats.update(f["name"] for f in p["formats"])
        rules = [r for r in p["rules"] if requested_rules is None or r in requested_rules]
        formats = [f for f in p["formats"] if requested_formats is None or f["name"] in requested_formats]
        if rules and formats:
            profiles[catalog] = {"distribution": p["distribution"], "rules": rules, "formats": formats, "launch": p["launch"]}
    if requested_rules is not None and not requested_rules <= known_rules:
        raise ValueError("unknown --rules entry")
    if requested_formats is not None and not requested_formats <= known_formats:
        raise ValueError("unknown --formats entry")
    if not profiles:
        raise ValueError("selection has no configurations")
    items = build_items(profiles, settings, args.iid)
    return {"settings": settings, "profiles": profiles, "items": items, "smoke": args.smoke,
            "weighted_constants": {r: sampling.constants(r) for r in sorted({i['rule'] for i in items if i['method'] == 'weighted'})},
            "iid_control": args.iid, "protocol": {
                "bias": "Target-weighted full-bank signed ULP mean with independent-stratum sample-variance SE.",
                "magnitude": "Unweighted candidate/reference absolute oracle-error maxima on the restored target multiset.",
                "restoration": "Independent K~Binomial(N,p0); first K near proposals accepted with probability r/1.001 and first N-K rest-conditionals. Insufficient supply is INCONCLUSIVE.",
                "selection": "Original quantized-input domain and finite P/Q/oracle; exclusions reported without replacement.",
                "stopping": "Run each selected configuration once to its full source count, irrespective of gate outcomes.",
                "limitations": "Finite observed maxima are not a universal tail bound. Analytic Gaussian transport is not an exact finite-state RNG certification."}}


def part_plan(item, settings):
    width = math.prod(settings["execution_shape"])
    allocations = [("near", settings["near_samples"]), ("rest", settings["samples"])] if item["method"] == "weighted" else [("iid", settings["samples"])]
    result = []
    for stratum, count in allocations:
        while count:
            take = min(width, count)
            result.append((stratum, take)); count -= take
    return result


def fixed_audits(parts, settings):
    indices = set(np.linspace(0, len(parts)-1, min(len(parts), settings["audit_parts"]), dtype=int).tolist())
    for j, part in enumerate(parts):
        if j == 0 or part[0] != parts[j-1][0]:
            indices.add(j)
            if j:
                indices.add(j-1)
    return indices


def launch(torch, triton, kernels, item, profile, inputs):
    rule, fmt = item["rule"], item["format"]
    engine = implementation(item["catalog"])
    domain = primary.domains.mask(torch, rule, inputs)
    oracle = engine.oracle(torch, rule, inputs)
    errors = None
    if engine is supplement:
        (p, q), compiled, errors = engine.launch_pair(torch, triton, kernels, rule, inputs, profile, fmt)
    else:
        (p, q), compiled = engine.launch_pair(torch, triton, kernels, rule, inputs, profile, fmt)
    dtype = torch.bfloat16 if fmt["output"] == "bf16" else torch.float32
    ulps = primary.ulp(torch, oracle.abs().clamp(max=torch.finfo(dtype).max), fmt["output"])
    return p, q, oracle, ulps, domain, errors, compiled


def ptx_files(fmt):
    return {"reference.0.ptx", "candidate.0.ptx"} | ({"oracle.0.ptx"} if fmt["compute"] == "fp64" else set())


def run_instance(torch, triton, kernels, manifest, digest, item, output):
    directory = output/item["id"]; directory.mkdir(exist_ok=True)
    record = {"state": "RUNNING", "item": item, "manifest_sha256": digest}
    primary.write_json(directory/"record.json", record)
    start = time.perf_counter()
    if item["unsupported"]:
        record.update(state="UNSUPPORTED", reason=item["unsupported"])
    else:
        try:
            settings = manifest["settings"]; fmt = item["format"]
            generators = {k: torch.Generator(device="cuda").manual_seed(v) for k, v in item["seeds"].items() if k != "mixture"}
            restorer = sampling.Restoration.create(settings["samples"], item["target_near_count"]) if item["method"] == "weighted" else None
            parts = part_plan(item, settings); audits = fixed_audits(parts, settings)
            rows, strata, counts, fingerprints, saved_audits, lowerings = [], [], [], [], [], {}
            source_checks = []
            torch.cuda.reset_peak_memory_stats()
            for j, (stratum, count) in enumerate(parts):
                shape = settings["execution_shape"] if count == math.prod(settings["execution_shape"]) else [1, count]
                profile = {**manifest["profiles"][item["catalog"]], "shape": shape}
                inputs, latent, ratio, _ = sampling.conditional_sample(torch, supplement, profile, fmt, item["rule"], stratum, generators)
                before = asdict(restorer) if restorer else None
                chosen, uniform = restorer.select(torch, stratum, shape, ratio, generators["acceptance"]) if restorer else (torch.ones(shape, device="cuda", dtype=torch.bool), None)
                values = launch(torch, triton, kernels, item, profile, inputs)
                if restorer and (j == 0 or stratum != parts[j-1][0]):
                    statistics.check_weighted_source(inputs, values, item["rule"])
                    source_checks.append({"part": j, "samples": count, "stratum": stratum})
                files = {f"{side}.{k}.ptx": ptx.encode() for side, programs in values[-1].items() for k, ptx in enumerate(programs)}
                if set(files) != ptx_files(fmt):
                    raise ValueError("unexpected compiled PTX files")
                for name, data in files.items():
                    key = f"ptx/{count}/{name}"; h = primary.sha(data)
                    if key in lowerings and lowerings[key] != h:
                        raise ValueError("compiled PTX changed within one configuration")
                    if key not in lowerings:
                        path = directory/key; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                        lowerings[key] = h
                row, fingerprint, arrays = statistics.gpu_observe(torch, values, ratio, inputs, chosen)
                if restorer and int(row[0]) != count:
                    raise ValueError("weighted design requires its full finite domain")
                metadata = {"part": j, "stratum": stratum, "drawn": count, "before": before}
                # Include every active rejection step and every partial prefix, independent of errors.
                if j in audits or uniform is not None or 0 < int(row[11]) < count:
                    data = statistics.audit_data(torch, arrays, row, latent, ratio, chosen, uniform, item["rule"], sampling)
                    name = f"audit-{j:05d}.npz"; save_npz(directory/name, data)
                    saved_audits.append({**metadata, "file": name, "sha256": file_digest(directory/name)})
                rows.append(row); strata.append(0 if stratum in {"near", "iid"} else 1)
                counts.append(count); fingerprints.append(fingerprint)
                if (j+1) % 128 == 0:
                    primary.write_json(output/f"progress-{item['id']}.json", {"parts": j+1, "target_parts": len(parts), "elapsed_seconds": time.perf_counter()-start})
            rows, strata, counts = np.asarray(rows), np.asarray(strata, dtype=np.int8), np.asarray(counts, dtype=np.int64)
            restoration = restorer.finish() if restorer else None
            probabilities = [sampling.constants(item["rule"])[k] for k in ["p0", "p1"]] if restorer else [1.]
            result = statistics.evaluate_parts(rows, strata, counts, probabilities, settings, manifest["smoke"], restoration)
            if restoration is None or restoration["state"] == "COMPLETE":
                if int(rows[:, 11].sum()) != settings["samples"]:
                    raise ValueError("peak raw sample count differs from the fixed budget")
            save_npz(directory/"statistics.npz", {"rows": rows, "strata": strata, "drawn": counts})
            primary.write_json(directory/"fingerprints.json", fingerprints)
            record.update(state="COMPLETE", result=result, restoration=restoration,
                lowerings=lowerings, audits=saved_audits, independent_source_checks=source_checks,
                statistics_sha256=file_digest(directory/"statistics.npz"),
                fingerprints_sha256=file_digest(directory/"fingerprints.json"),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        except Exception as error:
            record.update(state="ERROR", reason=f"{type(error).__name__}: {error}")
    record["elapsed_seconds"] = time.perf_counter()-start
    primary.write_json(directory/"record.json", record)
    print(record.get("result", {}).get("decision", record["state"]), item["id"], record.get("reason", ""), flush=True)
    return record


def read_manifest(bundle):
    manifest = primary.read_json(bundle/"manifest.json")
    if manifest.get("format") != BUNDLE_FORMAT or manifest.get("schema") != 1 or type(manifest.get("smoke")) is not bool:
        raise ValueError("expected a two-gates bundle with matching format and source identity")
    validate_settings(manifest["settings"])
    if np.__version__ != manifest["backend"]["numpy"] or mpmath.__version__ != manifest["backend"]["mpmath"]:
        raise ValueError("replay requires the recorded NumPy and mpmath versions")
    expected = {name: file_digest(path) for name, path in source_paths().items()}
    if manifest["sources"] != expected:
        raise ValueError("source hashes differ: use the exact experiment sources")
    for name, h in expected.items():
        if file_digest(bundle/"sources"/name) != h:
            raise ValueError(f"saved source hash mismatch: {name}")
    if manifest["items"] != build_items(manifest["profiles"], manifest["settings"], manifest["iid_control"]):
        raise ValueError("frozen plan, sampling design, mixture count or seed mismatch")
    if manifest["weighted_constants"] != {r: sampling.constants(r) for r in sorted({i['rule'] for i in manifest['items'] if i['method'] == 'weighted'})}:
        raise ValueError("frozen proposal probabilities or weights differ")
    return manifest


def replay(bundle):
    """CPU-only replay of all parts, fixed audits, identities and gate decisions."""
    m = read_manifest(bundle); digest = file_digest(bundle/"manifest.json")
    results = []; total_parts = total_audits = 0
    for item in m["items"]:
        directory = bundle/item["id"]; row = {"id": item["id"], "rule": item["rule"], "format": item["format"]["name"]}
        if not (directory/"record.json").exists():
            results.append({**row, "state": "MISSING"}); continue
        rec = primary.read_json(directory/"record.json")
        if rec.get("item") != item or rec.get("manifest_sha256") != digest:
            raise ValueError("record identity mismatch")
        if rec["state"] == "UNSUPPORTED" and rec.get("reason") == item["unsupported"] and item["unsupported"]:
            results.append({**row, "state": "UNSUPPORTED", "reason": item["unsupported"]}); continue
        if rec["state"] in {"RUNNING", "ERROR"}:
            results.append({**row, "state": rec["state"], "reason": rec.get("reason")}); continue
        if rec["state"] != "COMPLETE" or item["unsupported"]:
            raise ValueError("invalid result state")
        parts = part_plan(item, m["settings"])
        expected_ptx = {f"ptx/{n}/{name}" for _, n in parts for name in ptx_files(item["format"])}
        if set(rec["lowerings"]) != expected_ptx or any(file_digest(directory/p) != h for p, h in rec["lowerings"].items()):
            raise ValueError("compiled PTX hash mismatch")
        for name in ["statistics", "fingerprints"]:
            suffix = ".npz" if name == "statistics" else ".json"
            if file_digest(directory/(name+suffix)) != rec[name+"_sha256"]:
                raise ValueError("observation hash mismatch")
        with np.load(directory/"statistics.npz", allow_pickle=False) as f:
            rows, strata, drawn = f["rows"], f["strata"], f["drawn"]
        if (rows.shape != (len(parts), len(statistics.FIELDS)) or not np.array_equal(drawn, [n for _, n in parts])
                or not np.array_equal(strata, [int(st == "rest") for st, _ in parts])
                or len(primary.read_json(directory/"fingerprints.json")) != len(parts)):
            raise ValueError("saved parts differ from the full fixed source budget")
        if item["method"] == "weighted" and not np.array_equal(rows[:, 0], drawn):
            raise ValueError("weighted evidence must cover its full finite domain")
        required = fixed_audits(parts, m["settings"]); near_left = item["target_near_count"] or 0
        rest_left = m["settings"]["samples"]-near_left
        metadata = {}
        for j, (st, count) in enumerate(parts):
            before = {"peak_count": m["settings"]["samples"], "target_near_count": item["target_near_count"], "near_remaining": near_left, "rest_remaining": rest_left} if item["method"] == "weighted" else None
            selected = int(rows[j, 11])
            if st == "near":
                if near_left:
                    required.add(j)
                if not 0 <= selected <= min(count, near_left):
                    raise ValueError("invalid near selection count")
                near_left -= selected
            else:
                if selected != (count if st == "iid" else min(count, rest_left)):
                    raise ValueError("invalid target/rest prefix count")
                if st == "rest":
                    rest_left -= selected
            if 0 < selected < count:
                required.add(j)
            metadata[j] = {"part": j, "stratum": st, "drawn": count, "before": before}
        if len(rec["audits"]) != len(required) or {a["part"] for a in rec["audits"]} != required:
            raise ValueError("missing or repeated required raw audit")
        for a in rec["audits"]:
            j = a["part"]
            if any(a[k] != v for k, v in metadata[j].items()) or a["file"] != f"audit-{j:05d}.npz":
                raise ValueError("raw audit identity mismatch")
            if file_digest(directory/a["file"]) != a["sha256"]:
                raise ValueError("raw audit hash mismatch")
            with np.load(directory/a["file"], allow_pickle=False) as f:
                statistics.replay_audit(dict(f), rows[j], a, item["rule"], sampling)
            total_audits += 1
        restoration = None
        if item["method"] == "weighted":
            restoration = asdict(sampling.Restoration(m["settings"]["samples"], item["target_near_count"], near_left, rest_left))
            restoration["state"] = "COMPLETE" if near_left == rest_left == 0 else "INCONCLUSIVE"
        if restoration != rec["restoration"]:
            raise ValueError("restoration state mismatch")
        ps = [sampling.constants(item["rule"])[k] for k in ["p0", "p1"]] if restoration else [1.]
        result = statistics.evaluate_parts(rows, strata, drawn, ps, m["settings"], m["smoke"], restoration)
        if result != rec["result"]:
            raise ValueError("stored decision or statistics disagree with replay")
        total_parts += len(parts); results.append({**row, "state": "COMPLETE", **result})
    return {"format": BUNDLE_FORMAT, "manifest_sha256": digest, "complete": all(r["state"] in {"COMPLETE", "UNSUPPORTED"} for r in results),
            "rows": results, "accepted": [r for r in results if r.get("decision") == "ACCEPT"],
            "part_statistics_verified": total_parts, "raw_audits_verified": total_audits,
            "scope": "All saved part statistics and required raw audits. Not CPU execution of every source lane."}


def worker(bundle, number):
    import torch
    import triton
    torch.set_num_threads(1)
    m = read_manifest(bundle); digest = file_digest(bundle/"manifest.json")
    for key, value in {"torch": str(torch.__version__), "triton": triton.__version__, "cuda": torch.version.cuda,
                       "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}.items():
        if m["backend"][key] != value:
            raise ValueError("device/software differs from frozen manifest")
    engines = {c: primary.load_module(implementation(c).KERNELS) for c in m["profiles"]}
    for c, kernel in engines.items():
        catalog = primary.registry.load_catalog() if c == "primary" else supplement.load_catalog()
        if kernel.SUPPORTED != set(catalog):
            raise ValueError("candidate catalogue and executable templates disagree")
    failed = False
    for item in m["items"][number::m["settings"]["workers"]]:
        p = bundle/item["id"]/"record.json"
        if p.exists() and primary.read_json(p)["state"] in {"COMPLETE", "UNSUPPORTED"}:
            continue  # The parent independently verified all retained rows before spawning.
        rec = run_instance(torch, triton, engines[item["catalog"]], m, digest, item, bundle)
        failed |= rec["state"] == "ERROR"
    return int(failed)


def run(args, plan):
    import torch
    import triton
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise ValueError("GPU run requires an NVIDIA CUDA device")
    backend = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "torch": str(torch.__version__), "triton": triton.__version__, "cuda": torch.version.cuda,
               "numpy": np.__version__, "mpmath": mpmath.__version__, "enable_fp_fusion": False}
    backend["drivers"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=10).stdout.strip().splitlines()
    manifest = {"format": BUNDLE_FORMAT, "schema": 1, **plan, "backend": backend,
                "sources": {n: file_digest(p) for n, p in source_paths().items()}}
    output = args.output
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with (output/".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if primary.read_json(output/"manifest.json") != manifest:
                raise ValueError("resume requires identical configuration, sources, device and software")
            replay(output)
        else:
            for name, path in source_paths().items():
                saved = output/"sources"/name; saved.parent.mkdir(parents=True, exist_ok=True); saved.write_bytes(path.read_bytes())
            primary.write_json(output/"manifest.json", manifest)
        primary.write_json(output/"run-status.json", {"state": "RUNNING", "configurations": len(plan["items"])})
        processes, logs = [], []
        for n in range(plan["settings"]["workers"]):
            log = (output/f"worker-{n}.log").open("a"); logs.append(log)
            processes.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker", str(output), str(n)], stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        for log in logs:
            log.close()
        report = replay(output); primary.write_json(output/"report.json", report)
        counts = Counter(r.get("decision", r["state"]) for r in report["rows"])
        primary.write_json(output/"run-status.json", {"state": "COMPLETE" if report["complete"] and not any(codes) else "ERROR",
            "worker_exit_codes": codes, "counts": dict(counts), "elapsed_seconds": time.perf_counter()-start,
            "part_statistics_verified": report["part_statistics_verified"], "raw_audits_verified": report["raw_audits_verified"]})
        print(json.dumps({"complete": report["complete"], "counts": dict(counts), "bundle": str(output)}, indent=2))
        return int(not report["complete"] or any(codes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ["check", "run"]:
        p = sub.add_parser(name); p.add_argument("--catalog", choices=["all", *PROFILES], default="all")
        p.add_argument("--config", type=Path); p.add_argument("--profile", type=Path)
        p.add_argument("--rules"); p.add_argument("--formats"); p.add_argument("--smoke", action="store_true")
        p.add_argument("--samples", type=int); p.add_argument("--workers", type=int)
        p.add_argument("--iid", action="store_true", help="explicit direct-target-sampling control")
        if name == "run":
            p.add_argument("--output", type=Path, required=True); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("import", help="independently replay a complete bundle on CPU")
    p.add_argument("bundle", type=Path); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("worker", help=argparse.SUPPRESS); p.add_argument("bundle", type=Path); p.add_argument("number", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            return worker(args.bundle, args.number)
        if args.command == "import":
            report = replay(args.bundle)
            with args.output.open("x") as f:
                json.dump(report, f, indent=2, allow_nan=False); f.write("\n")
            print(f"Replayed {len(report['rows'])} configurations; saved {args.output}")
            return int(not report["complete"])
        plan = make_plan(args)
        if args.command == "check":
            print(json.dumps({**plan, "counts": dict(Counter(i["unsupported"] is None for i in plan["items"]))}, indent=2))
            return 0
        return run(args, plan)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
