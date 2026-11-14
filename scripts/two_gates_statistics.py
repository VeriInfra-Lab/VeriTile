"""Streaming, weighted bias and unweighted restored-subset magnitude gates."""
import hashlib
import math
import numpy as np

BIAS_MULTIPLIER = 5.0
PEAK_RATIO_LIMIT = 10.0
FIELDS = ["valid", "mean", "m2", "reference_peak", "candidate_peak", "in_domain",
          "weight_sum", "max_abs_delta", "selected_valid", "selected_reference_peak",
          "selected_candidate_peak", "selected_raw"]


def check_weighted_source(inputs, outputs, rule):
    """Independent NumPy execution of the three unchanged FP32 source pairs."""
    x, y, z = (a.cpu().numpy() for a in inputs)
    if rule == "CANCEL":
        p, q, oracle = (x-y)+y, x, x.astype(np.float64)
    elif rule == "ADD-ASSOC":
        p, q, oracle = (x+y)+z, x+(y+z), (x.astype(np.float64)+y)+z
    elif rule == "MUL-DISTRIB":
        p, q, oracle = x*(y+z), (x*y)+(x*z), x.astype(np.float64)*(y.astype(np.float64)+z)
    else:
        raise ValueError("no independent weighted-source check for this rule")
    for got, wanted in zip(outputs[:3], (p, q, oracle)):
        if not np.array_equal(got.cpu().numpy(), wanted):
            raise ValueError("GPU source or oracle differs from independent NumPy execution")


def merge(rows):
    n = int(rows[:, 0].sum())
    if not n:
        return {"n": 0, "mean": 0., "m2": 0.}
    mean = math.fsum(float(v[0])*float(v[1]) for v in rows)/n
    m2 = math.fsum(float(v[2])+float(v[0])*(float(v[1])-mean)**2 for v in rows)
    return {"n": n, "mean": mean, "m2": m2}


def moments(values):
    mean = float(values.mean()) if len(values) else 0.
    return {"n": len(values), "mean": mean, "m2": float(np.square(values-mean).sum())}


def evaluate_parts(rows, strata, drawn, probabilities, settings, smoke=False, restoration=None):
    rows, strata, drawn = np.asarray(rows), np.asarray(strata), np.asarray(drawn)
    if (rows.ndim != 2 or rows.shape[1] != len(FIELDS) or len(rows) != len(strata)
            or len(rows) != len(drawn) or not np.isfinite(rows).all()):
        raise ValueError("invalid part statistics")
    if (np.any(rows[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] < 0)
            or np.any(rows[:, 0] > rows[:, 5]) or np.any(rows[:, 5] > drawn)
            or np.any(rows[:, 8] > rows[:, 0]) or np.any(rows[:, 8] > rows[:, 11])
            or np.any(rows[:, 11] > drawn) or np.any(rows[:, 9:11] > rows[:, 3:5])):
        raise ValueError("invalid counts, moments or subset peaks")
    for col in [0, 5, 8, 11]:
        if not np.equal(rows[:, col], np.floor(rows[:, col])).all():
            raise ValueError("fractional sample count")
    if np.any(rows[rows[:, 8] == 0, 9:11] != 0):
        raise ValueError("nonzero peak without a selected valid sample")
    groups = [merge(rows[strata == k]) for k in range(len(probabilities))]
    source, valid, in_domain = int(drawn.sum()), int(rows[:, 0].sum()), int(rows[:, 5].sum())
    sampling = {"drawn": source, "valid": valid, "in_domain": in_domain,
                "excluded_domain": source-in_domain, "excluded_nonfinite": in_domain-valid,
                "peak_drawn": int(rows[:, 11].sum()), "peak_valid": int(rows[:, 8].sum())}
    bank_peaks = {"reference_peak": float(rows[:, 3].max()), "candidate_peak": float(rows[:, 4].max())}
    reason = None
    if any(g["n"] < 2 for g in groups):
        reason = "fewer than two valid samples in an estimator stratum"
    if restoration is not None and restoration["state"] != "COMPLETE":
        reason = "insufficient near-bank supply; mixture count and peak budget were not changed"
    if reason:
        return {"decision": "SMOKE_ONLY" if smoke else "INCONCLUSIVE", "sampling": sampling,
                "bias": None, "magnitude": None, "bank_peaks": bank_peaks, "reason": reason}
    mean = math.fsum(p*g["mean"] for p, g in zip(probabilities, groups))
    variance = math.fsum(p*p*g["m2"]/(g["n"]-1)/g["n"] for p, g in zip(probabilities, groups))
    se = math.sqrt(variance); bound = abs(mean)+settings["bias_se_multiplier"]*se
    p, q = float(rows[:, 9].max()), float(rows[:, 10].max())
    ratio = 0. if q == 0 else q/p if p else math.inf
    bias_pass = bound <= settings["bias_tolerance_ulp"]
    peak_pass = q <= settings["peak_ratio_limit"]*p
    return {"decision": "SMOKE_ONLY" if smoke else "ACCEPT" if bias_pass and peak_pass else "REJECT",
            "sampling": sampling, "bias": {"status": "PASS" if bias_pass else "FAIL", "mean": mean,
                "standard_error": se, "upper": bound, "tolerance": settings["bias_tolerance_ulp"],
                "strata": [{"probability": p, **g} for p, g in zip(probabilities, groups)]},
            "magnitude": {"status": "PASS" if peak_pass else "FAIL", "reference_peak": p,
                "candidate_peak": q, "ratio": ratio if math.isfinite(ratio) else "+inf",
                "limit": settings["peak_ratio_limit"]}, "bank_peaks": bank_peaks}


def gpu_observe(torch, values, ratio, inputs, chosen):
    p, q, o, u, domain, errors, _ = values
    valid = domain & torch.isfinite(p) & torch.isfinite(q) & torch.isfinite(o)
    if bool((valid & (~torch.isfinite(u) | (u <= 0))).any()):
        raise ValueError("invalid retained ULP spacing")
    delta = (q.double()-p.double())/u
    zero = torch.zeros_like(delta)
    weighted = torch.where(valid, delta if ratio is None else delta*ratio, zero)
    n = valid.sum(); mean = weighted.sum()/n.clamp(min=1)
    m2 = torch.where(valid, (weighted-mean)**2, zero).sum()
    ep, eq = (torch.abs(p.double()-o), torch.abs(q.double()-o)) if errors is None else errors
    if bool((valid & (~torch.isfinite(ep) | ~torch.isfinite(eq) | (ep < 0) | (eq < 0))).any()):
        raise ValueError("invalid retained oracle errors")
    selected = chosen & valid
    row = torch.stack([n.double(), mean, m2, torch.where(valid, ep, zero).max(),
        torch.where(valid, eq, zero).max(), domain.sum().double(),
        n.double() if ratio is None else torch.where(valid, ratio, zero).sum(),
        torch.where(valid, delta.abs(), zero).max(), selected.sum().double(),
        torch.where(selected, ep, zero).max(), torch.where(selected, eq, zero).max(), chosen.sum().double()]).cpu().numpy()
    count = inputs[0].numel(); ix = torch.arange(0, count, max(1, count//32), device="cuda")[:32]
    columns = [x.reshape(-1)[ix].double() for x in [*inputs, p, q, o, u, domain, delta, chosen]]
    if ratio is not None:
        columns.append(ratio.reshape(-1)[ix])
    fingerprint = hashlib.sha256(torch.stack(columns).cpu().numpy().tobytes()).hexdigest()
    return row, fingerprint, (delta, ep, eq, valid)


def audit_data(torch, arrays, row, latent, ratio, chosen, uniform, rule, sampler):
    """CPU observer check, with lossless observations retained for replay."""
    delta, ep, eq, valid = (x.cpu().numpy().reshape(-1) for x in arrays)
    selected = chosen.cpu().numpy().reshape(-1)
    d = delta[valid]
    if ratio is None:
        vals, counts = np.unique(d, return_counts=True)
        data = {"delta_values": vals, "delta_counts": counts.astype(np.int64)}
        y = d
    else:
        t = latent.cpu().numpy().reshape(-1)
        w = sampler.weights(t, rule, sampler.constants(rule))
        if not np.allclose(w, ratio.cpu().numpy().reshape(-1), rtol=1e-14, atol=1e-14):
            raise ValueError("CPU/GPU weight mismatch")
        data = {"delta": d, "latent": t}; y = d*w[valid]
    stat = moments(y)
    want = [stat["n"], stat["mean"], stat["m2"]]
    if not np.allclose(row[:3], want, rtol=1e-10, atol=1e-10):
        raise ValueError("GPU moments disagree with original CPU observer")
    peaks = np.array([ep[valid].max(initial=0.), eq[valid].max(initial=0.)])
    subset = valid & selected
    peaks_selected = np.array([ep[subset].max(initial=0.), eq[subset].max(initial=0.)])
    if (not np.array_equal(peaks, row[3:5]) or not np.array_equal(peaks_selected, row[9:11])
            or int(subset.sum()) != int(row[8])):
        raise ValueError("GPU peak or selected count disagrees with CPU observer")
    data.update(peaks=peaks, selected_peaks=peaks_selected, valid_count=np.array(len(d), np.int64))
    if uniform is not None or (selected.any() and not selected.all()):
        data.update(reference_error=ep, candidate_error=eq, valid=valid, selected=selected)
        if uniform is not None:
            data["accept_uniform"] = uniform.cpu().numpy().reshape(-1)
    return data


def replay_audit(data, row, metadata, rule, sampler):
    active_near = metadata["stratum"] == "near" and metadata["before"]["near_remaining"] > 0
    if active_near or 0 < row[11] < metadata["drawn"]:
        if not {"selected", "valid", "reference_error", "candidate_error"} <= data.keys():
            raise ValueError("missing complete restoration-mask audit")
        if active_near and "accept_uniform" not in data:
            raise ValueError("missing independent acceptance randomness")
    if "delta" in data:
        w = sampler.weights(data["latent"], rule, sampler.constants(rule))
        stat = moments(data["delta"]*w)
    else:
        v, k = data["delta_values"], data["delta_counts"]
        if (v.ndim != 1 or k.dtype != np.int64 or v.shape != k.shape or (k <= 0).any()
                or not np.isfinite(v).all() or (len(v) > 1 and (v[1:] <= v[:-1]).any())):
            raise ValueError("invalid exact delta frequency table")
        n = sum(map(int, k)); mu = math.fsum(float(x)*int(y) for x, y in zip(v, k))/n if n else 0.
        stat = {"n": n, "mean": mu, "m2": math.fsum((float(x)-mu)**2*int(y) for x, y in zip(v, k))}
    if not np.allclose(row[:3], [stat["n"], stat["mean"], stat["m2"]], rtol=1e-10, atol=1e-10):
        raise ValueError("raw audit moments disagree with part statistics")
    if not np.array_equal(row[3:5], data["peaks"]) or not np.array_equal(row[9:11], data["selected_peaks"]):
        raise ValueError("audit peaks disagree with part statistics")
    if "selected" in data:
        chosen = sampler.cpu_selection(metadata["stratum"], metadata["drawn"], metadata["before"],
            data.get("latent"), data.get("accept_uniform"), rule)
        if not np.array_equal(chosen, data["selected"]):
            raise ValueError("independent restored mask disagrees with GPU selection")
        mask = chosen & data["valid"]
        if int(mask.sum()) != int(row[8]) or int(chosen.sum()) != int(row[11]):
            raise ValueError("restored count disagrees with audit")
        peaks = np.array([data[k][mask].max(initial=0.) for k in ("reference_error", "candidate_error")])
        if not np.array_equal(peaks, row[9:11]):
            raise ValueError("restored maximum error disagrees with raw audit")


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
