"""Conditional proposals and error-independent restoration of target samples.

The Gaussian model is analytic. FP64 reconstruction and machine RNG probabilities
are not formal certifications of an exact discrete Gaussian implementation.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math

import mpmath as mp
import numpy as np

H = 2.0**-12
ENVELOPE = 1.001
WEIGHTED_RULES = {"CANCEL", "ADD-ASSOC", "MUL-DISTRIB"}
NNEG = 2**40 + 2**16
NPOS = 2**41 - 2**16 - 1
MICROCELLS = 4*NNEG + 3 + 2*NPOS
ZLO = -1. - H - H*2.0**-24
ZHI = -1. + H - H*2.0**-25


def weighted(catalog, rule, fmt, distribution):
    return (catalog == "primary" and rule in WEIGHTED_RULES
            and all(fmt[k] == "fp32" for k in ("input", "compute", "accumulator", "output"))
            and distribution == {"family": "normal", "mean": 1.0, "std": 1.0})


def role_seeds(seed, identity):
    return {role: int(hashlib.sha256(f"two-gates:{seed}:{identity}:{role}".encode()).hexdigest()[:15], 16)
            for role in ("iid", "near_sum", "near_perp", "rest_sum", "rest_perp", "mixture", "acceptance")}


@lru_cache(maxsize=3)
def constants(rule):
    with mp.workdps(100):
        h = mp.mpf(2)**-12
        if rule == "CANCEL":
            lo, hi = mp.mpf(ZLO)-mp.mpf(2)**-53, mp.mpf(ZHI)-mp.mpf(2)**-54
            p = (mp.erf(hi/mp.sqrt(2))-mp.erf(lo/mp.sqrt(2)))/2
            width = hi-lo
            if width != MICROCELLS*mp.mpf(2)**-54:
                raise ValueError("incorrect cancellation rounding-cell widths")
            factor = width*mp.exp(-mp.mpf('.5'))/(mp.sqrt(2*mp.pi)*p)
            upper = h+h*mp.mpf(2)**-24
            bound = factor*mp.exp(upper-upper*upper/2)
            extra = {"lo_100digits": str(lo), "hi_100digits": str(hi)}
        else:
            if rule not in WEIGHTED_RULES:
                raise ValueError("no weighted design for this rule")
            mean, var = (mp.mpf(3), mp.mpf(3)) if rule == "ADD-ASSOC" else (mp.mpf(2), mp.mpf(2))
            p = (mp.erf((h-mean)/mp.sqrt(2*var))-mp.erf((-h-mean)/mp.sqrt(2*var)))/2
            factor = 2*h*mp.exp(-mean*mean/(2*var))/(mp.sqrt(2*mp.pi*var)*p)
            bound = factor*mp.exp((mean*h-h*h/2)/var)
            extra = {"sum_mean": float(mean), "sum_variance": float(var), "sum_sd": float(mp.sqrt(var))}
        if not bound < mp.mpf(str(ENVELOPE)):
            raise ValueError("invalid rejection envelope")
        return {"p0": float(p), "p1": float(1-p), "p0_100digits": str(p),
                "weight_constant": float(factor), "weight_constant_100digits": str(factor),
                "ratio_upper_bound_100digits": str(bound), **extra}


def weights(latent, rule, c, array=np):
    if rule == "CANCEL":
        t = array.where(latent == -1., -2.0**-55, latent+1.)
        exponent = t-.5*t*t
    else:
        exponent = (c["sum_mean"]*latent-.5*latent*latent)/c["sum_variance"]
    return c["weight_constant"]*array.exp(exponent)


def mixture_count(count, probability, seed):
    return int(np.random.default_rng(seed).binomial(count, probability))


def cancellation_near(torch, shape, cells, perpendicular):
    needed = math.prod(shape)
    pieces, have, drawn = [], 0, 0
    while have < needed:
        size = max(256, 2*(needed-have)+4096)
        k = torch.randint(0, 2**44, (size,), device="cuda", dtype=torch.int64, generator=cells)
        drawn += size
        k = k[k < MICROCELLS]
        take = min(needed-have, k.numel())
        pieces.append(k[:take]); have += take
    k = torch.cat(pieces)
    z = torch.where(k < 4*NNEG, ZLO+(k//4).double()*2.0**-52,
                    -1.+((k-(4*NNEG+3))//2+1).double()*2.0**-53)
    z = torch.where((k >= 4*NNEG) & (k < 4*NNEG+3), -1., z).reshape(shape)
    a = (z*1.+1.).float()
    if not bool(((a >= -H) & (a < H)).all()):
        raise ValueError("cancellation proposal outside its stratum")
    bc = [(torch.randn(shape, device="cuda", dtype=torch.float64, generator=perpendicular)*1.+1.).float()
          for _ in range(2)]
    return [a, *bc], z, {"proposal_integers": drawn, "rejected_tuples": 0}


def conditional_sample(torch, supplement, profile, fmt, rule, stratum, generators):
    """Preserve source dtypes and casting; stratify only the validated target."""
    shape = profile["shape"]
    if stratum == "iid":
        return supplement.sample_inputs(torch, profile, fmt, rule, generators["iid"]), None, None, {}
    c = constants(rule)
    gsum, gperp = generators[stratum+"_sum"], generators[stratum+"_perp"]
    if rule == "CANCEL":
        if stratum == "near":
            inputs, latent, draws = cancellation_near(torch, shape, gsum, gperp)
        else:
            inputs = supplement.sample_inputs(torch, profile, fmt, rule, gsum)
            rejected = 0
            bad = (inputs[0] >= -H) & (inputs[0] < H)
            while bool(bad.any()):
                ix = bad.nonzero(as_tuple=True); n = len(ix[0]); rejected += n
                for value in inputs:
                    value[ix] = (torch.randn((n,), device="cuda", dtype=torch.float64, generator=gsum)*1.+1.).float()
                bad = (inputs[0] >= -H) & (inputs[0] < H)
            latent, draws = None, {"rejected_tuples": rejected}
    else:
        rejected = 0
        if stratum == "near":
            k = torch.randint(0, 2**52, shape, device="cuda", dtype=torch.int64, generator=gsum)
            latent = ((k-2**51).double()+.5)*2.0**-63
        else:
            latent = torch.randn(shape, device="cuda", dtype=torch.float64, generator=gsum)*c["sum_sd"]+c["sum_mean"]
            bad = (latent >= -H) & (latent < H)
            while bool(bad.any()):
                ix = bad.nonzero(as_tuple=True); n = len(ix[0]); rejected += n
                latent[ix] = torch.randn((n,), device="cuda", dtype=torch.float64, generator=gsum)*c["sum_sd"]+c["sum_mean"]
                bad = (latent >= -H) & (latent < H)
        u = torch.randn(shape, device="cuda", dtype=torch.float64, generator=gperp)
        v = torch.randn(shape, device="cuda", dtype=torch.float64, generator=gperp)
        if rule == "ADD-ASSOC":
            center = latent*(1./3.); x = u*math.sqrt(2./3.); y = u*(1./math.sqrt(6.)); z = v*(1./math.sqrt(2.))
            inputs = [(center+x).float(), ((center-y)+z).float(), ((center-y)-z).float()]
        else:
            center = latent*.5; z = v*(1./math.sqrt(2.))
            inputs = [(u*1.+1.).float(), (center+z).float(), (center-z).float()]
        draws = {"rejected_tuples": rejected}
    ratio = weights(latent, rule, c, torch) if stratum == "near" else None
    return inputs, latent, ratio, draws


@dataclass
class Restoration:
    peak_count: int
    target_near_count: int
    near_remaining: int
    rest_remaining: int

    @classmethod
    def create(cls, count, near):
        if not 0 <= near <= count:
            raise ValueError("invalid mixture count")
        return cls(count, near, near, count-near)

    def select(self, torch, stratum, shape, ratio, generator):
        """Never reads program outputs, oracle values or error magnitudes."""
        if stratum == "rest":
            take = min(math.prod(shape), self.rest_remaining)
            chosen = (torch.arange(math.prod(shape), device="cuda") < take).reshape(shape)
            self.rest_remaining -= take
            return chosen, None
        if stratum != "near":
            raise ValueError("invalid restoration stratum")
        if self.near_remaining == 0:
            return torch.zeros(shape, device="cuda", dtype=torch.bool), None
        if not bool((torch.isfinite(ratio) & (ratio > 0) & (ratio <= ENVELOPE)).all()):
            raise ValueError("invalid proposal/target ratio or rejection envelope")
        uniform = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
        accepted = uniform < ratio/ENVELOPE
        rank = torch.cumsum(accepted.reshape(-1).long(), dim=0).reshape(shape)
        chosen = accepted & (rank <= self.near_remaining)
        self.near_remaining -= int(chosen.sum().item())
        return chosen, uniform

    def finish(self):
        return {"state": "COMPLETE" if self.near_remaining == self.rest_remaining == 0 else "INCONCLUSIVE",
                **asdict(self)}


def cpu_selection(stratum, count, before, latent=None, uniform=None, rule=None):
    """Independent CPU reconstruction for complete mask audits."""
    if stratum == "iid":
        return np.ones(count, dtype=bool)
    if stratum == "rest":
        return np.arange(count) < min(count, before["rest_remaining"])
    if before["near_remaining"] == 0:
        return np.zeros(count, dtype=bool)
    ratio = weights(np.asarray(latent).reshape(-1), rule, constants(rule))
    accepted = np.asarray(uniform).reshape(-1) < ratio/ENVELOPE
    return accepted & (np.cumsum(accepted, dtype=np.int64) <= before["near_remaining"])
