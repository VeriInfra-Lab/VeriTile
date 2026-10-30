# FP32 intermediate-rounding experiment — 2026-10-21

**Outcome: INCONCLUSIVE in both sampled operand profiles. No new numerical assumption was admitted.**

We tested removal `R(R(u) + v) -> R(u + v)` and, separately, insertion in the reverse direction. `R` is FP32 nearest-even rounding with gradual underflow. Both directions use the same paired observations.

This is a CPU experiment. GPU testing was unavailable (`torch.cuda.is_available() == False`; no NVIDIA device). Each profile contains 4096 independent replicates of 65,536 tuples, or 268,435,456 tuples. Operands are independent Normal(mean=1, standard deviation=1) draws, seed 20261021. `u` retains binary64 precision; `v` is either quantized to FP32 or retained in binary64. The draw budget was fixed before sampling.

| Addend v | Bias mean (local ULP) | Five-SE bias interval | B (must be ≤ 0.05) | Removal magnitude U (PASS ≤ 10) | Insertion magnitude U | Decision |
|---|---:|---|---:|---:|---:|---|
| fp32 | -0.056385305 | [-0.408100392, 0.295329783] | 0.408100392 | 1.028600327 | 2.275054384 | INCONCLUSIVE |
| fp64 | -0.063408849 | [-0.451212354, 0.324394657] | 0.451212354 | 1.013230063 | 2.294967492 | INCONCLUSIVE |

The magnitude gate passed for both directions and both profiles. The largest observed removal error ratio was exactly 1 in each profile; the fitted upper estimates U may be slightly larger. Bias intervals overlap both the acceptance band and its exterior. This is neither a bias PASS nor evidence of a bias FAIL.

Near-cancellation samples make the true sum small and therefore make its local FP32 ULP very small. These samples can contribute large normalized differences, even when absolute errors are small. They were retained; no cancellation guard or output-dependent filtering was added.

For example, the FP32-addend profile contained the following pair (hexadecimal floating notation preserves the sampled values):

```text
u = 0x1.4a426ba66be3cp-2
v = -0x1.4a426c0000000p-2
reference = 0x0.0p+0
candidate = -0x1.6650700000000p-28
exact_sum_high = -0x1.6650710000000p-28
exact_sum_low = 0x0.0p+0
```

In this sample the reference is zero while the candidate retains a small negative sum. Its absolute local-ULP difference is 11741240.0. This diagnoses the observed statistical uncertainty; it is not a reason to remove the sample.

The intermediate rounding was nontrivial on 268,435,455 of 268,435,456 tuples in each profile. Outputs differed on 61613653 tuples with an FP32 addend and 50490872 tuples with a binary64 addend.

Validation: 2074 exact-rational rounding checks passed, including six cases that require correcting binary64-to-FP32 double rounding. A separate-process replay verified frozen source/observation hashes and reproduced all four gate outcomes. This is statistical replay of the observations, not independent GPU execution or raw-output replay.

Artifacts: [configuration](experiment.json), [full results](summary.json), [replay](replay.json). Raw per-replicate observations and frozen sources are in `Logs/fp-rounding-add-20261021-cpu/`. The [runner and commands](../README.md) reproduce the experiment.

The existing FMA-CONTRACT report is unchanged. Its product-derived `u = a*b` profile is distinct from this direct binary64 sampling. This result neither revokes that report nor establishes a universal arbitrary-expression rounding law. The current experiment is smaller than the catalogue GPU sampling budget and is not imported into Lean or the per-task completion ledger.
