# Removing a rounding before addition

This experiment compares the following local scalar expressions, with `R`
meaning FP32 round-to-nearest, ties-to-even, and gradual underflow:

```text
reference = R(R(u) + v)
candidate = R(u + v)
```

The additions inside `R` denote exact sums. The CPU implementation preserves
each binary64 sum as a `TwoSum` expansion and corrects binary64-to-FP32 double
rounding at midpoints. An independent exact-rational nearest-neighbour check
covers random inputs, normal and subnormal midpoints, cancellation, and the
existing FMA counterexample. Signed zero is compared numerically, not by bits.
Absolute oracle errors are evaluated in compensated binary64 arithmetic.

`u` is a binary64 draw, so the proposed intermediate FP32 rounding actually
changes it. There are two separately evaluated profiles: `v` quantized to FP32,
and `v` retained in binary64. The latter probes an addend that also has unrounded
precision. These are sampling representations of the arguments to `R`, not
additional dtype variants of any benchmark kernel.

Both raw operands are independent `Normal(1, 1)` draws (standard deviation 1).
Each profile uses seed 20261021, 4096 independent replicates, and 65,536 scalar
tuples per replicate: 268,435,456 tuples per profile. The budget is fixed before
sampling. There is no conditioning on outputs, extra draw, or retry-until-PASS.
Nonfinite outputs on valid inputs are errors, not excluded observations.

The experiment calls the shared [numerical gates](../../../scripts/numerical_gates.py):
mean local-ULP bias budget 0.05 with a five-standard-error band, and peak absolute
oracle-error ratio with PASS/WARN thresholds 10/100. Reverse-direction insertion
is evaluated separately from the same paired observations. A forward PASS does
not supply a reverse experimental result.

This is a **CPU numerical probe**, with an explicitly smaller replicate shape
than the catalogue's 4096-by-4096 GPU runs. Its raw gate decision does not import
a Lean assumption, establish GPU execution, or establish a universal relation
for arbitrary operand expressions or distributions. The existing FMA report
remains a separate experiment for `u = a*b` with FP32 operands.

```bash
python3 experiments/floating_point/rounding/run.py check
python3 experiments/floating_point/rounding/run.py run \
  --output Logs/fp-rounding-add-20261021-cpu \
  --replicates 4096 --samples 65536 --seed 20261021
python3 experiments/floating_point/rounding/run.py replay \
  Logs/fp-rounding-add-20261021-cpu
```

The output directory must be new. It contains frozen source/configuration,
per-replicate NPZ observations, source and observation hashes, diagnostics, and
full gate statistics. Replay runs in a separate process, verifies hashes, and
recomputes both directions' gates without executing the returned sources.

See [the recorded result](report/summary.md) for this run's outcome. Raw data
remain in the ignored `Logs/fp-rounding-add-20261021-cpu/` directory.
