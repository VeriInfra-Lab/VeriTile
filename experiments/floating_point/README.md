# Floating-point rule experiments

The two-gates runner is [check_two_gates.py](../../scripts/check_two_gates.py).
It checks atomic source-expression pairs on an NVIDIA CUDA GPU, then replays
all saved statistics and required raw audits on CPU. The experiment does not
change the Lean arithmetic model or automatically update its admission records.

## Run the experiment

The shared settings are in [two_gates_config.py](two_gates_config.py):

| Setting | Default |
|---|---:|
| Original-distribution peak input groups per configuration | `2**32` |
| Additional near-zero proposals for the three weighted FP32 rules | `2**25` |
| Execution shape, for bounded memory | `[1024, 1024]` |
| Runs per configuration | One, to the full fixed budget |
| Bias tolerance / standard-error multiplier | `0.05 ULP` / `5` |
| Candidate/reference peak-error limit | `10` |
| Worker processes sharing the GPU | `3` |

Install the pinned GPU environment, inspect the effective plan, then check a
small smoke run before starting a fresh full output directory:

```bash
python3 -m pip install -r experiments/floating_point/requirements.txt
python3 scripts/check_two_gates.py check
python3 scripts/check_two_gates.py run --smoke --output Logs/two-gates-smoke
python3 scripts/check_two_gates.py run --output Logs/two-gates
```

The default inventory has 168 configurations across the primary, supplemental
and typed-count profiles: 115 runnable and 53 explicitly unsupported precision
combinations. A numerical `REJECT` is an outcome, not an execution error.
`UNSUPPORTED`, `INCONCLUSIVE`, `ERROR` and `SMOKE_ONLY` remain distinct.
Smoke results never enter the accepted list.

Select configurations or inspect the direct-sampling control with:

```bash
python3 scripts/check_two_gates.py check --catalog primary --rules CANCEL,ADD-ASSOC,MUL-DISTRIB --formats fp32
python3 scripts/check_two_gates.py run --catalog primary --rules CANCEL --formats fp32 --output Logs/two-gates-cancel
python3 scripts/check_two_gates.py run --catalog primary --rules CANCEL --formats fp32 --iid --output Logs/two-gates-cancel-iid
```

`--config` selects shared settings. `--profile` selects a relation/precision
profile and requires an explicit catalogue. Primary, supplemental and typed-count
profile files supply distributions, source precisions, rules and launch settings.
The effective
parameters, sampling design, independent RNG roles and exact mixture count are
frozen before execution. No gate outcome changes the sample budget or seed.

`--samples` explicitly changes the total target-input budget; it is recorded in
the manifest. `--workers` controls process concurrency. The execution shape only
partitions memory and work; it is not a count of statistical replicates. Changing
any frozen setting, including the execution shape, requires a new output directory.

## One shared sample bank for both gates

For primary FP32 `CANCEL`, `ADD-ASSOC` and `MUL-DISTRIB` under independent
`Normal(1,1)` operands, the bank contains `2**25` near-zero proposals and
`2**32` rest-conditionals: 4,328,521,728 evaluated input groups in total,
0.78125% more than the target peak count. Other configurations sample their
declared target distribution with unit weights. The conditional design is not applied to
other precisions or input distributions.

Inputs are generated in FP64 and cast to the declared source precision. The
source programs define the casts, operation order, P/Q kernels and input domains;
the checker uses output-format ULP spacing and the FP64/residual oracle.
The near interval has half-width `2**-12`. For CANCEL it partitions rounded FP32
`a`; for ADD-ASSOC it partitions the latent pre-quantization three-operand sum;
for MUL-DISTRIB it partitions the latent sum of the two addends.

Let `p0,p1` be target stratum probabilities and `r=f_near/q_near` the near
conditional target/proposal ratio. Using signed shifts
`delta=(candidate-reference)/ULP(oracle)`, the mean uses **every bank
sample**, without clipping, self-normalization or error-based selection:

```text
mean = p0 * average(r * delta_near) + p1 * average(delta_rest)
SE^2 = p0^2 * sample_variance(r * delta_near) / n_near
     + p1^2 * sample_variance(delta_rest) / n_rest
bias passes when abs(mean) + 5 * SE <= 0.05 ULP
```

For the magnitude gate, draw `K ~ Binomial(2**32,p0)` from an independent CPU
random stream before observing errors. Accept near proposals with probability
`r/1.001`, retain the first K accepted points and the first `2**32-K` rest points.
This selection never reads either program's output or error. It restores the
**joint input multiset**, so both error maxima use the same selected inputs.
A physical shuffle is unnecessary for maxima. Missing near supply yields
`INCONCLUSIVE`; K is never truncated or resampled, and the peak budget is never
silently shortened.

Take the **unweighted** maximum absolute error of each source implementation
against its oracle, and require `candidate_peak <= 10 * reference_peak`.
The peaks may occur at different inputs. Full-bank peaks remain separate
diagnostics and never replace the restored-subset peaks in the decision.
A zero candidate peak has ratio zero; a positive candidate peak against a zero
reference peak has infinite ratio.

Valid inputs satisfy the rule's declared domain and finite-output selection.
The budget counts raw input groups; excluded groups are reported without
replacement. The three weighted designs have a fully finite valid domain, so
their restored peak sets contain exactly `2**32` valid groups. For other rules,
`peak_drawn` and `peak_valid` can differ. Integer-count probes use
bounded integer/constant input distributions.

The acceptance-rejection and mixture construction preserves the target multiset
law in the ideal-density model, apart from a bounded finite-supply failure event.
FP64 reconstruction and finite-state Gaussian RNG probabilities are not formally
certified. Finite observed maxima are empirical checks, not universal tail
bounds or unconditional IEEE equality.

## CPU replay, evidence and resume

Copy the complete output directory and use the exact experiment sources and
recorded NumPy/mpmath versions. CPU replay does not require Torch or a GPU:

```bash
python3 -m pip install -r experiments/floating_point/requirements-cpu.txt
python3 scripts/check_two_gates.py import Logs/two-gates --output Logs/two-gates-replayed.json
```

The full-run [validation receipt](validation/two-gates.json) binds its
measured source hashes and results.

The manifest binds resolved settings, profiles, seeds, mixture counts, source
snapshots, device/software identity and compilation options. Each configuration
retains all part moments, counts, both full-bank and selected peaks, input/output
fingerprints, PTX and required raw audits. Audits include fixed positions, stratum
boundaries, every active near rejection and every partial prefix selection.
Each raw audit checks the GPU observer against NumPy; the weighted pairs also
receive independent NumPy execution of their first source part in each stratum.

Replay checks identities, hashes, every part's contribution, exact fixed budgets,
all mandatory audits, restoration masks and final decisions. It does not claim
independent CPU execution of every one of the billions of source inputs.
The bank streams through bounded GPU memory; storing the entire input population
is unnecessary. Frozen sources and seeds permit regeneration.

Use the same GPU command with `--resume` to retain independently verified
complete/unsupported rows. Unfinished or errored rows restart from their frozen
seeds. Resume requires matching configuration, source and environment identities.
An execution error produces a nonzero exit; a completed numerical rejection does
not. Logs and raw measurements stay outside Git; the scripts and configuration
needed to reproduce them are tracked source files.

The lightweight CPU regression suite runs in core CI; full GPU experiments are
manual. Reports identify the measured source, precision, distribution and
acceptance scope. Lean admission records require evidence for the exact rule
instance used by the proof.

See [TwoGatesAcceptance.md](../../documents/TwoGatesAcceptance.md) for the contract
and [the supplemental guide](supplement/README.md) for source-relation details.
