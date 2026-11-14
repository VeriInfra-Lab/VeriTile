# Dot/add contraction

`DOT-ADD-TO-DOTACC` compares a standalone dot followed by an FP32 add with a dot
seeded by the live accumulator. This fixed profile has 32×32 FP32 inputs and
accumulator, K=32, TF32 dot-input precision, four warps and three stages. It uses
the [shared two-gates settings](../two_gates_config.py) and its own typed
[catalogue](rules.json). The scalar catalogue remains separate.

```bash
python scripts/check_two_gates.py dot check
python scripts/check_two_gates.py dot run --smoke --output Logs/dot-smoke
python scripts/check_two_gates.py dot run --output Logs/dot
python scripts/check_two_gates.py dot import Logs/dot --output Logs/dot-cpu-replay.json
python scripts/export_dot_rules.py --bundle Logs/dot
```

Each configuration runs once to 2^32 independent whole input tiles, with
independent FP64 Normal(1,1) entries cast to FP32. Inputs have unit weights;
the scalar near-zero proposal does not apply. Execution parts of 8192 tiles
only bound memory. `--samples` is diagnostic when it differs from 2^32;
diagnostics and smoke cannot enter the accepted list.

For each tile, take the mean signed output-ULP difference as one independent
bias observation. Compute the standard error across tiles. The 1024 output
lanes share operands and are not independent samples. The peak gate uses the
largest absolute FP64-oracle error across every output lane of all 2^32 tiles,
on the same inputs for both implementations. Thresholds remain
`abs(mean)+5*SE <= 0.05 ULP` and `candidate_peak <= 10*reference_peak`.

Triton's dot/add combiner contracts both expressions even when ordinary FP
fusion is disabled. The reference therefore stores its FP32 dot result to an
FP32 temporary and adds the live accumulator in a separate kernel. The store
adds no precision conversion. Required PTX must contain TF32 MMA on both dot
sides and a separate FP32 add. The oracle uses FP64 products, sequential K
addition from zero and then the live accumulator; FP fusion is disabled.

Bundles bind actual GPU/compiler versions, profile, sources, all compiled PTX,
per-part moments and peaks, every part's group-observation fingerprint, fixed
raw audits, and special-value diagnostics. Each mandatory audit retains all
group observations for the selected part and complete inputs/outputs/oracle
for 32 predetermined groups. CPU replay checks group correlation, the ordered
oracle, ULP spacing, raw observations, exact fixed budgets and final decisions.
It checks evidence consistency and does not certify remote GPU execution.

NaN/Inf/signed-zero/subnormal diagnostics remain separate from the Normal
statistics. Classification or NaN-payload mismatches block the declared
all-word scope. Diagnostic agreement is not an exhaustive all-word proof;
numerical admission remains an explicit external modeling premise in Lean.
Evidence from an SM89 compiler profile does not validate an SM90 profile.
Numerical rejection is retained and never exported as a passing assumption.

The shared Lean template is `VeriTile.Triton.FP.DotContraction`. Export writes
`DotAdmission.lean` and a compact receipt after replaying the complete bundle.
The candidate always remains inspectable. Only passing, scope-compatible
evidence enters `DotAdmission.all`; constructing `AcceptedAtom` still requires
the explicit external `EvidenceValidated` premise. CI checks the generated
table against its receipt without rerunning the GPU experiment.

## Selected evidence

The [receipt](report/receipt.json) records a complete SM89/Triton 3.5.1 run of
2^32 whole input groups and independent local CPU replay. The outcome is
**REJECT**: the bias mean is -0.121701681 ULP, SE is 0.00588068783,
and `abs(mean)+5*SE` is 0.15110512 ULP against the 0.05 limit. The
magnitude ratio is 1, so that gate passes. No groups were excluded.
`DotAdmission.all` is empty; the candidate and its rejected evidence remain
available for inspection. This local numerical rejection is not a whole-kernel
refutation.
