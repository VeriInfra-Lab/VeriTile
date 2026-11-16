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

## FP64 profile

The [FP64 profile](fp64/config.py) uses FP64 operands, live accumulator and
result, K=32 and a 32×32 tile. It retains independent Normal(1,1) entries,
unit weights, the same role seeds and the same two thresholds. Statistical
independence is still per whole tile. The FP32/TF32 result above stays attached
to its own profile and is never replaced by FP64 evidence.

```bash
python scripts/check_two_gates.py dot-fp64 check
python scripts/check_two_gates.py dot-fp64 run --smoke --output Logs/dot-fp64-smoke
python scripts/check_two_gates.py dot-fp64 run --output Logs/dot-fp64
python scripts/check_two_gates.py dot-fp64 import Logs/dot-fp64 --output Logs/dot-fp64-cpu.json
python scripts/export_dot_fp64_report.py --bundle Logs/dot-fp64
python scripts/export_dot_fp64_report.py --check
```

The FP64 observer uses double-double products and accumulation so that its
reference retains more precision than either tested implementation. Every
mandatory raw output audit independently reconstructs the dot plus live
accumulator with exact integer products and rational arithmetic, checks FP64
ULPs and bounds the reference error below 2^-35 output ULP. The same audited
lanes are then connected to the per-tile observations and complete gate
statistics. Both peak maxima cover the full 2^32 whole input groups.
The raw bundle includes frozen sources, implementation digests, fixed audit
locations, every part's statistics and observation fingerprint, special-value
diagnostics, elapsed time and peak memory.

Only a complete 2^32-group run can yield formal statistical acceptance;
`--smoke` and a different `--samples` budget remain diagnostic. FP64 acceptance
supplies no FP32 assumption, strict IEEE equality or whole-kernel refutation.


## Selected FP64 evidence

The [FP64 receipt](fp64/report/receipt.json) retains a complete 2^32-group
SM89/Triton 3.5.1 run, server replay and independent local replay. The Normal
statistical outcome is **ACCEPT**: mean 0.0018087388155 ULP,
SE 0.00184924112698, upper 0.0110549444504 against 0.05,
and peak ratio 1 against 10. No groups were excluded.
Exact rational checks cover all 524,288 mandatory raw output lanes.
The double-double reference's largest audited error is
2.13162820728e-14 output ULP.

The zero-padding diagnostic A=-0, B=+0, accumulator=-0 returns +0 in the
reference and -0 in the candidate, across 1024 output lanes of one diagnostic
group. **All-word admission is disabled.** This classification result remains
separate from the passing Normal statistics. FP32/TF32 rejection and its empty
admission table remain unchanged. FP64 control is not a dependency of the FP32
source-model disproof in VeriTileBench.
