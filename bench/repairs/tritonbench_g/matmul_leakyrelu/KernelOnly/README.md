# matmul_leakyrelu kernel proof

`Correct.matmul_leakyrelu_real` proves that the actual original separate-dot/add
source implements the independent full-K Real matrix product, followed by the
slope 1/100 exactly when activation is `"leaky_relu"`. The interpretation uses
identity casts and retains FP16-tagged output cells. `OriginalIO`, `OriginalFacts`,
`OriginalSafety`, `OriginalIOSafety` and `OriginalReal` connect the actual source
schedule to complete typed inputs/outputs, access safety and the exact exterior
frame. No contracted-candidate theorem substitutes for this original result.

`FPEquiv.source_pair_equiv` compares `SourcePair.original` with the existing
`SourcePair.contracted` source: separate FP32 dot/add becomes a live-accumulator
dot, and `0.01*x` becomes `x*0.01`. The complete ordinary shared quotations remain
`FPValue.typed` and `FPFused.typed`, exposed through `sourceDsl` and `dsl`.
`matmul_leakyrelu_contracted.py` expresses that existing candidate as a complete
Python source with the exact same checked wrapper. The dot change eliminates the
separate accumulator addition and enables the existing accumulator-fed dot
instruction profile. No measured speedup is claimed.

The main judgment is the complete `ProgramIO.Equivalent`: the original logical
input binding and bounds imply both actual safe successful full schedules,
equality of every FP16-tagged output cell, and both exact raw exterior frames.
`Pair.equivalent` derives it from local typed primitive relations, complete grid
coverage and actual execution theorems. It does not assume a successful kernel,
whole output equation or whole-program equality.

The sole experimental entry is the local K32 FP32 dot/add-to-dotAcc relation in
`SourcePair.Rules.assumptions`. It retains the FP32 dot output, explicit carrier cast, separate
FP32 addition, arbitrary incoming accumulator and candidate typed dotAcc. Its
numerical validation is **deferred**, with both gates `NOT_RUN` and no artifact.
The main takes an explicit future `Rules` evidence/admission parameter; the pending record has no fabricated acceptance proof. The conditional `≡[R]` conclusion is not unconditional IEEE word equality.
`NUMERICAL_VALIDATION.md` records precision, domain, backend, sampling and the
project acceptance protocol. The existing graph counterexample remains valid.
The slope multiplication change instead reuses strict `FP.ScalarOpsLaws.mul_comm`
through an explicit concrete FP32 multiplier binding; no duplicate experiment is
required for it. Other scalar operations, casts and ordered dot ports stay fixed.

The original domain is preserved: arbitrary strided B including transpose, slice
and broadcast, read-only A/B aliasing, all admitted payloads including NaN/Inf,
all tails, K=0, empty shapes, both activation branches and every permutation of
the complete grouped 32x32x32/GROUP4 launch. The original checked dimensions,
signed32/padded-pointer bounds, physical byte envelopes, owning-device execution,
FP32 inputs/live accumulation and FP16 result stores are unchanged. The kernel IO
premises are the retained output/input region separation; no new numeric input
predicate, finite-only restriction or positive-size requirement is introduced.

Optional candidate Real/native correctness, Tensor byte applications and audits
remain in `Misc`, with their original declarations and scope. Main files and
`Proofs` do not import `Misc`. The existing canonical source pin and source/guard
controls remain intact. New source-delta controls check exactly two changes in
both complete Python ASTs and shared ordinary quotations. The candidate's 108 CPU
source executions check geometry, masks, stores, frames, K=0, empty shapes,
nonfinite data, aliasing and three PID orders; these are not contraction numerical
validation or a native TF32/GPU test.
