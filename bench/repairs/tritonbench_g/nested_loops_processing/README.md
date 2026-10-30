# Nested-loop processing repair

The public entries are kernel-only: `Correct.kernel_correctness` uses the
shared exact-cell `ProgramIO.ImplementsOnCells` relation; `FPEquiv.kernel_equiv`
states `original M rows cols ≡[assumptions M rows cols] indexCommuted M rows cols`,
with an empty experimental rule list. The shared implementation view reuses
the complete `ProgramIO.EquivalentOnCells` judgment. Both connect to the same readable
`Kernels.source` typed DSL, including all three loops, 14 cached loads,
24 masked stores, 24 barriers, and final pointer increments. The genuine
`Kernels.commuted` variant commutes the initial column multiplication and
has the same complete final state. The Python wrapper is unchanged by this
proof refactor: it constructs int32 arange/zeros, launches, prints, and returns
`None`. The older whole-wrapper statements are retained in
`Misc/Legacy/HistoricalCorrect.lean` and `Misc/Legacy/HistoricalFPEquiv.lean`.

`IO.lean` fixes the exact-cell input mode and independent initialization domain
in both implementation types. `Proofs/Pair.lean` reuses `IOProof.equivalent`;
`FPEquiv.kernel_equiv_io` in `Misc/NativeCorrect.lean` retains the explicit complete-cell formulation
as an ordinary theorem. The change preserves successful runs, source access
safety, every output cell and both exterior frames.

## Independent result and precision

The wrapper's payload is homogeneous int32. There is no floating-point or
integer payload arithmetic. The original generic low-level 15-storage-profile
proofs remain separate and unchanged. The Real target decodes the copied int32 words
as signed integers, including modular arange wraparound.

Store `t`, for `0 <= t < 24`, writes its 2-by-2 tile at column shift `2*t`.
Writing `i=t//12`, `j=(t%12)//6`, and `k=(t%6)//3`, its cached input shift is
`14*i`, `14*i+6*j+2`, or `14*i+6*j+2*k+4`, according as `t%3` is 0, 1, or 2.
The resulting sequence is:

```
0,2,4,0,2,6,0,8,10,0,8,12,14,16,18,14,16,20,14,22,24,14,22,26
```

Out-of-allocation input reads contribute zero. Only in-allocation output
cells are written; later stores win, and unwritten cells retain their initial
zero. `Math.target` reuses the independently proved ordered-copy formula.
It does not execute the repaired source.

## Checks, buffers, and source execution

Dimensions must be actual Python integers (excluding bool), lie in
`[0,2^63)`, and have product at most `2^62`. Nonempty input and output
buffers must be four-byte aligned and have complete byte windows fitting
uint64. Empty extents skip the native launch before pointer queries;
`KernelResult.empty_launch` covers this branch without buffer premises.

`KernelBinding.Buffer` records ordinary shared `TensorMetadata` observations
and actual stride queries for the constructed contiguous CUDA int32 tensors.
Constructor laws supply strides `(cols,1)` and the shared device. Its `Dense` address map
is derived from the actual checked `dataPtr`. `Binding.state` reads actual
words from the same physical-memory API and buffer addresses used in the
physical execution proof. Arange and zero initialization are constructor
API facts. There is no assumed final-output equation.

`SignedStep` through `SignedFinish` prove successful `FP.Structural.exec` of
the current pasted source, including its signed registers and runtime argument
initialization. `TypedStep` through `TypedFinish` explicitly describe the
historical natural-address skeleton. `SignedMemory.finalState_memory` and
`SourcePreservation` connect both complete original/candidate executions to
that skeleton, preserving success/failure and every memory cell. Internal
registers with different types are not equated. `TypedMemory.final_output` then
connects every cached load and store to the independent copy oracle;
`final_frame` preserves full cells,
including dtype tags, outside the output allocation. `KernelResult` proves
the signed Real result, exact int32 words, four bytes per word, and agreement
with physical execution for every permitted within-store lane permutation.
The required closure contains no allocator, wrapper, print, or native-launch
state machine. `Kernels.lean` imports no proof module.

## Repair and runtime boundaries

The original launch used multiple CTAs despite ignoring program ID and had
unmasked out-of-bounds accesses. The existing repair uses one CTA, widened
signed-int64 indices, extent masks, and clamped pointers even for inactive
lanes. For one-column layouts it suppresses one duplicate lane. Barriers
after each store establish order between overlapping stores; one CTA alone
would not establish that order.

The finite-address proofs cover the final shift 48 and all input shifts
through 26. The natural index channel is justified by those nonnegative
signed-int64 bounds. `CtaBarrierContract` states local barrier completion,
arrival, and release laws and derives cross-store order. Primitive memory
read/write, little-endian int32 storage, constructor metadata, and native
compiler binding to these source/barrier semantics remain model boundaries.
There are no deferred numerical obligations.

## Validation

The earlier validation recorded 24 required kernel modules and 34 repair modules.
The current full-directory replay is recorded separately in the migration checkpoint. The public
entries and source/memory/binding bridges pass standard-base axiom audits;
the target definitions pass the non-circularity checks. A combined import
audit checks coexistence with the preserved historical entries.

The focused check is:

```
python bench/repairs/tritonbench_g/nested_loops_processing/check_contract.py
```

It executes the actual repaired Python AST with a small adapter over CPU
Torch int32 storage and mocked CUDA allocation/launch/print. The prior CPU validation reported
12 guard rejections, 20 admissions, and 16 nonempty program executions.
This historical result is not a claim that the current environment has Torch
or that a new CPU/GPU run has passed.
Each execution observes 14 loads, 24 stores, and 24 barriers, with reversed
within-store lane order. Cases cover empty/wide-empty, one-column aliases,
overlapping stores, partial tiles, larger shapes, output guard words,
printed-object identity, and `None` return. This is not Triton's native
interpreter. GPU execution and generated native-code inspection were not
performed for this validation.

`IO.lean` names the original source and genuine commuted variant, the complete
int32 input and output windows, and their shared single-launch geometry. The
logical domain explicitly retains signed-int32 arange input and zero initial
output; exact input-cell binding preserves the original cell-tag requirement.
Every output allocation cell is observed, including unwritten zeros.

`Correct.lean` exposes the independent ordered-copy integer formula;
`Misc/TargetProperties.lean` retains its exact Real interpretation as `Correct.target_real`.
`FPEquiv.lean` proves both source traces safe and
successful, all typed outputs equal, and both exterior frames. Both public
statements have surface audits excluding raw execution/state/cell plumbing.
`Proofs/SourceSafety.lean` checks actual clamped loads/stores, all three loop
levels and barriers for both variants. Bounds on positive allocations imply
bounds of clamped addresses in every actual loop state.

The former complete raw endpoints and their supporting memory/address/barrier
aliases live in `Misc/KernelRealResult.lean` and `Misc/KernelNativeResult.lean`.
`Misc/TensorBridge.lean` retains its full prior statements and now refers to these
identified shared foundations. The IO proof reuses the same successful source
execution and complete output-cell theorem; it does not replace the Tensor or
physical results with a smaller theorem. No Python source, contract, dtype,
barrier, launch, numerical assumption or experiment status changed. Scoped
Lean builds check this integration; they are not independent replay or GPU tests.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/TargetProperties.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.

Additional optional reader paths: `Misc/KernelRealResult.lean`, `Misc/KernelNativeResult.lean`, `Misc/Legacy/CorrectPublic.lean`, `Misc/Legacy/FPEquivPublic.lean`, `Misc/Legacy/HistoricalCorrect.lean`, `Misc/Legacy/HistoricalFPEquiv.lean`. Their declaration bodies are retained. Historical readers remain separate from current primary/auxiliary checks.

## Source readability cleanup (2026-10-22)

`Kernels.lean` now contains complete ordinary `triton` quotations for both the
original and existing index-commuted kernel. Transparent source aliases retain
the complete function interface and existing proof unfolding. The whole-source
macro was removed in that organization pass. The current raw-body migration
retains the Python int64 casts and signed address registers; its connection to
the older natural-address proof is a complete execution/memory theorem in
`SourcePreservation`, not whole-state or AST identity. All three loops, cached
loads, clamping, unique-lane masks, stores and 24 uniform barriers remain.
`SourceSafety` separately checks the actual signed source and its runtime
argument invariant through every loop successor.

The scoped source-only delivery records targeted dependency builds and a fresh
joint check of the existing public and optional entries. This organizational
change does not reopen proof completion, change contracts or add a numerical,
host, compiler or hardware acceptance requirement.

### 逐句 Python 函数体

原始 Python 不变，Kernels 直接粘贴三层循环的完整函数体，保留三个
运行整数参数在开头的 int64 cast、四个 arange cast、每次递增、clamp、
全部 unique/mask、缓存读取、每次 store 和 barrier。三项调用实参在体外；
保留旧连续输入范围的实际 minor stride = 1，raw quotation 也接收完整实参。
新 index_commuted.py 仅对应既有 Lean 初始整数乘法候选，来源明确标注。
