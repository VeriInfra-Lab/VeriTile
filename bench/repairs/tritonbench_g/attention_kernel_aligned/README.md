# attention_kernel_aligned

The public specifications use finite typed IO for both original FP16 and BF16 sources. The original Python, original quoted source definitions, input contracts and historical evidence are unchanged; the scheduling candidate is added separately.

## Review entry points

- `IO.lean` exposes the actual source schedule, channel dtypes, original four-dimensional strides, finite input/output windows, masks, encoding, output observation and independent formulas. Q/K/V have shape `[B,H,N,D]`; bias has `[B,H,N,128]`. There is no private scratch. Read/read aliases use simultaneous input bindings; output separation is required only on nonempty grids, as before.
- `Correct.lean` contains one mathematical implementation specification for each storage branch. The shared `ProgramIO.Implements` judgment includes actual source access safety, successful execution of the complete supplied grid permutation, every logical output, and the exact exterior cell frame.
- `Misc/NativeCorrect.lean` contains one optional ordinary native observation theorem for each branch. `ProgramIO.Observes` preserves arbitrary symbolic input payloads and observes the final output at its original physical owner. It retains the instruction model and existing precision-specific compiler plan; the BF16 serial reduction route remains admitted. This is the existing independent native observation claim, not a new kernel transformation or a native-to-Real equality.
- `Misc/Auxiliary.lean` retains the exact previous raw Real/native statements, the polymorphic empty-grid result, and the raw native reference. Empty grids still execute identically and have vacuous output windows in the public IO specifications.
- `Misc/TensorBridge.lean` is unchanged. Its complete checked Tensor, pointer, word-readback, output-allocation, and exterior-frame statements remain available in both precision namespaces. It and the finite IO results reuse the existing whole-source execution foundation; both new implementation pairs reuse that complete execution and IO foundation.

## Independent mathematics and precision

For each batch/head, the exact-Real formula is

```
s[i,j] = (scale * 1.44269504) * sum_d Q[i,d]*K[j,d]
       + bias[i,j//BN] + bias[i,64+(j%BN)%64]
out[i,d] = sum_j 2**s[i,j]*V[j,d] / sum_j 2**s[i,j]
```

The noncausal formula includes every key. Bias is not multiplied by log2(e). `Real.expected` supplies this mathematical target. Finite-input locality connects only the declared tensor coordinates to the older proof; the public IO theorem does not require infinitely many exterior memory cells to hold input values.

The independent native recurrence retains every original mask, cast, dot, reduction, online update and final division. FP16 QK has FP16 dot output and FP16 bias arithmetic. BF16 QK accumulates in FP32, narrows to BF16, then performs its explicit BF16 additions. Both sources scale Q in FP32 before narrowing, narrow probability before PV, and retain FP32 PV/online state and final division. NaN/Inf words and arbitrary symbolic native payloads are not excluded.

The existing `Guard` preserves the selected tile widths, original stride/index/launch conditions, resource exclusions and `N <= 64*BN`. The native compiler plans retain their original routing boundaries, including BF16 `nativeSerial=true`. No Python assertion, dtype requirement, scalar ABI, separation premise, numerical law or compiler applicability claim is added.

## Proof organization and validation

`Proofs/IOExecution.lean` and its BF16 counterpart reuse `RawKernel.complete` and the existing mathematical proof. `GraphLocality` restricts the input dependency to the finite original windows, and `IOResults` connects exact typed outputs and the full old frame. `AccessWindows` derives bounds for every active source address. `SourceSafety*` follows the actual prologue, loop predecessor states, masked Q/K/V/bias loads and final stores; `IOSafety` composes these across the entire grid. The small proof-only prefix in `SafetyFragments.lean` is checked by exact definitional equality against the authoritative source and is never selected as a public or runtime kernel.

The current scoped validation and source/dependency hashes are recorded in `.veritile-cache/public-io-format-20261020/attention_kernel_aligned/report.json` at the repository root. Public surface, axiom and independent-target audits accompany the Lean entries. The old packets in `kernel-evidence/` remain historical evidence and are not recopied or relabeled as a new replay.

This format pass adds no numerical assumption, numerical experiment, GPU measurement or independent replay claim. Existing primitive/compiler boundaries and historical execution evidence remain separate from formal IO completion. The deferred shared launcher grid-product issue remains outside this task.

## Complete implementation pairs (2026-10-21)

`FPEquiv.query_early_equiv` and `BF16.FPEquiv.query_early_equiv` each state
`Pair.original p work ≡[[]] Pair.queryEarly p work` for the actual full source
at that storage precision. `Kernels.lean` contains complete ordinary readable
`typedSurfaceQueryEarly` quotations and matching `queryEarly` parameter adapters.
The unchanged Q scale/load/narrowing statements move immediately after
`Q_block_ptr`, before independent K/V pointer construction and m/l/acc
initialization. This permits earlier Q issue and overlap with independent
prologue work; no measured speedup is claimed. The full candidate Python module
is `attention_kernel_aligned_query_early.py`, with the same change in both actual
storage kernels and no host change.

`Proofs/PairPrefix.lean` proves exact final prologue state equality and equivalent
actual load safety for either branch. `Proofs/Pair.lean` connects these to complete
kernel runs, every original source invocation and the full supplied grid order.
Both implementations succeed safely, every typed output cell agrees, and both
preserve the entire raw exterior frame. The original arbitrary native expression
carrier includes all input words and special values; no finite-value restriction,
new numeric law, experimental rule or deferred numerical experiment is introduced.
The BF16 serial reduction route, scaleWord including signed zero, casts, masks,
dot result precision, all original strides and aliases, original nonempty output
separation and empty grids retain their full scope. Existing compiler applicability
and optional physical-owner observations remain unchanged.

The original proof-only prologue/body/epilogue decompositions have moved from
`Kernels.lean` to `Proofs/Parameters.lean` with their original full names and
identical definitions. They do not construct either authoritative source. The
main closure does not import `Misc/`; independent Real theorems and optional
Native/Tensor/audit entries are checked jointly. This adds no GPU run, numerical
experiment, compiler conformance task or new host acceptance requirement.
