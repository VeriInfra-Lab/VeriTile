# Local implementation and audit

Current selected public entries: [Contract.lean](../../../bench/tritonbench_g/rms_matmul_rbe/Contract.lean), [Correct.lean](../../../bench/tritonbench_g/rms_matmul_rbe/Correct.lean), [FPEquiv.lean](../../../bench/tritonbench_g/rms_matmul_rbe/FPEquiv.lean). The [contract review](../../contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](../../../bench/tritonbench_g/rms_matmul_rbe/Correct.lean) and [FPEquiv.lean](../../../bench/tritonbench_g/rms_matmul_rbe/FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

The typed source represents both Python JITs and both weight branches. The
candidate changes only the final natural output-row product. Q output storage
is independent of the byte-weight flag; K/V retain the source E5M2 casts.

## Checked public results

- Historical callee and QKV specifications remain inventory-visible in
  `Correct.lean`, with all 70 historical declaration headers and 52 historical
  docstrings unchanged.
- `half_formal_real_correct` and `qkv_half_formal_real_correct` include full
  source trace safety, successful finite-flat execution, independent initial
  input formulas and the raw memory frame outside active writes.
- `fp8_formal_real_correct` and `qkv_fp8_formal_real_correct` provide the same
  obligations for actual typed signed-byte inputs with finite E5M2 decoding.
  Their independent `PaddedRMSGemm.output` formula takes supplied arrays, not
  an executed result or decoder fallback.
- `FPEquiv.source_equivalence` and `FPEquiv.qkv_equivalence` preserve complete
  typed states and failure for arbitrary backends, including all byte values,
  with no FP assumptions.
- Independent precision templates cover all 48 callee configurations and the
  three source QKV loops/stores, including fp64-local multiplication and the
  source's separate fp32 dot-result/carry operations.

## Nonempty finite witnesses

The callee and QKV witnesses use logical M=N=K=1, BM=1, BN=BK=2, a nonzero
batch id, unrelated weight/output strides and allocated unmasked K padding.
All allocation, access, injectivity and execution hypotheses are discharged.
The non-FP8 QKV example produces Q=1, K=2, V=3. The raw-byte example produces
Q=2, K=3, V=4; masked Q/K/V cells retain integer 37, and the raw weight and RMS
bytes remain 64 and 60. These observations come from the same full flat run.

## Verification receipts

The final focused public build of `Correct` and `FPEquiv` passed (3482 jobs),
including the shared int32 exponent-range check for integer power. All 41 local
modules are reachable from the public entries, which expose eight specifications.
The final audit checked 38 declarations against the standard axiom base,
checked all 69 historical public names, and reported no FP assumptions in either
equivalence theorem. `rmsSpec`, the pure `PaddedRMSGemm.output` and the typed
numeric witness's target memory passed the no-kernel-reference audits; the
numeric target also has no reference to the structural execution function.
The source comparison confirms all 70 historical headers and 52 docstrings
remain unchanged. No `sorry`, custom axioms or `native_decide` are used.

Session receipts: `/tmp/rms_matmul_rbe_final_public_stable.log`,
`/tmp/rms_matmul_rbe_final_audit.log`,
`/tmp/rms_matmul_rbe_final_history_audit.json` and
`/tmp/rms_matmul_rbe_import_closure.json`.

The separate typed numeric witness passed its focused build (3380 jobs).
It runs the complete original and candidate QKV bodies for both weight profiles,
returning Q=2, K=3, V=4 in their actual source storage types. The proof composes
each callee's 12-statement prologue, actual ceil-K loop, three arithmetic
assignments, four address assignments and final store. Checked source
partitions retain every operation; the loop body has nine statements in the
half branch and fourteen in the byte branch. Each pass accepts the previous
pass's actual state, preserving its earlier outputs and all raw input cells.
The numerical backend uses real arithmetic, real matrix products and reductions,
integer power and finite E5M2 decoding, with identity casts rather than IEEE
rounding. These witnesses are included in the final public build and audit.

## Boundaries

`source_scope.md` records unmasked padding, the wrapper's absent K-divisibility
check, unused rotary parameters, formal floating X profiles, native admission,
IEEE rounding and storage-byte packing. No independent comparator replay or
GPU-validation result is claimed. These boundaries are not discharged by the
exact-real witness or by backend-parametric equivalence.
