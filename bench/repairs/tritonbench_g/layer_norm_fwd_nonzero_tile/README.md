# Layer normalization with a positive tile

`Kernels.lean` contains the complete `_layer_norm_fwd_1pass_kernel` function
bodies from `layer_norm_fwd_nonzero_tile.py` and the explicitly materialized
`layer_norm_fwd_index_commuted.py` candidate. Each retains all 24 top-level
statements, the `row.to(tl.int64)` cast, input casts, seed narrowing, both
seven-round random streams, optional branches and original stores. The
candidate changes only the operand order of the initial X-address product.
The complete Python module AST, including the wrapper, is checked.

Pointer dtypes, scalar call bindings, constexpr flags and the checked arange
bound are outside `triton { ... }`. The function body contains no `tl.region`
declaration, synthetic `BLOCK_N_arg` assignment or added output cast. Runtime
N and Triton's native integer-one specialization use separate full-body
quotations. `jit-correspondence.json` records four exact raw mappings.

The selected source profile uses FP32 tensor pointers, int64 seed storage and
Boolean dropout masks. Its generic primitive result preserves the actual
runtime integer-to-FP conversion for N; the integer-one specialization uses a
literal. An arbitrary primitive interpretation is never assumed to identify
these conversions or casts. Other requested output encodings have their own
existing `StorageCompiler` artifact interface; those optional claims are not
an unconditional hardware/compiler-conformance guarantee for this profile.

The Python input guard, original wrapper and positive tile rule are unchanged.
`tile N = max 1 (min 16384 (nextPower N))` preserves N=0 with a tile of one,
leaves accepted nonzero widths unchanged, and rejects oversized widths before
launch. All source flags remain represented. Public width, read/write
separation and active-output geometry conditions remain explicit; neither
positive rows nor positive columns are added as assumptions.

## Proof connections

- `Source.lean` binds the public records to the actual pasted source.
  `Proofs/Typed*.lean` proves successful execution, all seven outputs,
  unchanged input/return views and the exact exterior tagged-memory frame.
  Each ordinary-grid row runs from the same initial memory; the disjoint merge
  reads its actual execution result.
- `NativeIO.lean` holds the complete ordinary source IO. Its opaque reference
  retains explicit input casts, actual runtime N conversion, reduction order,
  both streams and zero-column statistics. The reference follows actual lowering, including the implicit FP32 destination
  conversion of conditional affine values; no explicit cast is added to the body.
- `Correct.lean` uses `ActualLogical.io`, the exact Real interpretation of that
  same pasted source and complete IO judgment. `RealModel.lean` supplies only
  mathematical primitive meanings; `Proofs/NativeMathematics.lean` proves the
  independent logical-column sums, canonical Philox dropout, normalization and
  affine formulas with Mathlib. The logical input/output maps are bijections,
  including inactive channels. This is a Real specification, not an FP law.
- `FPEquiv.lean` compares the actual original and index-commuted ordinary grids.
  `Proofs/RawPair.lean` proves equal complete states, including failure, from
  integer multiplication commutativity. `RawPairSafety.lean` separately proves
  the changed prefix safe and transports the identical full suffix. The grid
  result additionally provides both successful executions, access safety,
  every active output and both exterior frames. It needs no numerical rule.

The historical `AlgorithmProgramIO` helpers in `IO.lean` and
`Proofs/RealImplementation.lean` explicitly retain their previous `repaired`
source view. They are not used to assert that its AST equals the pasted body.
`Misc/RawRealResults.lean` distinguishes the historical algorithm theorem from
the current actual-source Real IO result. Retained compiler-artifact and requested-storage results remain separate optional
interfaces in `Misc/`; their artifact/source views are not used to certify the
current pasted body. Its current Real physical application instead follows the
actual typed grid and its complete memory frame.

## Validation

`bench/tests/test_layernorm_fwd_body.py` checks the complete bodies and the sole
candidate AST change, with mutation checks for removed casts, changed RNG
rounds/streams, masks, optional stores and the native integer-one call.
`bench/tests/CheckedArangeCallBody.lean` checks symbolic bound aliases and
rejects unchecked or merely same-named adapters.

The focused replay of all 293 related modules and their joint import passes,
including both public entries and the current physical-memory application.
Cumulative acceptance is recorded separately in the repository's source
correspondence inventory and checked review receipts. Historical `.veritile-cache` reports
are diagnostic history, not build dependencies or current replay evidence.
No new GPU or numerical experiment is part of this source migration.
