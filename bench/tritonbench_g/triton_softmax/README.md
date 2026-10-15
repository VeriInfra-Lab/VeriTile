# triton_softmax

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

`Kernels.lean` retains the complete `softmax_kernel` source: both row pointers,
inline lane/mask expressions, negative-infinity padding, max, subtraction,
exponential, sum, division and masked store. The candidate commutes only the
integer multiplication forming the input row address. The public correctness
entry is `softmax_typed_real_correct`. The required mathematical proof remains
an internal theorem in `Proofs/RealIO.lean`, rather than a second public
specification; the obsolete combined module is removed.

The typed source family has six independently chosen floating input/output
storage formats: fp32, fp64, fp16, bf16, float8e4nv and float8e5. The wrapper's
`empty_like(x)` chooses the same dtype for both. Smaller floating inputs promote
to fp32 for max and the subsequent arithmetic; fp64 remains fp64. Explicit
casts at stores expose Triton's implicit destination conversion through pointer
aliases. `Proofs/Precision.lean` checks the actual load and complete arithmetic
sequence, including dtype, padding and masks, for every source pair and candidate.

`Proofs/Real.lean` and `Proofs/RealIO.lean` provide the independent `softmaxSpec`
and mathematical flat-memory IO bridge. The public typed contract covers both source
variants using equality of their complete real projections and execution of
the entire address-rewritten candidate. It includes successful execution,
actual active-address bounds, partial rows and every cell outside active writes.
Both strides, program IDs and sizes remain symbolic; block size is positive.
The shared flat bridge explicitly requires zero undefined values. The real
proof does not assume the result or a complete source execution.

`FPEquiv.lean` states equality of complete typed executions, including registers,
all memory cells and failure, with an empty numerical-atom list and empty
floating domain. Exact natural-number multiplication commutation is sufficient.
The opaque exp, reductions, floating division and every storage conversion stay
unchanged. `Proofs/ExecutionBackend.lean` constructs actual successful executions
of all 72 variants in an exact-real abstract backend, from empty registers with
a nonzero program ID, unequal row strides and a partial tail.
`Proofs/Execution.lean` additionally proves that both complete fp16 variants
write one half to each active output cell and retain the masked sentinel.
This backend is not an IEEE implementation or a GPU experiment.

The source computes only the first `BLOCK_SIZE` contiguous columns. The wrapper
caps `BLOCK_SIZE` at 1024, so rows wider than 1024 keep an unwritten suffix and
normalize only the prefix. `Proofs/SourceBehavior.lean` records that every column
beyond the selected tile is unchanged. The mathematical theorem describes this
source behavior; it does not claim a full-row softmax for such inputs. The
wrapper also does not pass an inner-column stride, so the source reads contiguous
physical columns, irrespective of a tensor view's logical inner stride.

`source_dtype_receipt.json` records offline compilation of the unchanged JIT AST
with Triton 3.8.0 for CUDA sm90, `BLOCK_SIZE=8`, four warps and signed i32 runtime
strides/count. All six wrapper-correlated floating profiles compiled. Of all 36
independent floating pairs, 34 compiled; fp64-to-float8 destination conversion
was rejected by native lowering. All nine tested same-storage integer/bool
profiles were rejected at `tl.exp`'s floating dtype check. These observations
certify only the recorded specializations, not arbitrary tile widths or targets.
The Lean carrier theorems also describe the two target-rejected mixed pairs;
those do not constitute a native source execution claim.
Regenerate this receipt with
`python3 bench/tritonbench_g/triton_softmax/Proofs/check_native_profiles.py`.

Natural-number addresses retain the existing nonnegative-stride contract.
Host launch/grid composition, cross-program aliasing/races, native shape and
fixed-width address admission, and concrete IEEE numerical correctness are
outside these per-program statements.

All 11 Lean modules passed local elaboration with Lean 4.29.0 on 2026-10-13,
including the complete real contracts, every typed precision assertion, the
72 successful execution witnesses and the concrete fp16 output check. The FP
assumption printer reports `none`. All 23 Python source-audit gates also passed
for the selected fp32 source view; this is separate from the other dtype evidence.
A fresh dependency build against the updated shared DSL also passed (3463 Lake
jobs); the named targets and logs are recorded in the
[local build checkpoint](../../review-checkpoints/2026-10-13/current-port-builds/receipt.json).
Independent comparator and trust replay remain pending. These local results
alone do not claim current full-port acceptance or additional native target coverage.

All 36 original floating source views subsequently passed a complete
statement/expression AST comparison with the pinned Python JIT and all 24
production mechanical source gates. The comparison normalizes direct parameter
antiquotations and checked destination-store conversions only; 108 arithmetic
mutations were rejected. Frozen sources, selected profiles, logs and hashes are
in the [source-profile checkpoint](../../review-checkpoints/2026-10-13/softmax-all-source-profiles/receipt.json).
Reproduce with `python3 scripts/check_triton_softmax_source_profiles.py --output NEW_DIRECTORY`.
This includes the two native-rejected mixed pairs as source syntax; their
compiler rejection and the independent replay boundary remain as stated above.
