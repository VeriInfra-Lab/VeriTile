# rms_rbe_matmul

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

Current review: the typed source retains both weight arms, all E5M2 byte
decodes and casts, native integer power, separate dot/accumulator addition,
the actual dynamic ceiling-divided loop and the wrapper-selected output dtype.
The formal family contains six floating input labels and both variants; native
frontend admission for these input/dot combinations remains pending.

`Kernels.lean`, `FPEquiv.lean` and all 24 dtype-erasure projections compile.
Full source/candidate FP equivalence preserves every state, memory and failure
for arbitrary backend algebras, with no extra FP assumptions. The candidate
commutes only the final output-row index product. The complete non-FP8
exact-real execution also compiles: successful execution, the independent
input-memory formula over all loaded K lanes, active output readback and the
complete raw-cell frame. Its logical K divisor and padded final block are
retained without a divisibility premise; only active output lanes require
injectivity, allowing N smaller than BN.

The original 53 declaration signatures and docstrings remain unchanged across
`Proofs/LegacyReal.lean` and `Correct.lean`; the historical specification is
declared directly in `Correct.lean` so corpus discovery still includes it.
The relocation passed the combined 3482-job build. The full-source projection
passed 3417 jobs and the new transport/result/execution chain passed 3420 jobs
with standard-axiom module audits. These are local checks, not independent replay.
Actual source access safety and finite-flat IO now also compile. The new
`half_formal_real_correct` public contract and finite witness pass the combined
3451-job public build. The witness covers both variants and all six formal
input labels, batch id one, independent strides, logical K=1/BK=2 and N=1/BN=2.
It checks the actual nonzero output one plus integer sentinels before the batch
output and in its masked tail. Both public entries reach all thirteen local
modules at that checkpoint. The subsequent FP8 public build passes all 3461
jobs and now reaches twenty-one local modules. It includes exact typed
precision for all 24 templates, complete byte-source execution and access
safety, independent finite byte-array inputs and the public
`fp8_formal_real_correct` contract. Byte inputs must be actual integer cells
with finite E5M2 decoder results. The finite byte witness retains the same
batch, strides and partial tiles, produces output two, preserves both output
sentinels and leaves the actual matrix/RMS bytes 64/60 unchanged.
The separate typed backend numerical witness now also passes (3380 jobs).
It composes all twelve prologue statements, the actual ceiling-counted loop,
three normalization statements, four address/mask statements and the final
store. The loop's nine/fourteen original assignments are checked in consecutive
groups and reconnected by a source partition equality. Both weight arms and
both variants produce output two at batch offset 17, with the fp16/E5M2
destination tag, unchanged bytes and integer sentinels checked on that same
full execution. This algebra uses real arithmetic and identity casts, not
physical IEEE rounding. Native admission and the separate rotary JIT remain
unresolved.

The final fourteen-declaration audit passes with standard Lean axioms, no FP
assumptions and a non-circular input-memory formula. The two generated
specification sheets have no self-reference or missing summary. Current receipts
are `/tmp/veritile-rms-rbe-half-public-final.log` and
`/tmp/veritile-rms-rbe-half-public-audit.log`. The four public coverage records
are now migrated with all thirty-five source/support fingerprints validated;
independent replay and the stated full-port gaps remain pending. The FP8
milestone receipt is `/tmp/veritile-rms-rbe-fp8public-first.log`; its regenerated
specification sheets also have no self-reference or missing summary. The
subsequent `/tmp/veritile-rms-rbe-fp8public-audit.log` checks twenty-seven
declarations against the standard axiom base, reports no FP assumptions and
checks both input-memory and pure-array formulas for non-circularity.
The typed execution/readback receipt is
`/tmp/veritile-rms-rbe-typed-values-second.log`. Both public entries now import
all thirty-five local modules. Their final dependency refresh passes all 3476
jobs in `/tmp/veritile-rms-rbe-final-public-build.log`.
The final `/tmp/veritile-rms-rbe-final-public-audit.log` passes thirty-three
standard-axiom checks and both formula non-circularity checks, with an empty
FP assumption list.

The wrapper does not check that K is divisible by its block width 64. The
source performs unmasked loads through `cdiv(K, 64)` blocks, so a partial final
block can exceed an exactly sized allocation. The historical exact-multiple
hypothesis is a restriction of the existing theorem, not a guarantee supplied
by the wrapper. Historical module docstrings are retained for compatibility;
this review supersedes their host-guarantee claim.

- Source file: `rms_rbe_matmul.py` (upstream `data/TritonBench_G_v1/rms_rbe_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 190 lines, 2 `@triton.jit` kernels (one dead in-file; see below)
- Status: active completion; historical public theorem
  `rms_matmul_rbe_closed_form_correct` is preserved in `Correct.lean`.

Target JIT: **`rms_matmul_rbe`** (the file's second kernel) — a batched
RMSNorm-fused GEMM: `out[b] = (rms(x[b]) · rms_w) @ w`, `llama_ff_triton`'s
kernel minus the SwiGLU gate plus a batch grid axis. The file's first
kernel, `rbe_triton` (rotary embedding), is **dead code in this file**: no
host function launches it and it calls `get_freq_multi_tokens`, which is
undefined in the file — the upstream file itself cannot run it. It is not
modeled.

Headline: every active output cell holds the fp16 cast of `n_i · S[i,j]`
with `S = Σ_t Σ_e X[b, r(i), ·]·RMS[·]·W[·, c(j)]` (batch offset
`pid_batch·stride_x_batch` in every X read, `% M`/`% N`-wrapped indices) and
`n_i = 1/√((Σ X²)/K + EPS)` — derived independently of the kernel. Side
conditions: `K = BLOCK_SIZE_K · numKBlocks` (unmasked loads), `son = 1` +
`BN ≤ som` store-lane injectivity, clean-input `hundef`.

The historical translation-surface blocker describes the older mathematical
slice. The new typed source preserves `USE_FP8`, the unused parameters
`start_token_position`/`RBE_EPILOGUE`/`THETA`, actual `tl.cdiv(K, BLOCK_SIZE_K)`
and both fp16/E5M2 output choices. The absent `get_freq_multi_tokens` callee
still prevents completion of the separate rotary JIT. No rotary epilogue is
executed by the GEMM body when `RBE_EPILOGUE` is true. See `source_scope.md`
for the remaining native precision, packed-byte and allocation boundaries.
