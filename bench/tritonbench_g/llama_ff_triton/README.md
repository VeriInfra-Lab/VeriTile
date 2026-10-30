# llama_ff_triton

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

`Kernels.lean` contains both complete wrapper storage profiles: fp16 A/output,
with either fp16 or signed int8 W1/W3/RMS weights. `FPEquiv.source_equivalence`
checks complete state/failure equivalence with an empty domain and no FP
assumptions. `Proofs.Precision` checks both variants' arithmetic and conversion
nodes. The dimension-general public `fp16_flat_correct` and `fp8_flat_correct`
specifications in `Correct.lean` use `Spec.Real` with the named
`Contracts.ExactIO` and `Contracts.ByteIO` surfaces in `Proofs/Contracts.lean`.
Both retain actual source trace safety, successful finite-flat execution, the
independent padded-array SwiGLU formula and a full typed-cell frame.
The supporting `FiniteWitness.execution` gives nonempty instances for both
storage profiles and both variants: K=1/BK=2, N=1/BN=2, nonunit strides and five
nonzero disjoint allocations, with output `sigmoid(1)` and every other typed
cell unchanged.

The supporting `NumericExecution.numeric_run` and `NumericExecution.numeric_cells`
check successful execution and output readbacks of the complete typed source
under a concrete numerical algebra. The finite and typed numerical witnesses
remain in the import closure of `Correct.lean`, whose final public theorem is
the general `fp8_flat_correct`. The algebra's dot is an actual matrix product,
reduction is a sum, integer power/rsqrt/sigmoid perform their
real mathematical operations, and E5M2 decoding preserves the finite decoder
boundary. Casts retain values, so this is an exact-real algebra rather than an
IEEE implementation. Both storage profiles and both variants write the actual
nonzero `sigmoid(1)` result; the masked output, gap sentinel and original typed
weights remain unchanged. Consecutive source stages are composed back into the
original whole-source `numeric_run`; no slice replaces the executed program.

The source streams `cdiv(K, BK)` blocks, with four unmasked loads per block.
It accumulates native `libdevice.pow(a.to(fp32), 2)` in fp32, multiplies A by
RMS weights in fp16, performs two independent fp16-input/fp32-output dots with
separate fp32 accumulator additions, and finishes with fp32 RMS normalization
and the SwiGLU gate before the fp16 destination conversion.

The int8 specialization retains all three E5M2 byte reinterpretations. RMS
weights then convert directly to fp16; W1 and W3 convert through fp32 to fp16.
The existing shared byte decoder and independent backend callbacks are used.
Whole-source FP equivalence preserves all 256 signed-byte payloads, backend
calls, memory, registers and failure; it only commutes a natural output-row
index product. It assumes no power, dot, rounding or activation identity.

The Python wrapper fixes BM=BN=16, BK=64 and EPS=1e-6. It does **not** check
that K is a multiple of 64 or that the RMS allocation covers all K accesses.
When K is not divisible by BK, the actual source reads through
`cdiv(K,BK)*BK`, including padding, and that padding contributes to both the
RMS moment and GEMM sums while the RMS divisor remains K. Successful finite
memory execution therefore needs bounds for every actual rounded-up access;
a logical-K-only formula needs additional padding hypotheses. No such check
is silently attributed to the wrapper.

Exact-real claims for int8 weights require valid finite E5M2 interpretations
of all read bytes. Nonfinite E5M2 values remain available to FP backends but
have no successful real projection. Negative signed strides, fixed-width
address overflow, GPU launch races and native IEEE conformance remain outside
the current natural-address/real model.

All 57 historical declaration signatures and adjacent docstrings are retained
in `Proofs/LegacyReal.lean` and `Correct.lean`. They describe the historical
real model with floating-exponent power, a supplied K-block count and only the
fp16 weight arm; they do not establish the full typed source above. The old
claim that the host guaranteed K divisibility was incorrect. The historical
output theorem also assumes BN <= output row stride, which excludes valid
small-N wrapper layouts; the new contracts use active-lane injectivity from positive column stride and
`N * soutn <= soutm`, so partial output tiles are supported.

Validation: the final public-entry build passes all 3478 jobs. The dependency
closure of `Correct` and `FPEquiv` reaches all 37 local Lean modules, including
precision checks and complete typed numerical execution. Eighteen explicit
axiom checks use only the standard Lean base; both independent output
definitions pass non-circularity checks and the FP assumption list is empty.
The 57 historical headers and adjacent docstrings compare exactly with the
pre-migration file. All four public coverage declarations are migrated with
validated source/support fingerprints, and both generated specification sheets
have no self-reference or missing summary. Independent replay and final corpus
acceptance remain pending.
