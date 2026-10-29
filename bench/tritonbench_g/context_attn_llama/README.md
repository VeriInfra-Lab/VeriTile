# context_attn_llama

## Current validation (2026-10-14)

The complete `Correct`, `FPEquiv`, and typed execution witnesses compile for
all fp16/bf16/fp32 source storage branches. The passing closure, including the
registered frontend regressions, contains 3,607 Lake jobs and 318 unchanged
captured inputs; see [context-fp-typed](../../review-checkpoints/2026-10-14/context-fp-typed/README.md).

The FP public premise is now `TypedScalarArithmetic.Rules`: existing scalar
rules plus a separately validated, exactly typed fp32 multiplication atom.
The half/bfloat branches guard the actual typed operands before the existing
result cast. The old real-channel atom alone does not imply that law; a Lean
countermodel checks this distinction. See the
[binding and evidence note](../../review-checkpoints/2026-10-14/typed-fp32-mul-binding.md).
No storage branch or loop was removed. This is conditional abstract FP
program equivalence, without a claim of IEEE equality or new GPU validation.

Historical comparator counts below describe earlier snapshots. The current
host cannot perform official replay because user-systemd and kernel Landlock
isolation are unavailable.

## Contract and historical evidence

- Source: `context_attn_llama.py::_fwd_kernel`.
- `Kernels.lean`: full original and scale-first candidate for fp16/bf16/fp32 storage. The candidate only swaps qk and sm_scale in the eager where arm.
- `Correct.lean`: both complete sources equal an independent initial-array base-2 weighted sum in real semantics, retaining finite causal and padded denominator weights. The contract includes successful finite flat execution, actual address bounds and a full tagged-cell frame.
- `FPEquiv.lean`: complete program equivalence using only admitted `mul_commute`, with finite actual fp32 qk/scale operands on every reached lane. Page gathers, binary dot, opaque accumulator-dot, exp2, casts, masks, stores and failure are preserved.

The real contract keeps dimensions, strides, grouped heads, program IDs and
signed metadata/pages symbolic. It requires positive BN, valid active metadata,
page-table, Q/K/V/output addresses, and injective active output stores. Inputs
may alias output: every load precedes the sole store. Experimental shapes do
not restrict the symbolic derivation.

`Proofs/` contains source decomposition, execution, numerical identities,
loop invariants, memory safety and finite-memory transport. A real witness
uses grouped heads, a cached prefix, three key blocks, a partial tail and
nonconstant gathered page indices. A consistent abstract FP backend witnesses
a nonempty domain and complete source execution from empty registers.
These are model-level results; they do not establish GPU execution, IEEE
bitwise equality, fixed-width index-overflow safety or concurrent launch safety.

Historical validation (not current acceptance): both repository proof closures (3570 jobs), source/type
regressions, source-only corpus gates, 23 audit/layout tests and independent
comparator/trust replay of all 97 files across these two cases. Final target
inventories and source fingerprints match `Logs/comparator-check-b23tm5xw`.
