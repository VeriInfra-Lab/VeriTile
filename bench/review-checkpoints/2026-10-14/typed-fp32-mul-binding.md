# Typed fp32 multiplication binding — 2026-10-14

The context_attn_fwd and context_attn_llama half/bfloat input branches perform
an actual typed fp32 score multiplication before casting its result into the
real carrier. The previous scalar atom constrains only the real arithmetic
channel; the structural algebra intentionally has no implicit channel coherence.

`TypedScalarArithmetic.Rules` now carries the existing scalar rules plus a
separate `validatedTyped` premise for exactly the two typed fp32 fragments.
Both fp32 input registers must be finite and the fp32 output register is observed
directly. The fragment selects the default fp32 profile; explicit precision tags
retain their meanings. Its key ends in `:typed-fp32` and its configuration records
this new binding. The old `ScalarArithmetic.Rules` atom catalog is unchanged.
This is a conditional rule-model statement. The extra validation premise has
not been proved from the old rule, and no IEEE identity is asserted.

The published fp32 MUL-COMMUTE row remains the numerical provenance. Its source
`experiments/floating_point/kernels.py` has SHA-256
`f239cb37af1eab11170c912303a82fa4201c681caa1273db7acb0e0e746aa98e`, matching the
frozen report metadata. An offline compile of that unchanged source, both sides,
sm90, block 1024, 4 warps, FP fusion disabled, yields respectively one fp32
`arith.mulf a,b` and one fp32 `arith.mulf b,a`. Full artifacts and hashes are in
[typed-fp32-native/receipt.json](typed-fp32-native/receipt.json). This verifies the
source/type mapping only: no device was launched and report samples were not
replayed. The report validation trust boundary remains explicit.

The two public FPEquiv entrypoints now take `TypedScalarArithmetic.Rules`.
For fp16/bf16 storage their reached-site guards inspect the actual typed fp32 qk
and typed scalar before multiplication, including every eagerly evaluated where
lane. For fp32 storage they retain the original real-channel rule and guards.
All three storage variants, signed loops, masks, casts, dots, gathers, stores,
complete final state and execution failure remain in the statement.

`TypedFP32Mul.lean` proves that a model can satisfy all old scalar rules while
its typed fp32 multiplication is noncommutative; thus the new atom is not
derivable from the old assumptions. It also rejects real-register guards for
missing typed inputs and proves compatibility of the old real defaults. Both
full kernels have successful typed execution witnesses under a model satisfying
the combined rules; these are abstract executions, not GPU runs.
