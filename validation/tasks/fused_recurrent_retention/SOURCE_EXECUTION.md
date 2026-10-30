# Repaired source quotation and local compiler model

`SurfaceSource.lean` contains both complete FP32 JIT quotations from
`fused_recurrent_retention_repaired.py`. `Kernels.lean` imports these source
declarations. The quotation retains all arithmetic, masks, storage casts,
pointer updates, flags, the backward barrier, arbitrary nonnegative `do`
strides and the optional final-state cotangent and its strides. The one dummy
`scale` assignment provides quotation typing and is removed by `drop 1`;
execution receives the actual FP32 launch scalar in its initial register state.

`Proofs/QuotedExecution.lean` proves successful execution of both complete
quotations in `FP.Structural.exec`. This theorem starts with the launch scale
binding and otherwise arbitrary registers and memory. It supplies no successful
intermediate execution, carry, output, event stream, or whole-kernel equality.
Successful structural execution alone is not a numerical or pointer-safety
postcondition.

`Proofs/StagedArithmetic.lean` specifies a local compiler arithmetic model.
A scalar node has a word and static producer metadata. Metadata selection
never examines a word. The model retains product operands only at the source
multiply/add and product-reduction sites, implementing these routes:

- `beta * h + k * v` rounds `k * v` and fuses `beta * h` with that result.
- `d_h += (q * scale) * do` rounds `q * scale` before its FMA with `d_h`.
- `d_h *= beta` materializes its result before the next loop iteration.
- The output, dQ, dK and dV product reductions use four distinct `DotPlan`s.
  Plans contain only product/add/FMA nodes and indices; coverage proves that
  each reduction lane occurs exactly once. Plan selection is independent of
  numerical inputs.

The tags are derived from the source expression: the decay subtraction, the
launch scale, forward state update and reverse adjoint update. Loads and casts
produce ordinary word nodes, preventing input words from forging metadata.
No FMA equivalence with separately rounded multiply/add is assumed.

`Proofs/StagedNative.lean` relates the model to seven fixed scalar ports: add,
sub, multiply, FMA, unsigned32-to-FP32, exp2 FTZ, and the host scale conversion.
The scalar laws admit every FP32 bit pattern. The two unary functions are fixed
before tensor values are supplied. There is no equality between native FP32
arithmetic and mathematical real arithmetic.

The local compiler model and scalar ISA conformance are explicit boundaries.
They are not a machine-checked proof of Triton/LLVM/PTX lowering for every
configuration. The offline probes in
`.veritile-cache/kernel-evidence/fused_recurrent_retention/` invoke the actual
repaired JIT binder, retain scalar-one specialization, and compile eight
forward/backward cases for SM90 using the source's one warp and one stage.
They cover singleton, 16-wide, 32-wide and tail tiles and both optional-state
flags. The PTX exhibits the documented FMA routes. These are compilation and
source-binding checks; no GPU numerical execution is represented by them.

The numerical execution chain is proved in the `Quoted*` modules:

1. `QuotedForward` and `QuotedBackward` derive the complete forward and both
   backward loops from initial memory, including the repaired final cotangent,
   barrier, pointer rebasing, output casts, stores and dead pointer advances.
2. `QuotedLoopStores` derives each actual execution's memory as its finite
   typed stores. These records contain no arithmetic callbacks.
3. `QuotedGridExecution` composes actual `FP.Structural.exec` launches, with
   fresh PID/scale registers and current memory. Input-region preservation is
   proved, so all programs observe the same original inputs.
4. `QuotedPayloads`, `QuotedGridCoverage` and `QuotedOwnership` prove each
   store's value, exhaustive coverage, and unique actual writers, including
   masked DK/DV tails. No store-value or whole-grid equation is a premise.
5. `QuotedGridResult` proves the entire resulting typed memory equals the
   derived payload on exactly the written addresses, and initial memory
   everywhere else. Its input binding observes the actual tensor storage and
   offsets, retaining read-only aliases and arbitrary cotangent strides.
6. `QuotedStorage` lowers these proved exhaustive typed stores to the existing
   fresh tensor allocations, reading every value from the resulting Structural
   memory. The existing host sum plans then produce the API output tensors.
   It proves equality to the existing `sourceForward/sourceBackward` functions;
   that equality is a conclusion, not a boundary assumption.
7. `QuotedCalls` returns the existing Tensor metadata and object handles with
   that actual execution memory. Standard `Correct.lean` exposes
   `checked_quoted_returned`; standard `FPEquiv.lean` exposes
   `checked_quoted_fp_returned`, full memory/byte equality, and forward/backward
   complement frames. Independent dense Real targets and padded-lane native
   targets are audited separately from execution.

The wrapper assertions, Python implementation, independent mathematical target
and original kernel files are unchanged. `StoreDomain` contains only the
existing shape admission and q/v dense-stride facts; checked calls derive them.
There is no nonfinite-value exclusion, shape specialization, extra stride
restriction or assumed execution result. Padded FP32 lanes retain arithmetic
such as zero times infinity rather than applying Real-only simplifications.

Typed-region to physical allocation binding, Tensor metadata/identity,
allocator freshness, saved-version observations, host reduction plans and the
compiler/ISA scalar model remain the existing explicit local boundaries. This
work adds no host runtime, allocator, GPU launcher or universal byte-heap
model. A sequential grid order models execution; unique source writers and
input preservation justify the existing independent-program scheduling model.

The unused `Model` import of historical `Retained` proofs is removed. Those
proofs and all original files remain unchanged; `Reader` now audits the two
complete repaired quotation executions. The public closure does not rely on
the historical backward theorem with zero final cotangent.
