# chunk_cumsum_vector

`Correct.lean` displays `prefixIO`: the exact source under its Real
interpretation, input/output buffers, actual strides, all time rows, active
feature-column masks and no scratch. `prefixFormula` independently states the
inclusive time prefix. Its single `Correct.memory_correct` specification uses
the shared `UKernelIO.RegionalScheduledImplements` relation. One invocation
executes the complete dynamic time loop, including every partial final block;
this retains the previous per-program scope.

`IO.lean` displays `prefixNativeIO`, the existing `sumLeftVariant`, and
`carryDomain`. `FPEquiv.memory_equiv` compares the actual `original` and
`sumLeft` implementations as `original ≡[[]] sumLeft`. The shared
`ProgramIO.Implementation` view retains both complete invocation schedules and
their typed observers. Its existing `ProgramIO.EquivalentOn` proof remains as
ordinary `FPEquiv.memory_io_equiv` in `Misc/KernelFPInterface.lean`: both sources succeed, all active typed output
cells agree, both source traces are safe, and both exterior frames hold. The only source change remains the existing operand-order
swap in the FP32 carry update. The load promotion, explicit zero padding,
TF32-disabled dot, reduction order and output casts remain unchanged.

The checked Python path remains FP32. All nine configurations and every dtype
parameter already admitted by the generic source theorems are retained. The
logical `carryDomain` is exactly the prior `Rewrite.FiniteCarry` domain:
`NativeIO.finite_binding_iff` proves equivalence under the common input binding.
It constrains only active carry updates read by a later output block; inactive
lanes, the final unused update and other payloads gain no new finite-value
restriction. ADD-COMMUTE now uses the shared strict FP32 certificate for the
nearest-even software profile without input or result flushing. The public
theorem retains its primitive-model bindings through `Models []` of the fixed
strict theory and has no experimental `Rules` parameter. This change adds no
numerical experiments and preserves the existing carry domain.

`Proofs/FPAccess*` and `Proofs/RealAccess*` prove actual load/store safety along
both complete source executions, reusing the established loop invariants.
`Proofs/PublicRealIO.lean` and `Proofs/PublicNativeIO.lean` connect those proofs,
initial logical input values, every output and the exact exterior frame to the
shared IO relations. No execution, output or access-safety conclusion is assumed
as an input-domain or numerical condition.

The former physical-heap Real result, including `Stream.StableInput` and the
entire physical exterior frame, remains unchanged as
`RealInterface.memory_correct` in `Misc/KernelRealInterface.lean`. The former
raw native successful-execution/memory-equality/frame result is retained as
`FPInterface.memory_equiv` in `Misc/KernelFPInterface.lean`, using the same
empty experimental rule set. These reuse the existing proofs; they are ordinary
helpers, not extra public specifications.

`Misc/TensorBridge.lean` retains the execution and memory conclusions of
`TensorBridge.Correct.kernel_correct`,
`TensorBridge.FPEquiv.chunk_sum_left_equiv`, admission checks and all native
index/address bindings. Its FP consumer uses the empty experimental rule set
through the preserved helper interface and the same logical-input binding law.
Neither primary reader imports the bridge,
and no proof helper imports a primary reader.

Validation for this pass is targeted Lean compilation and standard-base axiom
and independent-formula audits. No new independent replay or GPU run is claimed.
The strict-rule migration builds the 37-module local import closure of the
selected FP entrypoint and registered Tensor consumer. The direct
`#print_fp_assumptions memory_equiv` audit reports no experimental assumptions
and identifies the shared strict ADD-COMMUTE certificate; the standard-base
axiom and independent-domain audits pass.

The 2026-10-21 pair-entry review reuses `NativeIO.equivalent` and the complete
existing proof chain through `Rewrite.successful_pair`. The empty displayed
rule set means no experimental assumptions; `Models []` and the original
finite-carry domain still bind the fixed strict FP32 theory. Neither the
backend, input domain, source precision nor per-program coverage changed.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.
