# Typed retention and parameterized isfinite execution in isolation

The 3,532-job local Lake build passes with 243 unchanged captured inputs.
Retention's complete forward and backward typed fp32 execution succeeds from
arbitrary interpreter states, including both backward loops and the original
debug barrier. Its backward finite-allocation transport and independent
dq/dk/dv value/frame contract also pass. The exact debug barrier is a no-op
only in the sequential per-program interpreter; native synchronization and
inter-program scheduling are outside this result.

For isfinite, complete original fp32 and fp64 source executions now prove
bit classification for arbitrary caller-supplied input patterns, all runtime
sizes, block widths, natural strides, launch IDs and loop counts in the
selected rank-one Bool-output profile. Both source branches are retained.
Value contracts require distinct input/output regions, positive output
stride and actual typed input cells. The contract uses the independent
exponent-bit classifier, preserves the full tagged frame and quantifies the
indices covered by this program. It does not assume input finiteness.

This is local Lean elaboration and axiom-gate evidence. Typed finite flat
memory transport for isfinite, public entry integration, native refinement
and independent comparator replay are separate work. All 48 original
isfinite native configurations remain rejected in the earlier native receipt.
