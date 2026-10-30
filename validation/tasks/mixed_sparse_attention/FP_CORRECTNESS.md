# Finite-precision source-value contracts

The new public forward contract targets the previously accepted fp16 storage
configuration and complete active-program scope: eight fixed block iterations,
one column iteration when BLOCK_N≥16, division, final fp16 cast, and masked
output stores. Its result is an independent initial-memory recurrence with the
actual operator order and a complete output memory function. The separate
materialized-accumulator epilogue scope is retained as its own public contract.
These declarations are backend value contracts, not native GPU correctness.

The input Q/K/V and output are fp16; the scale argument and accumulator
arithmetic are fp32. The graph retains the scale multiplication by 1.44269504,
Q conversion, each fp16×fp16→fp32 dot, fp32 output conversion, row max, exp2,
probability cast to fp16, accumulator update, row sum, normalization division,
and final storage conversion. Dot uses the source's default options; no
allow_tf32=false option is inserted. V uses the source's K batch/head base;
output uses the source's Q batch/head base, even though additional unused
stride arguments are present in the Python signature.

The independent recurrence keeps every source iteration. Inactive fixed block
steps still read masked block metadata as undef and still execute the numeric
update. No l_i*0=0 identity, infinity-infinity cancellation, positive/nonzero
denominator premise, zero-valued undefined tail, or masked-dot identity is
introduced. The column-index load uses its scalar cond mask, separately from
the lane n_mask. The existing scalar-mask shape normalization is explicit in
the source profile; it is not evidence of native JIT/launch validity.

There is a local source expansion correction for exactly the two m_i_new
statements. The common frontend expands tl.maximum into comparison plus
where, which does not preserve max-number behavior on NaNs and signed-zero
ties. The local profile restores Op.max2 with the original fp32 precision,
operands and shape. `parser_maximum_nodes` checks the exact two original AST
nodes; `local_expansion_exact` proves that each loop body changes only its
single maximum statement. The selected Triton 3.8 implementation defaults to
PropagateNan.NONE and emits maxnumf. Primary source hashes and excerpts are
recorded in FP_SOURCE_PROFILE.json and Proofs/MAXIMUM_SOURCE.txt. No claim
that comparison/where equals maximum is made.

The whole-loop proof gives successful execution and complete memory/frame from
ordinary initial inputs and an available dot primitive. It never assumes a
final accumulator, denominator or output. A nonempty 16×16×16 instance executes
both loop phases. It remains an interpreter witness rather than native kernel
execution evidence.

The epilogue software instance uses actual ScalarOps fp32 loads and division.
Its sole numerical callback is the explicit final fp16 conversion `toHalf`;
the output is independent of all unused `rest` callbacks. A kernel-checked
example divides 1 by 3, gets fp32 bits 0x3eaaaaab before conversion, writes the
converted result and preserves the masked neighboring output. Since fp16
conversion is explicit rather than implemented by the current fp32/bf16 bit
interpreter, this entry and its witness remain backend value contracts.

Native correspondence still needs the precise fp16 storage and conversion
policy, fp32 scale/literal representation, scalar rounding/FTZ and fusion
settings, exp2 implementation, maxnum/reduction NaN and tie behavior, exact dot
and reduction configurations, and integer/address/launch interpretation.
Unmasked query reads and metadata accesses require valid allocated input
windows; integer metadata and address arithmetic must fit the native domain.
The helper source pin is provenance for this chosen profile, not proof that
another runtime uses it. No experiment or numerical algebraic law is substituted
for these primitive/configuration bindings, and no closed native bit interpreter
is claimed for the complete attention kernel.
