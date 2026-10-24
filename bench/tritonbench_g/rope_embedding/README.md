# rope_embedding

`FPCorrect.lean` adds the actual fp32 software contract for all four previously
accepted half-store stages: forward first/second and backward first/second.
Each stage executes its typed source, reads Q/cos/sin from the independent
initial state, and proves successful execution, active output readback, and
preservation of every unwritten cell. The load-to-`sin1.dtype` register casts
are retained. No numerical algebraic identity or assumed output is used.

The reference is the ordered operation graph
`sub(mul(q1,c),mul(q2,s))` or `add(mul(q2,c),mul(q1,s))`, with each operation
performed by the concrete `FP.Software` fp32 implementation and its supplied
configuration. Backward uses `FP.Value.neg` on the sine before multiplication.
The true floating negation preserves the sign-bit behavior of zero and NaNs;
it is not `0 - sine`. The default IEEE witnesses exercise all four stages with
`1 ± 2^-25` rounding and an unwritten tail. A backward witness produces `-0`
where replacing fneg with subtraction would produce `+0`; the primitive bit
witness also checks the sign flip of a signaling NaN payload.

`rope_embedding_source_projection` proves the full statement list of each
selected stage agrees with the existing source after dtype erasure. The
accepted scope remains four independent half-store stages. It does not assert
combined in-place execution of both stores or all heads of the original loop.
The supplied cosine/sine tables remain inputs. Binding the software execution
to a specific Triton/CUDA compiler and runtime remains the shared native
refinement boundary; it is not an additional mathematical law assumed here.

## Checked FP32 entrypoints

The separate [checked repair](../../repairs/tritonbench_g/rope_embedding/README.md)
preserves original forward clone, actual squeeze returns, and backward reshape
storage reuse. Its complete source and physical-return theorems derive finite
ABI conditions from executable guards. Original conditional proofs remain.

The selected repair now exposes kernel-only `Correct.lean` and `FPEquiv.lean`
entries, with a local complete typed source, independent polynomial, successful
whole-memory native-graph relation, and shared Tensor pointer/value binding.
The prior host results remain optional helpers. See the checked repair README
for the exact current domain, original short-sine counterexample, and explicit
compiler/primitive model boundary; no numerical rewrite experiment is pending.
