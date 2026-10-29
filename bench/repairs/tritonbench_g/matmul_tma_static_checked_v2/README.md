# Static TMA matrix product

`KernelOnly/Kernels.lean` contains the complete original `matmul_tma_load_store`
function body from `matmul_tma_static_checked_v2.py`, inside `triton { ... }`.
Pointer types, scalar bindings and constexpr values come from the caller.
The three block pointers, layouts, two unmasked loads, default `tl.dot`, optional
FP16 conversion and final store follow the Python body statement for statement.
Both transpose flags and both output precisions remain supported.

The original Python has no explicit FP32 output cast. `candidate_kernels.py`
materializes the historical formal output-conversion variant as a separate
candidate: only the false arm adds `c = c.to(c_ptr.type.element_ty)`. That pointer
is FP32 on this path. This expresses the previous redundant-conversion comparison
in the corrected direction. It is not a new numerical optimization or speed claim.
The manifest checks both complete bodies and external call bindings; mutation
checks reject omitted casts, invented dot options and changed block-pointer order.

## Mathematics and complete execution

`KernelOnly/Correct.lean` exposes the independent Real matrix product through
logical matrix IO. Its summation covers the complete inner dimension. The public
judgment includes access safety, successful execution, every output and all
unrelated typed memory cells. `Proofs/SourceExecution.lean` evaluates the complete
actual source, including both constexpr branches and destination storage
conversions. `Proofs/RealExecution.lean`, `PublicIO.lean` and `LogicalIO.lean`
connect it to the independent formula without adding input restrictions.

`KernelOnly/FPEquiv.lean` compares the two complete programs with `≡[[]]`.
Its native interpretation retains the fixed K16 first/carried instructions and
FP32 accumulator, with the existing instruction-local realization boundary.
The new `Proofs/SourceNative.lean` explicitly defines FP16-to-FP16 storage as the
identity on the encoded word; the existing native model supplies FP32 identity
and the actual FP32-to-FP16 encoding. No identity is assumed of an arbitrary
backend cast. The default dot context is retained in the source; the selected
native dot interpretation ignores that inherited context by definition.
`Proofs/SourcePair.lean` connects the complete sources, including access safety.

The main Real and FP input domains remain unchanged. The optional native result
and physical Tensor applications remain in `KernelOnly/Misc/`. The concrete
Real/native execution connections in `Proofs/SourceModels.lean` justify reuse of
the historical physical refinement; no equality of the old and new ASTs is
claimed. The twelve recorded compiler profiles remain a separate corollary and
do not become a restriction on the general theorem.

## Wrapper and evidence scope

The Python wrapper is unchanged. Its five positive-power-of-two/K-minimum checks
and three block-tensor limits (`M*K`, `K*N`, `M*N` at most 1048576) retain the
existing static domain. `KernelOnly/Contract.lean` connects those executable
checks with the mathematical contract and derives the dimension bounds.
The earlier candidate and its independent NEEDS_REVISION review are preserved.

Historical interpreter and offline compilation records remain historical
observations. They do not assert universal compilation, device capacity or GPU
instruction conformance. Source migration replays Lean proofs and source checks;
it does not run a new GPU or numerical experiment. Current build receipts are
recorded separately under `bench/review-checkpoints/`.
