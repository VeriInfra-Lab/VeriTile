# Scalar-base power on signed FP32 views

`Kernels.lean` contains the unchanged `pow_func_scalar_tensor` helper and complete
original/existing index-commuted writer bodies in `triton { ... }`. Both branches,
block-pointer creation, boundary checks, casts and the grid-stride loop remain
visible. `val0` remains a runtime FP32 scalar and Python retains its original
`do_not_specialize` decorator. Calling metadata supplies pointer storage,
signed pointer origins/strides, scalar values and constexpr settings outside
the pasted body. `jit-correspondence.json` checks all three source mappings.

The caller explicitly selects the original module's primary CUDA libdevice
`_pow` import. The checker records the two ImportError fallback paths as source
provenance; it does not assert that other implementations are the same
intrinsic. A different import binding requires its own supported target.

The existing candidate commutes only `tile_id0 * tile_size0` in both branches.
No numerical power law is used. The public FP result compares complete original
and candidate execution, including every typed memory cell, for arbitrary
interpretations of the unchanged floating operations.

The original input `.to(in0_ptr.type.element_ty)` is retained. The historical
flattened source omitted this same-format cast. `Proofs.Source.sourceCastBridge`
composes that event with the historical operand conversion solely to reuse its
memory/frame proof; it never assumes an arbitrary cast is identity. Actual
kernels execute the original interpretation. `Proofs.SourceResults` separately
checks the Real erasure and existing FP32 word interpreter. Historical source
is archived in `Proofs/HistoricalBody.lean`; `SourcePreservation` rejects an
AST-identity claim and checks complete source statement counts.

`Correct.lean` retains the independent formula `Real.rpow base (xs i)` for every
logical output over the complete grid, including input preservation, output
exterior memory frame and active access bounds. Its mathematical domain allows
positive bases, zero with nonnegative exponents, and negative bases with
integral exponents. `FPEquiv.lean` keeps the existing integer-index comparison
and introduces no finite-value or numerical-transformation premise.

`IO.lean` retains signed views, typed FP32 input/output channels, logical masks,
capacities, active footprint separation and the capped CTA schedule. The checked
Python contract remains unchanged: FP32 rank-one CUDA views, a Python float
scalar, `0 < n < 2**31`, aligned valid signed-stride backing and disjoint active
input/output cells. Inactive backing may overlap. No new Python guard or
restriction is introduced.

Actual block-pointer access safety, full tile/grid execution and both original
branches are proved from the pasted bodies. The complete signed pointer sum is
formed before natural cell-index conversion. Arbitrary masked-load observations
remain permitted. Optional native, Tensor and historical host results remain
under `Misc/` and `Proofs/`; primary entries do not import optional readers.

Migration validation is recorded in the matching review checkpoint after the
final source replay. Source-only checks do not claim a GPU run or numerical
power experiment. The original scalar ABI and libdevice interpretation remain
explicit model boundaries.
