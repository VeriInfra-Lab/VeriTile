# Source and proof scope

The complete original and candidate writer quotations are in `Kernels.lean`.
Both retain the exact finite-classifier helper, both constexpr branches, the
grid-stride loop, unused cdiv and task parameter, all eleven source parameters,
the two input/output element casts, and every original boundary check. The
candidate changes only `j * num_ctas` to `num_ctas * j` in the grid tile ID.

The mathematical contract uses fp32 or fp64 bit patterns, Bool output,
natural strides and a rank-one layout with both original order arguments
explicitly proved zero. Input and output regions are distinct; output stride
is positive. Input stride may be zero. Every shape, iteration count, pid and
program count is allowed by the per-program execution theorem, including
zero sizes and zero iterations. Initial registers and the unspecified-value
oracle are arbitrary. An input realization premise preserves dtype tags and
binds the actual source channel to arbitrary input bits.

Written indices are exactly the selected source tiles intersected with
`x < size`. Their output cells contain Bool exponent-field classification of
the initial input pattern: exponent different from 255 for fp32 or 2047 for
fp64. Both signs of zero, subnormal and normal values are finite; both signs
of infinity and every NaN payload are nonfinite. Every other tagged memory
cell retains its initial value.

`Spec.Real` is used as a wrapper for this mathematical typed-bit proposition.
It does not erase or reinterpret `isFinite` as a real-valued predicate.
`ComputeExpr.isFinite.toAlgorithm?` rejects Real projection. The ordinary
FP comparison has an empty numerical-assumption list, applies to arbitrary
algebras and states, and is accompanied by actual successful nonempty runs
of both complete public programs.

The typed flat-allocation bridge proves execution of the actual relocated
complete sources, including every body statement and grid iteration. It
requires disjoint allocated regions, active-address extent bounds and explicit
agreement of the flat and regional unspecified-value oracles. It preserves
the independent bit formula and full tagged frame, including unallocated
background and foreign regions. Regional oracles remain arbitrary in the
regional contract; the additional flat compatibility condition is explicit,
and padding is not replaced by zero. Host grid coverage, wrapper signed strides and caller-selected output
types, bf16 widening, fixed-width addressing and native classifier refinement
are not covered by these entrypoints.

The dedicated strict source-pair adapter checks complete ASTs, aliases and
import chains with a single allowed candidate AST change. The ordinary global
source audit also passes, with explicit finite-helper dispatch metadata and
strict checks for all imports, helper bindings, complete writer statements
and selected fp32/fp64/bf16 storage. No source exemption is used.
The original native receipt records 48 compilation failures at
`out0_bptr.type.element_ty` on Triton 3.8.0/sm90. The candidate has not been
natively compiled. No GPU execution or independent comparator acceptance is
asserted. Public proof validation and overall source acceptance remain
separate gates.
