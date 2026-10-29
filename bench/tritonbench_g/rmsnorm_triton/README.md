# rmsnorm_triton

Migration is in progress. The `Correct` and `FPEquiv` public targets compile,
including the shared integer-power extension, complete-source FP equivalence,
exact-real safety and full output/frame contracts, nonempty numeric executions,
finite-allocation witness and exhaustive floating precision checks. Historical
contracts remain available after the split. Dtype/address scope gaps and final
review still prevent a whole-wrapper completion claim.

The source has two complete tiled loops, two program IDs and seven strides.
Both input passes explicitly convert X to fp32. Variance calls
`tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)`, then fp32 sum/division and
`tl.math.rsqrt`; pass two multiplies x32 by rstd32 and the uncast weight.
Only fp64 weight widens the final floating product. Weight loads have no
`other`; input loads explicitly use zero. The candidate commutes only the
first natural row-address product.

The prepared floating source family has 216 independent input/weight/output
storage combinations; the Python wrapper uses output storage equal to input.
This is not exhaustive dtype coverage: the wrapper does not reject discrete
inputs or outputs. Declared signed input followed by `x.to(tl.float32)` fails
in the current DSL (`tl.cast: signed integer casts are not modeled yet`).
Mixed/discrete output conversions inherit the recorded shared conversion
obligations from `var_len_copy/conversion_gaps.md`. Natural address geometry
also leaves negative signed strides and fixed-width overflow outside scope.

Pinned Triton 3.5.1 dispatches `(fp32,int32)` power to `__nv_powif`, distinct
from `__nv_powf` and multiplication:
[official libdevice source](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/third_party/nvidia/language/cuda/libdevice.py).
The DSL now retains `Op.libdevicePowi`, its integer exponent, and an independent
optional backend callback. Backend support is mandatory for successful
execution; multiplication and floating-exponent power are not fallbacks.
The identity `x^2=x*x` is used only in exact-real mathematical proofs, never
as an assumed FP law. The shared regression fixture checks dispatch, precision,
unsupported-backend failure, exponent bounds and the fp32/fp64 base restriction.

`FPEquiv.source_equivalence` compiles for all 216 modeled floating storage
profiles with an empty domain and no FP assumptions. It preserves complete
state and failure behavior for arbitrary backends, including native powi.
`Proofs.Execution` proves successful one-iteration and two-iteration runs for both
variants and mixed fp64/fp16 input/weight profiles, with both program IDs equal
to one, independent padded strides, output value two, and masked sentinels.
Its numerical algebra uses real square root, integer power and reduction;
casts are identities, so this witness does not establish IEEE conformance.
`Proofs.Precision.computation_profiles` checks the exact powi/input-cast,
accumulation, reduction/division, rsqrt, weight promotion and destination-store
ASTs for all 216 profiles and both source variants.

`Correct.typed_real_correct` proves access safety, successful flattened
exact-real execution, every active independent RMS output, and preservation
of every other typed memory cell for all modeled profiles and both variants.
It allows arbitrary row length and positive tile width, natural strides and
both program IDs, with finite access bounds, distinct output/input regions
and a positive output column stride. `Proofs.Witness.execution` instantiates
that contract in finite disjoint allocations with nonzero program IDs, two
iterations, nonunit column strides, output value two and typed sentinels.
Both witness modules use direct kernel calls with tile width two; the Python
wrapper instead fixes tile width 1024 and output storage equal to input.

All 72 historical declaration headers and adjacent docstrings are preserved.
The historical kernel remains an explicit exact-real square model. Public
theorems pass the standard-base axiom audit, and FP equivalence reports no
assumptions. Comparator validation is unavailable; independent final review
and acceptance registration remain integration work.
