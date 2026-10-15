# isfinite_kernel

- Source file: `isfinite_kernel.py` (upstream `data/TritonBench_G_v1/isfinite_kernel.py`)
- Corpus: TritonBench-G v1
- Status: public typed-bit correctness, FP entries, complete typed-flat contracts and ordinary source audit pass locally.
- The complete Python file is now imported verbatim from upstream `603e28a`,
  SHA-256 `9661c6a64cb20de85a75775c6daec1dc7e16397d1aa48b83296e9f37039a0ba6`.
  Both `isfinite_func` and `isfinite_func_kernel_rank_1` remain intact.

The [bit-classification checkpoint](../../review-checkpoints/2026-10-13/isfinite-bit-classification/receipt.json)
passes 20 local axiom gates for bf16/fp32 decoded values and fp32/fp64 exponent
classification. Cases include both signs of zero, subnormal and normal finite
values, infinities, quiet NaNs and signaling NaNs. The fp64 rational decoder
alone is insufficient: it also rejects zero and subnormal encodings.

`Proofs/BitClassification.lean` contains the namespaced leaf. The shared AST
now has a compute-only classifier with an explicit fp32/fp64 tag and an optional
typed backend callback. Missing support fails, and mathematical real projection
explicitly rejects the operation. Exact source-helper binding, both pointer
element casts, branch dtype joins and `tl.advance` provenance are integrated.
The [17 frontend regressions](../../review-checkpoints/2026-10-13/combined-frontend-compatibility-root/receipt.json)
and [aggregate rebuild](../../review-checkpoints/2026-10-13/combined-frontend-aggregate-root/receipt.json)
pass; the latter contains 4,780 Lake jobs and 1,425 unchanged recursive inputs.
The complete [Kernels](Kernels.lean), [Correct](Correct.lean) and
[FPEquiv](FPEquiv.lean) entries now pass an
[isolated canonical rebuild](../../review-checkpoints/2026-10-13/isfinite-public-stage/receipt.json).
The [ordinary source audit](../../review-checkpoints/2026-10-14/isfinite-ordinary-source-root/receipt.json)
now passes all 178 source views and 101 Python regressions, using an explicit
strict finite-helper adapter for fp32/fp64/bf16 comparison views. This source
comparison does not extend the fp32/fp64 proof coverage to bf16.
Source discovery now pairs all 178 imported Python sources with Lean entries.

The [complete source execution checkpoint](../../review-checkpoints/2026-10-13/isfinite-full-source-bit-witnesses/receipt.json)
passes fp32/fp64 witnesses through both original branches: a sixteen-lane
single tile and two eight-lane grid iterations over fourteen values. These
cover both signs of zero, subnormal, finite, infinity, quiet NaN and signaling
NaN. All seventeen observed sentinel cells in stride holes, the masked tail
and a foreign region are preserved. The omitted-padding oracle returns
signaling NaN. The exact helper dispatch, both casts and every source statement
remain present. These concrete typed interpreter runs use explicit bit
backends.

The [parameterized source checkpoint](../../review-checkpoints/2026-10-13/retention-typed-isfinite-parametric-stage/receipt.json)
also proves actual complete fp32/fp64 executions for arbitrary input bit
patterns, sizes, block widths, natural strides, launch IDs and loop counts
in the selected rank-one Bool-output profile. The value contract requires
distinct input/output regions, a positive output stride and actual typed input
cells. It classifies each covered input with the independent exponent-bit
predicate and preserves every unwritten tagged cell. Zero sizes and empty
iterations are included; no finiteness premise is imposed on input bits.

The [source/candidate comparison](../../review-checkpoints/2026-10-13/isfinite-source-candidate-pair/receipt.json)
checks a complete candidate that only commutes the grid tile-index product,
and rejects fifteen mutations. The [candidate proof checkpoint](../../review-checkpoints/2026-10-13/retention-public-isfinite-candidate-stage/receipt.json)
proves whole typed state/failure equality for arbitrary algebras and initial
states. Both fp32/fp64 candidates also have actual successful runs and the
same independent bit-value/frame contract as their original source.

The [native compilation checkpoint](../../review-checkpoints/2026-10-13/isfinite-native-rejection/native-receipt.json)
retains 48 rejections from the unchanged helper/writer on Triton 3.8.0 / sm90.
The original `out0_bptr.type.element_ty` expression fails because the block
pointer aggregate type has no `element_ty` attribute. Both original import
chains were retained and their selected aliases match primary CUDA libdevice.
This establishes an observed compatibility gap; no GPU execution or native
refinement is claimed.

Structurally this kernel is the same FlagGems pointwise-codegen template as
`relu_strided_buffer` (a `one_tile_per_cta` constexpr branch, `tl.make_block_ptr`
with `boundary_check`, an inlined `@triton.jit` helper, and a grid-stride else
branch over `tl.num_programs(0)`). The complete quotation and successful
nonempty writer examples are checked through the public entries in isolation;
the [root aggregate replay](../../review-checkpoints/2026-10-14/public-barrier-aggregate-root/receipt.json) passes.
The [public root typed finite-allocation rebuild](../../review-checkpoints/2026-10-14/isfinite-complete-flat-root/receipt.json)
passes 3,431 jobs with 144 unchanged inputs. The new bridge
proves actual complete flat executions of both original/candidate fp32/fp64
sources, a common final state, independent exponent-bit outputs and the full
tagged memory frame. Active cells require allocation bounds and disjointness.
Unspecified loads require explicit agreement between regional and flat oracles;
nonzero and NaN padding are retained. This is interpreter execution, without
a native memory or libdevice refinement claim. The [concrete flat witnesses](../../review-checkpoints/2026-10-14/isfinite-typed-flat-witness-stage/receipt.json)
discharge the allocation and padding conditions for both branches over fourteen
special-value patterns, with nonzero pointer bases and stride-two output.

The wrapper permits caller-selected output types and signed strided buffers;
those domains must be explicit. A well-typed payload invariant is required for
a backend mixing different floating formats. Host launch, empty-input wrapper
behavior, fixed-width address arithmetic, native libdevice refinement and
independent comparator replay remain outside the current primitive evidence.
