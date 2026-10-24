# Rotary transform with a checked static tile

This separate candidate adds one source-formation guard after the original empty/zero-rotary return. For `R = 2*cos.shape[1]`, `BK = next_power_of_2(max(32,R))`, it requires `BM*BK <= 2^20` for interleaved layout and `BM*(BK//2) <= 2^20` for split layout. The original `BM` rule remains: 4 for interleaved, 8 for split `R <= 64`, otherwise 4. The always-formed `BK`, `BK/2`, and `BM` vectors also fit under this check.

This narrows the launch domain. It does not repair the historical source over its full domain. All 86 original source, proof, compiler-control, and defect-evidence files remain unchanged. Interleaved `R=262144` and split `R=524288` are admitted; the old witnesses `R=262146` and `R=524290` are rejected. Original empty and zero-rotary calls return before the new guard.

The Python JIT body, wrapper preparation, dtypes, arithmetic order, and other checks are unchanged. The entire JIT body AST is also identical to `rotary_transform_ops`; only that sibling's full quotation and 14 necessary execution bridges are adapted here. This candidate uses rotary-transform's original unbounded block-width definition and split threshold 64, not rotary-ops' width-256 contract or threshold 128. Its two unused runtime arguments, `nheads` and `CACHE_KEY_SEQLEN`, remain in the actual Python binder.

`Kernels.lean` contains the complete original and candidate quotations for all four metadata widths; `KernelSource.lean` is a compatibility import. These quotations include both layouts, packed/fixed metadata, scalar/tensor offsets, conjugation, early return, loads and stores. `TypedExecution` and `TypedGrid` prove actual shared structural execution and complete-grid composition; complete launch permutations have the same values and frame. `SourceTiles.quotation_shapes_bounded` traverses the quotation and specializes only literal constexpr branches.

`KernelEntry` decodes signed int32/int64 cumulative and offset metadata from original caller bytes, preserving cumulative storage and the existing strided-offset preparation. Its payload values come from original input and strided coefficient views. Suffix readiness is proved from the original out-of-place copy and is reflexive inplace. No output formula is an entry premise.

The complete `Misc/TensorBridge.lean` applications reach actual Tensor metadata and storage. The Real theorem permits arbitrary real values read through the caller Tensor pointers, with integer metadata supplied separately. The native theorem preserves all raw FP32 words and derives the independent coordinate target from initial Tensor reads. Both execute the full quotation and serialize only that final state's output values using existing local memory primitives; they prove returned shape, dtype, device, object identity, restored caller device, all logical output coordinates, and exterior frame. Native output words additionally determine all four little-endian output bytes. Serialization is an output-observation refinement, not a claim that the host performs an extra output copy.

`Misc/TensorBridge.lean` retains these complete Tensor applications and the additional byte observation, signed-metadata, checked tile/index, and pointer corollaries as ordinary theorems. `Kernels.lean` is source-only. `Proofs/SourceExports.lean` retains the former shared source-binding, grid execution, and grid-order compatibility exports. Neither primary entry imports `Misc/TensorBridge.lean`.

The native model selects the original default FMA graph: low `fma(co,a,neg(mul(signedSin,b)))`; split high `fma(signedSin,a,mul(co,b))`; interleaved high `fma(co,b,mul(signedSin,a))`; conjugation uses `sub0`. The four local instruction callbacks and compiler realization of this pattern are explicit model boundaries. There is no universal native-to-Real equality, whole-output premise, numerical experiment, or deferred scalar obligation.

Ordinary Tensor observations, existing allocation/identity behavior, local memory primitives, and the FP32 word/byte codec remain runtime representation boundaries. The existing finite index/pointer checks are reused. Static source formation does not certify compiler resource availability, allocator behavior, or the separately postponed grid-product compatibility work.

Validation uses actual Torch CPU objects and the actual wrapper/JIT argument binder with only CUDA classification/context and the launcher captured. It covers all flags, all four int32/int64 metadata-width pairs, strided preparation, coefficient fallback, empty returns, block selection, and both exact tile boundaries. Four saved width-514 SM80/SM90 compiler controls are reused. No GPU, kernel interpreter, new compiler, or numerical run is performed. Evidence and fresh candidate module receipts are in `.veritile-cache/kernel-evidence/rotary_transform_tile_checked/delivery.json`; independent review and root source selection remain separate.

`Correct.lean` and the auxiliary `Misc/NativeCorrect.lean` expose shared `ReadProgramIO` judgments. Their visible channels contain the original FP32 input, the cosine/sine tables, the selected signed cumulative/offset metadata, and the pre-existing output suffix. `PublicIO.domain` states logical metadata agreement and suffix equality; it contains no execution or completed-output premise. The schedule selects the complete original grid and both metadata widths. Writable output masks cover exactly the rotary prefix, while result-read masks cover every logical column. Untouched suffix cells keep their original tags and use the same typed-read convention as the former raw endpoint.

`Correct.lean` exposes `rotation`, the independent rotation formula; `Misc/NativeCorrect.lean` exposes `nativeRotation`, the existing raw-word instruction graph, and retains `rotary_transform_native` as an ordinary theorem with its original statement and proof. The native input domain still admits arbitrary `Native.Node` values and every FP32 word; observation embeds their exact `.value` in `Native.atom`. No native-to-Real equality or new numerical law is asserted. The public statements contain no initial/final state, raw runner, or per-address execution result.

`Proofs/SourceSafety.lean`, `StoreSafety.lean` and `CompleteSafety.lean` follow the actual quotation's prefixes, branch bodies and existing successor witnesses. `ScheduleSafety.lean` carries these bounds and signed metadata across the complete grid, including padded coefficient reads and masked data tails. `PublicIO.lean`, `KernelValues.lean` and `IOResults.lean` connect the finite inputs, independent formulas, old complete-grid results and exact frame to the shared interface. `KernelEndpoints.real/native`, `every_token`, and `Native.source_graph_lowering` retain the original raw results and supplemental facts. The complete `TensorBridge.rotary_transform_correct` and `TensorBridge.rotary_transform_fp32` applications remain unchanged and use the same execution foundation.

`FPEquiv.source_candidate_equivalence` now compares two actual complete
native FP32 grids with `≡[[]]`. The candidate reuses `rk` when computing
`rk_repeat` in the interleaved branch, eliminating a repeated lane-vector
generation. No measured performance claim is made. Both entire sources are
ordinary readable Triton quotations, with no whole-source macro, AST generator
or proof import in `Kernels.lean`. Every original load, mask, cast, conjugation,
FMA distinction, store and branch remains intact.

`Proofs/IndexCSE` proves exact complete-state execution and trace-safety equality
for every scalar backend, all original flags and both signed metadata widths.
`Proofs/ProgramPair` lifts it across the actual complete grid, retaining the
original logical metadata/suffix domain, every native input payload, all result
reads (including untouched suffix cells), both safe successful executions and
both exact exterior frames. No floating numerical rewrite law or experimental
obligation is used. A fresh historical-source check proves complete
`ComputeKernel` equality of all four expanded original width profiles to the
prior macro source. Original Python, contract, Real target, native formula,
checked domain, Tensor statements and numerical/hardware boundaries remain
unchanged.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
original source/native theorem scopes are preserved. The required entries do not re-export
optional native conclusions. The new main source pair is proved independently of these optional applications.
