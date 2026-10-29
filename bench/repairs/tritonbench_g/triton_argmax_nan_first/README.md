# First-NaN, first-index argmax

The current source quotations in `Kernels.lean` match the three Python
`@triton.jit` bodies in `triton_argmax_nan_first.py` statement by statement.
`CandidateKernels.lean` matches both changed bodies in `triton_argmax_shifted.py`.
The candidate replaces multiplication by 1024/128 with a left shift, retaining
the original formula for other block sizes. Python wrappers and admissions are
unchanged.

## Source correspondence

The quotations retain signed index casts, runtime scalar parameters, dtype
queries, constexpr declarations, NaN selection, both clamp operations,
reductions, rereads, masks, stores and statement order. `IO.lean` binds runtime
`M`, `N`, `K`, `mid_size` and `program_base` at each invocation. Block sizes,
index-width flags and pointer element types are compile-time parameters.

The shared frontend `VeriTile/Triton/DSL/SourceDType.lean` specializes pointer
properties and constexpr expressions and inserts the destination conversion
implicit in Python `tl.store`. It does not replace the runtime NaN/clamp sequence
with a conversion macro. Signed, unsigned and boolean scratch keep their
respective channels. The selected boolean byte ABI truncates to i8 before the
nonzero test. Unsupported store control flow and ambiguous pointer alias
rebinding are rejected.

Run from the repository root:

```bash
python3 bench/repairs/tritonbench_g/triton_argmax_nan_first/check_pair_source.py
python3 -m unittest bench.tests.test_jit_correspondence
```

The first command checks the complete Python candidate delta and all five
Python/Lean AST pairs listed in `jit-correspondence.json`. Only quotation
punctuation, declared constexpr antiquotes, and `tl.full` shape/keyword spelling
are normalized. Removing casts, NaN selection, clamps, changing masks/stores,
or turning runtime parameters into constexpr splices fails the check. This is
a source correspondence guard, not a proof of the Python/Triton compiler.

## Public proofs

Both public entries now execute the current quotations, through the canonical
`IO.lean` and `PairIO.lean` bindings. All execution and mathematical bridges are
in `Proofs/`.

| Entry | Coverage and connection |
| --- | --- |
| `Correct.global` | Complete first grid and second stage; independent first maximum over the logical Real vector; `JITRealGrid`, `JITRealPublicIO`, `JITReaderCorrect` |
| `Correct.dimension` | One dimension CTA with arbitrary BM/BN/base/pid and the entire N loop; independent first maximum for each active tensor row; `JITRealDimension`, `JITRealDimensionIO`, `JITReaderCorrect` |
| `FPEquiv.global_equiv` | Complete native global schedules for all 13 requested scratch types; `JITIntegerStage1`, `JITInteger`, `JITSafety`, `JITNativePublicIO`, `JITProgramPair` |
| `FPEquiv.dimension_equiv` | Complete native dimension grids and N loops, including empty outer products; `JITDimension`, `JITNativeDimensionIO`, `JITProgramPair` |

The full IO judgments include successful execution, memory safety, independent
output formulas and preservation of other memory. The pair proofs compare the
actual original and candidate programs; they do not assume their outputs or
successful execution as premises. Axiom checks cover all four public results.
The experimental FP rule set is empty: the candidate changes integer indexing,
with no floating numerical rewrite.

Global coverage, active CTAs, positive block sizes and the original region
separation conditions remain explicit. Native global IO also retains the
initialized INT64 gather tail. `BoundSource.Binding` supplies the existing local
floating codecs/rounding and integer scalar conversion laws. These boundaries
are unchanged; no whole-kernel equivalence assumption is added.

## Mathematical and model boundaries

Floating reductions select the first NaN, otherwise the least index of the
numerical maximum. Global execution first converts/stores each chunk winner to
the requested scratch type, then selects from those stored values and gathers
the original index. Consequently lossy scratch can change the result relative
to the original-input FP32 argmax. For example FP16 scratch on
`[1.0001] + [-9.0]*3 + [1.0002] + [-9.0]*11` selects 0, while the input FP32 argmax
is 4. The native target retains this two-stage behavior.

The Real theorem retains the existing explicit idealization: requested
integer/bool conversion is projected to the FP32 Real-value branch via
`JITIO.projectedKind`. It proves the independent Real first maximum of that
projection. It does **not** claim that a finite Real clamp is identity or that
all native scratch formats compute the original-input Real argmax. In the
native/FP proof, the actual integer NaN0/clamp/truncate sequence is executed and
proved instruction by instruction.

The signed index carrier uses the existing mathematical-Int, no-wrap index
model. Keeping `.to(tl.int64)` visible does not add a hardware overflow or CUDA
compiler refinement theorem. Floating primitive/codec semantics retain their
existing explicit model boundaries. No new GPU experiment is required or
claimed by this source repair.

`Proofs/ProjectedKernels.lean`, `ProjectedCandidateKernels.lean`,
`ProjectedIO.lean` and `ProjectedPairIO.lean` preserve the earlier projected
representations for mathematical and byte-memory helper proofs. They are not
the current public kernel definitions. Optional historical Tensor/byte results
in `Misc/TensorBridge.lean` still concern that projected representation and its
existing byte transport; they are not a new compiler/host refinement result.

## Verification and wider audit

The current reproducible checks, exact source pins, proof logs and limitations
are recorded in `bench/review-checkpoints/2026-10-24/jit-correspondence/`.
Required Lean sources are in Git; no `.veritile-cache` file supplies a proof
module. The official comparator is unavailable in this environment, so its
independent trust check is recorded as unavailable, not passed.

The corpus inventory screens all 183 selected tasks. It finds 45 other tasks
whose Python JIT contains explicit int64 casts absent from the selected direct
Lean quotation file. Those are source correspondence issues, not automatically
semantic counterexamples. The remaining 137 tasks have not received a complete
statement correspondence certificate from this screen. Their earlier Lean
proof results do not certify Python/Lean statement identity.
