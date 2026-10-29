# rotary_transform_ops kernel verification

The original Python kernel and guards are unchanged; the new candidate is described below. The domain remains FP32
CUDA contiguous payloads, canonical strides, zero storage offsets, no lazy
flags, `0 < D <= 256`, and rotary width at most D. CU retains its original
signed int32/int64 storage and unit stride; offsets retain their original
signed int32/int64 storage. Fixed/packed sequences, scalar/tensor offsets,
split/interleaved pairs, conjugation, and inplace/out-of-place execution are
covered. Empty sequences, masked lanes, coefficient fallback, empty inputs,
and zero rotary width retain the existing behavior. No clone, conversion,
packing, allocator change, guard narrowing, or runtime compatibility condition
is introduced.

`KernelSource.lean` contains four complete ordinary Triton blocks for the JIT body, including signed metadata
loads, early return, both arithmetic branches, masks and stores. Its four
storage-width declarations share the exact signed-value interpreter; the
original word widths are separately bound to caller bytes. FP32 compute tags
use the interpreter's erased `.real` carrier. That carrier is opaque and does
not assert real-number arithmetic.

`TypedSetup`, `TypedLoads`, `TypedStores`, and `TypedControl` use slices of this
same quotation. Their execution witnesses compose into the complete source
execution. `ProgramModel` is an independent matrix-load/store normal form;
`TypedResult` connects the executed quotation to it. Geometry, masked scatter,
coverage, noninterference and permutation results connect that memory result
to the independent coordinate formulas. No source-success, completed-result,
whole-output-equality, or numerical-test premise is used.

`Correct.lean` exposes `rotary_transform_ops_correctness`: the shared
`ReadProgramIO.ImplementsOn` judgment, wrapped in `Spec.Real`, implements the
independent coordinate formula. `Misc/NativeCorrect.lean` retains the ordinary
auxiliary theorem `rotary_transform_ops_native_observation`, with its original
full name, statement and proof, using the same shared judgment and the exact
native word formula. `FPEquiv.rotary_grid_equiv` compares the complete original
grid with an exact adjacent-partner index candidate under the original full IO. `IO.lean` declares the six finite input channels
(data, cosine, sine, cumulative lengths, offsets, preinitialized suffix), the
complete output view, source quotation, complete grid, and writable prefix.
Both original signed metadata widths remain explicit parameters.

`Interface.Ready` states only the original input metadata and suffix-copy
conditions. For in-place execution suffix equality follows from simultaneous
input/output alias binding. Result reads cover every logical column; writable
windows cover only the rotary prefix, preserving the original exact raw-cell
frame. Empty input or zero rotary width skips the grid. Actual quotation
access safety includes coefficient reads in padded rows and is proved through
every intermediate grid state. A declared memory window alone is not used as
an access-safety proof.

`Misc/TensorBridge.lean` retains both complete actual-Tensor applications unchanged.
The former raw-state endpoints, arbitrary grid-permutation result and
coverage/index facts retain their names in `Proofs/KernelCompatibility.lean`;
these compose the existing source proofs. Previous wrapper statements remain
under `Proofs/HistoricalPublicCorrect.lean` and
`Proofs/HistoricalPublicFPEquiv.lean` in namespace `HistoricalPublic`.

`KernelTensor` reads FP32 words and original signed metadata at actual
`Torch.TensorMetadata.dataPtr` values. The unchanged accepted check derives
input metadata bindings and finite index bounds. Tensor read-law and formed
pointer lemmas expose the relation to logical input values, signed32/signed64
indices, and uint64 pointer arithmetic. The initial output view includes the
already existing suffix-copy state.

The native reference retains all FP32 encodings, including exceptional words.
Its selected default graph is `fma(cos,a,neg(mul(signedSin,b)))` for low output,
`fma(signedSin,a,mul(cos,b))` for split high output, and
`fma(cos,b,mul(signedSin,a))` for interleaved high output. Conjugation uses
`sub.f32(+0,sin)`. The four raw-word primitive callbacks and this fixed compiler
pattern are explicit model boundaries. There is no deferred equality between
two supplied instruction implementations in the public endpoint. Fallback
`cos=1, sin=0` still performs the arithmetic; raw-word identity is not claimed.

The saved compiler evidence contains 16 actual wrapper/JIT-binder/default
Triton 3.7.1 SM80 profiles, covering all combinations of the four compile-time flags
and original metadata widths, including mixed widths. Their 16 TTIR and 16 PTX
artifacts are pinned in `kernel-evidence/profiles.json`. This evidence is
specific to those compiler profiles and the selected primitive pattern; it is
not a universal compiler-correctness or GPU-execution claim. The carried CPU
storage replay passed 76 complete source/host paths and 16 prewrite
rejections. Its provenance and limits are recorded in
`kernel-evidence/regression-provenance.json`. CUDA execution and native IEEE
conformance were not tested. The mathematical theorem is a separate real-number
interpretation and does not assert floating-point error bounds.

The native auxiliary endpoint in `Misc/NativeCorrect.lean` is an observation of the existing primitive/compiler model,
not equivalence between two kernels. `observeNative` preserves the exact raw
word while discarding only internal product provenance; every existing input
Node, and therefore every FP32 word, remains admitted. There are no floating
transformation assumptions to report with `#print_fp_assumptions`, which does
not apply to this shared IO judgment. No numerical experiment, independent
native replay, or GPU test was added by this format revision.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean), [KernelAudit.lean](Misc/KernelAudit.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.


The new complete source `XorSource.lean` replaces the adjacent-branch index
`rk + ((rk + 1) % 2) * 2 - 1` with `rk ^ 1`. Both compute exactly the partner
sequence `1,0,3,2,...`; source lanes range from zero through 255. This replaces
multiple integer operations with one xor; no measured performance improvement
is claimed. Every FP32 operation, source cast, FMA/lowering distinction, load,
store, mask, early return and signed metadata case remains unchanged. The
Python entry is `apply_rotary_xor`, using `rotary_kernel_xor`, with exactly the
original validation, suffix copy, empty handling, inplace choice and launch.

The main empty-rule-set comparison retains the original `Interface.Ready`
metadata/suffix invariant, original bounds/input binding and all raw FP32 words.
Its `ReadProgramIO.EquivalentOn` judgment concludes both actual safety traces,
both successful complete grids, all result reads including untouched suffix,
and both complete exterior frames. The xor identity is exact integer
arithmetic, with no numerical relaxation, experimental rule or new deferral.
The original four metadata source macros have also been materialized as
ordinary readable blocks with the same literal bodies and typed ASTs. This
source presentation change retains the existing independent Real and native
proofs. Main entries and their proof helpers import no optional `Misc` result.
