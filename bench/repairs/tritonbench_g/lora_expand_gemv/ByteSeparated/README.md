# Separate byte-interval candidate proofs

The original selected program is `../lora_expand_gemv_byte_separated.py`.
`Kernels.lean` contains its complete ordinary typed quotation for FP32 X/Y and
FP16 W, including both ADD branches, and retains the existing production pin.
`Pair.original_source` proves it equals the complete historical source by `rfl`.
The older byte-separation checker still verifies that the historical repair
changed only the wrapper's separation-check block.

`Contract` retains every original nonseparation check. `Guard` proves that
the actual nonzero-length/strict-endpoint overlap predicate rejects exactly
the complement of half-open byte-interval disjointness. Storage ranges describe
the complete `untyped_storage().data_ptr()/nbytes()` observations.

`Checked.valid` derives the old kernel geometry/metadata/access conditions
from the candidate checks. `TensorBridge.checked_real` then reuses the complete
typed-source and finite-sum proofs for every permutation of the real grid:
all outputs, sentinel rows, and all other symbolic cells are covered. K remains
1..4096; K tails, both metadata widths, both add branches, empty grids and
singleton 4D weights retain their original meaning.

`Binding` binds X and Y to actual `VeriTile.Torch.FP32.Tensor` objects and
`TensorLaws`. Metadata, object identity, storage identity, offsets, bases and
capacity agree with the API observations. A physical FP32 storage may have
unused trailing bytes: its element capacity is floor(nbytes/4), while the
guard and frames retain the full byte interval. Nonempty view bounds and
logical reads are derived from the ordinary core Tensor laws. Empty views do
not require an unused data pointer to lie inside nonempty storage. W and signed
indices use their ordinary two-byte/four-byte/eight-byte contiguous windows.
Neither binding contains an identity-to-disjointness law or a kernel-result
premise.

`Native` interprets the emitted stores from final symbolic source memory;
it does not manufacture stores from the independent result formula. Physical
leaves read the supplied pointer bytes. The independent reference uses the
retained FP16 product/reduction/conversion graph. `Scheduling` proves complete
source success and the same memory for every grid permutation. `Physical`
derives the whole original X/W/index backing frame from the checked byte
intervals and the derived store footprint; it also proves empty execution and
arbitrary partial-failure read frames. `Access` retains all formed signed-int32
source bounds and derives active 64-bit pointer/address and metadata spans.

`TensorBridge.checked_caller_bytes` combines successful source execution, every
active output byte, the complete original read-backing frames and all bytes
outside active output stores. Its compiler binding retains the historical
shape-only reduction/fusion choices and six local scalar instruction families
(narrow, widen, half mul/add/FMA, FP32 add), plus positive-zero conversions.
No split/fused equality or finite-value restriction is introduced. Actual
GPU scalar/compiler conformance remains this explicit existing boundary.
The Real theorem is a separate exact-arithmetic interpretation.

Python validation uses this candidate's unchanged kernel AST, real Torch
storage and the established CPU CUDA descriptor/context adapter. It retains
the 25 complete calls, 38 rejection classes and partial-write frame, and adds
11 interval fixtures: full/partial DLPack aliases, INT32/64 metadata aliases,
same-backing disjoint views, adjacent imported backings, shared zero-byte
storage, empty views of nonempty storage, FP32 storage tails, and overlap only
in a backing tail. This is CPU source/control evidence, not GPU arithmetic.

No old canonical file, defect packet, accepted theorem, core module, Lake
configuration or ledger selection is changed. Source selection and final
acceptance belong to the integration review.

`FPEquiv.grid_equiv` now compares two complete actual programs as
`original ≡[[]] unitSplit`. The directly related candidate is
`../lora_expand_gemv_unit_split.py` and the full ordinary quotation
`CandidateKernels.unitSplitSource` (declaration `ByteSeparated.unitSplitSource`).
It specializes `split_n_length = tl.cdiv(N, SPLIT_N)` to `split_n_length = 1`.
The selected wrapper already passes `SPLIT_N=N`, and returns before a zero-size
launch. This removes a uniform integer division and exposes unit split offsets;
no performance improvement is claimed without measurements. Grid dimensions,
program IDs and `num_programs` are unchanged.

`Proofs/Pair.lean` proves the exact integer site equality, then the unchanged
branches/loops/suffix, every actual invocation and the complete original grid
schedule. Nonempty-CTA bounds imply `N>0`; empty B or N remains admitted without
an added positivity premise. Source safety, success, every output and both
complete exterior frames are part of the shared exact-cell equivalence.
There is no floating rewrite and no experimental numerical rule instance.
The complete original graph-label input domain and arbitrary native bit
interpretations remain unchanged; no finite-value restriction is added.

`Misc/UnitSplitBytes.lean` applies the pair to the retained native byte observer.
`sourceStores_eq` proves equality of the complete raw byte states, evaluating
actual candidate final source memory with the same scalar/tree/fusion choices.
Thus the original sentinel, output, complete read-backing and exterior-byte
results transfer with their original explicit bindings. This adds no compiler,
hardware, Tensor or host validation requirement. The original byte interval
admission, storage tails, both signed metadata widths and all special words
are retained. `../check_unit_split_source.py` restores the one assignment and
kernel name and checks equality of the complete Python AST, including every
wrapper check, scalar binding, grid, branch, loop and store.

`Correct.kernel_real` and the ordinary theorem `FPEquiv.kernel_native` in
`Misc/NativeCorrect.lean` use the shared `ProgramIO.ObservesOnCells` judgment. `IO.lean` lists the actual source schedule,
dense signed index channel I[B], strided FP32 X[B,K], FP16 W[L,N,K], and in-place
FP32 Y[B,N]; no scratch is excluded from the frame. Every permutation of the
complete B×N grid is retained. Access safety follows the actual sentinel guard,
even/masked X branch, masked W loads, optional Y loads and all stores.

The original graph input model is explicit as `LoraIO.domain`: channel payloads
are the address labels defined by `labeledInputs`, encoded by identity. `data`
and `bits` interpret those same labels as mathematical values or native words;
neither chooses source instructions or reduction plans. Exact tagged input
cells preserve the original Y word even when index -1 skips the entire body.
The independent Real formula and native conversion/product/reduction/add graph
remain separate observations; there is no native=Real or new optimization law.

`Proofs/IOCorrect.lean` composes actual source safety, successful grid execution,
all outputs and the complete exterior frame. `KernelRealResults` and
`KernelNativeResults` retain the old raw endpoints as ordinary helper theorems.
`Misc/TensorBridge.lean` retains all three original Tensor statements, compiler
binding and physical byte/backing observations, using the same source and
finite-grid execution foundations. The unit-split candidate is a separate file; the selected byte-separated
original and its source pin remain unchanged.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Existing optional declaration namespaces, theorem statements and proof bodies
are retained. The required entries do not import `Misc/` or re-export optional
native conclusions. The new byte application is optional and introduces no
additional acceptance obligation.
