# Checked delta-forward repair

**Current status: source defect pending repair.** The checked source admits
`K=1` and `BT=1` specializations whose dot reduction dimension is below the
NVIDIA Triton frontend minimum. Actual wrapper/JIT negatives and a legal
same-configuration control are recorded in [SOURCE_DEFECTS.md](SOURCE_DEFECTS.md).
Existing conditional Lean results are retained; this is not a completed
full-domain kernel revision. The canonical Python and its contract are
unchanged.

`chunk_fwd_h_fn_repaired(k, w, u, BT, initial_state=None, final_state=None)`
returns the actual fresh `(history, corrected_values)` tensor objects. The
original benchmark Python is unchanged.

The checked API accepts ordinary contiguous FP32 CUDA K/W `[B,H,T,K]` and U
`[B,H,T,V]` tensors on one device, with `1 <= K <= 256`, a positive builtin
integer BT, and power-of-two `BC=min(BT,BV)`. BK and BV retain the original
selection. Optional states have shape `[B,H,K,V]`. Contiguous views with storage
offsets and harmless singleton strides are admitted, and read-only inputs may
overlap. Inputs are passed directly without cloning or layout conversion.
The optional final-state span must be disjoint from all nonempty read spans;
this executable guard permits the original direct final store.

The repair supplies zero padding to the four boundary-checked loads, passes
independent canonical W strides, checks CUDA y/z grid limits and conservative
formed-offset bounds below signed 2^31, and checks every original/fresh byte
pointer for FP32 alignment and room below signed 2^63. These bounds cover
masked lanes, source i32 arithmetic, descriptor i64 products, four-byte scaling,
and the complete four-byte word. The tensor owner's CUDA context restores the
caller's device on success or failure. B/H/V-zero grids return fresh empty
outputs without launch; T-zero calls with a nonempty state grid retain the
original seed-load/final-store source path. Allocation and compiler/device setup
failures remain ordinary explicit host failures. The concrete small-dot
frontend rejection documented above is a source-domain defect and cannot be
discharged as an ordinary setup failure.

The floating assignments, casts, dot options, stores, and all six original
autotune configurations are retained. Inner windows have width
`ceil(BT/BC)*BC` and stride BT; their overlap is preserved. Corrected rows use
the last visiting chunk `floor(t/BT)`. For one inner subblock, default lowering
uses the live state as the update dot's carry. With multiple subblocks the
final outer addition remains separate. The native adapter represents both
cases without a split-versus-fused equality assumption.

`Correct.returned` constructs the complete typed Real grid from original
physical input addresses and proves every history, corrected, and optional
final observation at the checked returned tensor byte addresses, against an
independent rounded-window recurrence. `FPEquiv.returned` connects the same
checked host, its exact allocator objects, complete typed source execution,
original raw input words, finite native pointers, every returned FP32 word,
and caller/exterior byte frames. Concrete words use identity bit leaves and
five fixed scalar semantic families: literals, binary operations, unary
operations, casts, and FMA. Dots expand to ordered scalar FMAs. There is no
whole-dot, whole-output, or whole-kernel numerical port. The Real interpretation
is separate from concrete FP32 payloads.
Primitive/ISA behavior, compiler correspondence and ordinary Tensor APIs are
model boundaries, not deferred numerical experiments.

`Contract.lean` is the canonical source-contract entry and imports
`Admission.lean`, which encodes the executable guards.
`Proofs/NativePointers.lean` derives
source coordinates, loop bounds, signed i32 bases and i64 descriptor arithmetic.
`Proofs/Storage.lean` binds original byte inputs and ordinary allocation metadata;
`Proofs/RealPhysical.lean` proves the independent Real observations.
`Proofs/CheckedHost.lean` records allocation/setup failures and restores the caller
context. Runtime ports provide ordinary allocator metadata and Unit setup
results only. `Proofs/ReturnedMemory.lean` publishes source writes through the exact
little-endian codec and sequential four-byte stores, then proves physical
readback and frames. The 24 existing helper modules remain unchanged because
chunk-linear-attention also imports them. Only `Proofs/NativeGraph.lean` was restored
from the remaining historical closure; the legacy finite-profile and abstract
word-memory endpoints are not imported.

Validation:

- `python check_min_dot.py`: two admitted small-reduction specializations
  rejected by the exact JIT frontend, plus one legal SM80 compilation through
  actual wrapper argument binding; see the defect note for scope and pins.
- The following checks are retained historical evidence; they have not been
  rerun or upgraded into full-domain acceptance by this defect diagnosis.

- `python check_contract.py`: 12 preserved floating assignments/store values,
  all six autotune configs, 60 metadata/alias rejections, and 22 exact CPU
  controls executing the actual wrapper and Triton interpreter. Controls cover
  rounded overlap, partial tiles, both state options, direct read aliases,
  singleton strides, empty grids, zero time, signed-zero seeds, and device
  restoration. Only CUDA admission/context are adapted for CPU execution.
- `python check_lowering.py`: Triton 3.7.1 offline SM80, four warps, K/BK=16,
  V=17/BV=64, T=81, BT=16 and 80. TTIR/PTX confirms NC=1 live carry,
  NC=2 separate outer add, zero-seeded correction, scalar FP32 FMA, and no MMA.
  This is narrow source-lowering evidence, not an exhaustive compiler proof.
- Both public Lean endpoints and their local dependencies are compiled with
  Lean 4.29.0 and audited for standard-base axioms and independent targets.
- No GPU execution was available.

Original Python SHA256:
`e440d0b109b4498b030bef6afa996423d0ae892d7518962d3a7c27b0a22de585`.
