# FP32 destination-row copy with a head-only mask

## Target, precision and contract

The independent formula is `Out[DestLoc[row], head, dim] = K_initial[row, head, dim]`,
with all other memory preserved. Payloads default to FP32 because the production
source does not specify dtype; destinations retain signed int32. There is no
floating arithmetic and every FP32 bit pattern is admitted.

The common contract is [kv1's direct-buffer contract](../destindex_copy_kv1/README.md):
contiguous strided CUDA tensors, rank-three FP32 payloads, rank-one int32 indices,
matching payload dimensions, enough rows, bounded unique destinations, one device,
no negative-view flag, signed-int32 extents, bounded tile size, natural pointer
alignment and output/read span separation. Metadata readback checks the indices.
For nonempty invocations the feature width must also be a power of two: the
original Python supplies `BLOCK_DMODEL=head_dim` directly to `tl.arange` and the
kernel has no feature mask. Empty work is admitted and launches no kernel.
There is no payload copy, layout conversion, padding or compatibility expansion.

`Contract.lean` records this extra width condition openly; the Python assertions
and `TensorBridge.python_checks_iff` agree. Shared kv1 `TensorBinding.checked_read`
connects actual pointers, contiguous offset views and the absent negative bit to
logical FP32 Tensor values through core Torch storage laws. Initial metadata
readback and pointer capacity are separate ordinary storage bindings.

## Sources and kernel results

`Kernels.lean` displays the complete selected FP32 head-only-mask source and its
existing integer index-product commutation variant as ordinary `triton`
quotations. These are the sources used by both public proof entries.
They are definitionally equal to the preserved `Legacy` definitions as complete
typed kernels, including input/output ports.
The older generic dtype sources remain available for compatibility.

- `Correct.destindex_copy_kv2_real` uses the shared `ProgramIO.ImplementsOn`
  judgment: complete actual source-grid execution, access safety, independent
  row-copy values and the full typed frame, including empty grids. The exact
  feature-width equation and arbitrary complete schedule remain visible.
- `FPEquiv.index_commuted_equiv` proves successful execution of both sources and
  equality of all typed memory with `assumptions = []`.
- `FPEquiv.destindex_copy_kv2_fp32` in [NativeObservation.lean](Misc/NativeObservation.lean)
  uses the same finite typed IO for every
  `BitVec 32` payload, proving complete FP32 source-grid outputs and the typed frame. `TensorBridge.destindex_copy_kv2_fp32` retains finite-pointer
  copying and exterior byte preservation for nonempty invocations. `Nonempty` records the
  actual launch branch, not a restriction on the wrapper's admitted inputs.

`Tile` executes the full source; `SourceAgreement.same_source_memory` derives that the
omitted feature mask is true for every admitted feature lane before reusing kv1
coverage and physical-copy proofs. `Kernel` holds the mathematical source result.
Historical wrapper/object/return theorems stay optional in `Host`; the public
entries do not import it. No numerical law is assumed, including on masked
lanes. Basic load/store meaning is a modeling boundary, not an experiment TODO.

## Original-source issue and validation

The original strided-destination counterexample remains: backing `[0,1,2]`
viewed with stride two requests `[0,2]`, but raw loads read `[0,1]`. Source
`[11,22]` and output `[100,200,300]` therefore produce `[11,22,300]` instead of
`[11,200,22]`. The checked contract rejects the unsupported layout; original
benchmark files are unchanged. Nonempty widths 3/5/96/129 are explicitly rejected
because they violate the original direct-arange width requirement.

`check_contract.py` verifies unchanged JIT AST, guard correspondence, ten
admissions, thirty-four rejections, raw words and frames, empty work, offset views,
head tails and the counterexample using CPU backing storage and mocked CUDA.
Both public Lean entries have targeted builds, standard-base axiom audits,
non-circular formula checks and an empty FP-assumption report.
No GPU run or independent numerical replay is claimed. Shared grid-product
overflow and segmented launch compatibility remain deferred globally.

Modeling references: [shared physical load/store contracts](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts)
record Triton masks, PTX byte addressing/alignment and ordinary Torch API laws;
[shared FP32 Tensor storage laws](../../../../VeriTile/Torch/FP32.lean)
provide the actual-pointer and logical-value connection reused here.

`Misc/TensorBridge.lean` exposes `TensorBridge.python_checks_iff` as an ordinary
theorem for Python-check correspondence. `Correct.lean` and `FPEquiv.lean` retain
the mathematical and source-equivalence results and do not import this bridge.
[NativeObservation.lean](Misc/NativeObservation.lean) retains the auxiliary native kernel result.

`IO.lean` shares kv1's finite typed source/metadata/output channels, then binds
the actual kv2 quotation and backend for every scheduled program. Payloads have
shape `[count, heads, dims]`; signed metadata has shape `[count]`. The independent
output formula copies the source channel, while the output window applies the
logical destination index. Nonnegative, injective signed destinations are the
logical counterpart of the former Nat destination map, not a new payload guard.

`Proofs/PublicIO.lean` assembles actual source safety and full-grid execution.
`Proofs/LogicalGrid.lean` transfers kv1's finite logical output/frame theorem only
after the actual kv2 source memory effects have been proved equal. It does not
assume global typed reads at all natural addresses. The existing `RawGrid.complete`
and all Tensor/finite-pointer/byte-frame applications remain available under
supporting proofs and `Misc/TensorBridge.lean`; no source quotation or contract changed.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
