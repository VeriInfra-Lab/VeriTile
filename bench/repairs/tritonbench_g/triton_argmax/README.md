# triton_argmax guarded repair

Current selected public entries: [Contract.lean](../triton_argmax_nan_first/Contract.lean), [Correct.lean](../triton_argmax_nan_first/Correct.lean), [FPEquiv.lean](../triton_argmax_nan_first/FPEquiv.lean). The [contract review](../../../validation/contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

**Current kernel-only acceptance status: source defect pending repair.**
The dimension kernel returns `0` for the admitted FP32 input `[1, NaN]`, while
the independent first-NaN argmax target and `torch.argmax` give `1`. This affects
both the original source and the selected repair. See
[source-defects.md](source-defects.md), the actual Triton CPU-interpreter receipt
[source-defect-validation.json](source-defect-validation.json), and
[Proofs/Counterexamples.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/triton_argmax/Proofs/Counterexamples.lean). No `nonNaN` input
guard or numerical deferral is introduced. This task is not counted as complete
under the current standard.

The previously accepted Real and native source-graph results described below
remain available with their original scopes. They do not establish that the
admitted all-word native kernel meets an independent argmax target.

The public API is `argmax(inp, dim=None, keepdim=False, *, dtype=None)`;
`argmax_repaired` is an alias. Input is strided FP32 CUDA. The global branch
reads its original contiguous, non-negative-view input directly, including
nonzero storage offsets. The dimension branch retains the original
`inp.contiguous()` call and accepts its strided/broadcast input domain. It does
not add a payload gather, snapshot, or copyback.

## Branch and precision behavior

Global argmax keeps the original `next_power_of_2(ceil(sqrt(numel)))` block
selection, all first-stage programs, the intermediate storage dtype, the second
reduction, and the final indirect index gather. Supported intermediate dtypes
are float16, bfloat16, float32, float64, int8/16/32/64, uint8/16/32/64, and bool.
Omitted dtype selects FP32. Integer stores retain the accepted scalar graph:
NaN to zero, FP32 lower/upper clamps, then truncation. Bool truncates through
an int8 byte and loads as byte-nonzero; fractional positives and negative
nonzero bytes therefore differ. Floating conversion can change the winning
block, so equality with a single FP32 argmax is not asserted.

Second-stage padding is dtype-specific. The complete padded intermediate-index
allocation is zero-initialized, making a floating NaN-induced selection of a
padding lane a defined gather. Ordered comparison trees and coupled value/index
selection remain explicit, including NaNs, infinities, signed zeros and ties.
Native second-stage registers have `max(32, storage_bits)` bits, with signed
extension, unsigned extension, and boolean normalization proved separately.

Dimension argmax normalizes a valid positive or negative axis, retains the
128-by-128 source tile and every axis tile, and updates only on strict FP32
improvement. Optional dtype is ignored, including unsupported tokens. It returns
the allocated int64 output when keepdim is true and its ordinary squeeze view
otherwise. Global output is the allocated int64 scalar or all-ones shape.
An empty global reduction or selected axis rejects. Zero outer/inner extent
returns the allocated empty result without a launch or payload-pointer query;
the original contiguous call still occurs.

## Executable admission and actual memory

`Contract.lean` names the branch-sensitive input contract and dynamic argument
classification. `Proofs/Admission.lean` proves the Boolean checker equivalent to
that contract. Guards cover dtype/device/layout, axis validity, used negative
view flags, source block sizes, signed64 control/index arithmetic, checked
pointer alignment and uint64 spans. Launch segmentation at 65,535 programs
preserves the full logical schedule; `num_ctas=1` is explicit. This launch
segmentation and all three JIT bodies are retained verbatim from the accepted
revision002 repair. This pass adds no shared launcher-product workaround or new
runtime-compatibility claim.

`Proofs/TensorBinding.lean` binds inputs to the shared FP32 Tensor, actual storage,
physical coordinates, storage offset, device and data pointer. Allocations and
squeeze views carry their observed objects, shapes, dtypes, pointers, storage
identity and capacities. Alignment is a receipt of the executable pointer
check, not an unconditional allocator axiom. Freshness and the ordinary
contiguous allocation footprint give the complete caller-byte frame.
`Proofs/NativeSafety.lean` derives exact active native pointers, full word spans,
mask comparisons, padded index bounds and safe intermediate-index gathers.

The ordinary Torch API boundary covers allocation, contiguous/squeeze semantics,
metadata queries, the observed Python square-root calculation and storage
ownership. These laws contain no argmax value or completed execution premise.
The pure memory boundary consists only of scalar read/write/frame laws.

## Independent public results

`Correct.lean` proves global and dimension results against an independent least
Real maximizer at the input Tensor's logical physical coordinates. Global Real
execution retains scratch initialization, every first-stage store, second-stage
loads, reduction, gather and output store. Its accepted idealization erases the
whole precision-adapter macro. It does not treat each finite clamp as an
identity, or identify every native dtype result with the Real result.

`FPEquiv.lean` independently proves the complete native byte execution against
an explicit native target, including casts, decoded registers, all reductions,
the full dimension loop, all programs and returned storage. Both public Tensor
theorems include finite pointer safety, shape/dtype and caller-frame guarantees.
The native dimension theorem also gives exact post-contiguous state identity
for empty output. The optional Real/native bridge requires only reached scalar
comparison agreements, including comparisons after the intermediate cast.

`Kernels.lean` is a source-only ordered graph. It imports neither public proof.
The native reduction closure retains finite width-typed emitted expressions,
transport/alias extraction, comparison/select sites and a derived ordered tree;
it has no reduction-result or whole-output callback. Attaching a concrete
compiler artifact uses the explicit selected-program extraction interface.
Scalar floating conversion/comparison and reached maxnum/minnum/truncation
interpretations remain local ports. The two zero-clamp facts are scalar facts.
No new GPU instruction conformance or floating error bound is claimed.

## Validation

Run `python bench/repairs/tritonbench_g/triton_argmax/check_contract.py`.
The CPU harness executes the actual JIT function AST with an explicit balanced
comparison tree and typed storage. It passes 28 source/API/dtype/axis/empty
admissions and 24 rejection controls, verifies exact allocation/squeeze
identity, segmentation boundaries, and caller storage preservation. It covers
intermediate rounding, bool-byte semantics, padding gathers, multiple axis
tiles, ties, noncontiguous/broadcast inputs, negative empty views, pointer
alignment/span and overlapping allocations. This is source validation, not GPU
execution.

The native graph, Tensor binding, native safety and both public proof entries
compile with standard-base axiom audits. Kernel AST SHA-256 values (excluding
source positions) are:

- Stage 1: `02e20deb1dbd20fc69c64be7a95e19f20edb275b48ad44bb1124b21fa41588ed`
- Stage 2: `7c3ffb73548ecd995186c6667e2aa6391b700d590e764d24868eb6be5d3064d0`
- Dimension: `284d53fd882833bf23d240fa794a9f2e244fc8c0cfcc1201fcc9fe765ff2d60e`
