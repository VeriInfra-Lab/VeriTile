# Rowwise quantization: complete kernel, direct FP32 input

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

The main `FPEquiv.rowwise_grid_equiv` compares the complete original grid with
an early-maximum-store candidate. Both actual source schedules are safe and
succeed, agree on the INT8 and FP16 typed output channels and preserve both
exterior memory frames. The empty experimental rule set is justified by exact
store commutation; every original FP32 word remains admitted.

For each logical row, define `m = max_c abs(x[r,c])`. The independent Real
formula is nearest-even `127 * (x[r,c] / m)`, encoded as INT8, with zero codes
when `m = 0`. The maximum output has the source's direct FP16 storage cast.
Real is the mathematical interpretation; it does not assert exact equality
between native rounded arithmetic and Real arithmetic.

The selected source remains `rowwise_quantization_triton_repaired.py`, SHA256
`17c364be76127d5aa3df9524611ca4da9fc5e28cc31aa32822f35cb148b20c35`.
It uses FP32 input/absolute value/maximum/division/multiplication, signed64
`llrint`, low-eight-bit INT8 storage and FP16 maximum storage. No BF16 variant,
copying wrapper, allocator proof or GPU runtime implementation is added.

## Input contract

Python and `Contract.lean` admit exactly rank-two strided FP32 CUDA Tensors,
default contiguous and without unresolved negative view bits. Nonzero storage
offsets and empty batches are admitted. Columns satisfy `0 < columns <= 2**20`,
rows satisfy `rows < 2**31`, and nonempty input data pointers are four-byte
aligned. Every FP32 bit pattern remains admitted: there is no finite-input or
normal-value filter. `checkInputs_iff` proves the Lean guard/contract equivalence;
`check_contract.py` checks the canonical Python guard AST against these fields.

The wrapper supplies these assertions, the launch dimensions and two fresh
outputs on the input device. Their dtype, shape, bounded storage, alignment and
freshness are ordinary API binding facts. Public specifications prove the
kernel, not Python wrapper/allocator execution.

## Source execution and proof organization

`Kernels.lean` contains the complete readable `triton {}` source and retained
scalar/byte definitions. Both original and existing early-maximum-store candidate
are complete raw Python bodies in this shared file; Candidate.lean is a
transparent alias. Pointer storage types, the unused runtime argument and
constexpr parameters are outside the quotation. The signed64 program-ID cast,
tl.full multiplication, integer-zero masking, ordinary division, llrint and
both implicit store conversions are retained. The Real proof and optional
native proof use these actual bodies. Natural logical addresses are connected
to the signed source addresses by exact integer identities, with the existing
native machine-index refinement including padded inactive lanes.
The maximum reduction uses one explicit `Launch.tree`, selected independently
of input payloads; its leaves cover every padded lane exactly once.

The selected source uses signed64 `llrint` and the typed `output` register
marker. `Proofs/IntegerBinding.lean` proves that its existing local executor is
exactly the shared `QuantizeIntegerStage.exec` on this quoted source. The shared
invocation names the scalar signed64 primitive and exact register binding;
missing registers and unsupported opaque markers still fail. No replacement
source or new primitive law is introduced.

The accepted raw representation retains signed64 `.int` cells. Shared
`IntegerOutputView.unsignedWidth 8` observes their low byte, preserving the raw
state, dtype tag, source execution and exterior frame. FP16 maxima keep their
own cell tag; native payloads use the already-proved `packHalf`/`unpackHalf`
encoding. This makes both actual output interpretations explicit without
silently changing the old store semantics.

- `KernelResult` derives successful execution, all active stores, the FP16
  maximum store and the row memory frame from the complete quoted source.
- `KernelGrid` and `KernelWhole` connect all scheduled row executions, input
  preservation, complete outputs and the exterior logical-memory frame.
- `Correct.lean` exposes the independent maximum/nearest-even Real formula and
  one shared `IntegerProgramIO.ObservesInteger` specification over named IO.
  Arbitrary Real inputs are not encoded into FP32 bit patterns.
- `Misc/NativeCorrect.lean` exposes the optional independent source-order native formula and the same
  shared IO judgment, preserving all FP32 words and both output storage types.
- `IO.lean` names the complete arbitrary row permutation, source, input/output
  windows, actual precision, scalar model and explicit byte output view.
- `Proofs/SourceSafety.lean` checks actual load/store accesses through the shared
  integer executor; `Proofs/IOImplements.lean` connects complete existing grid
  execution, logical binding, both typed outputs and the exact frame.
- `Misc/TensorBridge.lean` retains admission equivalence, actual input Tensor binding,
  both output storage addresses, byte values and byte frame. Its output bindings
  include backing storage, storage offset, physicalIndex, observed data_ptr,
  dtype, shape and device. No output equation or successful execution is an
  input premise.
- `KernelTensor` reuses the existing byte-grid proofs as storage lowering and
  proves their values equal the two stores of the quoted source. It adds no
  general byte-heap framework. Existing host results are retained under
  `Proofs/HistoricalCorrect.lean` and `Proofs/HistoricalFPEquiv.lean`.

## Precision and model boundaries

`KernelNativeProfile.round64` states nearest-even conversion for defined FP32
inputs. NaN, infinity and out-of-signed64-range conversions retain an explicit
signed64 undefined-result policy, without removing these inputs from admission.
Maximum/division/multiplication and the direct FP16 cast are named scalar native
lowering choices. None receives a row, a grid, a Tensor or an output state.
The source's only floating comparison is equality with the literal zero;
the profile installs that exact zero test, without asserting an equality law
for other right operands. Their chosen implementation and the structural reduction tree are documented
compiler/primitive model boundaries, not whole-result assumptions.

`KernelEncoding.narrow64` proves that signed64 interpretation followed by the
low-byte store preserves the INT8 encoding. `half_roundtrip` proves the exact
FP16-payload embedding. These integer/encoding facts and primitive semantics
are not numerical experiments. FP16 maxima can overflow for finite FP32 inputs;
no unsupported finite-output guarantee is made.

There are **no numerical rewrite assumptions** in this delivery. Quantitative
native-to-Real error is not part of the selected target. Native GPU execution
and independent replay are reported separately; neither is inferred from Lean
compilation or from a CPU AST check.

## Evidence and references

Current source/proof replay and its limits are recorded in the tracked
bench/review-checkpoints receipts. No source dependency or proof result is
supplied by .veritile-cache. The raw-body checker compares both complete
twelve-statement functions and their ABI against unchanged Python; regression
mutations check casts, integer literal type, division operands, masks and
implicit storage conversion. The CPU checker retains fourteen output-pair/frame
cases and thirteen guard rejections; its availability and actual result are
recorded separately. Mocked CUDA/launch checks do not establish GPU behavior
or numerical-assumption admission.

The semantic boundary follows the [CUDA libdevice llrintf specification](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrintf.html),
[Triton maximum](https://triton-lang.org/main/python-api/generated/triton.language.maximum.html),
[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
and [Triton store](https://triton-lang.org/main/python-api/generated/triton.language.store.html).
One explicit offline SM80 specialization (`columns=17`, `P2=32`, eight warps,
one stage; the source's first autotune configuration) is recorded in
`native-probe.json` with hashes for TTIR/TTGIR/LLVM/PTX. It emits
`div.full.f32`, `mul.f32`, `cvt.rni.ftz.s64.f32`, an INT8 low-byte store and
`cvt.rn.f16.f32` followed by a 16-bit store. The probe uses an explicit signature;
it does not trace the runtime JIT binder, cover every autotune configuration,
or execute a GPU. The ordered native scalar profile retains the selected
lowering, including the emitted multiplication operand arrangement.

The installed Triton 3.7.1 `language/extra/cuda/libdevice.py` dispatches FP32
`llrint` to `__nv_llrintf` with int64 output. `language/standard.py` defines
`tl.max` through its pairwise maximum reducer. Pointer/contiguity/negative-view
API references remain in shared `VeriTile/Torch/FP32.lean`.

The original benchmark's flat-pointer layout defects remain separately
recorded. For example, transposing `[[1,2],[3,4]]` gives logical rows `[1,3]`
and `[2,4]`, but its flat source loads `[1,2]` and `[3,4]`. The first maximum is
therefore 2 instead of 3 (and its first INT8 code is 64 instead of 42). This selected revision retains its previously authorized contiguous
contract, widened indices and zero-denominator handling; it does not silently
reclassify the original source as correct.

## Strict kernel and Tensor interfaces

`Specification.rowwise_grid_real` and the optional `Floating.rowwise_grid_fp32` in
`Misc/NativeCorrect.lean` expose
named IO and independent formulas. Their shared judgment concludes successful
complete source execution, access safety, both output arrays and exact exterior
memory preservation. `Launch` retains the original row/column/width scalars and
payload-independent reduction tree/coverage; `Schedule` retains any permutation
of the complete row grid. Public statements do not expose raw state, runners,
post-state witnesses or per-address cell predicates. Both have surface audits.

`TensorBridge.rowwise_kernel_correct` and `TensorBridge.rowwise_fp32` retain
their original complete statements, including native byte readback and physical
frames. The Real application now calls the existing `KernelRealResult` foundation
directly; native application and pointer/index facts are unchanged. Primary
readers do not import Tensor applications, and proof helpers do not import readers.
Python, Kernels, Contract, FP32/FP16/INT8 precision, accepted domains and numerical
boundaries remain unchanged. No GPU experiment or independent replay was added
by this format conversion.


## Early maximum store candidate, 2026-10-21

`Candidate.lean` transparently references the complete candidate body in `Kernels.lean`. It moves the
FP16 row-maximum store directly after the maximum reduction, before `safe_max`,
division, scaling, `llrint` and the INT8 output store. The original input tile
has already been loaded. The FP16 maximum and INT8 output use distinct regions,
so the two stores commute exactly. This shortens the maximum value's live range
across quantization; no measured performance improvement is claimed. The
Python candidate is `_quantize_rowwise_early_max` with wrapper
`quantize_rowwise_early_max` in the existing source file. Its contract, allocation,
empty handling, launch and all autotune configurations match the original.

`Proofs/StoreSchedule` proves exact full-state equality of the two actual source
executors for each row, preserving signed64 `llrint`, low-byte INT8 observation
and the explicit FP16 conversion. `Proofs/StoreSafety` checks the candidate's
actual accesses under the original bounds. `Proofs/StoreIO` composes the actual
row permutation, with source safety, successful runs, both typed outputs and
both complete exterior frames inside the main `IntegerProgramIO.Equivalent`
judgment. Original numerical primitives and their exceptional behavior are
fixed model bindings. No numerical relation or deferred experiment is added.
The independent Real specification and optional single-source native results
remain at their original scopes, outside the main proof's dependency on `Misc`.
