# Quantized cache transform: kernel-only proof

The canonical candidate scatters INT8 codes and FP32 scales into the two
caller-owned caches and returns `None`. Its Python, guards, launch options and
source arithmetic are unchanged in this proof update. The source SHA256 is
`1bdf2b5ec557dcb552340ee55b376d6e8ebed924b80ef5ce40323fe84b7702ee`.

For source token `t`, head `h` and logical width `D`, the independent Real target
uses `A = max_{0 <= c < D} abs(K[t,h,c])`, `scale = A / 127`, and denominator
`1` when `A = 0`, otherwise `scale`. Each code is the signed-byte observation of
the toward-zero integer conversion of `K[t,h,c] / denominator`. A zero row
therefore stores scale zero and codes zero. The cache row is the original
signed-INT32 destination entry. The mathematical input and scale cells are
arbitrary Reals; no finite decoder restricts this separate interpretation.

The checked domain retains contiguous FP32 input, INT32 destinations, INT8
codes and FP32 scales, all strided and on one CUDA device, with no unresolved
negative view. Ranks are 3/1/3/3. Heads and logical width are positive; source
rows cover the destination count; output geometry agrees; scales have shape
`[capacity, heads, 1]`. Nonempty direct data pointers are naturally aligned.
All four direct data spans are disjoint, and destination entries are bounded
and unique. Program count is below `2^31`, element extents below `2^63`, and
the padded head-by-feature tile is at most `2^20` elements. Nonzero storage
offsets are supported. An empty request launches nothing.

`Kernels.lean` includes the readable typed JIT body with the exact checked
wrapper stride arguments. It retains INT64 program/control promotion, the
INT32 metadata read, both padded masks, zero-filled source tails, the complete
maximum, the explicit FP32 scale casts, the zero-maximum divisor-one branch,
and every INT8/FP32 store. `KernelExecution` constructs a successful execution
of every statement, including all integer conversions, without an
execution-success or defined-conversion premise. `KernelResult` proves its
complete final memory and all active stores and exterior cells.

`KernelGridResult` composes actual source executions in any complete program
order. Checked destination uniqueness gives disjoint output rows, and source
reads remain unchanged. The same final state has every logical code and scale
and a complete exterior frame, including padding and unused cache rows.
`KernelRealResult` proves padded maximum neutrality against the independent
logical maximum and connects this execution to the Real target.

The independent native target retains the selected complete ordered maximum
tree and individual absolute-value, maximum, division and equality operations.
The tree's leaves are a permutation of every launched column; compiler tree
selection never receives floating Tensor values. FP32 scale transport is an
identity cast, while the two divisions remain distinct. No maximum
associativity, commutativity, whole-reduction equality or IEEE replacement of
native `div.full.f32` is assumed.

The retained SM80 profiles emit `cvt.rzi.s16.f32` followed by a low-byte INT8
store. `PtxSigned` defines this selected native conversion for all FP32 words,
including NaNs, infinities and out-of-range finite values, and proves its
agreement with ordinary toward-zero INT8 conversion on the latter's defined
domain. It does not replace this instruction pair with direct INT8 saturation.
Scale underflow and nonfinite inputs do not cause additional admission checks.
The native instruction definitions, including division, FTZ and NaN behavior,
are explicit target-model boundaries; there is no FP-to-Real error theorem or
claim of a unique portable LLVM conversion result outside its defined domain.

`KernelCompiler` binds the selected object bytes to this source hash, the
FP32/INT32/INT8/FP32 pointer signature, all original integer ABI slots, both
static blocks, the original one-warp/one-stage defaults, the address recipes,
branches and instruction structure. Scalar instruction names and complete
reduction trees are decoded from the selected object bytes and static ABI. It
is an explicit decoder interface, not a claimed complete installed-binary
decoder implementation.

`KernelTensor` constructs initial source cells from the four caller data
pointers and ties physical input reads to the original logical Tensor elements.
Its ordinary Tensor API laws retain object and backing-storage identities,
nonzero storage offsets, contiguity and physical indices. Every selected output
coordinate is proved in bounds and mapped to its original caller Tensor
address. `KernelStores`
observes the same final code and scale cells at their checked native addresses,
with exact low-byte and four-byte FP32 readback. No allocator or additional
whole-byte-heap execution is required. `Correct.lean` and `Misc/NativeCorrect.lean` expose
the complete mathematical and auxiliary native kernel certificates. `Misc/TensorBridge.lean` exports the ordinary
original-Tensor applications and Python-check correspondence. Historical wrapper declarations remain in
`Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`.

Validation artifacts are in
`.veritile-cache/quantize-kv-transform-kernel-build-logs/` and
`.veritile-cache/quantize-kv-transform-kernel-profiles/`. The workspace Lean
driver records exact source and object hashes. All 26 primary dependency
modules and the combined public-statement, axiom and noncircularity audit
passed. The allowed axiom
base is `propext`, `Quot.sound` and `Classical.choice`.

`check_contract.py` passes the existing 9 admissions, 50 rejections, actual JIT
AST CPU execution, backing-byte checks and the original metadata-stride defect
control. CUDA context and launches are mocked. Four unchanged-source offline
Triton 3.7.1 SM80 profiles cover width one, padding in both dimensions and signed
i32/i64 argument profiles; 20 compiler artifacts are retained with hashes.
The i64 profile is a bounded ABI control, not a claim about wrapper inference
for its small shape.
An additional one-million-element compiler control was stopped after prolonged
compilation; it supports no compiler-rejection or source-defect claim. These
bounded checks are not GPU execution or universal compiler acceptance evidence.

`IO.lean` now displays the public shared `ProgramIO`: complete actual token
schedule, FP32 logical input, signed destination metadata, INT8 codes and FP32
scales. Output addresses depend explicitly on the input destinations. The
logical domain states nonnegative, injective destinations; the shared judgment
supplies allocation bounds and input binding and proves actual source safety,
success, both complete outputs and every exterior cell. Coverage conditions
and arbitrary complete program permutations remain public.

`Correct.kernel_real` uses independent logical-row quantization. The ordinary
auxiliary `FPEquiv.kernel_native` is preserved in `Misc/NativeCorrect.lean` and uses
the exact existing native instruction/reduction formula. `FPEquiv.address_early_equiv` compares the actual complete source with the
address-scheduling candidate. Neither reader theorem exposes raw State/run/Result
details. The old exact
raw statements remain in `Proofs/PublicRealResult.lean` and
`Proofs/PublicNativeResult.lean`; the new adapters reuse them directly.
`Proofs/AccessSafety.lean` proves every actual masked read/scatter and the
unmasked destination load. `Proofs/PublicIO.lean` follows actual predecessor
states and preserves destination metadata throughout the complete grid.

The complete TensorBridge applications are unchanged, including native byte
store observations and full typed frame. Targeted builds cover the new IO,
helpers, both reader entries and TensorBridge. No numerical law, source change,
GPU run or additional compiler experiment is introduced by format migration.

## Address scheduling candidate (2026-10-21)

`KernelSource.addressEarly` is a second complete ordinary Triton quotation.
It schedules the existing pure `o_ptrs`/`os_ptrs` assignments immediately before
`abs_data`, to expose address formation alongside the FP32 quantization pipeline.
No measured speedup is claimed. `quantize_kv_transform_address_early.py` keeps
the complete checked wrapper and differs only by this source scheduling, as
verified by the whole-module AST check.

`FPEquiv.address_early_equiv` uses the shared actual `ProgramIO` pair syntax.
The proof establishes complete state equality and trace-safety equivalence and
lifts them to every token launch in the original arbitrary complete permutation.
The independent Real target, all FP32 words, explicit total signed-byte conversion,
nonnegative/injective destinations, head/column coverage, both code/scale outputs
and both exterior frames are retained. No floating arithmetic, reduction order,
cast, INT64 control, mask or store is changed. The experimental rule set is empty.
