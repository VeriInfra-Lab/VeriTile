# Direct rotary embedding with optional paged K cache

`Correct.lean` exposes shared `ProgramIO.Implements` / `ImplementsOn` Real
specifications for the distinct cached and uncached kernels. Ordinary single-source
native results are retained in `Misc/NativeCorrect.lean`; `FPEquiv.lean` records
the missing optimization-pair targets.
`IO.lean` lists source identity, actual IO channels, masks, in-place aliases,
metadata-dependent cache windows, and the empty scratch footprint. Mathematical
and selected-native targets use logical operands. The four former raw typed
statements are preserved in `Proofs/RealRaw.lean` / `Proofs/NativeRaw.lean`.
The full checked Tensor/byte applications remain unchanged in
[Misc/TensorBridge.lean](Misc/TensorBridge.lean), sharing the same successful source
execution and memory results. Primary readers do not import the bridge.

This repair preserves the original two Triton kernels, launch branches, warp choices, default fusion, in-place Q/K behavior, and `None` return. It adds independent sine strides, padded pair masks for even non-power-of-two widths, signed-64 index arithmetic, and K-head masks in the cached GQA branch. It performs no payload snapshot, clone, contiguous conversion, staging output, or copyback. The original Python remains unchanged.

Without a cache, Q and K are rotated in place; K work belongs to the original GQA group leaders. With a cache, Q is rotated in place and K's initial values are rotated into the selected paged-cache slots. The original K tensor remains unchanged. Cached stores retain their original order before the Q stores. All Tensor argument objects are retained.

The explicit production contract is canonical, zero-offset, ordinary FP32 CUDA payloads and int64 cache metadata on one device, with no lazy negative/conjugate flags. Q/K have equal token count and even feature width; both head counts are positive and Q heads are divisible by K heads. Coefficients have enough rows/pairs and may have independent canonical row widths. Empty token or feature dimensions return without a launch. Nonempty calls require the Q CUDA device to be current, as required by the original ambient-device launch; no device switch is introduced.

Executable guards validate actual scalar alignment, backing containment, full backing-range separation, exact launch strides, padded signed-64 byte offsets, and formed unsigned-64 pointers. Cache metadata has one row and one length per token, positive lengths, a selected page within the table, a signed selected block within the cache, and distinct `(block, slot)` destinations. All guards run before writes. Int64 backing alignment is checked before the guard reads metadata on the CPU.

`Kernels.lean` contains the complete readable typed Triton DSL for both
repaired kernels: all loads, signed address operations, masks, branches and
stores. `Correct.lean` proves shared logical IO implementation for each same source
over arbitrary Real payloads, the independent split-half rotation, actual
source access safety, all outputs and the exact exterior memory frame. The
cached frame includes every input K cell. Signed metadata channels are
separate from the Real payload carrier.
`TensorBridge.rotary_emb_nopad_checked_uncached_real` and
`TensorBridge.rotary_emb_nopad_checked_cached_real` construct source payload memory
from actual Tensor caller bytes through an explicit total `Word → ℝ` map.
Tensor laws and checked input bounds derive the independent logical-element
split-half formula, complete execution, and the full frame. Cached metadata
is read as signed integers from caller bytes, and every input K cell remains
unchanged. The map adds no finite-payload guard or FP-to-Real arithmetic law.

`Misc/NativeCorrect.lean` retains successful complete-source implementation of
independent selected native FP32 graphs on every raw word. The uncached graphs are
`fma(c,a,neg(mul(s,b)))` and `fma(s,a,mul(c,b))`; cached pairs use
`fma(a,c,neg(mul(b,s)))` and `fma(b,c,mul(a,s))`. These operand and contraction
orders retain default fusion. This task has no faithful historical source
optimization pair; the auxiliary theorem implements the independent ordered graph.
Loads, same-format casts, and selected local mul/neg/fma semantics are primitive
model boundaries. No numerical conformance obligations or finite payload
restrictions remain in the public theorem. Sine and cosine are loaded tables.

The checked FP entry points use actual Torch Tensor objects and shared
`TensorLaws`, derive logical reads from their data pointers, and bind signed
int64 metadata to current caller bytes. They connect typed results to actual
physical output words and the exterior byte frame. Cache placement is proved
separately: `past = length - 1`, selected page `past / blockSize`, slot
`past % blockSize`, and the signed block-table entry determine the exact caller
cache destination. Store ownership is derived from checked geometry and
metadata; it is not a public result premise.

The earlier scalar, vector, grid, byte, and wrapper closure remains under
`Proofs/Model.lean` and optional `HostReal.lean` / `HostFP.lean`. It supplies the
physical whole-grid binding and preserves the established return/object/device
claims without making wrapper behavior the kernel acceptance statement.
`Contract.lean` retains the same executable input scope. Original defects stay
separate from the repaired equivalence. Compiler/source correspondence and
CUDA behavior remain explicit runtime boundaries; no universal GPU-to-IEEE
conformance is claimed. `check_cuda_lowering.py` retains offline evidence for
the selected operand and contraction orders.

Validation:

```bash
python bench/repairs/tritonbench_g/rotary_emb_nopad/check_contract.py
python bench/repairs/tritonbench_g/rotary_emb_nopad/check_cuda_lowering.py --output-dir /tmp/veritile-nopad-lowering-evidence
```

The CPU source replay checks 16 complete cache/no-cache paths and 27 prewrite rejections, including token/pair tails, GQA, D=256/512 warp branches, empties, scalar/int64 misalignment, duplicate cache slots, and ambient-device mismatch. It reproduces the original independent-sine-stride error and cached-GQA corruption. CUDA is mocked. Offline Triton 3.7.1 compilation checks both kernels for SM80/SM90 with i32/i64 scalar arguments at D=12, QH=4, KH=2, four warps; it executes no GPU code and does not establish universal compiler conformance.

The shared `rotary_emb/check_interpreter.py` also passed ten direct Triton CPU
interpreter launches, including three uncached and three cached nopad launches.
They cover GQA leaders, padded heads/pairs, independent coefficient strides,
nontrivial cache pages/slots, K preservation, metadata and backing sentinels.
No GPU execution is claimed.

The main interfaces use `KernelInterface.Geometry` plus the actual two program
IDs and necessary positive-head/token premises. Cached metadata nonnegativity
is a logical input domain. No device, Tensor, API flags, raw execution state or
checked-admission predicate occurs in the headline. Coefficients have their own
read windows even when Q is masked or the current program is not a K leader;
cached metadata reads occur for a valid head even with zero feature width.
Early head return suppresses every cached access. Q/K input and output aliases,
all four output halves and the cache-before-Q order remain exact.

`Proofs/SourceSafety.lean` checks the real source access trace for both kernels.
The IO binding and interface helpers reuse `Typed.*Execution`, `*_memory`, and
the existing ownership/value/frame proofs. Native payloads retain all words and arbitrary initial product
provenance, with exactly the original selected mul/neg/FMA model and atom
outputs. All old raw statements and complete TensorBridge statements remain
available. This format pass adds no numerical assumptions, experiments or GPU
claims and changes no Python, kernel, contract, dtype or mathematical domain.


The 2026-10-21 implementation pairs are complete for both distinct source
stages. `Kernels.uncachedShared` and `Kernels.cachedShared` are complete ordinary
Triton quotations. Each derives the high-half Q address from `off_q0` plus the
half-dimension stride, removing repeated token/head stride calculations. This
is an address common-subexpression elimination candidate, without a measured
performance claim. Every mask, floating operation, GQA branch, signed metadata
load, cache address and cache-before-Q store remains unchanged.

`FPEquiv` exposes actual `original ≡[[]] candidate` results separately for the
uncached and cached complete tiles, at every original program ID and unchanged
public IO domain. They conclude safe successful execution of both programs,
all four active output halves and both exact exterior frames. Cached K and
metadata are preserved, and padded-head early return remains included.
`Proofs/AddressCSE` proves exact full-state execution and access-safety equality
for arbitrary backends, registers and layouts using integer distributivity.
`Proofs/ProgramPair` also composes run equality and safety equivalence across
any list of actual invocations. No floating rewrite rule or numerical
experiment is needed. The independent Real specifications, original native
formulas and optional Tensor applications remain retained at their scope.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.
