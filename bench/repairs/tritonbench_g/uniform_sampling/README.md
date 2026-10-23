# uniform_sampling guarded FP32 repair

Read `Kernels.lean`, `Correct.lean`, then `FPEquiv.lean`. The source module contains
both complete kernels. The existing candidate changes only `pid * BLOCK + lane`
to `BLOCK * pid + lane`; every random operation, floating operation, mask and
store remains in the same order. The Python wrapper continues to call the
original repaired kernel.

The source retains signed64 seed and offset arguments, signed64 program/lane
widening before products, modular uint32 low-counter addition, all four
Philox4x32-10 fields, and the canonical rounded uniform decoder. Each field then
uses a separate FP32 `upper - lower` and an explicit FP32 FMA. Four masked FP32
stores clamp their eagerly evaluated pointer to `N - 1`. Explicit Nat typing of
the literal in the clamp and the int64 promotion preserve integer addressing.
The two scalar bitcasts materialize the runtime FP32 argument words. Extraction
from Python float objects belongs to the scalar ABI model boundary; the source
parameters preserve signed zero, infinities and NaN payloads.

`IO.lean` defines the single shared output interface: the actual source, one
output channel with shape `[4, B]`, field/lane store addresses and tail masks.
`Correct.samplingIO` is only its Real interpretation; the FP entry reuses the
same interface with its selected arithmetic interpretation. Connection, access
safety and execution proofs remain in `Proofs/PublicIO.lean`.
`Correct.uniform_sampling_kernel_correctness` uses shared `ProgramIO.Implements`
to prove actual source access safety, successful execution, all four active
output values and the complete exterior memory frame. Its independent
mathematical target is `Correct.formula` (the existing `KernelMath.expected`): for field `f` and lane `i`, use canonical Philox word
`ScalarSpec.word seed offset (pid*B+i) f`, decode its actual rounded uniform
payload, then apply `lower + u * (upper - lower)` in Real arithmetic. The ideal
fraction `foldSign(word)/2^31` is a different historical specification and is
not substituted for the source decoder. The public Real theorem names ordinary
Real lower and upper values under explicit finite denotations of the extracted
scalar words. This is a mathematical
interpretation domain; it adds no Python guard and assumes no numerical rewrite
law. The totalized projection used internally is not the public mathematical
target for NaN or infinity. Operational equivalence below covers every word.

`FPEquiv.originalIO` and `FPEquiv.commutedIO` display the two actual source
schedules on the same typed output interface. The single public
`FPEquiv.index_commuted_equiv` displays the actual
`original M ... ≡[[]] indexCommuted M ...` pair. The shared
`ProgramIO.Implementation` view reuses the existing `ProgramIO.Equivalent`
proof: both actual source traces are safe, both implementations succeed, all
active typed outputs agree, and both complete exterior frames hold.
The only optional primitive required is the selected FP32 fused operation.
Random words, scalar extraction, subtraction, FMA and store conversion are
identical between the versions; no numerical rewrite or finite-input premise is
used. The earlier unconditional source-state comparison, including failure,
and stronger complete-memory execution pair are retained as ordinary theorems
in `Proofs/KernelEquivalence.lean`. `Proofs/KernelReal.lean` retains the complete
original operational Real result, native integer bounds and eager clamped
pointer-capacity conclusions. `Proofs/SourceSafety.lean` checks the actual 23
register-only prefix statements and each of the four stores in its reached
offset state; access safety is not inferred from the equality of two runs.

The checked wrapper admits a strided, contiguous CUDA FP32 output Tensor with
no unresolved negative view bit, fewer than 2^63 elements, and an aligned
nonempty data pointer. Bounds are Python floats, including exceptional values.
A supplied generator must belong to the output CUDA device. Scalar/empty shapes
and contiguous offset views remain admitted. `InputContract` and
`TensorBridge.python_checks_iff_preconditions` expose these exact assertions;
`KernelMemory.checked_output_memory` connects the guards to shared Torch FP32
storage, actual pointers and logical values.

The per-launch kernel domain is nonempty `N <= 2^34`, block size 512 or 1024,
signed64 seed/offset ABI values and a program ID in `ceil(N/(4*B))`. Existing
chunk proofs establish these bounds for the wrapper's carry-boundary splitting.
The wrapper retains B512/4 warps for total N<=512, B1024/8 warps for
512<N<=1024 and B1024/16 warps above that. All local counters fit uint32 and all
four widened output offsets fit signed64. Every clamped pointer, including an
inactive lane's pointer, stays within the nonempty chunk.

Allocation, reservation orchestration and returned-object identity are outside
the public kernel requirement. Existing reservation, scratch/copy, failure and
generator-state theorems remain optional under `Proofs/HostCorrect.lean` and
`Proofs/HostFPEquiv.lean`; the former lane/field source graph is preserved in
`Proofs/LegacySource.lean`. The unchanged C++ helper reserves the aligned padded
counter interval under the generator mutex, and the unchanged wrapper splits
launches at low-word carries. These optional helpers are excluded from the new
public proof import closure.

There are no deferred numerical transformation experiments for this candidate.
Primitive RNG, rounding, FMA and scalar ABI conformance are shared semantic model
boundaries. No native compiler/device or statistical conformance is claimed.
Historical conditional ideal-Real correspondence helpers remain optional and do
not add acceptance requirements.

The prior acceptance compiled all 38 then-existing task Lean modules, including
the public entries and optional historical helpers. The public and historical entries also import together.
Public axiom and specification-independence audits pass. The source-only module
imports only the DSL; public Real and FP entries are independent, and their
closure excludes the historical host proofs. `check_contract.py` passed the
exact one-product candidate AST check,
12 guard rejections, 13 wrapper admissions and 14 existing wrapper program
executions. It additionally passed 252 paired complete kernel cases and 560
program executions, comparing every memory byte and exterior guard storage.
Cases cover both block sizes, all warp/tail boundaries, negative and extreme
seed words, adjacent low-word carry boundaries and exceptional bounds. CPU
replay uses the published zero-key Philox10 vector and the same CPU fused
primitive for both versions; CUDA and reservation APIs are mocked.

`Misc/TensorBridge.lean` holds the optional Tensor/check correspondence helpers;
`Correct.lean` and `FPEquiv.lean` retain the independent mathematical target and
the actual floating implementation-pair entry.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean) and [Misc/Coverage.lean](Misc/Coverage.lean). The latter retains `Correct.output_coverage` as an ordinary theorem with its original statement and proof. Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.

Source readability cleanup (2026-10-22): both authoritative source functions in `Kernels.lean` are complete ordinary `triton` quotations. `Proofs/LegacySurface.lean` archives the previous macro; `Proofs/SourcePreservation.lean` proves equality of both complete functions by `rfl`, including every runtime bound word, seed/offset, block size and arange-size proof. Philox fields, conversions, separate subtraction/FMA, eager clamp pointers, four stores, public declarations and existing completion status are unchanged.
