# Checked direct context attention

The repaired function updates the supplied output Tensor and returns `None`.
It preserves the original FP32 source arithmetic: both dots, the probability
cast, base-2 exponentials, finite `-1e8` sentinel, accumulator dot, 64/128 tile
choice, warps4/8, stages1, and default fusion. Only integer IDs, metadata and
lane indices are widened before pointer arithmetic. The original Python file
and existing original proof APIs remain unchanged.

The explicit domain is FP32 CUDA `Q/Out[T,H,D]`, independent
`K[B,Hkv,Sk,D]` and `V[B,Hkv,Sv,D]`, and three contiguous int32 metadata vectors
of length B, all on one device. Q and Out are contiguous. K/V have dense inner
axes and disjoint batch slices; the batch stride may include a gap. This keeps
the original adjacent K/V head slices of a shared cache allocation. Read-only
inputs may alias. Output backing storage must be disjoint from every read
role, using full byte intervals, including externally wrapped partial
overlaps. Nonempty data pointers must be four-byte aligned. Lazy negative and
conjugate flags are rejected. Nonzero storage offsets are supported.

D is 16, 32, 64, 128 or 256. Head counts are positive and H is divisible by Hkv.
For each batch, metadata must satisfy `0 <= prompt <= sequence <= min(Sk,Sv)`,
`start >= 0`, `sequence-prompt <= max_input_len`, and
`start+sequence-prompt <= T`. Positive output-row intervals cannot overlap.
`max_input_len` is a built-in nonnegative signed-int32 Python integer and
`B*H <= 65535`. Conservative formed-pointer budgets include inactive lanes and
keep all relative element/byte arithmetic within signed int64. Ordinary
storage laws bind active addresses to finite native windows.

All validation precedes writes. Empty batches or zero active query lengths
skip the kernel. Only small integer metadata is observed on the CPU; query
and cache payloads remain in their original storage. There are no payload
copies, private output allocations, or copyback. The owning device is selected
for the call and the caller's device context is restored.

For local query i, let
`E = min((floor(i/BM)+1)*BM + prompt, sequence)` and
`J = ceil(E/BN)*BN`. The independent mathematical target is
`sum_j(2**score[j] * value[j]) / sum_j(2**score[j])` over `j < J`.
Values use the corresponding grouped KV head and are zero when `j >= E`.
Scores use the Q/K dot and the actual host scale when `j <= i+prompt`, and
the finite value `-1e8` otherwise. Future and padded lanes therefore retain
positive Real weights. This source-specific formula is not strict causal
softmax for arbitrary Real scores. The replay harness includes a case where
the finite sentinel visibly changes the FP32 output.

`Checked.whole_mathematical_real` connects arbitrary Real source arrays to
that independent formula through the complete source/grid execution. It
proves the full logical frame and unchanged read roles and retains the None
return. It requires no finite encoding of Real payloads. The older encoded
Real helper is retained separately.

`Checked.whole_fp32_bytes` derives source execution, integer control, formed
offset bounds, memory events, complete output stores, read-role preservation,
empty behavior, supplied output identity, None and device observations from
the checked contract. `CheckedTensor.whole_tensor_fp32` adds exact logical
readback on the same shared Torch Tensor object using the four-byte codec and
ordinary Tensor laws. Its output is `CheckedTensor.referenceWord`, evaluated
from an independent online recurrence rather than read from the source result.
`FPReference.advance`, `iterate`, and `output` specify each masked tile update;
`FPReference.program` proves the prelude, loop invariant, and epilogue bridge,
and `FPReference.whole` lifts it to every program in the host grid. Both the
reference recurrence and final word have checked noncircularity audits. `CheckedTensor.logical_source` binds original logical
input values to the same source-byte reads. `Checked.role_alignment` derives
alignment from the actual checked pointers. The shared descriptor contract
also binds all Q/K/V/Out source stride arguments to the same observed Tensor
strides; `Checked.source_view_offsets` proves the resulting coordinate/address
agreement, including cache batch gaps.

The existing native graph retains the selected SM90 instruction plan: fixed
k8 TF32 WGMMA outputs, explicit FMA, max/sum reduction order, exp2 and division.
WGMMA scale-d explicitly clears the first QK instruction and preserves the
live PV accumulator, including when that accumulator's bits happen to be zero.
Every later k8 instruction accumulates the preceding instruction result.
`Checked.whole_fp32_if_local` transports the entire byte result under nine
instruction-local laws. No arbitrary dot, reduction, final accumulator, or
whole-output equality is a premise. The host scale preserves Python's ordered
binary64 power/division/multiplication through the existing host-arithmetic
interface. No native-to-Real equality or error bound is asserted.

Ordinary Tensor/storage observations, int32 metadata observations, native
b32 transport, compiler realization of the selected instruction profile, and
successful race-free launch delivery are explicit boundaries. `Snapshot` in
the reused proof namespace denotes a read-only observation of original bytes;
the Python wrapper does not snapshot payloads. The shared launcher's separate
grid-product overflow issue remains tracked in `bench/TODO.md`; these source
proofs do not establish runtime applicability for affected grids.

Validation uses targeted Lean builds with standard-base axiom and independent
target audits. `check_contract.py` verifies that removing the explicit int64
index widenings leaves the original floating-source AST exactly unchanged.
It executes the actual repaired source in Triton's CPU interpreter, with a
labeled CUDA metadata/context facade, covering both tiles, multiple widths,
65-row tails, grouped heads, cached prefixes, shared cache views, finite
sentinel behavior, backing frame, empty calls, device restoration and rejection
before launch. No GPU run, new compiler audit, or numerical conformance result
is claimed.

`Correct.lean` has one public specification, `PublicKernel.correct_real`, using
shared `AlgorithmProgramIO.ImplementsOn`. `RealIO.lean` names the exact original
FP32 source with Real dtype erasure, six finite typed input windows, all active
output lanes and the ordinary grid stage. The independent base-2 formula and
the original nonnegative formed-address and unique-writer conditions are
explicit. `Proofs/RealLocality.lean` proves that only the declared finite input
windows affect this formula. The ordinary grid retains its original semantics:
every CTA reads the same initial arrays, then disjoint output frames are merged.
`Proofs/RealIOComplete.lean` connects source execution, actual access safety,
every output and the full raw-cell exterior frame.

`Misc/NativeCorrect.lean` retains the ordinary theorem `PublicKernel.correct_fp32`,
using shared `ArrayProgramIO.Observes` and the existing fixed SM90 primitive model.
`NativeIO.lean` exposes six named typed arrays, finite dynamic read footprints,
exact store masks and addresses, and the actual original-source launch list.
`NativeFormula.lean` gives an independent ordered array recurrence. Arbitrary
finite launch lists, duplicate output addresses, and aliases between data,
output and integer metadata remain admitted. An earlier output may change a
later launch's reads; an aliased real-cell store makes a later integer read
observe the existing typed default zero. Overlapping stores observe the final
writer. Access safety follows actual source successors, while successful
execution, native outputs and frame hold independently of capacity bounds.

This native theorem observes a source under its established primitive model;
it does not introduce a second optimization kernel or a floating algebraic
rewrite. The existing primitive and compiler boundaries remain unchanged.
There is no new numerical assumption, finite-word restriction, native-to-Real
claim, numerical experiment or GPU result.

Operational details and optional raw statements are in `Proofs/`. The original
per-program and full flat-grid Real signatures and arbitrary raw native
singleton/grid signatures are retained as ordinary theorems in
`Proofs/RawCompatibility.lean`, exported by `TensorBridge.raw_real`,
`raw_real_grid`, `raw_fp32` and `raw_fp32_grid`. Native raw consequences consume
the shared IO theorem through exact original schedule transport. Real flat
consequences reuse the same existing `EpilogueResult.correct` and
`WholeSafety.complete` execution/safety foundation and existing flat transport.
No source execution is reproved for the optional facade.

The checked full-grid Tensor/byte results remain `TensorBridge.correct_real`
and `TensorBridge.correct_fp32`; the input binding remains
`TensorBridge.input_tensor`. Supplied output identity, None return, complete
physical frame, offsets/strides, input aliases, index ABI and empty behavior are
preserved. The redundant historical reader import was removed from
`Proofs/Real.lean`, so these live helpers no longer depend on that public reader.


The 2026-10-21 complete program pair is
`FPEquiv.block_end_factored_equiv`: `original ≡[[]] blockEndFactored`.
`Kernels.lean` directly quotes both complete FP32 programs. The candidate
factors the common prompt offset out of `minimum`, replacing
`min(block_start+BM+prompt, sequence+prompt)` with
`min(block_start+BM, sequence)+prompt`. This saves one integer addition per CTA;
no measured performance improvement is claimed. The existing scale-first
candidate is retained historically and is not needed for this strict index law.

`SourcePair.original_preserved` proves complete equality with the original
shared FP32 program. Exact integer ordering under translation gives complete
state equality and source-safety equivalence; all FP32 operations, finite
sentinel, exp2, probability cast and live dot seed remain. `PairIO` and
`PairEquivalence` retain every finite launch list, repeated jobs, all raw typed
array aliases (including metadata/output aliases), evolving read windows,
every active final output and both exterior frames. The only additional
explicit execution condition remains the original positive BN; dot support is
the existing primitive-model interface. There is no finite-word restriction,
floating rewrite, experimental rule or new numerical validation obligation.
The independent Real `Correct` formula, ordinary grid and original address/
ownership domain are unchanged. Optional native results retain all consumers.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/Public.lean](Misc/Public.lean), [Misc/Empty.lean](Misc/Empty.lean), [Misc/KernelOnly.lean](Misc/KernelOnly.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.

[Misc/SourceCompatibility.lean](Misc/SourceCompatibility.lean) retains the optional
`PublicKernel.shared_source` proof alias; it is not a second kernel specification.

`InvocationIO.lean` holds the shared `NativeArrayBridge.invocation` ABI. Finite ordinary Real IO and evolving raw-array native IO remain distinct views in `RealIO.lean` and `NativeIO.lean`; the index-inverse lemma lives with the binding proofs.
