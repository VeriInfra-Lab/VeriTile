# attention_score: independently checked static-tile candidate

Preserved raw endpoint readers are also in Misc: `Misc/RawCorrect.lean`, `Misc/RawNative.lean`. Their full declarations are unchanged.

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/PhysicalIOBinding.lean`, `Misc/NativeCorrect.lean`.

Current source-pair review (2026-10-21): The current `FPEquiv.lean` headline compares the complete selected repaired FP32 `original` source with the already specified `scalarLeft` source. It uses the retained reached-site finite qk/scale domain and the shared strict FP32 MUL-COMMUTE software certificate; actual experimental rules used: zero. `PairIO.lean` exposes the programs/domain and `Proofs/SourcePair.lean` proves the source-specific context. The previous unrestricted full-grid `kernel_native` declaration keeps its full name, statement and proof as an ordinary theorem in `Misc/NativeCorrect.lean`. The main pair is a source-kernel equivalence uniform in PID, not a newly proved native whole-grid pair or a hardware FMA/TF32 claim.

This sibling adds one guard to the selected repaired Python source:
`block * d <= 1048576` before each nonempty launch attempt. The JIT body, its
heuristics, input objects, allocation, window handling and existing single
`OutOfResources` retry are unchanged. The previously selected source and its
96 historical files remain frozen; its admitted D32768/B64 counterexample is
still a recorded source-domain defect.

The added condition follows from every source tensor shape. Query and key
block pointers/loads have `[B,D]` and `[D,B]` tiles. Dot, distance, mask,
probability and broadcast results have `[B,B]`; vectors/expanded vectors and
scalars have `[B]`, `[D]`, `[B,1]`, `[1,B]` and `[]`. Existing B in {16,32,64}
bounds B² by 4096, so B*D<=2²⁰ suffices for the static tensor element limit.
This is a source-formation requirement, not a per-device resource guard.
B16/D65536, B32/D32768 and B64/D16384 retain their own boundaries. Empty query
or empty output returns still skip the launch checks. The guard raises
`ValueError`, so it is not reclassified as a resource retry.

Installed reference: Triton 3.7.1 `_utils.TRITON_MAX_TENSOR_NUMEL` and
`validate_block_shape`; `language.core.block_type` invokes that validator,
and `language.semantic.make_block_ptr` constructs the block type. Their exact
source hashes and two original actual-binder/offline SM80 compiler controls
are pinned in the frozen original diagnosis:
`../../../../.veritile-cache/kernel-evidence/attention_score/delivery.json`.
Triton API references: [make_block_ptr](https://triton-lang.org/main/python-api/generated/triton.language.make_block_ptr.html),
[dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html),
[load](https://triton-lang.org/main/python-api/generated/triton.language.load.html).
Core `VeriTile/Torch/FP32.lean` records the Tensor API/storage binding references.

## Mathematical target and precision

For Q[Z,H,N,D], K[Z,HK,NK,D], M[Z,H,R] with R>=N, HK>0 dividing H,
the output at (z,h,j) is

```
sum i<N, allowed(i,j):
  exp2((sum d<D: Q[z,h,i,d] * K[z,h//(H/HK),j,d])
       * ((1/sqrt(D))*1.4426950408889634) - M[z,h,i])
```

No window permits all pairs. The ordinary window permits
`0 <= i-j+offset < size`; source complement mode permits
`i-j+offset >= size`, not the Boolean complement of that interval.
This formula is independent of execution, program blocks and returned memory.

Real correctness is ideal arithmetic. Native inputs/output remain FP32;
QK uses the selected TF32 dot model, a fixed m16n8k8 graph chain, FP32 scaling,
the observed scale-first FMA with sign-negated M, `ex2.approx.ftz.f32`, and the
existing explicit B16/32/64 column reduction graph. Scalar/fixed-MMA primitive
interpretations are explicit model data. The runtime scale parameter retains
the actual launch scalar and its FP32 interpretation; there is no assumption
that Python sqrt/divide equals exact Real reciprocal sqrt. No native-to-Real
exactness law, finiteness restriction or numerical experiment obligation is
introduced. Narrow saved compiler profiles are evidence for the chosen
lowering model, not universal CUDA/compiler conformance proofs.

## Proof organization and boundary

`Kernels.lean` exposes the existing readable widened quotation; the actual
selected Python JIT AST is unchanged. `Contract.lean` states the existing
input/range/address contract plus the exact conditional tile guard and proves
empty-domain preservation.

`RealIO.lean` declares the exact Real source projection, Q/K/M/Out flat
channels, identity windows, active masks and complete original job schedule.
`Misc/NativeIO.lean` declares the same authoritative typed source and schedule,
exact input-cell encoding and graph observation. `Correct.lean` has one public
`AlgorithmProgramIO.ImplementsOn` specification; `Misc/NativeCorrect.lean` retains the ordinary
`ProgramIO.ObservesOnCells` theorem. These shared judgments include
actual source access safety, successful complete execution, every output and
exact raw exterior memory preservation. Main signatures retain the prior
machine-range domain and tile bound. They expose no raw state or returned-memory
existential.

`RealInterface.formula` is the independent finite mathematical sum using the
supplied scale. `Misc/NativeReference.lean` is an independent source-ordered graph
formula, including the existing TF32 dot, scalar operations and ordered
reduction. `Proofs/RealIOProof.lean` and `Misc/NativeIOProof.lean` connect these
formulas to the selected full source grid. `Proofs/NativeSafetySteps.lean`,
`NativeSourceSafety.lean`, `NativeSafetyLoop.lean` and `Misc/NativeIOSafety.lean`
prove the actual source reads/writes safe through their reached states; access
safety is a proved conclusion, never an input or numerical assumption.

For empty dispatch only, Out is also an explicit input channel and its domain
requires the original preallocated zero value. The schedule is then empty.
Nonempty execution reads only Q/K/M and writes all output elements. Native
empty initialization retains the original staging literal zero; all nonempty
input words remain unrestricted. This convention introduces no new initializer
or wrapper implementation.

`Misc/PhysicalIOBinding.lean` binds the original zero-initialized physical state to
both IO interfaces, supplies concrete bounds witnesses, and preserves every
read alias through the same physical reader and ABI. `Misc/RawCorrect.lean`
and `Misc/RawNative.lean` retain the exact old physical endpoints, including
machine closure, source tile formation, all input words and the exterior frame.
They use the same existing source execution foundation as the new IO proofs.
`Misc/TensorBridge.lean` retains the complete actual Tensor theorems and derives them
from these helpers; it is optional and is not imported by the public entries.

`Proofs/Tensor.lean` binds the same observed objects to the shared core Tensor
API, storage offsets, capacity and logical addresses. `Proofs/Memory.lean`
connects exact input bytes to the FP32-profile load channel and serializes the
executed output graph to FP32 bytes. Read aliases and nonzero contiguous
storage offsets remain permitted; the prepared output allocation is fresh.
The Tensor application theorems explicitly cover nonempty launches. Raw and
public IO results also retain their original empty branches.
`TensorBridge.empty_query_target` states the zero mathematical target; empty
output has no element to prove.

`Proofs/Reused/` is an isolated namespace port of the necessary existing
attention-score proof chain. This was required by a demonstrated backend ABI
mismatch: the old cached `NativeStaging.algebra.fp32Load` fails its current-field
probe. The port rebuilds the source proofs against the current core without
changing core, historical sources, or their recorded acceptance. It does not
add a separate Python implementation or handwritten source AST.

The executable grid model uses ordinary prepared/zeroed output storage and
source jobs. Theorems assume no whole-kernel result, successful execution or
host return. Allocation/API implementation, device resources and general
compiler conformance remain explicit model boundaries. The postponed shared
grid-product issue is unchanged. There are no deferred numerical experiments.

## Scoped validation

`check_candidate.py`: 16 actual-wrapper cases, 8 actual JIT binder captures;
three block boundaries and rejections, all empty branches, grouped heads,
window/complement, read alias/storage offset, and the single 64-to-32 retry.
Only CUDA classification/context and launcher are replaced; no GPU or
interpreter kernel runs. Two existing compiler controls are reused after
checking unchanged source AST and artifact hashes. New compilation experiments
are unnecessary for this host-guard-only change.

The frozen delivery packet records targeted Lean builds, current backend ABI
probe, axiom/noncircular audits, source/import hashes, preserved baseline and
these validation limits. Counts of built modules are not task-completion counts.

All 61 candidate Lean modules have fresh successful build receipts; the 54
reused modules differ from their preserved sources only in import paths and
namespace qualification. `import-provenance.json` records three historical
attention-fwd cache/source differences, recovered Git sources matching their
saved Lake checksums, and exact loaded object/interface hashes. A conservative
type/body/projection audit finds no dependency of either public endpoint on
attention-fwd declarations. The retained access-safety proof uses the listed
generic helpers and does not depend on the three changed historical modules.
The packet does not claim that their current source produced those caches.

## Public IO format validation (2026-10-20)

The new Real/native IO helpers, main entries, retained raw endpoints, physical
binding and Tensor consumer have targeted current builds. Main statement-surface,
noncircular-formula and standard-base axiom checks pass. Python, `Kernels.lean`
and `Contract.lean` are unchanged. This formatting work adds no numerical
assumption and performs no new compiler/GPU experiment or independent replay.


Source-only organization (2026-10-22): `SourceDefinitions.lean` now owns the
three unchanged complete ordinary FP32 quotations (original, scalar-left and
widened). `Kernels.lean` imports that source authority explicitly;
`Proofs/Reused/Kernels.lean` remains a compatibility import. All complete
source functions are preserved by `rfl`; public declarations, input domains,
precision, candidate, numerical rules and existing completion status are
unchanged. Geometry, execution and source-shape proofs stay under `Proofs/`.
