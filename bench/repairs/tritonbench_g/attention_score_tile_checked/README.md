# attention_score: checked static tiles and literal Triton bodies

`Kernels.lean` contains the complete function body of `_score_kernel` from
`attention_score_tile_checked.py`, plus the complete body of
`_score_scalar_left_kernel` from `attention_score_scalar_left.py`. Each quote
has the same 21 top-level Python statements. Caller types, runtime arguments
and constexpr settings live outside `triton { ... }`; source casts, block
pointers, masks, branches, loop bounds and stores remain inside it.

`attention_score_scalar_left.py` is an authored counterpart of the existing
scalar-left Lean rewrite, using the independent score-sum specification and
the unchanged selected Python source. Its only numerical change is
`qk * qk_scale` to `qk_scale * qk`. It is not recovered historical Python.
`jit-correspondence.json` and the body checker record both mappings and caller
bindings. The original selected Python file is unchanged.

The execution proofs use these actual signed-index quotations. In particular,
the key block offset retains `.to(tl.int32)`; its exact natural-address
projection follows from the existing nonempty-launch padded-key bound.
`Proofs/Reused/IndexModel.lean` contains compatibility aliases to the actual
quotations. It does not substitute a separately handwritten natural-index
program. `SourceDecomposition.lean` derives proof slices from the complete
source decomposition. The public Real, FP and raw endpoint statements keep
their existing domains, full output ranges and exterior memory frames.

The main `FPEquiv.lean` compares the two complete source kernels using the
reached-site finite qk/scale domain and the strict FP32 multiplication
commutativity certificate. It uses no experimental numerical rule. Its
source-kernel result is uniform in program IDs; optional native execution and
Tensor endpoints remain in `Misc/`. Native instruction models and source
correspondence are separate from hardware conformance claims.

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

The tile limit was checked against the pinned Triton 3.7.1 sources during the
original host-guard review. Its old delivery packet lived under
`.veritile-cache/kernel-evidence/attention_score`; that historical local packet
is not a current build input or a reproducible receipt in this checkout.
Current builds use the source files tracked in Git. The source-only recovery
and its limits are recorded in
[`source-closure`](../../../review-checkpoints/2026-10-23/source-closure/README.md).
Triton API references: [make_block_ptr](https://triton-lang.org/main/python-api/generated/triton.language.make_block_ptr.html),
[dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html),
[load](https://triton-lang.org/main/python-api/generated/triton.language.load.html).

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

`Kernels.lean` owns both complete quotations and imports no proof module.
`SourceParameters.lean` owns caller parameters and shape data;
`SourceDefinitions.lean` is a compatibility import. `Proofs/SourceBridge.lean`
contains execution and tile-formation connections. The selected original
Python JIT AST is unchanged. `Contract.lean` states the existing
input/range/address contract plus the exact conditional tile guard and proves
empty-domain preservation.

`IO.lean` owns the Real source projection, Q/K/M/Out channels, active masks,
complete original schedule and logical tensor views. `RealIO.lean` is a thin
compatibility import. `Misc/NativeIO.lean` declares the same authoritative
typed source and schedule, exact input-cell encoding and graph observation.
`Correct.lean` displays the independent finite score sum through the logical
IO judgment; `Misc/NativeCorrect.lean` retains the ordinary
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

`Proofs/Reused/` retains the isolated namespace of the existing attention-score
proof chain. Its source execution lemmas now follow the literal signed-index
program, including the retained key-offset cast, loop and output cast.
Shared access-safety helpers live under `VeriTile/Triton/Memory`; they do not
load historical attention-forward proof objects from a cache.

The executable grid model uses ordinary prepared/zeroed output storage and
source jobs. Theorems assume no whole-kernel result, successful execution or
host return. Allocation/API implementation, device resources and general
compiler conformance remain explicit model boundaries. The postponed shared
grid-product issue is unchanged. There are no deferred numerical experiments.

## Validation and historical evidence

`bench/tests/test_attention_score_body.py` checks both complete mappings and
mutation failures for deleted casts, masks, loop statements and changed caller
bindings. Syntax correspondence is separate from the Lean execution, safety,
full-grid formula, source-pair equivalence and axiom checks.

`check_candidate.py` retains the original host-wrapper checks: 16 wrapper
cases and 8 JIT binder captures, including tile boundaries, empty branches,
grouped heads, windows, read aliases/storage offsets and the single 64-to-32
retry. Those checks replace CUDA classification/context and the launcher;
they do not run a GPU or interpreter kernel. Torch/Triton availability is
reported separately in the current validation receipt.

Earlier host-guard and IO-format reviews covered their then-current source
and imported objects. Their counts, compiler controls and saved cache hashes
do not certify this literal-body migration. The current migration rebuilds
the changed proof chain from tracked source and records focused and cumulative
results separately. No new numerical experiment or hardware-conformance
claim follows from a successful Lean build.

An additional replay of the unselected historical attention-score modules
found four existing failures: `attention_score.Proofs.RealProgram`,
`attention_score.Proofs.LegacyNaturalWindow`,
`attention_fwd_triton3.Proofs.RealHeaderSource` and
`attention_fwd_triton3.Proofs.FPModel`. All four reproduce on the preceding
Liger source revision with identical error headers. None belongs to this
selected repaired task's import closure. They remain recorded as failed
historical diagnostics; this migration does not certify those old versions.
