The raw Python body retains both explicit query-half casts to `sin1.dtype`.
The ordinary FP32 stores do not contain a further cast. The proof execution
summary follows that exact order; it does not assume arbitrary casts are
identities. The historical manually typed source contains two extra output
casts, so `fast_rope_embedding/Proofs/LegacyStoreView.lean` records only its
syntactic difference. Current execution, safety, Real and native proofs use
the raw source directly. The independent mathematical formula, native word
formula, public input domain and original/candidate bodies are unchanged.

# Fast RoPE: checked FP32 kernel

`Correct.lean` and `Misc/NativeCorrect.lean` expose the unchanged `_rope_embedding`
source through the shared `ProgramIO` judgment and independent logical arrays.
`IO.lean` displays Q's in-place two-half tile, separate cosine/sine rows, the
four-head-group mask, source strides, direction, and actual source invocation.
Both statements prove active-access safety, complete source execution,
independent outputs, and every unwritten typed cell.

`Proofs/AccessSafety.lean` checks both coefficient loads and each iteration of
the dynamic head loop in its actual predecessor state, including both Q loads
and both stores. `Proofs/Interface.lean` reuses the existing source/context proof;
`Proofs/RealIO.lean` and `Proofs/NativeIO.lean` connect independent scalar pairs.
The former raw statements and coverage results remain in `Proofs/RawReal.lean`
and `Misc/RawNative.lean`. Only the active row-stride product is constrained,
so arbitrary unused singleton strides remain supported. Native inputs retain
all FP32 words and arbitrary product markers; source loads/casts reset those
markers and the resulting exact native pair is stored with no product marker.

`Misc/TensorBridge.lean` retains the complete admitted Tensor applications,
including logical outputs, all four native bytes, access bounds, and physical
frames. Its existing binding layer now uses that same raw source proof.
The previous host proofs remain in `Proofs/HostContract.lean`,
`Misc/HostCorrect.lean`, and `Misc/HostFPEquiv.lean`.

For row `r`, head `h`, and `j < floor(D/2)`, let `a=Q[r,h,j]`,
`b=Q[r,h,j+floor(D/2)]`, and take `c,s` from coefficient row `r % S`.
The independent Real target is `(a*c-b*s, b*c+a*s)` in forward and
`(a*c+b*s, b*c-a*s)` in backward. Both input halves are loaded before either
store. The odd final channel is unchanged. For fixed current coefficients,
backward is the adjoint; no unit-circle assumption is made. For example,
`(a,b)=(1,2), (c,s)=(2,0)` gives `(4,8)` after forward then backward, so the
operation is not generally its inverse. Backward reads the current coefficient
objects, not an implicit forward snapshot.

`Contract.lean` records the actual Tensor metadata, pointers, row strides, and
ordinary indexing laws. Admission follows the existing FP32/device/layout,
packed-table, separation, launch-size, padded-i32-offset and u64-pointer checks.
The source Q stride is never silently replaced with `H*D`: contiguity and
ordinary row indexing derive equality of the *active row product*.
A singleton row may have any nonnegative stride. The machine-index proof uses
modular multiplication at either 32 or 64 bits; it does not require an unused
singleton stride to fit i32. Empty payloads and `D<2` have no declared program.

The native scalar reference preserves same-FP32 load and output casts,
coefficient-first operands, and the observed default-fusion graph:

```
t = backward ? sub(+0, s) : s
first  = fma(c, a, neg(mul(t, b)))
second = fma(t, a, mul(c, b))
```

`Numerics.operations` gives the actual typed source this explicit lowering
interpretation; it resets product tags at loads/casts. `softwarePrimitives cfg`
uses the shared FP32 `add/sub/mul/fma` semantics, with genuine one-round FMA.
It does not assume exact Real arithmetic. As a concrete distinction,
`a=c=1+2^-23, b=1, s=1+2^-22` gives first-output word `0x28800000` with FMA,
but `0x00000000` with a separately rounded multiply/subtract. Source backward
`sub(+0,+0)` and sign-bit negation of `+0` also differ. Primitive ISA semantics
and compiler conformance to the selected lowering remain explicit model
boundaries; the finite compiler observations are supporting evidence, not
premises claiming a universal compiler or GPU proof.

`TensorBridge.fast_rope_kernel_correct` combines actual
Tensor input binding, actual-stride address bounds, successful execution of
the complete typed source, every Q logical coordinate, and the full regional
and physical word frame. The Real interpretation of input words is explicit.

`TensorBridge.fast_rope_kernel_fp32` starts from one little-endian
byte heap and combines actual Tensor binding and address bounds with successful
source execution, every output word decoded from the completed byte heap,
all four bytes of every Q coordinate, the regional frame, and every byte
outside the actual store footprint. `KernelBytes.byte_initial` proves that the source-state interpretation starts
with exactly the original byte heap. `byte_store` proves that the encoding
commutes with each source store, `source_store_written` places every active
source store in that footprint, and `decode_encode` links word and byte
observations. No completed output, successful execution,
local numerical exactness, or reference-computing memory callback is assumed.

Both endpoints quantify over every valid `(row, group)` program and both
directions. Each program executes the original up-to-four-head loop and
describes the entire Tensor: its own write footprint gets the formula, other
rows/groups and odd tails retain their inputs. Coverage, cross-program
non-overwrite, and no-launch theorems accompany the endpoints. They do not
claim execution of the Python launch or autograd engine.

`Kernels.lean` contains the readable full source quote.
`KernelRegressions.original_typed_source` proves its exact identity with the
retained original typed fast JIT. `KernelSource.lean` reuses the full-loop
source proof with actual strides; `KernelMemory.lean` and `Misc/KernelResult.lean`
derive the Tensor and memory observations; `KernelAccess.lean` checks formed
padded offsets and both possible Q stride integer widths. The mathematical
and native targets pass noncircularity checks, and public and supporting
theorems are audited against the standard Lean axiom base.

Validation is scoped to this task. `check_contract.py` checks the unchanged
JIT/settings/outer-apply ASTs and replays eleven sequential Q/K forward cases,
both backward callbacks, eleven rejection/cache cases, current coefficient
mutation, reshape reuse, exact `None` results and backing frames on CPU.
`check_kernel_edges.py` replays the actual JIT on ten singleton-stride/direction
cases, including `2^63-1`, checks odd tails and backing bytes, and records the
FMA, non-inverse, signed-zero and rejected-offset examples. These use CPU
adapters and do not establish GPU numerical conformance.

`check_cuda_lowering.py` compiles this exact fast JIT offline with Triton 3.7.1:
SM80 `(D,H)=(35,7)` and SM90 `(64,5)` with i32 Q stride, plus SM80 `(35,7)` with
i64 Q stride, each in both directions. All six PTX profiles pass checks of
both complete ordered stored-expression graphs, six masked FP32 accesses and
absence of FTZ modifiers. The fast source's local group constant is retained;
no extra constexpr group parameter is supplied. No GPU execution is claimed.

The selected production Python, its guards, shared core, Lake configuration,
and progress ledger are unchanged by this kernel-only proof update.

`Misc/InterfaceChecks.lean` is the optional reader audit. It imports the public
entries and Tensor application; supporting modules under `Proofs/` do not.

## Implementation-pair review (2026-10-21)

The source-to-native results above are retained as ordinary theorems in
`Misc/NativeCorrect.lean`; their full names and statements remain unchanged.
`FPEquiv.lean` records the pending pair status without importing the optional
native theorem. It has no implementation-equivalence headline. Status: `proof_connection_pending`.

Original task provides head-group integer commutation across 27 storage triples; selected repaired source is FP32 with runtime strides and grouped in-place IO. No candidate or source/IO connection for this selected view is present.

The independent `Correct.lean` mathematical target and its complete existing
input domain, source precision, launch coverage and frame results are retained.
No missing program connection is treated as a numerical assumption.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.

The primary `FPEquiv.lean` compares the readable FP32 original and retained
head-group commutation candidate from `Kernels.lean` on exactly the existing
per-launch logical IO scope. `Proofs/IOPair.lean` uses the shared
`ProgramIO.Equivalent` judgment: both actual source kernels are safe and
succeed, their active typed output cells agree, and each preserves the complete
exterior memory frame. All FP32 words and both directions are admitted under
the retained launch and stride conditions. The numerical backend is the
existing explicit native lowering profile; the integer rewrite itself holds
for every floating algebra and uses no numerical rule.
`Proofs/SourcePair.lean` proves complete identity with the retained sources and
preserves execution and trace safety. `Proofs/ProgramPair.lean` retains arbitrary
launch-list execution/safety composition as an internal helper. The primary
experimental rule set is empty.


Function-body copy migration (2026-10-25): `Kernels.rope` and its retained
head-group product candidate now contain the actual selected Python bodies,
including the docstring, `ROPE_GROUP_SIZE = 4`, both directions, the dynamic
four-head loop, both `.to(sin1.dtype)` casts and both writes. Pointer storage,
four runtime integer call values and shape/direction/block constexpr roles
are supplied outside the quotation. The original Python is unchanged;
`index_commuted.py` is clearly marked newly authored Python for the existing
Lean candidate, not a historical reference.

The source frontend evaluates immutable constexpr Nat division by a positive
literal at call time, while retaining runtime division and reassigned values.
It resolves a value's `.dtype` using the retained source precision rather than
its Real carrier, before direct-load assignment dispatch, preserving both the
load precision and explicit cast. Complete-program regression checks distinguish
constexpr/runtime/mutable inputs, preserve FP32/half/bfloat16 load conversions,
and reject unknown native dtype targets. SourcePair's full original/candidate
program equalities pass for both direction flags and every old public argument.
All task modules, including complete Real/FP, loop/safety/output/frame, native,
physical bytes, host failures and Tensor scopes are replayed from those actual
bodies. No numerical rule or cast-identity premise is added.

Fast RoPE assigns the group-size register; the other RoPE takes a constexpr
parameter. `Proofs/Source.lean` proves this actual prelude and full execution
separately, sharing only the definitionally identical loop and mathematical
helpers. Their whole source ASTs are no longer asserted to be identical.
