# attention_fwd_triton1: live carry-tile check candidate

This separate candidate repairs the accepted static-tile admission defect in
`attention_fwd_triton1/Whole001/attention_fwd_triton1_repaired.py`. The original
42 task files, original canonical source selection and frozen diagnostic
packet are preserved. Source selection and acceptance are recorded in the
shared ledger; the IO-format pass does not change either.

The retained original wrapper adds only the check `dim * dim > 1048576` followed by a
`ValueError`, after the original allocations and empty return. The existing
power-of-two check means the actual binder passes `BD = dim = next_power_of_2(dim)`.
Thus the new live-domain condition is exactly `BD * BD <= 1048576`. Empty B/H/T
calls retain their original behavior. All of that wrapper’s other guards, allocations, flags,
launch options, precision sites and kernel arithmetic are unchanged. A separate
address-optimization candidate is described below.

`Contract.lean` has the original checked input/setup conditions plus this
empty-aware tile check. Its equivalence, empty preservation and live bound
are proved. The bound addresses Triton 3.7.1's static frontend element limit;
it does not assert that every admitted launch fits every GPU's resources.

## Proof entry points

- `Correct.kernel_real` states the shared `ProgramIO.Implements` judgment for
  the independent attention sum and prefix history formula.
- `Misc/NativeCorrect.lean` retains the ordinary auxiliary theorem
  `FPEquiv.kernel_native`, with its original full name and complete statement.
  It states `ProgramIO.ImplementsOn` for the independent ordered FP32 reference.
  Its numerical premises are the existing local mulRN/fmaRN conformance
  equations and reached operand domains.
- `FPEquiv.grid_equiv` compares the complete original and `shiftedSource`
  schedules with `≡[[]]`. For the checked fixed BT=32 launch, the candidate
  replaces four temporal block-pointer multiplications `i * 32` by `i << 5`.
  `Proofs/Pair.lean` proves exact state and safety preservation for every chunk
  and head, all input payloads, both flags, all outputs and exterior cells.
  The historical scalar-left floating candidate remains separate; its finite
  numerical-site condition is not imposed on this new full-domain pair.
- `TensorBridge.Correct.kernel_correct` in `Misc/TensorBridge.lean` proves complete shared execution,
  actual Tensor logical output and pre-update history reads, exact caller-cell
  frames, checked machine arithmetic, and the static live tile bound.
- `TensorBridge.FPEquiv.kernel_equiv` in `Misc/TensorBridge.lean` proves the corresponding actual
  `Torch.FP32.logicalValue` readback against the concrete source FP32 profile.
  Its numerical premises are local mulRN/fmaRN conformance and reached operand
  domains. `TensorBridge.FPEquiv.source_fp32` gives a closed software-profile witness.
- `Misc/Audit.lean` checks the theorem chain's axioms and non-circular statements.

The independent Real result is chunked linear attention:

```
O[head,r,j] = (1/sqrt(D)) *
  sum(u < min(T, (r/32+1)*32), sum(e < D, Q[head,r,e]*K[head,u,e]*V[head,u,j]))
H[head,chunk,i,j] = sum(u < min(T, chunk*32), K[head,u,i]*V[head,u,j])
```

It has no softmax and no token-causal truncation within a chunk. Native FP32
uses the actual host/JIT scale word, retained casts and an explicit ordered
scalar FMA interpretation. It does not assert FP32 equality to ideal Reals.

## Connection to the source and actual storage

`source` aliases the entire unchanged canonical typed quote. `Program.quoted`
is a definitional equality to all of its statements. `SharedStep.body_run`
and `SharedLoop.kernel_run` prove successful execution in the shared
`FP.Structural` interpreter, including pointer formation, zero-filled tails,
pre-update H stores, all branches, six source FP32 casts, dots with TF32
prohibited, and final O stores. `SharedGrid` threads complete prior states
through every head. `Schedule` additionally proves execution, output,
history and frame properties for arbitrary head orders.

The symbolic `Term` graph keeps explicit dot and addition nodes separate.
`Term.evaluate` defines the selected scalar lowering, including carry-seeded
FMA fusion of source add/dot pairs. `Projection.graph_output` and
`Projection.graph_carry` prove this graph interpretation equals the independent
input-only recurrence; no whole-result equation is an assumed compiler gate.
The subsequent Real and scalar-profile transports reuse the unchanged
canonical algebraic results.

`Payload` contains an actual `Torch.FP32.Tensor`, its ordinary Tensor API laws
and the observed conjugate flag. `TensorBinding` records actual pointer
quotients and dense output/history metadata, including the input device and
ordinary view flags. Admission derives pointer
alignment; Tensor API laws derive byte addresses and logical reads. The
existing ABI requires fresh O/H regions and allows Q/K/V aliases. Physical
realization writes only the proven O and enabled-H cell starts. Every other
caller cell retains its exact tag and payload, including non-FP32 cells.
Disabled H and input regions are framed explicitly. Public O object identity
is retained.

These Tensor application endpoints begin after ordinary allocation has supplied the
five launch tensors. No allocator receipt, host execution, successful kernel
execution, final heap or returned-result equality is an endpoint premise.
Python execution, Torch API implementation and compiler/runtime conformance
remain ordinary external bindings; the internal quote-to-graph and
quote-to-handwritten-recurrence connections are proved.

## Validation and evidence

`check_candidate.py` checks whole-module AST equality after removing the sole
new guard, all 42 preserved files, and 17 public-forward cases. Live D=16 through
1024 reach the installed actual JIT binder; D=2048 and 4096 reject. Empty B/H/T
cases at D=2048 preserve allocation order and O identity. Four T=33/D=32 flag
combinations and the T=1/D=16 scalar-one specialization compile offline with
Triton 3.7.1 for SM80. TTIR/TTGIR/LLVM/PTX artifacts retain dots and FP32 FMA
without TF32. Actual host scale values and their FP32 words are recorded.

The harness uses ordinary CPU Torch allocations with CUDA classification,
device context and the final launcher replaced. Boundary D=1024 is an
admission/binder check, not a resource-capacity compilation claim. No GPU or
kernel interpreter executes, and no numerical experiment is claimed.

Evidence, build receipts, resolved import closure, reproduction scripts and
the frozen candidate delivery are under
`.veritile-cache/kernel-evidence/attention_fwd_triton1_tile_checked/`. Builds
use `workspace_lean_local_first_v2.py` with normal escalation. Existing
repository and external cached dependencies are pinned separately from fresh
task module builds; the packet does not claim a cold rebuild of the library.

## Separate kernel interfaces

`IO.lean` declares the actual quote, full head schedule, Q/K/V input windows,
O and optional H output windows, exact masks, symbolic cell type, encoding and
native/Real observations. There is no private scratch. `RealIO.lean` and
`NativeIO.lean` expose independent logical formulas and the selected precision
model. The symbolic `.real` cell carrier retains the explicit FP32 source
loads, casts and computation; the physical endpoints materialize FP32 cells.

`Proofs/IOPhaseSafety.lean`, `IOComputeSafety.lean` and `IOProgramSafety.lean`
prove bounds for every active source load/store at its actual predecessor
state. `IOInterface.lean` joins that safety with the existing complete grid
execution, all O/enabled-H outputs and exact exterior frame. Empty schedules,
partial final chunks, both flags and arbitrary initial register contents are
covered. These are proved program obligations, not numerical assumptions.

The earlier raw ABI conclusions remain in `Proofs/PhysicalReal.lean` and
`Proofs/PhysicalNative.lean`. The complete checked Tensor results, including
machine closure, static tile admission, input/disabled-history frames and the
all-word software witness, retain their original statements and domains in
`Misc/TensorBridge.lean`. They use the same execution and projection foundations;
no replacement Python or kernel is introduced.

The Correct and retained NativeCorrect proof closures are reader-independent. The optional
legacy physical/Tensor endpoints retain this protected import boundary:
`Admission → attention_fwd_triton1.Contract → Whole001.Proofs.Host →
Whole001.FPEquiv → Whole001.Correct`. The protected contract and frozen files are unchanged.
`Proofs/NumericalReference.lean` extracts the unchanged two scalar laws and
reached domains for the live IO closure. `NumericalCompatibility.lean` proves
that the scalar/output definitions coincide and the numerical structures are
equivalent; it is imported only by the optional legacy endpoints.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean), [Audit.lean](Misc/Audit.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

## Fixed-tile address optimization

The new `attention_fwd_shifted_kernel` and `_forward_shifted_with_state` retain
the complete checked launch and FP32 computation. The four temporal offsets use
a shift because BT is fixed at 32; the history offset `i * BD` is unchanged.
This is an integer address strength reduction. No measured speedup is claimed.
`check_pair_source.py` checks exactly these four kernel changes and an otherwise
identical checked wrapper without importing Torch or executing a GPU.

The main equivalence is proved for the retained source graph and any shared
primitive observer. It uses exact natural-number shift arithmetic and no
floating numerical law, finite-value restriction or experimental admission.
The independent Real formula and optional native/Tensor theorems retain their
previous domains. The native conformance premises of those auxiliary theorems
are not premises of the new program pair.
