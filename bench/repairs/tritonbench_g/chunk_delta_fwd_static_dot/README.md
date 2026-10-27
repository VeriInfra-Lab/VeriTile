# Python body correspondence

`../chunk_delta_fwd/Kernels.lean` contains the complete raw selected Python
body and its existing address-grouping candidate. This directory's
`Kernels.lean` explicitly shares those two definitions. `callSignature` supplies
the FP32 pointer types, runtime strides and constexpr arguments outside the
quotations. Block-pointer destinations are named as output ports; conservative
read-port collection also counts descriptor construction bases. The public IO
retains the exact four input channels, three outputs, optional masks and frames.

The quoted bodies preserve the tuple program-ID assignment, both state flags,
both `range(...)` loops, positional block-pointer arguments, boundary tuples,
all pointer/value dtype queries, explicit casts and both `allow_tf32=False` dots.
The original `chunk_delta_fwd_static_dot.py` and wrapper are unchanged.
`candidate_kernels.py` is newly written Python for the already established Lean
candidate: only three state/history integer address products are regrouped.
The original function signature and all autotune configurations are retained.
`check_pair_source.py` compares the complete function AST; the correspondence
manifest checks both verbatim quoted bodies and caller bindings.

Proof decomposition follows the actual tuple prefix and counted loops.
`StructuralLoopNormalization` connects counted loops to the existing unit-stride
range proofs, including execution failure and trace safety. The canonical
program is never rebuilt from a proof-only statement-list transformation.
`TypedStores` resolves known value `.dtype` queries while retaining every cast;
negative guards reject loop-carried dtype changes. No cast identity or floating
arithmetic rewrite is introduced, and public domains and numerical models stay
unchanged. Validation and its exact source hashes are in the latest body-review
receipt; earlier CPU/compiler observations below are historical evidence.

Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

Current two-program result (2026-10-22):
`FPEquiv.chunk_delta_addressFold_equiv` compares the complete original selected
kernel with `Kernels.addressFold`, with empty experimental rule set `≡[[]]`.
At the initial, per-chunk history and final block-pointer bases, the candidate
replaces `i * K * V` by `i * (K*V)`, folding the constant product during kernel
specialization. This removes one runtime integer multiplication at each of
these sites, including each history-loop iteration. No performance measurement
is claimed. Both programs are complete ordinary `triton` quotations (the original
is transparently shared from the predecessor authority module).

The theorem retains every FP32 input word, all four W strides, both state flags,
the original shape conditions, every complete CTA permutation, both dots and
their exact FP32 casts and lowering/FMA order. It concludes access safety,
successful complete executions, all history/corrected/optional-final output
cells, and both exterior frames. `PairIO.lean` contains the concrete two-program
views. `Proofs/PairContext.lean`, `PairSource.lean` and `PairCorrect.lean` prove
compositional source execution/safety equality and connect it to the retained
full-grid IO theorem. The source equality holds under arbitrary numerical
backends, and the main view preserves the established native scalar ISA model.
No floating numerical law or experimental assumption is used.

The preserved benchmark's `sumLeft` candidate was considered first. Under the
selected native lowering, `H + (0 + dot)` uses H as the single-subblock FMA
seed, whereas `(0 + dot) + H` materializes a zero-seeded dot followed by add.
These cannot be identified by a strict add-commutation theorem: for finite
FP32 x=`0x3f800001`, y=`0x3f7ffffe`, H=`0xbf800000`, nearest-even software
`fma(x,y,H)` has bits `0xa8800000`, while `add(fma(x,y,0),H)` has bits zero.
The local exact diagnostic is kernel-checked in the campaign delivery directory;
it diagnoses the fusion boundary and does not claim to be a whole-kernel
counterexample. The new address candidate preserves the selected domain and
all established arithmetic staging. Existing independent `Correct.lean` and
optional `Misc/NativeCorrect.lean` results retain their declared scopes.

# Delta-forward static-dot contract candidate

This is a separate input-contract candidate for `chunk_delta_fwd`. It preserves
the checked kernel computation and changes only which nonempty launches are
admitted. The original Python source, diagnosis and conditional theorem scopes in
`../chunk_delta_fwd/` remain unchanged. Its shared Lean source is migrated to
the raw function body and its proofs reconnected; accepting the selected
contract does not prove the original broader domain correct.

The Python entry is `chunk_fwd_h_fn_static_dot(k,w,u,BT,initial_state=None,
final_state=None)`, also exported as `chunk_fwd_h_fn`. Its sole new assertion is:

```python
if B * H * V and (BK < 16 or BC < 16):
    raise ValueError("launched FP32 dots require BK >= 16 and BC >= 16")
```

`BK=next_power_of_2(K)`, `BV=16/32/64` according to the unchanged BK selection,
and `BC=min(BT,BV)`. `Contract.lean` states exactly:

```lean
def StaticDot (r : Request) : Prop :=
  r.B*r.H*r.V=0 ∨ (16≤r.dims.BK ∧ 16≤r.dims.BC)
def InputContract (r : Request) : Prop := r.Admitted ∧ StaticDot r
```

`check_iff` connects the executable check to that contract. `static_dot_iff`
proves its exact dimension form: an empty B/H/V grid, or `K>=9` and `BT>=16`.
The original limits, power-of-two BC requirement, FP32/contiguity/device/view
checks, direct input pointers, alias rules and integer/address bounds remain
in force. No input copying or finite-value restriction is added.

The distinction between empty grid and zero time is intentional. When B, H or
V is zero, the wrapper skips launch and retains the old small K/BT domain.
When only T is zero, it still launches; Triton frontend checks the dot bodies
before eliminating their zero-trip loop. Actual source probes confirmed both
small-dimension failures at T=0, with and without optional states, and accepted
K=9/BK=16 controls. This is a static dot API condition, not a device-resource
restriction. The current NVIDIA FP32 minimum is `(M,N,K)=(1,1,16)`, where K
denotes the dot's reduction dimension. Correction reduces BK; update reduces
BC.

## Mathematical and native targets

The independent Real recurrence uses chunk stride BT and rounded window width
`ceil(BT/BC)*BC`, clipped at T:

```
history[c] = H_c
corrected_c[t] = U[t] - W[t] @ H_c
H_(c+1) = H_c + sum_(t in chunk c's rounded window) K[t] outer corrected_c[t]
```

Optional initial state supplies `H_0`; otherwise it is zero. Returned corrected
rows use the last visiting chunk, `floor(t/BT)`. Optional final state receives
`H_NT`. Rounded overlap is preserved rather than silently changing the formula.
`Correct.chunk_delta_kernel_correct` states `kernelIO p schedule ⊨ formula p`
for every complete CTA permutation. The shared judgment connects the actual
source to bounded logical inputs, access safety, successful full-grid execution,
the independent recurrence and the exact regional frame.
`TensorBridge.RealApplication.chunk_delta_kernel_correct` retains all Tensor
observations and the physical exterior frame.

`Proofs/NativeReference.lean` defines a separate recurrence from four logical
input readers. It contains no execution state, source-loop summary, source
output formula or output-equality premise. It retains physical padded lanes,
FP32 casts, zero-seeded correction dots, and the actual accumulation order.
The shared staged-to-instruction interpretation expands every dot into ordered
scalar FMAs. The reference's scalar expansion theorems explicitly preserve the
one-subblock live-state carry and the separate final addition for multiple
subblocks. This distinction is not justified by a split/fused equality axiom.

`Proofs/NativeBinding.lean` retains the original seed, correction, accumulator,
chunk recurrence and output connections. `Proofs/NativeIOBinding.lean` connects
the same recurrence to bounded logical inputs, without assuming anything about
unused surrounding input storage. `FPEquiv.chunk_delta_kernel_native` states a
shared source-to-native IO observation for every output word and the exact
regional frame. It does not assert an algebraic rewrite or native-to-Real equality.
`TensorBridge.NativeApplication.chunk_delta_kernel_native` retains supplied Tensor
byte addresses, optional final state, input byte frame and exterior frame.
`TensorBridge.NativeApplication.source_load_word` connects current physical input bytes to actual
evaluated FP32 source operands; raw input words are identity leaves.

## Organization and boundaries

`Kernels.lean` reuses the authoritative readable `triton` quotation from
`../chunk_delta_fwd/Kernels.lean`. The static-dot wrapper uses the same original Python kernel AST as its
predecessor; the separately named address-fold candidate differs only at the
three documented integer products. There is no parallel handwritten AST.
Canonical entries are `Contract.lean`, `Correct.lean` and `FPEquiv.lean`.
The optional native theorem is in `Misc/NativeCorrect.lean`. `IO.lean`
exposes the four input channels, three outputs, source flags, weight strides
and complete schedule. `Proofs/ReferenceTargets.lean` provides independent finite-input Real
and native formulas. Operational helpers are under `Proofs/`. Existing execution, ownership, independent Real, finite pointer and scalar
byte-store results retain their scope; the execution and safety decompositions
are reconnected to the raw tuple assignment and counted loops.

The inherited `Allocations` structure is used only as supplied pre-kernel
output tensor metadata and separation. No successful wrapper execution,
allocator computation or setup result is a premise of the public theorems.
Tensor results use inherited contiguous descriptors and exact base-plus-four-
times-flat-index byte observations; this is not a new shared
`Torch.FP32.logicalValue` theorem. Existing natural-alignment and finite-pointer
proofs remain applicable.

Scalar ISA meanings, compiler realization beyond inspected profiles and
ordinary Tensor API binding are explicit model boundaries. They are not
experimental numerical assumptions. There is no claimed native-to-Real error
bound and no numerical experiment TODO. All admitted FP32 words remain in
scope. Runtime grid-product overflow and segmented-launch compatibility remain
deferred outside this candidate.

## Historical validation

- `python check_contract.py`: exact kernel AST identity and wrapper AST
  difference restricted to the documented guard, 12 preserved floating
  assignments/store values, all six retained autotune configs, 60 inherited
  metadata rejections, 7 new domain admissions, 6 new domain rejections and
  26 exact CPU interpreter controls. These cover K=9, K=8 rejection, partial
  tiles, rounded overlap, seed/final combinations, read aliases, harmless
  singleton strides, empty grids and signed-zero seeds at T=0.
- `python check_lowering.py`: seven actual candidate wrapper/JIT bindings and
  offline SM80 compilations using the first original configuration
  (`num_warps=1`). Covers T=0, K=9, optional-state branches, single-subblock
  carry, multiple-subblock outer add and rounded overlap at K=129/BT=17.
  Saves TTIR, TTGIR, LLVM and PTX. CPU-backed Tensor proxies adapt CUDA identity,
  context and allocation; the actual wrapper guards and real JIT binder run.
- Targeted Lean builds and declaration audits check the public entries,
  independent target bindings and standard-base axiom footprints. Import
  source/object hashes and available workspace receipts are frozen separately.

GPU execution, independent native numerical replay and an independent cold
Lean rebuild have not been performed. Offline compiler profiles do not claim
that every retained autotune configuration fits every device.

`Misc/TensorBridge.lean` exposes `TensorBridge.python_checks_iff_preconditions` as an ordinary
theorem for Python-check correspondence. `Correct.lean` and `Misc/NativeCorrect.lean` retain
the complete mathematical and native kernel results and do not import this bridge.

The primary interfaces take Parameters/ShapeValid and a complete CTA schedule;
the shared IO judgment binds logical channel values to the actual source. Main
statements contain no raw State, Memory, run relation or existential post-state.
`Proofs/SourceSafety*.lean` proves every active block-pointer load/store along
the actual nested source loops; `Proofs/IOSafety.lean` obtains those bounds from
the public channels. `Proofs/GridSchedule.lean` reuses the existing whole-program
execution result for every invocation and retains arbitrary grid permutations.
The complete former raw statements remain ordinary theorems in
`Misc/RawCorrect.lean` and `Misc/RawNative.lean`.

The Tensor application Result structures and full declarations remain unchanged
under TensorBridge.RealApplication and TensorBridge.NativeApplication. These
reuse the same whole_grid source foundation through the existing
RealPhysical/Checked source_complete and physical input/output binding lemmas;
the static-dot contract and all original byte/pointer observations are retained.
The IO-format pass performs targeted Lean builds and source/structure checks.
It adds no GPU run, numerical experiment, independent replay or source repair.
