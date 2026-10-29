# Static-tile checked attn_fwd_triton candidate

This is a separate candidate for the current repaired source. The original
107 source/proof/document files and the accepted static-tile defect packet
remain unchanged. Selection belongs to the root review.

The only Python AST change is this check in `_launch_contract`:

```python
if 128 * bd > 1048576:
    raise ValueError("source tile exceeds Triton static tensor element limit")
```

Here `bd = max(16, triton.next_power_of_2(dim))`. The additional condition
applies only to nonempty calls. Both original allocations (`O`: BF16 and the
unused `M`: FP32) still occur before the original empty return; empty calls
still bypass scale-layout/length and launch checks. All original dtype,
layout, axes, signed offset envelope and unsigned aligned pointer checks stay
in place. No total grid-product check is added.

Q/output pointers, Q, accumulator and output tiles have shape `[128, BD]`;
K and V use `[BD,64]` and `[64,BD]`; scores, probabilities and masks use
`[128,64]`. `Contract.all_tiles_fit` covers those and all source vectors.
The new condition excludes the formerly admitted D=8193, BD=16384 frontend
failure. D=8192 is an admission/binder boundary, not a GPU resource guarantee.

`Kernels.lean` reuses the complete unchanged shared quotation. Both helper
calls are expanded, and the production source retains final division. The
reciprocal experiment is not selected.

`Correct.lean` has one public specification, `whole_real`, using the shared
`AlgorithmProgramIO.Implements` judgment. `RealIO.lean` declares the five input
roles, independent strides and offsets, complete ordinary grid, output window,
source projection and independent formula. The implementation proof reuses
the existing `GridLaunch.launch` and `GridLaunch.safe` proofs. The judgment
includes actual source safety, successful execution, every output and the full
raw-cell frame. Its target is causal scaled softmax attention:

```
score(r,u) = sum(d<D, Q[r,d]*K[u,d]) * QS[r//128] * KS[u//64]
O[r,j] = sum(u<=r, exp2(score(r,u))*V[u,j]) / sum(u<=r, exp2(score(r,u)))
```

Batch/head scale bases are retained. There is no implicit `1/sqrt(D)` factor.
Read-only aliases and zero/broadcast strides remain admissible.

The native proof uses the candidate-local `Graph` backend, freshly elaborated
against the current structural Algebra record. The inherited backend cache
showed a real record-projection ABI mismatch and is not used by the candidate
endpoint. `Model.update`, `Model.fold` and `Model.reference` are independent
input-only recurrences. Exact source initialization, both helper loops and the
BF16 store connect those recurrences to the complete shared execution; the
reference does not choose the output of a source run.

FP32 compute loads in the shared structural interpreter read the erased
`.real` cell channel. `Physical.callerCells` binds exact FP32 input words to
that channel; it does not decode them to mathematical Reals. The auxiliary native
theorem `whole_native` in `Misc/NativeCorrect.lean` uses shared
`ProgramIO.Observes`. `NativeIO.lean` declares all five input channels, BF16
output, complete program schedule and the independent recurrence on logical
arrays. The existing `Operations` primitive model observes that recurrence;
no transformation identity, finite-operand restriction or numerical assumption
is added. Actual Structural access safety covers initialization, both loops
and the final store.

`Proofs/NativeSchedule.grid_memory_eq` proves that this schedule has exactly
the same final memory as the existing independent-program ordinary grid.
Its geometric input/output separation follows from the old physical `Views`
width equations: a live input is four bytes and a live output two bytes, so one
logical cell cannot be both. Read-only aliases and zero/broadcast strides remain.
Earlier output stores preserve every later program's input reads; the original
output-injectivity premise supplies unique ownership. The proof reuses
`Model.program` and the independent recurrence instead of reproving execution.

`Proofs/RawReal.lean` and `Proofs/RawNative.lean` retain the complete original
physical-memory statements and proofs. Their execution foundation is the same
`GridLaunch` / `Model.program` used by the new IO theorems; the checked grid
memory equivalence connects the native scheduling presentations.
`Misc/TensorBridge.lean` retains the complete former `whole_real`, `whole_native`
and `empty_return` application statements in the `TensorBridge` namespace.
Its Real theorem derives the six kernel windows from unchanged metadata checks
and storage bounds before applying the raw theorem. Its native theorem applies
the raw theorem to the descriptor pointers, binds each active read to the actual
`Torch.FP32.Tensor` logical value, and preserves object/storage identity,
metadata, storage offset and strides. Logical-index bounds, signed64 control,
static tile coverage and exact native byte addresses still follow from the
unchanged admission and envelope/pointer checks. Neither kernel entry imports
`TensorBridge`; `Misc/InterfaceChecks.lean` checks both layers explicitly.

The native operations retain default TF32 score dot with FP32 scaling;
FP32 max, exp2, denominator and accumulator; explicit probability/V FP16
casts; the FP16-result value dot and its widening; two selected FMA updates;
FP32 division and final BF16 conversion. Reduction/dot operation contracts
retain their complete ordered input tiles and configuration. Local instruction
semantics and the pinned unchanged-source SM90 compiler profile are explicit
external boundaries; no whole-output equality or successful execution is a
premise, and no unconditional native-equals-ideal-Real claim is made.

The bridge's returned descriptor is the original allocated BF16 output object.
The byte realization writes its two-byte little-endian outputs, preserves every
exterior byte and each entire input allocation, and preserves all other
physical cells. The existing tensor allocation/view/freshness facts are
ordinary API bindings, not a new allocator or host-runtime project.

`check_candidate.py` exercises actual wrapper checks, allocations and the
installed JIT binder on ordinary Torch CPU tensors with only the CUDA flag,
device context and final launcher replaced. Its 18 cases cover both boundary
sides, feature/key tails, two query blocks, independent strides, aliases,
broadcasts, signed scales, rejected metadata and all empty axes. No GPU or
interpreter kernel executes and no numerical experiment is performed. The
three frozen SM90 compiler controls remain applicable because both JIT ASTs
are exactly unchanged.

The evidence directory is
`.veritile-cache/kernel-evidence/attn_fwd_triton_tile_checked/`.
The final delivery packet, once frozen, records source/object pins, complete
public proof receipts, actual binary import-header checks and precise scope.

Current two-program result: `Equivalence.quotient_remainder_equiv` compares the
complete original quoted source with the direct ordinary `quotientRemainder`
quotation. It replaces `off_hz % H` with `off_hz - (off_hz // H) * H`, exposing
the quotient already computed for `off_z`. This is an integer CSE opportunity;
no measured speedup is claimed. The historical reciprocal candidate remains
available, with its original guarded proof.

The main pair retains the complete original native graph, grid, compound input
carrier, all masks and 64-bit index casts, TF32 score dot, FP16 value dot and
BF16 output. `Proofs/PairConnection.lean` proves whole-state execution and actual
access-safety equality for arbitrary numeric backends; `Proofs/PairIO.lean`
transports full-grid success, tagged outputs and both exterior frames. The
pair uses `≡[[]]` and exact Nat identities, with no new numerical assumptions
or experimental evidence obligations. The independent Real target and previous
optional native theorems retain their original scope.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/InterfaceChecks.lean](Misc/InterfaceChecks.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
