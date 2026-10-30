# flash_decode2_llama kernel proof

The retained raw `RealKernelResult` and `NativeKernelResult` reader applications are imported explicitly from `Misc/RealKernelResult.lean` and `Misc/NativeKernelResult.lean`. Their declarations are unchanged.

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/KernelAudit.lean`.

Current source pair (2026-10-21): `FPEquiv.kernel_equivalent` compares the
complete original with the new ordinary `Candidates.llamaSharedBase` source.
The original computes `offs_v = batch*(H*S*D)+head*(S*D)+offs_d` and separately
`offs_logic = batch*(H*S)+head*S`. The candidate computes `offs_logic` first
and reuses it in `offs_v = offs_logic*D+offs_d`, removing one integer multiply
and one add. No performance improvement is claimed without measurements.
The exact natural-index identity follows from distributivity and associativity;
no floating computation, cast, recurrence order, exp or FMA rule is rewritten.
The pair uses shared `ProgramIO.EquivalentOn`: both full grids are safe,
succeed, agree on active typed outputs and preserve both exterior frames.
The complete all-word native domain and the existing length domain are retained,
including every permutation of the original complete grid and empty grids.
`Proofs/SourcePair.lean` connects proof-only statement decomposition to both
complete quotations and proves exact execution and safety for arbitrary Algebra.
`Proofs/IOPair.lean` transports every actual invocation, preserving register
clearing, program IDs, numPids and the observer. No experimental rule is used.
Optional single-source results remain in `Misc/NativeCorrect.lean`.

The selected Python source and its guard are unchanged. Its SHA256 is
`2d8a58c053404bcc7b8b58117aaea53cb4eed4742ddcadc90c8eaa6075e34c85`, matching
`bench/tritonbench_g/task_progress.json` at this revision's start.

`IO.lean` exposes the actual complete source schedule, runtime geometry, all
E/M/L inputs and O outputs, dense windows, masks and empty scratch footprint.
`Correct.kernel_real` uses shared `ProgramIO.ImplementsOn` with the independent
weighted-summary formula and explicit logical length bounds.
The ordinary `FPEquiv.kernel_native` theorem in `Misc/NativeCorrect.lean`
uses the same shared IO and complete source with the
existing native instruction/compiler interpretation. Both require successful
execution and prove actual source-access safety through the dynamic loop,
every output and the complete exterior frame. No raw State/run/memory details
occur in these IO statements. The main `FPEquiv.lean` proves the actual source-to-source transformation.

Exact previous raw input/state results remain ordinary helpers in
`Misc/RealKernelResult.lean` and `Misc/NativeKernelResult.lean`. Earlier
host/return/byteheap wrappers remain in `Proofs/HistoricalCorrect.lean`,
`Proofs/HistoricalFPEquiv.lean`, and `Proofs/HistoricalPublic.md`.

The public results execute the complete `FlashDecode2.llama` quotation in the
shared Phi source file, including the dynamic online max/exp/denominator/vector
accumulator loop. They cover all batch/head/lane coordinates, any permutation
of the complete grid, empty grids, and every untouched tagged memory cell.
`Misc/TensorBridge.lean` retains all prior Tensor application endpoints as ordinary
theorems, including actual FP32 metadata, pointers, logical input reads,
output physical indices and the full tagged frame. Signed INT32 lengths, INT64 promoted offsets, pointer
alignment and storage bounds have separate proved endpoints. No new allocator,
returned-object, runtime-device or unified final-byteheap gate is added.

The admitted domain remains contiguous zero-offset ordinary CUDA tensors on
one device: Mid `[B,H,S,D]`, Log `[B,H,S]`, Length `[B]`, O `[B,H,D]`; payload
and output FP32, lengths signed INT32; D in `{16,32,64,128}`; positive signed-INT32
`block_seq`; existing grid, storage-disjointness and address bounds; and, on an
active grid, `0 < Length[b] <= min(S*block_seq, 2^31-1)`. Grid-product runtime
compatibility is outside this kernel revision. Mid and Log may alias read-only.

The independent mathematical target, for `K[b] = ceil(Length[b]/block_seq)`, is

```
sum(j < K[b], exp(Log[b,h,j]) * Mid[b,h,j,d])
/ sum(j < K[b], exp(Log[b,h,j]))
```

The main interface accepts independent typed real input arrays. The bridge binds
those observations to Tensor addresses without restricting them to a finite
floating decoder. The ideal
interpreter uses `Option Real` so negative infinity is not represented by a
finite real sentinel. Stage-1 summaries are the API inputs. Relating them to
raw-token attention would require a separate stage-1 specification.

The native result retains every FP32 bit pattern. Six total scalar primitive
families define its model: subtract, multiply, full divide, max, approximate
exp2, and RN FMA. Exp32 is explicitly `exp2(mul(x, 0x3fb8aa3b))`. The selected
compiler graph contracts the left product in both accumulator updates. The
maximal `block_seq = 2^31-1` single-iteration profile places zero second in the
FMA product; its admitted iteration count is proved to be one. Other selected
profiles retain carry first. No commutativity or actual/reference scalar
equality is assumed. These primitive and fixed compiler models are explicit
boundaries, not numerical experiments or native-to-real accuracy theorems.

Validation evidence is stored under `kernel-evidence/`: the unchanged actual
wrapper/JIT AST passes 11 accepted cases and 30 rejection controls; 20 actual
wrapper-to-JIT-binder default Triton 3.7.1 SM80 compilations cover all four D
values and five sequence-block boundaries. Their TTIR/PTX hashes and FMA/exp/div
operand checks are recorded. No GPU execution was performed. Samples support
the selected compiler patterns; they do not prove every Triton backend lowering.
The frozen delivery packet records targeted Lean receipts and the full resolved
workspace import closure. No Phi source file was changed; stale shared proof
objects were rebuilt against the current structural algebra. The Llama
contract locally imports the relocated SquareMatrix block definition.

The current format migration targeted-builds IO, source-safety and full-grid
composition, both public entries, the retained raw statements, KernelAudit and
TensorBridge. It adds no numerical assumptions or experiments. The generic
actual-source safety helper also covers the masked Phi quotation for reuse.
