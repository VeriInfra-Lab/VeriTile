# lightning_attention kernel-only revision

This package proves the three kernels in the unchanged
`../lightning_attention_repaired.py` (SHA-256
`e610983718f24e2c5bc0e79d0bbd6f2586bfd6b6968a640980538458f090bd32`).
The public namespace is `VeriTile.Bench.Repairs.LightningAttention.KernelOnly`.
The standard entries are `Contract.lean`, `Kernels.lean`, `Correct.lean` and
`FPEquiv.lean`. The single-source native result is an ordinary auxiliary theorem
in `Misc/NativeCorrect.lean`; optional native results
require an explicit Misc import. `Safety.lean` exposes address consequences and `Misc/Audit.lean`
checks the public proof closure. Historical Python, task proofs and shared
infrastructure are preserved.

## Independent mathematical result

For each head, with Q/K of shape `[N,D]` and V/G of shape `[N,E]`, the target is

```text
score(t,u) = sum_a Q[t,a] K[u,a]
O[t,c]    = sum_{u <= t} score(t,u) V[u,c]
loss      = sum_{t,c} G[t,c] O[t,c]
dQ[t,a]   = sum_{u <= t} (sum_c G[t,c] V[u,c]) K[u,a]
dK[u,a]   = sum_{t >= u} Q[t,a] (sum_c G[t,c] V[u,c])
dV[u,c]   = sum_{t >= u} score(t,u) G[t,c]
```

There is no decay, softmax, normalization, scaling or exponential. The public
`forward_correct` and `backward_correct` specifications use shared
`ProgramIO.Implements` judgments for these independent logical formulas.
`TensorBridge.correct` applies the same execution results at actual Tensor
logical addresses. Backward includes
`HasDerivAt` for every Q/K/V coordinate of the independently defined scalar
loss. Its physical Real reader is unrestricted; there is no finite-value or
FP32-representability premise, and it is not an IEEE byte encoder.

## Unchanged admitted domain

- Q/K are `[B,H,N,D]`; V/O/G are `[B,H,N,E]`. B/H/N are positive and D/E
  are powers of two at least 16. No finite width list or device resource cap
  is added.
- Inputs are ordinary same-device CUDA FP32 tensors, with resolved negative
  and conjugate flags. Q/K/V are contiguous. Valid singleton strides, nonzero
  storage offsets, and every alias among read tensors remain admissible.
- `NT = ceil(N/64) <= 65535`, `BM = min(E,32)`, and
  `NV = ceil(E/BM) <= 65535`. The unchanged arithmetic guard is
  `(B*H*N+128)*max(D,E) < 2^31`. Every passed pointer is four-byte aligned and
  satisfies `pointer + 4*formedLimit < 2^63`.
- Forward has one fresh output. Backward has three fresh outputs. Complete
  output backing ranges are separated from every read backing and from one
  another; read backings may overlap freely.
- Forward grid is `(B*H,NV)`, intra grid `(B*H,NT)`, inter grid `(B*H,)`.
  BLOCK=64, CBLOCK=32, NC=2, warps=4, stages=3, and fusion is enabled. Payload
  data and casts are FP32; source dots use default TF32 precision.

Requests are indexed by phase. A forward request contains only Q/K/V and O;
it has no future gradient or cotangent obligation. A backward request uses
the current saved Q/K/V and both original G and the dense G supplied by the
existing `G.contiguous()` operation. Original G can be noncontiguous or
broadcast. `Preparation` is only coordinate-wise equality between those two
current input views, using ordinary API behavior. Both full G backings are
included in the frame. The unchanged wrapper saves the exact original Q/K/V
objects and retains the sole incoming contiguous materialization; this
package does not certify host lifecycle or allocation success.

## Source execution and physical results

`Kernels.lean` uses the shared typed quotation and `FP.Structural.exec`.
Complete statement-by-statement quotations retain the signed forward mask,
the intra FP32 arange conversion/subtraction/comparisons, and the literal
descending inter ranges, including their mirrored iterator assignments.
Forward, intra, both inter scans and every program in the grid have
constructive execution proofs. The finite 64-lane mask decisions are proved
for Real and the actual software IEEE FP32 constant operations. They are
internal discharged lemmas, not public mask assumptions. Signed difference,
formed-row, false-mask lane, final advance, reverse initial-pointer and
reverse-update bounds are audited.

`forward_fp_equiv` and `backward_fp_equiv` expose shared materialized-IO
judgments for native source execution against an
independent stage/grid recurrence. That recurrence uses a fixed-K8
instruction graph with explicit load/cast materialization and clear/live
seed behavior. Intra results are threaded into inter. The sole numerical
premise, `Native.LocalConformance`, contains pointwise scalar add and bounded
single-coordinate MMA/WGMMA K8 laws. It contains no variable-size dot,
reduction, graph, stage, complete-execution or returned-result law.

The Tensor layer uses the actual shared `VeriTile.Torch.FP32.Tensor` API.
Source reads are proved to agree with its logical input words, including
original G. Output stores use only executed final cells. A proved exact
four-byte codec reconstructs every logical output word and byte. The ordinary
`TensorBridge.correct` and `TensorBridge.fp_equiv` applications establish the
exact exterior of the output footprint and
preserve every byte of all complete read backings, including unused prefix
and suffix storage. Tensor API metadata/pointer laws and backing separation
are input contracts, not execution hypotheses.

Task-local execution, algebra, Real and native-expression layers are fresh
against the current shared Algebra layout. Pure historical dimension,
target, mask and bounded instruction definitions are reused. Historical
imports transitively include older execution modules; the Audit checks
that the Tensor application proof bodies do not depend on the old
normalized quotations, `stageProgram`, or `runGrid`.

## Validation and scope boundaries

The evidence directory is
`.veritile-cache/kernel-evidence/lightning_attention/` at the repository root.

- The existing source/contract checker passed: all three original kernel
  ASTs unchanged, 31 metadata admissions, 20 rejections, seven complete
  forward/intra/inter interpreter calls with input frames, and the retained
  lazy-negative counterexample rejected by the checked contract.
- `check_source.py` compared all statements of all three Python kernels with
  the Lean quotations (26/33/66 statement nodes). Its only representation
  adaptations are the signed arange annotation, the DSL FP32 numeric
  conversion spelling, signed constexpr annotation, resolved FP32 store
  type, reserved identifier renaming and exact zero literal spelling.
  This is source-binding evidence, not a verified Python frontend.
- Exactly three new representative offline SM90 compilations used the
  actual JIT binder: forward/intra/inter at B=H=1, N=65, D=16, E=32 and Q
  storage offset 1. TTIR, TTGIR, LLVM IR and PTX are retained. No new GPU
  numerical cases or device execution are claimed.
- The retained earlier targeted build covered the original 60 task-local Lean
  modules. The public-IO format pass records its exact additional targets and
  receipts separately in `.veritile-cache/public-io-format-20261020/lightning_attention/`.
  Public results and Tensor applications use only `propext`, `Quot.sound` and
  `Classical.choice`. This format pass adds no compiler, GPU or numerical experiment.

`Native.Plan` is a static shape-only choice. The retained representative
lowering does not certify `observedPlan` or any compiler plan across all
admitted widths. Ordinary API/contiguous preparation, byte transport,
compiler realization of the selected instruction plan, and race-free
launch/device/resource delivery remain external boundaries. There is no
compiler, allocator, Host-success or whole-execution binding premise and no
native-to-Real equality claim. The shared launcher grid-product issue stays
separately tracked and is not used as a kernel proof premise.

The prior `delivery.json` records the kernel-only delivery, including its
source, import closure, build artifacts and 87 preserved historical files.
The current format pass pins its changed files and exact build receipts in
`.veritile-cache/public-io-format-20261020/lightning_attention/report.json`.
Root integration owns the canonical ledger, Lake registration and independent
review; this report does not claim a new independent replay or GPU run.


## Public IO organization

`SourceIO.lean` lists original kernel identities, channels, shapes, dense
head/time/feature addressing, input masks and complete launch schedules.
`Reference.lean` defines independent causal output, VJP and coordinate-loss
derivatives over logical arrays. `Correct.lean` has one specification for each
actual API phase: forward and the full two-stage backward operation. Real
geometry remains exactly `Host.Contract`; no native arithmetic equality is
asserted.

`NativeIO.lean` uses the shared `MaterializedProgramIO` interface. Each CTA
embeds the current typed word memory, executes the original source, then
decodes that actual successor before the next CTA. `NativeReference.lean`
retains the independent fixed-K8 recurrence. The native domain remains all
raw dimensions and words: padded forward channels use the original physical
E stride, including overlaps for geometry outside the Tensor contract.
Initial output/gradient channels expose arbitrary original scratch values.
There is no extra `Host.Contract`, finite-word restriction or numerical law.

`Proofs/ForwardSafety.lean`, `IntraSafety.lean` and the `Inter*Safety.lean`
helpers check the actual source loads/stores, masks and successor states of
all natural and signed loops. `StageSafety.lean` connects those proofs to the
shared launch schedule; `IOSafetyBounds.lean` connects them to public windows.
`IOBounds.lean` gives concrete capacities equal to the admitted Tensor extents,
so safety is not hidden behind impossible capacity premises.

`PhysicalIOBinding.lean` binds the actual original Real Tensor readers,
including the prepared incoming G, to the same logical arrays and formulas.
Native helpers connect the exact raw words and the actual per-CTA materialized
execution. `Misc/TensorBridge.lean` retains all previous output-word/four-byte,
exterior and complete read-backing results. The prior raw statements remain
ordinary helpers in `Misc/RawCorrect.lean` and `Misc/RawNative.lean`.
Public entries and helper modules do not depend on the optional Tensor bridge.
The original Python, all three original typed bodies, `Contract.lean` and the
numerical assumption inventory are unchanged. New candidates appear as complete
ordinary quotations alongside the originals in `Kernels.lean`.

## Completed program pairs, 2026-10-21

`FPEquiv.forward_equiv` compares the complete original forward grid with the
actual offset-reuse candidate. `FPEquiv.backward_equiv` compares both complete
backward grids in order. The actual candidate is
`lightning_attention_offset_reused.py`; its entire Python module differs only
in the three assignments `o_offset = v_offset`, replacing repeated
`o_offset = off_bh * n * e`. This common-subexpression elimination removes two
integer multiplies per CTA. No speedup measurement is claimed.

All three actual candidate bodies are complete ordinary `triton` quotations.
`Proofs/PairSource.lean` proves the short pure preludes equal, then lifts exact
full-state execution and trace-safety equivalence through the unchanged complete
suffix, for arbitrary original scalar operations. No floating operation, cast,
mask, reduction, dot, branch, store or loop ordering changes.
`Proofs/PairExecution.lean` retains each actual source invocation and the
unchanged codec and per-CTA materialization, then lifts through every stage grid.

`PairIO.lean` uses a narrow implementation view of the existing
`MaterializedProgramIO`: its only field is a schedule of actual shared
`MaterializedInvocation`s. The logical domain, typed channels, codec, observer,
read/write windows and fixed shared runner remain those of `NativeIO.forward`
and `NativeIO.backward`. Both actual schedules are safe and successful, every
observed typed output cell agrees, and both retain the complete exterior frame.
The native scope remains all original dimensions and raw words, arbitrary
initial output/gradient cells, overlapping forward padded windows, and actual
materialized intra-to-inter gradient dataflow. No `Host.Contract`, finite-data
restriction or new primitive-conformance premise is added to these pairs.
The independent Real phase specifications and their original geometry remain
unchanged; no host-composition or hardware-conformance task is added.

All three original complete ASTs are preserved by `rfl`; all six complete
original/candidate quotations match their actual Python source under the
existing explicit representation normalization. The existing complete
forward/intra/inter CPU interpreter and host checker are replayed on the
candidate. Targeted builds and fresh main/optional joint imports are recorded
in `.veritile-cache/finish-pending-105-20261021/lightning_attention/`.
The experimental rule set is empty, with no deferred numerical experiment or
GPU execution claim. Historical single-source native certificates retain their
full theorem names and statements in `Misc/NativeCorrect.lean`.

## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
