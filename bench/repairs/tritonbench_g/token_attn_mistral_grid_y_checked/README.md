# token_attn_mistral: CUDA grid-y check candidate

This independent sibling candidate adds exactly one guard immediately after the existing empty-batch return:

```python
if head > 65535:
    raise ValueError("query heads exceed the CUDA grid y limit 65535")
```

Its accepted domain is the preceding checked source's domain intersected with `B=0 or Hq<=65535`. Empty batches retain all previous behavior, including heads above 65535 and omission of live pointer and alias checks. Other validation, the wrapper's canonical strides, the original JIT kernel, and its return/mutation behavior are unchanged. There is no grid-product check, copy, finite-value restriction, alternate launch, or new host implementation.

The predecessor source is `../token_attn_mistral/token_attn_mistral_repaired.py`, SHA256 `36afb09dac2c46a8049cf250c8c592484147e3d6133ad915e6917d8920c9c553`. Its frozen defect packet is `.veritile-cache/token-attn-mistral-kernel-evidence/delivery.json`, SHA256 `7336fccc4e958aeb2d990b95a6d8066b2250b9f43c33e22bcf1a4a949720f2df`. That packet preserves the admitted `(1,65536)` launch capture and the official CUDA Programming Guide grid-y limit of 65535. The predecessor's 42 task files and defect evidence remain frozen. Root owns independent review and selection; this directory does not select itself or update the ledger.

`Contract.lean` uses shared Torch TensorMetadata, signed-int32 words at the actual caller pointers, and the observed Python scalar window. `windowIsInt` represents exact `type(window) is int`, so bool is rejected. `checkArguments` represents the tensor type test. Header, contents and live physical checks match the source: shapes, FP32/int32, device, ordinary contiguous views, padded extents, request/sequence/probability intervals, entire active table suffix, view-byte separation, alignment and address range. `B_Start_Loc` has no content condition. Same-dtype read aliases are admitted. The window and all metadata checks still precede the empty branch. `python_checks_iff`, `exact_domain` and `empty_domain_unchanged` give the public domain correspondence.

The kernel computes a gathered weighted sum without softmax. For each batch, `first=max(sequence-window,0)`, and the independent mathematical target is

`sum(n<attentionLength, if first+n<sequence then P[h,attentionStart+n] * V[table[request,first+n],h/(Hq/Hkv),d] else 0)`.

The two masks are deliberately distinct. P uses `n<attentionLength`; table and V use `first+n<sequence`. The source may load V beyond P's tail, so native evaluation retains `0*NaN`. Attention length need not equal the window suffix length. `Proofs/Targets.lean` states the independent Real sum and native graph directly in terms of the original caller data. The native target includes every padded 128-lane product, all reduction leaves, sequential accumulator updates and final FP32 cast.

`Kernels.lean` reuses the predecessor's single full source quotation. `Proofs/Model.lean` supplies its actual pointer tags, window, group, canonical strides and 128-token tile. `complete` invokes the original whole-grid execution proof for any permutation of the complete `(B,Hq)` grid. It derives every output store and the full tagged-memory frame without a supplied execution-success or final-result premise. `TensorBridge.empty_call` proves an empty work list and preserves every caller word.

`Proofs/Tensor.lean` retains original Tensor object/storage identities, offsets and physical indices through ordinary four-byte storage, pointer and contiguity laws. Checked dtypes distinguish FP32 and int32. `probability_bound` and `value_bounds` follow their separate source masks; `probability_logical`, `value_logical` and `gather_logical` connect each active input to its caller Tensor element. Public output readback uses the same caller logical Out. `RawFrame` covers all source locations outside stores; `observe` interprets exactly those stores in the caller output view and preserves its full exterior and all admitted read views, including unused metadata. It is a mathematical observation, not a copyback implementation.

`TensorBridge.kernel_correct` proves full quoted-source execution, every logical Real output, and both frames. `TensorBridge.kernel_native` proves every logical output word against `NativeTarget.reference`, using a single caller word map for both floating data and signed metadata. `Proofs/Indices.lean` proves submitted x/y axis bounds, padded formed indices, active V/output indices, dynamic scalar bounds, signed sliding-window subtraction, signed metadata interpretation and active aligned 64-bit addresses. Shared grid-product compatibility remains outside this repair.

Native results use one model with three scalar parameters (`addRN`, `mulRN`, `fmaRN`), complete shape-driven reduction trees and legal adjacent FP32 multiplication/addition contraction. Scalar instruction behavior, FP32 load/cast identities, selected compiler structure and source/ISA correspondence are explicit model boundaries. They are not pending numerical equalities, row-result callbacks or an assertion that native arithmetic equals Real arithmetic. This guard change adds no numerical transformation or numerical experiment obligation. Tensor API laws, stable input memory during checking/execution, and the formal source interpreter are also explicit boundaries; driver, resource and universal compiler/hardware conformance are not proved.

`check_candidate.py` proves the sole-guard module AST delta and unchanged kernel AST, then runs the predecessor's complete 55-case checker against this candidate using Triton's CPU interpreter. This includes independent-mask and NaN controls. Five additional launch-capture cases cover live heads 65535/65536/65537 and empty heads 65536/100000. CPU facades and launch capture do not execute a GPU. No GPU run or new offline compilation is claimed.

The public entries now use shared source-connected `ProgramIO` judgments.
`Correct.lean`’s `kernel_correct` implements the independent
weighted sum; `Misc/NativeCorrect.lean`’s `kernel_native` observes the independent native recurrence.
Both cover the entire submitted work list. `Ports.lean` declares seven finite
inputs, their actual metadata/strided/gathered addresses, separate masks, and the
full output channel; `IO.lean` declares the exact shared-source invocations.
`Proofs/SourceFormula.lean` contains the state-free mathematics and ordered
primitive formulas needed by the main proofs. `Misc/NativeReference.lean` and
`Misc/NativeIOResults.lean` retain optional native observation formulas and
results; finite input accessors remain with `Ports.lean`. The main
statements expose no raw state, runner, memory cells, or existential final state.

The `counts` parameter is each submitted program's attention metadata snapshot,
used only to size finite ports. The extent is `sum(counts) + block`; it is not a
new maximum-length restriction. `Ports.Domain` equates that snapshot with the
logical metadata channel. `Proofs/PhysicalIO.lean` constructs the snapshot,
logical inputs, simultaneous binding and a finite capacity witness for every
original raw heap, including arbitrary tagged native words. Probability reads
are active only below attention length. Page/value reads use the exact visited
block condition and the sequence-window mask, retaining every padded native
operation. Native block zero, width zero, all original strides/regions/work
lists, reduction trees, contractions and read-only aliases remain covered.

`Proofs/AccessSafety.lean` proves bounds on the actual evaluated source trace;
`Proofs/IOSafety.lean` carries those bounds through the complete launch schedule.
`Proofs/IOExecution.lean` reuses the original execution theorem and proves the
new shared schedule has exactly the old grid's memory result.
`Proofs/ReferenceBinding.lean` connects those results to the independent finite
formulas. `Misc/RawCorrect.lean` and `Misc/RawNative.lean` retain the former
raw statements; `Misc/TensorBridge.lean` retains the exact previous Tensor
applications, physical readback and frames under unchanged guards. Reader audits
live in `Misc/Audit.lean`; supporting `Proofs/` modules do not import public readers.
This organization change adds no Python guard, kernel variant, numerical
assumption, or numerical/GPU experiment claim.

## Affine address candidate (2026-10-21)

`token_attn_mistral_affine.py` is a concrete callable candidate with the complete
checked wrapper and JIT body. The single kernel change is

```python
p_offs = (cur_head * stride_ph + cur_batch_in_all_start_index * stride_pbs) + offs_n * stride_pbs
```

The original computes `cur_head * stride_ph + (start + offs_n) * stride_pbs`.
The candidate moves the uniform origin into scalar arithmetic, removing one
vector addition from probability address formation. It has the same grid,
source FP32 operations, masks, conversions, reduction order and stores. No
measured hardware speedup is claimed. `Kernels.affineSource` is the complete
ordinary Triton quotation; `IO.affine` invokes it for exactly the original work
list on the unchanged public typed interface.

`FPEquiv.affine_address_equiv` proves `original ≡[[]] affineAddress` for the entire
source grid and all original symbolic native payloads. The task-local strict
natural distributivity proof shows the modified assignment, including failure,
returns exactly the original state. It is lifted through every remaining source
statement and every invocation; both executions succeed, both access traces are
safe, all active output cells agree and both complete exterior frames hold.
Distinct probability and V/table masks and padded native products remain intact.
Empty grids, zero block/width behavior, original arbitrary strides/regions and
read aliases retain their prior scope. No new finite-value or numerical premise
is used. The original `Correct.kernel_correct` independent Real weighted sum and
optional native/Tensor applications are unchanged.

The previous sort-only schedule experiment is superseded by this executable
source optimization and is not counted as a completion certificate.


The complete original and existing affine candidate bodies now share the raw
source entry in the underlying task's `Source.lean`. `Kernels.lean` explicitly
forwards to that shared entry; it emits no replacement program or generated
body. The selected checked Python and existing affine Python remain unchanged,
and this directory's `jit-correspondence.json` checks both complete bodies.
Types, runtime call bindings and constexpr metadata stay outside the bodies.
The terminal `return`, all strides, gathers, masks, padding, reduction order,
FP32 arithmetic and output conversion remain. The existing FP32 input profile
and signed-int32 native metadata bounds are retained; unused generic dtype
helper parameters are removed without restricting any public theorem.

Both full executable statement lists are proved identical to their preceding
selected FP32 programs in `Proofs/SourceIdentity.lean` of the shared task.
Revalidation includes this directory's complete Real and FP specifications,
whole-state/failure and trace-safety affine equivalence, full-grid execution,
all physical frames and retained Tensor/native applications, together with the
underlying execution/host/native proof layers. No numerical assumption, input
restriction, new candidate or GPU experiment is introduced. Earlier checks
above remain historical; the current frozen-source receipts and unavailable
tool attempts are recorded separately in the task ledger.
