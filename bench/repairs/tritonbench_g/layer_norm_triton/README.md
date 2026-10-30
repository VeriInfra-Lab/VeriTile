# layer_norm_triton kernel proof

The selected source is `layer_norm_triton_repaired.py`, SHA256
`6f2f1898b81f0f51580f26c12c22fb8989cec2ed61a69a6e33f2d3a5aa9f20a7`.
The selected Python, original benchmark, guard, and historical host proofs are
unchanged. The kernel revision proves execution of the readable typed source,
its independent Real specification, and native FP32 expression evaluation with
explicit compiler policies. Final targeted validation is recorded in the delivery
receipt; historical host acceptance is preserved separately.

For each row, the independent mathematical target is
`mu=sum(x)/N`, `v=sum((x-mu)^2)/N`, `r=1/sqrt(v+eps)`, `z=(x-mu)*r`,
`y=z*w+b`. Backward is `dx=(dy*w-z*mean(dy*w*z)-mean(dy*w))*r`,
`dw=sum_rows(dy*z)`, and `db=sum_rows(dy)`. The Real interpretation uses local
primitive laws. TensorBridge Real applications obtain epsilon through the input-only
`SourceIdealInput.ScalarProjection` applied to the exact `Forward.epsilon.toBits`.
That same projected scalar drives source execution, the independent formula,
and saved/composed statistics; the kernel interfaces take the decoded Real scalar directly.
This input abstraction does not identify rounded FP32 arithmetic with Real arithmetic.

`Contract.lean` retains the exact selected guard. X/W/B and dY are contiguous,
strided FP32 CUDA Tensors on the required device, without lazy-negative flags;
nonempty observed data pointers must be four-byte aligned. Nonzero storage
offsets, Torch-contiguous singleton/empty strides, read/read aliases, NaNs,
infinities, subnormals, and signed zeros remain admitted. No original storage-base
alignment, finite-data, nonzero-native-denominator, or copying premise is added.
Normalized axes form the exact nonempty trailing suffix and affine shapes match
it. Epsilon is a finite positive builtin float; the native binding retains all
binary64 bits through the selected scalar ABI conversion, including values that
convert to FP32 zero.

The original width selection remains
`B=min(16384,next_power_of_2(N))`, with failure for `N>B`; the warp formula and
256/128/96/64 group thresholds are unchanged. Final reduction uses BM=32,
BN=128. Either zero dimension dispatches no kernel and preserves the supplied
memory. The wrapper's existing affine zero allocations are not reclassified as
kernel stores. The TensorBridge applications take ordinary Dense output allocation descriptors;
allocation and runtime failure semantics remain in the historical host proof.

`Kernels.lean` contains readable typed FP32 bodies for forward, grouped backward,
and final affine reduction. `Proofs/PythonSource.lean` retains the complete
immutable Python AST; `bind_source.py --check` checks that AST and both source
hashes. `SourceRows` connects the positive-step while traversal with the typed
range. `SourceStride` derives the effect of the actual flattened Tensor stride
from its Dense address law, including arbitrary singleton strides.
`SourceObserved` executes the actual stride and proves complete physical-memory
equality with the dense logical projection. `SourceNativeBinding` supplies the
signed64 row, scratch, and final-tile address equations, including masked lanes.

The primary mathematical interfaces use the shared typed IO judgments.
`IO.lean` declares forward/final channels, windows, masks and actual source
invocations; `GroupedIO.lean` additionally separates DX readback from writable
windows so untouched cells retain their original tags and values.

- `Correctness.forward_correct` exposes independent Y/Mean/Rstd formulas.
- `Correctness.grouped_correct` exposes every visited DX derivative and both
  independent partial-gradient sums. Its original row-address premise remains
  visible, including singleton-stride cases.
- `Correctness.final_correct` exposes both independent sums over all partial
  rows. The original Real block-row selection remains 32.

Each judgment proves actual source access safety, successful execution and the
raw exterior-cell frame. `Proofs/SourceSafety`, `SourceGroupedSafety`, and
`SourceBackwardSafety` follow the real loop indices and intermediate states,
including unconditional scalar Mean/Rstd accesses. `Proofs/RealEndpoints`
retains all former operational theorem statements as ordinary helper results.

`TensorBridge.Correctness` retains all five original mathematical applications,
including the forward/backward graph results and complete source forward,
backward and composed launches. Their unchanged statements retain actual
Tensor address correspondence, common physical memory, empty-launch identity,
caller/exterior bytes, and saved statistics derived from preceding source stores.

The native source path instantiates the same generic execution proofs with
`FastLayernorm.KernelNative.Term`. Source operations therefore build expressions
before any rounding. Reduction plans cover every lane; compiler choices inspect
only expression shape. The observer selects reduction trees, operand order,
constant specialization and adjacent multiply/add/subtract contraction. This
supports FMA directly and never assumes fused and split instructions are equal.
Explicit FP32 casts remain in the source tree and their identity lowering is
visible in the observer. Native storage evaluates each kernel's output words
before commit and seeds fresh leaves from actual physical storage at the next
kernel, preventing contraction across those kernel boundaries.

- `Equivalence.forward_native` in [Misc/ForwardNativeObservation.lean](Misc/ForwardNativeObservation.lean),
  and `grouped_native` and `final_native` as ordinary theorems in `Misc/NativeCorrect.lean`, expose
  independent ordered logical-array references through shared IO. Address-labelled
  word leaves remain admitted; the public logical carrier admits all expression
  terms, and injective `.bits` output embedding records exact native words.
  Grouped native IO retains zero group step, overlapping rows, arbitrary block
  coverage and preexisting DX values. Final native IO retains arbitrary BM/BN.
- `Equivalence.forward_equiv` proves the existing forward unrolling candidate
  equivalent under arbitrary scalar semantics, with both source traces safe,
  both actual executions successful, all output observations and both frames.
  Its headline is `ForwardPairIO.original ≡[[]] ForwardPairIO.candidate`, using
  the shared `ProgramIO.Implementation` view of the actual invocations and
  observers. `Equivalence.forward_io_equiv` in `Misc/NativeCorrect.lean` retains the
  complete raw `ProgramIO.Equivalent` result. No floating
  transformation law is used. The candidate is now a complete ordinary `triton`
  source in `TypedSource.forwardCandidate`; its identity to the existing unroll
  helper is proved by `SourceCandidate.forward_source`.
- `Equivalence.grouped_equiv` compares the complete original grouped-row program
  to `TypedSource.groupedBackwardCandidate`, which delays W loading until after
  xhat. It retains every DX result read, including untouched rows, partial sums,
  actual safety/success and the original write footprint.
- `Equivalence.final_equiv` compares the complete final reduction to
  `TypedSource.finalBackwardCandidate`, which stores sum_dw before computing
  sum_db. Both result reductions and store order are retained. The two backward
  schedules shorten vector lifetimes without a floating rewrite or performance
  claim. `layer_norm_triton_scheduled.py` contains all three actual candidates
  with the original wrapper and guard AST unchanged.
- `Proofs/NativeEndpoints` retains every original strict-UInt32/raw-state/native
  endpoint, including complete source-state equality and exterior word reads.
- `TensorBridge.Equivalence` retains every original strict, graph, bound-native,
  composed and physical candidate application. Selected-object/scalar ABI
  correspondence, word materialization at stage boundaries and all empty-grid
  and physical-byte observations are unchanged.


`SourceNativeBinding.Selected` ties policy to source hash, entry, object bytes,
ABI signature, options, address/control recipes, FP32 precision and primitive
identifiers. The decoder supplies structural plans rather than kernel results.
The primitive model supplies individual target instructions, rounding/FTZ/NaN
behavior and epsilon conversion. These are ordinary local compiler/API modeling
boundaries, not deferred numerical experiments or assumed whole-kernel laws.
The earlier strict UInt32 source results and independent `*_graph_*` statements
remain explicitly named for their narrower interpretation.

The literal source graph retains initial `0+a`, `0+x*x`, all explicit casts,
first-group no-add behavior, and count divisions. Historical `Proofs/Model.lean`
omits some forward zero additions and permits unit-division deletion; it is
preserved for history and is not used as a replacement for the new source graph.
The source candidates make only control-flow and independent scheduling changes. `deferred_experiments`
is empty; no floating arithmetic rewrite is claimed.

Historical public host statements remain byte-identical in
`Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`; the prior README is in
`Proofs/HostREADME.md`. Historical defects and ledger entries are not relabeled.

The strict interface split is checked with the isolated driver at
`.veritile-cache/kernel-interface-split-20261020/workspace_lean.py`; prior
validation used `/tmp/veritile_workspace_lean.py` and a writable workspace
output overlay, without a Lake build or shared-cache mutation. Public and helper
proofs include standard-base axiom audits. The CPU source checker passes 30
rejected calls and 17 admitted forward/backward cases spanning group branches,
aliases, offsets, empty axes, singleton rows, tails, nonfinite words and
width/version failures. `kernel_precision_check.py` checks the signed-zero mean
control (`00000000` versus `80000000` after deleting the initial addition) and
admitted binary64 epsilon `0000000000000001` converting to FP32 zero and yielding
a NaN singleton output. These are source/representation checks. No GPU execution
or independent native-lowering conformance experiment was run.

`Correct.lean` exposes the three independent mathematical stage specifications.
`FPEquiv.lean` now exposes actual `original ≡[[]] candidate` results for all three
retained per-program stage scopes. No new host-composition requirement is added;
optional complete Tensor/composed/native applications retain their prior scope.
The grouped pair supports arbitrary group steps and block coverage, and the final
pair arbitrary BM/BN, preserving the broader native source input scopes. Forward
retains its original positive-width/single-block admission. Both backward pairs
prove full-state execution equality and actual trace-safety equivalence for every
reached loop state. No experimental numerical relation is used.

The campaign checks all three original complete source ASTs by `rfl`, the direct
forward candidate against its old unroll AST, all candidate Python changes and
unchanged wrapper/guards, the existing CPU source checker and precision controls,
and joint main/optional Lean entries. No GPU execution or speedup is claimed.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/ForwardNativeObservation.lean](Misc/ForwardNativeObservation.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
