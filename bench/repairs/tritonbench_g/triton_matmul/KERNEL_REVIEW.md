# Complete kernel revision

Canonical source: `triton_matmul_repaired.py`, SHA256
`ae92db50d9682c192a9404f75686f30dd02e9881d6fd440d00dce82ccd68c84f`.
This is the ledger's current `contract_revision.source`; the older accepted
repair hash and benchmark original are retained history. Python is unchanged.

The independent Real specification is the full-K matrix product on arbitrary
Real values at the original physical A/B element addresses. The native target
uses the original padded source blocks, fixed K16 half or K32 E4M3 MMA groups,
live FP32 carry, and original FP16/E4M3 output encoding. Native equality to Real
is not claimed. No numerical transformation or new primitive experiment is
introduced by this revision.

## Domain and precision

The existing checked domain remains unchanged: matching strided FP16/E4M3FN,
same CUDA device, rank two, matching inner dimensions, contiguous resolved
views, signed64 metadata and byte envelopes, signed32 program-count envelope,
and aligned representable nonempty input byte windows. A/B aliases, storage
offsets, Tensor subclasses admitted by `isinstance`, and unused singleton
strides remain allowed. There is no finite-value restriction. M=0 or N=0 runs
no program and does not need input pointer-window checks. K=0 with nonempty C
runs the normal grid, loads no A/B entries and stores the cast initial zero.

FP16 uses (BM, BN, BK, group, stages, warps)=(128,256,64,8,3,8).
E4M3FN uses (128,256,128,8,4,8). The source has no autotuner. Canonical runtime
strides are (K,1,N,1,max(N,1),1). Scalar widening, quotient/remainder ceilings,
clamped row/column/K addends, grouped ownership, full K loop, live dotAcc and
final cast/store are covered by the retained readable `Kernels.lean` and the
existing finite signed64 source correspondence. No source body is replaced by
a mathematical result.

## Public proof route

The selected thin public entries are `KernelOnly/Contract.lean`,
`KernelOnly/Kernels.lean`, `KernelOnly/Correct.lean` and `KernelOnly/FPEquiv.lean`.
They reuse the proof modules below and preserve the historical entry files.

- `KernelContract.Output`: existing destination metadata, actual Tensor laws,
  aligned representable window and physical separation from A/B. No allocator,
  returned-object identity or execution/result obligation.
- `KernelSource.tensorAPI`: derives the old finite-source ABI from these direct
  conditions and the original shared physical input reader. No A/B separation.
- `KernelSource.executes`: derives complete shared `FP.Structural.exec` grid
  success, complete finite-source grid success, whole-memory equality and the
  full output/frame result. `KernelSource.safety` derives address and finite
  index correspondence. `source_store_unique` proves grid/lane ownership.
- `KernelCorrect.kernel_real`: derives the independent full-K Real product;
  `KernelTensor.real_tensor` reads C through its actual physicalIndex/storage
  offset and provides the exterior physical element-address frame. The Real
  physical reader is an interpretation of completed finite stores, not a finite
  encoding of arbitrary Real values.
- `KernelNative.correct`: derives all of the above execution routes for the
  fixed instruction model and actual physical byte stores. The original input
  byte binding and initial C byte binding are ordinary memory ABI facts.
- `KernelFPEquiv.kernel_native_if_local`: compares the native model with an
  independent instruction reference through only four instruction-local fields.
  `KernelWhole.input_bytes` derives preservation of both original input views.

The broad variable-shape `PrimitiveLaws` exists only as an internal reuse
interface; `KernelRealModel.ideal_primitives` and `Native.primitives` discharge it. Public
entries assume no successful source run, expected output, whole-dot equality,
whole `CompilerBinding`, allocator execution or returned-object proof.
Existing host modules are imported for definitions and reusable lemmas; their
whole-host correctness theorems are not used to discharge the new kernel results.

## Validation and classification

The existing checker passed 26 preallocation rejection cases, 18 actual Triton
CPU-interpreter calls, source/profile binding and byte/index boundary controls.
Actual JIT binder/signature packing plus offline SM90 compilation passed six
cases: both dtypes at (M,N,K)=(1,1,0),(1,1,1),(129,257,65). Runtime unit values
were specialized by Triton's real binder, confirming source `tl.cast` handles
the singleton specialization. These are structural/compiler controls, not
GPU numerical conformance experiments. CUDA is unavailable.

All preexisting 51 task files are preserved. Fourteen new kernel modules
(including four thin public entries) and 25
unchanged task dependencies are built with the approved workspace-local driver.
The latter rebuild repairs a stale `FPTypedSource` cache that predates its
existing definition extraction into `SourceDefinitions`; only the affected
task dependency closure is rebuilt. This rebuild also exposed an external
`context_attn_fwd.FPModel` cache built against the old `Algebra` constructor:
it lacks the current `fma` field, and its identity cast does not reduce. The
failed historical proof build and a read-only `#print`/`rfl` probe are retained.
The shared package and all historical sources/conditional results are preserved.

With root approval, `KernelRealModel` constructs this task's ideal interpretation
against the current `Algebra` and proves its cast, dot and full-K sum facts.
`KernelRealInputs` supplies the original physical input and independent target
definitions. The current `KernelOnly` public roots and their complete effective dependency
chain exclude the bad shared model and the historical Real/defect routes that
used it. This is a task-local isolation; no claim is made that the old shared
cache or old host Real routes have been repaired. Current receipts, source/import
closure, standard-axiom and noncircular audits, compiler artifacts and source
pins are frozen in the task delivery packet. Core, Lake and ledger are unchanged.
