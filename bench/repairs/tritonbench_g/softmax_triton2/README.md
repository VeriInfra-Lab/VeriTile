# Contiguous FP32 softmax kernel

The independent mathematical target in `Contract.reference` is
`exp(x[r,c]) / sum_j exp(x[r,j])`, over exactly the logical columns. The
canonical Python source is `softmax_triton2_repaired.py`; its floating-point
operations and public input guards are unchanged in this kernel revision.

The wrapper accepts strided, contiguous, rank-two FP32 CUDA tensors without an
unresolved negative-view bit. Nonempty inputs require a four-byte-aligned data
pointer, fewer than `2**31` rows, and `next_power_of_2(cols) <= 2**20`. Empty
axes return an empty allocation without a launch. Nonzero storage offsets,
singleton dimensions with unusual irrelevant strides, NaN/Inf words, signed
zeros and subnormals remain admitted. There is no finiteness check, input copy,
or new size restriction. The unchanged source passes canonical row width and
unit column strides, widens PID/lane indices to int64 before arithmetic, and
uses the original 4/8/16 warp selection.

`Kernels.lean` contains the readable typed Triton DSL source. It explicitly
marks FP32 regions and retains load, max, subtraction, exp, sum, division and
store order. Both `Contract.lean` and the independent public proof entries use
this source. `Proofs.Binding.source_identity` proves literal AST equality to
the existing reviewed `softmax_triton1` source, allowing reuse of its internal
row proofs. The native evaluator in `Proofs.Native` executes this task's source
directly. No sibling public correctness entry is imported.

The kernel-only results are:

- `Correct.softmax_kernel_correctness`: successful complete source execution,
  every active output, active access safety and every exterior flat-memory cell.
  The headline explicitly requires `0 < B` and `n ≤ B`, and displays a sum over
  exactly `Fin n`, rather than a padded sum over `Fin B`. `Fin.castLE` merely
  selects those logical columns from the input tile. The wrapper's existing
  `blockSize_bounds` already supplies coverage; no Python guard is added.
  `RealProof.masked_sum_eq_logical` proves the masked/logical sums equal,
  and `logical_formula_eq_reference` connects the displayed formula to
  `Contract.reference`. The previous general window theorem remains available
  as `RealProof.io_correct`, including when the tile does not cover `n`.
- `FPEquiv.softmax_kernel_fp32` in `Misc/NativeCorrect.lean`: optional single-source
  theorem using the shared `ProgramIO.Implements` judgment with
  source access safety, successful native execution, every owner-observed FP32
  output word and exact exterior cells. The primitive graph admits every word.
- `TensorBridge.softmax_checked_kernel_correctness`: every column of every valid
  row equals the independent unshifted Real formula, with the actual Tensor
  input storage and a frame over all other memory cells.
- `TensorBridge.softmax_checked_fp32`: successful typed source execution agrees with
  the independent ordered native graph at every output column; every full cell
  outside the active output row is preserved, including non-real cell tags.
- `TensorBridge.softmax_tensor_fp32`: the same result at the actual output Tensor's
  logical-to-physical indices, retaining storage offsets and the complete frame.
- The `TensorBridge.real_input_binding` and `TensorBridge.native_input_binding` results connect each active source load to
  the physical reader at the exact widened sequential pointer, the Tensor's
  logical value, four-byte alignment and the actual allocation bounds.

`Proofs.Binding.physical_index` also binds output indices to actual Tensor
storage. The reused `Contiguous.pointer_argument` identifies the source pointer
argument with `data_ptr`; `pointer_bounds` establishes the same address and
bounds for input and output objects; `mask_exact` covers the signed source mask.
Tensor storage laws and the physical reader are ordinary memory-model premises.
Masked lanes need no storage read or numerical value condition. Output contents
before execution are unconstrained. Kernel proofs require no allocation or
whole-wrapper execution theorem.

The retained single-source native target preserves production layout V1: register left folds,
simultaneous self-first warp butterflies, shared-memory warp aggregation,
subtraction, multiplication by raw FP32 `0x3fb8aa3b`, `ex2.approx`, and `div.full`.
It admits every FP32 bit pattern and retains exact negative-infinity padding.
Six raw-word primitive ports (`maxNum`, `subRN`, `mulRN`, `exp2Approx`, `addRN`,
`divFull`), compiler selection of this native graph, and ordinary memory/runtime
semantics are explicit model boundaries. They are not per-task numerical
experiments. No numerical rewrite is used inside these retained native theorems. The new
main comparison uses the reciprocal relation described below. The native
theorem does not assert bit equality with the mathematical Real target.

The earlier host, returned-object, whole-grid, byte-frame and optional scalar
port-transport results remain in `Proofs/LegacyCorrect.lean` and
`Proofs/LegacyFPEquiv.lean`, under `LegacyCorrect` and `LegacyFPEquiv` namespaces.
They are optional imports and their propositions are retained. The original
benchmark source remains unchanged, including its ignored column strides,
transposed-output defect and signed-int32 row-offset overflow.

Validation uses targeted Lean builds and standard-base axiom audits, plus
`python3 bench/repairs/tritonbench_g/softmax_triton2/check_contract.py`. The checker
covers source correspondence, unchanged guard order, 16 CPU Tensor admissions,
14 rejections, offsets/singletons, all pointer misalignments, empty out-of-storage
views, warp branches, large metadata, caller storage preservation and the
original transpose/stride-two/overflow counterexamples. CUDA launch/allocation
are mocked; no native GPU execution or new numerical conformance is claimed.

`IO.lean` holds the mathematical and native IO descriptions. `Correct.lean`
retains a thin named mathematical instance and displays the independent formula.
The native view retains arbitrary raw row/column strides, masks, explicit FP32
interpretation, exact launch setup, and per-column blocked-thread owner observation. The native
source interpreter uses temporary thread-valued nodes: input words are constant
nodes and output words are read at their actual owning thread. Execution remains
the fixed shared `Structural.exec`; no custom execution callback is supplied.

`Proofs/NativeIO.lean` composes the existing complete source row proof and native
graph-lowering proof with an actual-source access-safety proof. The optional native
theorem takes only raw geometry, row coverage and existing output injectivity,
and uses independent logical word arrays through the shared IO judgment. The
previous unrestricted raw-word evaluator result is retained as
`Proofs/KernelNative.complete`; Tensor applications and pointer/storage binding
results remain in `Misc/TensorBridge.lean`, with their exact prior statements.
The new regional native IO does not itself claim physical byte transport.

The format revision targeted-builds all edited/new Lean modules and the complete
Tensor consumer. Both public statement surface gates and independent formula
audits pass. It does not rerun the historical CPU contract suite, independent
native replay or GPU experiments.


Public FP pair, 2026-10-21: `FPEquiv.softmax_reciprocal_equiv` compares the
complete original source with `Kernels.guarded_reciprocal_kernel` under the
original per-launch typed IO, active bounds and logical input binding. The main
judgment concludes both source safety traces, both successful runs, every active
typed output cell equality and both exterior memory frames. The optional
`CheckedPair.checked_reciprocal_equiv` in `Misc/CheckedPair.lean` derives the count
width from the unchanged contract and retains the wrapper's empty or every-row
raw memory schedule comparison. The candidate Python entry is `softmax_reciprocal` in
the same source file. It performs one common scalar reciprocal followed by lane
multiplication when every computed numerator is finite and the computed
denominator is finite and at least one; every other row uses the original
ordinary division. This guard avoids the obvious reciprocal-overflow example
with a tiny denominator without excluding any input word or shape.

The source prefixes, padded reductions, output windows, casts and launch
selection are unchanged. `Proofs/GuardedReciprocal` proves every branch executes
successfully; `Proofs/Pair` connects actual source operands, runtime classifier
results and full-memory equality; `Proofs/PairSafety` retains the original active
access bounds in both branches; `Proofs/SourceIO` connects these safety results,
success, typed output agreement and both frames in the main judgment; `Proofs/GridPair` composes actual predecessor
memories with fresh per-launch registers across every requested row. The
runtime finitef and FP32 order callbacks have explicit exact `Ports` bindings
and soundness conditions; these are model implementation bindings, not caller
finiteness guards or numerical whole-program assumptions.

The sole numerical rewrite used is the already admitted FP32 `DIV-MUL-RCP`
relation, with FP32 input/compute/output and finite numerator, finite nonzero
denominator. Its existing supplemental report is
`summary.json#fp32/DIV-MUL-RCP` (snapshot
`9b01836a349a09e938f62e813c56a99aed2d0e39bf801fb5b050d57000b5604c`). The
runtime guard proves a subset of that rule's operand domain. No new unvalidated
numerical relation or experiment deferral is introduced. The theorem remains
conditional on this numerical rule set and its evidence; finite-sample
admission is not a universal IEEE bit-equality proof. Existing optional native
results retain their all-word scope and do not serve as this program pair.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/RealApplication.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。

`Misc/RealApplication.lean` retains `RealProof.access` and the optional
`RealProof.checked` Tensor application, with their complete names, statements
and proofs. The required row IO proof remains in `Proofs/Real.lean`; the main
entries do not import the optional application.
