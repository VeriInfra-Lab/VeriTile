# IV-dependent matmul: explicit launched-dot shape candidate

This sibling preserves the original benchmark, the selected i64 repair and its
accepted BK=1 compiler-defect packet. It adds one preallocation condition:
`M != 0 and N != 0` requires `BLOCK_SIZE_K >= 16`. Empty M/N calls still allocate
A, B and C in the original order and return C without submitting a kernel;
empty BK=1 calls remain admitted. Nonempty M/N with K=0 still submits the JIT,
so the necessary static dot shape applies there. All previous power-of-two,
size, signed-control, active-byte, grid, selector, dtype and CUDA checks remain.

All five selectors remain: `pre_load`, `post_load`, `post_pre_mixed`,
`post_load_two_iters`, and `post_load_three_iters`. Stage counts remain 3/4.
The JIT AST is unchanged. Inputs, output allocation and accumulation remain
FP32; the source uses its default TF32 dot profile, explicitly narrows the
final accumulator to FP16, and widens that half into the FP32 output. There are
no additional dtype variants, payload restrictions or numerical input bounds.

`Contract.lean` proves new admission, original admission, the necessary launched
FP32 dot shape, preservation of every empty call, and rejection of the old live
counterexample. This is an explicit static shape repair. It is not a promise
that every otherwise admitted block fits every device's resources.

`Kernels.lean` contains ten complete ordinary DSL quotations with the exact pointer
statements for all five constexpr selectors. The source graph keeps literal
Dot+Add; the native graph binds the observed live-carry MMA lowering separately.
`Proofs/SourceBridge.lean` proves both graphs equal the retained schedule ASTs.
The natural index quotation is connected to the existing checked signed64 ABI,
pointer staging and active-address correspondence. The source checker erases
only those integer widening casts and makes destination widening explicit.

`Proofs/RealExpressions.lean` and `RealSchedules.lean` execute the complete shared
interpreter over arbitrary Real inputs, including every masked block, alias,
lookahead update and terminal store. `RealProjection.lean` connects that actual
execution to the finite-pointer implementation. `GridExecution.lean` composes
the actual quoted programs through the entire ordered grid. `Misc/HostExecution.lean`
executes checked allocation/submission dispatch and returns the actual allocated
C object with its actual source-computed state.

The independent mathematical target is the dense full-K sum. The independent
native target accepts only original input functions and dimensions, then folds
fixed K8 instruction atoms with the live FP32 carry, masked zero operands and
the terminal half/single conversions. The new public native profile represents
one emitted `mma.sync.m16n8k8.f32.tf32.tf32.f32` result lane on raw operand words.
Saved SM80 PTX contains no separate TF32 conversion, and none is inserted by
this profile. IEEE FP16 narrowing and FP32 widening are concrete library models.
The ISA leaf models a single fixed instruction; it cannot inspect dimensions,
rows, memory, source execution or final outputs.

The retained Tensor endpoints are `TensorBridge.matmul_checked_real` and
`TensorBridge.matmul_checked_native`. They prove successful shared quotation/grid
execution, the exact returned Tensor, metadata and every result against the
independent targets. The native theorem also executes the finite physical
four-byte stores and proves every output word/byte, data-pointer address bound,
original A/B allocation snapshot readback, input preservation and caller frame.
No equality between floating results and arbitrary Reals is assumed or claimed.

Ordinary successful allocation/submission, Tensor metadata/freshness and the
selected compiler's register routing into the stated ISA operation are explicit
implementation boundaries. The static BK minimum is separate from device and
resource feasibility. No kernel-success result, output equality, arbitrary-K
dot equality, FP associativity, or Dot+Add/fused numerical equality is supplied
as a premise. No basic arithmetic obligation is classified as a numerical
experiment TODO. No GPU run was performed.

Validation includes 120 actual-wrapper metadata guard cases, ten ordinary
specialization/offline SM80 compilations (five selectors, K=0/1, BM=BN=1,
BK=16), ten complete Python/DSL statement comparisons and six rejected source
mutation controls. Lean source/object receipts and the immutable review packet
are under `.veritile-cache/task-deliveries/iv_dependent_matmul_dot_shape/`.
That packet records the historical independent-review handoff. The current
selected kernel and program-pair interfaces are described below; the old
full-domain defect stays valid.

## Kernel and Tensor interfaces

`Correct.kernel_real` now states `kernelIO rest g ⊨ formula g` for the
independent full-K matrix product. `FPEquiv.kernel_native` now compares the
complete grids of two distinct existing pointer schedules:
`original rest profile g ≡[[]] pointerScheduled rest profile g schedule`.
`different` explicitly excludes self-comparison. `Programs.lean` binds both
views to the actual native quotations; `IO.scheduledVariant` changes only the
pointer schedule. The common live-carry TF32 K8 arithmetic, FP16 narrowing,
FP32 store and exact active input-cell contract are preserved. The proof
reuses both complete `IOResults.native` results, including access safety,
successful runs, output equality and both exterior frames. No numerical rule
or Dot+Add/FMA equality is introduced. `NativeCorrect.kernel_native` retains
the ordinary single-source `ProgramIO.ImplementsCells` theorem against the
independent TF32 K8 target.
`IO.lean` visibly lists the two
FP32 input channels, FP32 output, dense row/column addresses, no private scratch,
source identity and every selected launch. Empty K retains the original loop
and final casts; empty output grids perform no invocation.

`IO.Layout` contains only the original region separation, byte bases/alignment,
physical non-overlap and addressability. Launch bounds remain explicit. The
Real IO binds decoded FP32 reads from arbitrary initial memory, preserving the
historical independent Real snapshot domain. Native IO explicitly binds exact
active FP32 cells as required by the historical allocation theorem. Neither
path restricts inactive cells or adds numerical operand bounds. Both inputs may
share a region; their simultaneous bindings retain any alias consistency.

`Proofs/SourceSafety.lean` checks actual masked loads and the final store.
`GridSafety.lean` follows all five staging variants through their existing
execution invariants and actual dynamic-loop successors. `IOBinding.lean`
connects every logical window to these source-access checks, proves exact
schedule equality and reconstructs the original allocation conditions.
`FormulaCongruence.lean` connects only active matrix operands to the independent
Real/native targets; native zero-padding does not read outside the logical K
domain. `IOResults.lean` composes these facts with the retained full-grid results
in `KernelRealResult.lean` and `KernelNativeResult.lean`. The public readers have
one primary specification each and no raw state/runner/post-state interface.

`TensorBridge.matmul_checked_real` and `matmul_checked_native` retain the
complete previous checked Tensor statements, input snapshots, returned identity,
physical stores and byte frames. Both layers reuse `Kernel.real_grid` and
`Kernel.native_grid`; the existing target and physical-address correspondence
lemmas connect the raw outputs to the Tensor applications. No Python, quoted
kernel, checked contract, dtype or runtime ABI was changed by IO unification.
No new numerical assumption, GPU run or independent replay was added.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/HostExecution.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。

The 2026-10-21 source-only cleanup expands the former whole-source macro into
five complete pointer-schedule quotations for each of the source Dot+Add and
compiled live-carry graphs. `Kernel.quotation_eq` still proves by cases and
`rfl` that every complete quotation is exactly the retained original program
for arbitrary parameters. Public declarations, all FP32/FP16 boundaries, input
domains, masks, schedules and the existing main equivalence are unchanged.
This is source readability work; proof completion and acceptance are unchanged.
