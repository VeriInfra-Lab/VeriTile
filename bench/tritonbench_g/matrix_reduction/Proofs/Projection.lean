import bench.tritonbench_g.matrix_reduction.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixReduction
set_option linter.unusedSimpArgs false
open VeriTile.Triton

/-- Four algorithm statements used only to decompose the actual erased source
in the execution proofs. `real_projection` checks this view against the quotation. -/
def realBody (X Y : RegionName) (sm sn M N : Nat) : List Stmt :=
  [ .assign .blockPtr [M,N] "x_ptr"
      (.makeBlockPtrDyn X (.constNat 0) [M,N] [M,N] [sm,sn] [0,0]),
    .assign .real [M,N] "x" (.load .real
      (.blockPtr (.ref .blockPtr [M,N] "x_ptr") []) .none .unspecified),
    .assign .real [M] "y" (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [M,N] "x")),
    .store .real [M] (.region Y (.arange M))
      (.ref .real [M] "y") .none ]

theorem real_projection (X Y : RegionName) (sm sn sy M N : Nat) :
    (load_reduce_kernel X Y sm sn sy M N).toAlgorithm? =
      .ok { inputs := [X], outputs := [Y], body := realBody X Y sm sn M N } := by
  simp [load_reduce_kernel, TypedSource.originalForKernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    realBody, Except.bind, bind, pure]
  all_goals repeat' constructor
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType]
#axiomsClean real_projection
end VeriTile.Bench.TritonBenchG.MatrixReduction
