import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import bench.tritonbench_g.fast_rms_layernorm.Proofs.ForwardRun
import bench.tritonbench_g.fast_rms_layernorm.Proofs.BackwardRun

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def update (r : RegionName) (o : Nat) (v : ℝ) (heap : RegionName → Nat → MemCell) :=
  fun q n => if q=r ∧ n=o then MemCell.real v else heap q n

def commit {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → ℝ)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (heap : RegionName → Nat → MemCell) :=
  indices.foldl (fun mem i => if active i then update r (off i) (val i) mem else mem) heap

theorem fold_mem {ι : Type} (r : RegionName) (off : ι → Nat) (val : ι → ℝ)
    (active : ι → Prop) [DecidablePred active] (indices : List ι) (u : BlockState) :
    (indices.foldl (fun s i => if active i then s.writeMem r (off i) (val i) else s) u).mem=
      commit r off val active indices u.mem := by
  induction indices generalizing u with
  | nil => rfl
  | cons i rest ih =>
    by_cases hi : active i <;> simp only [List.foldl_cons,hi,ite_true,ite_false]
    · rw [ih]
      simp [commit,List.foldl_cons,hi,BlockState.writeMem,update] <;> rfl
    · rw [ih]
      simp [commit,List.foldl_cons,hi]

/-- The copied Python body and the earlier algebraic presentation have the same
memory result. Register files intentionally differ: the source preserves signed
indices and every original assignment. This is an execution theorem, not an AST
identity or an assumption about a complete kernel. -/
theorem forward_memory (h : Call) (u : BlockState) :
    (exec (Kernels.forward h).eraseDType.toAlgKernel u).map BlockState.mem =
      (exec (ForwardRun.original h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps) u).map BlockState.mem := by
  cases mode : h.gemma
  all_goals
    simp [Kernels.forward,Kernels.regularSource,Kernels.gemmaSource,TypedSource.forward,TypedSource.gemma,ForwardRun.original,mode,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,BlockState.readMem,fold_mem]
    <;> simp [-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add]
    <;> simp [fold_mem,BlockState.writeMem,update]
    <;> rfl

theorem backward_memory (h : Call) (u : BlockState) :
    (exec (Kernels.backward h).eraseDType.toAlgKernel u).map BlockState.mem =
      (exec (BackwardRun.original h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps) u).map BlockState.mem := by
  cases mode : h.gemma
  all_goals
    simp [Kernels.backward,Kernels.inputGradientSource,Kernels.inputGradientBody,TypedSource.backward,BackwardRun.original,mode,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,BlockState.readMem,fold_mem]
    <;> simp [-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add]
    <;> simp [fold_mem,BlockState.writeMem,update]
    <;> rfl

#axiomsClean forward_memory
#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealSource
