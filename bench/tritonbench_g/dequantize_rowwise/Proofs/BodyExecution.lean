import bench.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.Mathematical_RealReference
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.Repairs.DequantizeRowwise.BodyExecution
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DequantizeRowwise
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private theorem masked_memory_congr {ι : Type} (work : List ι) (s t : BlockState)
    (region : RegionName) (offset : ι → Nat) (value : ι → ℝ) (active : ι → Prop)
    [DecidablePred active] (same : s.mem = t.mem) :
    (work.foldl (fun acc i => if active i then acc.writeMem region (offset i) (value i) else acc) s).mem =
    (work.foldl (fun acc i => if active i then acc.writeMem region (offset i) (value i) else acc) t).mem := by
  induction work generalizing s t with
  | nil => exact same
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases h : active i
    · simp only [if_pos h]
      apply ih
      funext r o
      simp only [BlockState.writeMem_mem]
      rw [same]
    · simp only [if_neg h]
      exact ih s t same

/-- The retained casts and runtime binding have the same regional Real memory
result as the independent natural-index execution, including failed execution. -/
theorem source_memory (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (u : BlockState) :
    (exec (source x scale out inv n block lanes).eraseDType.toAlgKernel u).map (·.mem) =
      (exec (RealReference.dequantize_rowwise_kernel x scale out inv n block lanes).toAlgKernel u).map (·.mem) := by
  simp [source, body, arguments, ComputeKernel.withScalarArguments, ScalarArgument.statement,
    ScalarArgument.fp32, ScalarArgument.integer,
    RealReference.dequantize_rowwise_kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeKernel.fromKernelBody, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    ComputeDType.eraseDType, FloatDType.toTileDType, VeriTile.Triton.eraseDType,
    exec, stepStmts, stepStmt, evalOp.eq_def, Tile.intToReal, Tile.bop, Tile.cop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Except.bind, bind, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  apply masked_memory_congr
  rfl

#axiomsClean source_memory
end
end VeriTile.Bench.Repairs.DequantizeRowwise.BodyExecution
