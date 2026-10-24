import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.BodyExecution

namespace VeriTile.Bench.Repairs.DequantizeRowwise.KernelCorrectness
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DequantizeRowwise
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Independent rowwise dequantization formula, including negative weights
and zero or negative scales. -/
def dequantized (scale : ℝ) (weight : Int) : ℝ := scale * (weight : ℝ) / 127

theorem source_memory (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) (u : BlockState) :
    (exec (source x scale out inv n block lanes).eraseDType.toAlgKernel u).map (·.mem) =
      (exec (RealReference.dequantize_rowwise_kernel x scale out inv n block lanes).toAlgKernel u).map (·.mem) :=
  BodyExecution.source_memory x scale out inv n block lanes u

def Result (p : Parameters) (u v : BlockState) : Prop :=
  (∀ column, column < p.cols →
    v.readMem outRegion (u.pid * p.cols + column) =
      dequantized (u.readMem scaleRegion u.pid)
        (u.readMemValue .int xRegion (u.pid * p.cols + column))) ∧
  (∀ r o, (r ≠ outRegion ∨ ∀ column, column < p.cols → o ≠ u.pid * p.cols + column) →
    v.mem r o = u.mem r o)

/-- Complete single-program execution and regional memory frame. Input
regions are read independently, so their backing views may overlap. -/
theorem correct (p : Parameters) (u : BlockState) :
    ∃ v, exec (kernel p).eraseDType.toAlgKernel u = some v ∧ Result p u v := by
  obtain ⟨v, he, ho, hf⟩ := RealReference.dequantize_rowwise_kernel_region_run
    xRegion scaleRegion outRegion (1 / 127) (p.rows * p.cols) p.cols p.width u
    (fun j => u.readMemValue .int xRegion (u.pid * p.cols + j.val))
    (fun _ => u.readMem scaleRegion u.pid) (by intros; rfl) (by intros; rfl)
  have memory := source_memory xRegion scaleRegion outRegion (1/127)
    (p.rows*p.cols) p.cols p.width u
  rw [he] at memory
  change (exec (kernel p).eraseDType.toAlgKernel u).map (·.mem) = some v.mem at memory
  cases ran : exec (kernel p).eraseDType.toAlgKernel u with
  | none => simp only [ran, Option.map_none, reduceCtorEq] at memory
  | some final =>
    have same : final.mem = v.mem := by simpa only [ran, Option.map_some, Option.some.injEq] using memory
    refine ⟨final, rfl, ?_, ?_⟩
    · intro column hc
      have h := ho ⟨column, lt_of_lt_of_le hc p.covers⟩ hc
      simpa only [BlockState.readMem, BlockState.readMemValue, same, dequantized,
        div_eq_mul_inv, one_mul] using h
    · intro r o miss
      rw [same]
      exact hf r o (miss.imp id (fun h j hj => h j.val hj))

/-- The original scale load is unmasked, including when the row has no
active output lane. Its capacity is therefore an independent scalar bound. -/
theorem traceSafe (p : Parameters) (bounds : RegionBounds) (u : BlockState)
    (weights : ∀ j : Fin p.width, j.val < p.cols →
      u.pid * p.cols + j.val < bounds xRegion)
    (scale : u.pid < bounds scaleRegion)
    (output : ∀ j : Fin p.width, j.val < p.cols →
      u.pid * p.cols + j.val < bounds outRegion) :
    Kernel.TraceSafe bounds (kernel p).eraseDType.toAlgKernel u := by
  unfold Kernel.TraceSafe
  simp [kernel, source, body, arguments, ComputeKernel.withScalarArguments,
    ScalarArgument.statement, ScalarArgument.fp32, ScalarArgument.integer,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeKernel.fromKernelBody,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, ComputeDType.eraseDType,
    FloatDType.toTileDType, VeriTile.Triton.eraseDType, eraseDType_float, Except.bind, bind,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, stepStmt, evalOp.eq_def,
    Tile.intToReal, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MemAccess.SafeAt,
    MaskOpt.Active, BlockState.setReg, -Nat.cast_add, -Nat.cast_mul,
    ← Int.natCast_add, ← Int.natCast_mul]
  exact ⟨weights, scale, output⟩

/-- Bind the typed INT8 row and the independently read FP32 scale to logical
inputs. Zero columns and read-only aliases retain the original regional scope. -/
theorem inputs_correct (p : Parameters) (u : BlockState)
    (weights : Fin p.width → Int) (scale : ℝ)
    (weightsRead : ∀ j : Fin p.width, j.val < p.cols →
      u.readMemValue .int xRegion (u.pid * p.cols + j.val) = weights j)
    (scaleRead : u.readMem scaleRegion u.pid = scale) :
    ∃ v, exec (kernel p).eraseDType.toAlgKernel u = some v ∧
      (∀ j : Fin p.width, j.val < p.cols →
        v.readMem outRegion (u.pid * p.cols + j.val) = dequantized scale (weights j)) ∧
      (∀ r o, (r ≠ outRegion ∨ ∀ j : Fin p.width, j.val < p.cols →
        o ≠ u.pid * p.cols + j.val) → v.mem r o = u.mem r o) := by
  obtain ⟨v, execution, values, frame⟩ := correct p u
  refine ⟨v, execution, ?_, ?_⟩
  · intro j active
    rw [values j.val active, scaleRead, weightsRead j active]
  · intro r o outside
    apply frame r o
    exact outside.imp id (fun outside j active =>
      outside ⟨j, lt_of_lt_of_le active p.covers⟩ active)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.KernelCorrectness
