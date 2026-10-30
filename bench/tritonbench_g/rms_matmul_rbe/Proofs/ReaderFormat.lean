import bench.tritonbench_g.rms_matmul_rbe.IO
import bench.tritonbench_g.rms_matmul_rbe.Proofs.HeadlineMath

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.ReaderFormat
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

/-- The named row is exactly the source-precision-decoded row, at every axis. -/
theorem rowInputs_eq (fp8 : Bool) (p : Fin 3 → Arguments)
    (raw : ArrayValues ℝ (IO.inputs fp8 p)) (phase : Fin 3) (batch row : Nat) :
    Logical.rowInputs (Logical.inputEquiv fp8 p raw) phase batch row =
      Target.rowInput mathArithmetic fp8 p raw phase batch row := by
  unfold Logical.rowInputs Logical.inputEquiv Logical.arraysOfBacking
  simp only [Equiv.coe_fn_mk, Equiv.symm_apply_apply]
  cases fp8 <;> rfl

/-- The signed-byte domain is unchanged, including all unused backing offsets. -/
theorem domain_eq (fp8 : Bool) (p : Fin 3 → Arguments) :
    (fun raw => Logical.Domain fp8 (Logical.inputEquiv fp8 p raw)) = IO.InputDomain fp8 p := by
  cases fp8 <;> rfl

/-- Full target equality, including every typed scheduled output lane. -/
theorem formula_eq (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : HostScalar)
    (order : List (Nat → Nat))
    (g : RmsMatmulRbe.Parameters ℝ → RmsMatmulRbe.RowInput ℝ → Bool → Nat → Nat → ℝ) :
    (Logical.qkvIO fp8 p epsilon theta order).lift
      (fun xs phase launch rowLane colLane =>
        g (parameters (p phase) epsilon.value theta.value)
          (Logical.rowInputs xs phase (Logical.batch order launch)
            (Logical.row p order phase launch rowLane))
          (decide (phase.val < 2)) (Logical.row p order phase launch rowLane)
          (Logical.column p order phase launch colLane)) =
    (fun raw => Target.result fp8 p order (fun e =>
      g (parameters (p e.phase) epsilon.value theta.value)
        (Target.rowInput mathArithmetic fp8 p raw e.phase
          (order.get e.launch 0) (Target.row p e))
        (decide (e.phase.val < 2)) (Target.row p e) (Target.col p e))) := by
  funext raw
  apply (Logical.outputEquiv fp8 p order).injective
  change (Logical.outputEquiv fp8 p order)
    ((Logical.outputEquiv fp8 p order).symm _) = _
  rw [Equiv.apply_symm_apply]
  funext phase launch rowLane colLane
  dsimp only [Logical.qkvIO]
  rw [rowInputs_eq]
  cases fp8 <;> fin_cases phase <;> rfl

/-- Equivalence of the complete old and new judgments, including domain,
source execution, safety, winner observation and exterior memory frame. -/
theorem statement_iff (fp8 : Bool) (p : Fin 3 → Arguments) (epsilon theta : HostScalar)
    (order : List (Nat → Nat))
    (g : RmsMatmulRbe.Parameters ℝ → RmsMatmulRbe.RowInput ℝ → Bool → Nat → Nat → ℝ) :
    (Logical.qkvIO fp8 p epsilon theta order ⊨{Logical.Domain fp8}
      fun xs phase launch rowLane colLane =>
        g (parameters (p phase) epsilon.value theta.value)
          (Logical.rowInputs xs phase (Logical.batch order launch)
            (Logical.row p order phase launch rowLane))
          (decide (phase.val < 2)) (Logical.row p order phase launch rowLane)
          (Logical.column p order phase launch colLane)) ↔
    (IO.qkvIO mathArithmetic Bool.false fp8 p epsilon.value theta.value order).ImplementsOn
      (IO.InputDomain fp8 p) (fun raw => Target.result fp8 p order (fun e =>
        g (parameters (p e.phase) epsilon.value theta.value)
          (Target.rowInput mathArithmetic fp8 p raw e.phase
            (order.get e.launch 0) (Target.row p e))
          (decide (e.phase.val < 2)) (Target.row p e) (Target.col p e))) := by
  change (IO.qkvIO mathArithmetic Bool.false fp8 p epsilon.value theta.value order).ImplementsOn
    (fun raw => Logical.Domain fp8 (Logical.inputEquiv fp8 p raw)) _ ↔ _
  exact iff_of_eq (congrArg₂
    (fun domain target =>
      (IO.qkvIO mathArithmetic Bool.false fp8 p epsilon.value theta.value order).ImplementsOn domain target)
    (domain_eq fp8 p) (formula_eq fp8 p epsilon theta order g))

end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.ReaderFormat
