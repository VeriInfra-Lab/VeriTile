import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.RuntimeReal
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.LegacySourcePreservation
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.ActualGrid
import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.RuntimeObserved

/-! The complete current source is executed directly in ActualRow/ActualGrid.
Historical literal/cast identities remain explicitly in LegacyPastedSurface.
The caller supplies the epsilon value, not a synthetic statement in the body. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton.SourcePreservation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Every memory cell agrees after moving the earlier scalar ABI conversion
from the body into its actual call argument. This connection uses only
the exact Real interpretation of the integer argument and floating values. -/
theorem real_row_memory (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State ℝ) :
    (ActualRow.sourceRun realAlgebra c (ActualRow.argument realAlgebra eps) pid s).val.mem =
      (Whole.sourceRun realAlgebra c eps pid s).val.mem := by
  funext region a
  by_cases same : region = c.Y
  · subst region
    by_cases active : pid*c.N ≤ a ∧ a < pid*c.N+c.N
    · have small : a-pid*c.N < c.N := by omega
      let i : Fin c.B := ⟨a-pid*c.N,small.trans_le c.complete⟩
      have eq : a=pid*c.N+i.val := by dsimp [i]; omega
      rw [eq,ActualRow.row_readback realAlgebra c (ActualRow.argument realAlgebra eps) pid s i small,
        Whole.row_readback realAlgebra c eps pid s i small,RuntimeReal.rowValue_argument]
    · have outside : Whole.Outside c pid c.Y a := by
        right
        intro i hi equal
        apply active
        rw [equal]
        exact ⟨Nat.le_add_right _ _,Nat.add_lt_add_left hi _⟩
      rw [ActualRow.row_frame realAlgebra c (ActualRow.argument realAlgebra eps) pid s c.Y a outside,
        Whole.row_frame realAlgebra c eps pid s c.Y a outside]
  · rw [ActualRow.row_frame realAlgebra c (ActualRow.argument realAlgebra eps) pid s region a (Or.inl same),
      Whole.row_frame realAlgebra c eps pid s region a (Or.inl same)]

/-- Both old/new complete call executions succeed and have the same full
memory observation; the actual call also initializes its scalar register. -/
theorem real_row_execution (c : Layout) (eps : ℝ)
    (pid : Nat) (s : State ℝ) :
    (run (FP.ProfiledRewrite.engine realAlgebra) (ActualRow.source c).surfaceBody
      (ActualRow.entry (ActualRow.argument realAlgebra eps) pid s)).map State.mem =
    (run (FP.ProfiledRewrite.engine realAlgebra) (c.source eps).surfaceBody (rowState pid s)).map State.mem := by
  rw [(ActualRow.sourceRun realAlgebra c (ActualRow.argument realAlgebra eps) pid s).property,
    (Whole.sourceRun realAlgebra c eps pid s).property]
  exact congrArg some (real_row_memory c eps pid s)

/-- The current graph retains runtime epsilon and integer N as distinct
source operands and lowers to the corresponding independent recurrence. -/
theorem native_row_observation (M : Observed.NativeModel α)
    (profile : Observed.ReductionFamily) (c : Layout)
    (xs ws : Nat → α) (i : Nat) :
    Observed.interpret M profile i (ActualRow.rowValue Observed.graphAlgebra c
      (.parameter "eps") (fun j => .input (xs j)) (fun j => .input (ws j)) i) =
      RuntimeObserved.reference M profile c.N c.B xs ws i :=
  RuntimeObserved.source_graph_lowering M profile c xs ws i

#axiomsClean real_row_memory
#axiomsClean real_row_execution
#axiomsClean native_row_observation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.SourcePreservation
